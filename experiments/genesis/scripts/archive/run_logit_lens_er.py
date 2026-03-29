import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

def calculate_shannon_er(distribution):
    """Calculate Shannon Effective Rank given a discrete probability distribution."""
    # Add epsilon to avoid log(0)
    p = distribution[distribution > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

class LogitLensHook:
    def __init__(self, target_layers):
        self.target_layers = target_layers
        self.handles = []
        # Structure: {layer_idx: [tensor_sequence_1, tensor_sequence_2...]}
        self.captured_acts = {layer: [] for layer in target_layers}

    def _make_hook(self, layer_idx):
        def hook_fn(module, inp):
            # Capture the input to the layer block 
            # (which is the output of the previous block's residual add)
            x = inp[0]
            # Capture only the generation token(s). If it's prompt, we take the last token.
            # For simplicity let's capture the last token of whatever sequence is passed
            self.captured_acts[layer_idx].append(x[:, -1, :].detach().float().cpu().numpy())
            return None
        return hook_fn

    def attach(self, model):
        for layer in self.target_layers:
            self.handles.append(
                model.blocks[layer].register_forward_pre_hook(self._make_hook(layer))
            )

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

def get_category_prompts(prompts_list, category):
    if category == "Mathematical":
        subset = prompts_list[0:10] + prompts_list[60:90]
    elif category == "Creative":
        subset = prompts_list[10:20] + prompts_list[90:120]
    else:
        subset = []
    # Return 30 prompts per category to keep it fast
    return [p["text"] if isinstance(p, dict) else p for p in subset][:30]

def main():
    print("Loading Genesis-152M Logit Lens...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    
    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_data = json.load(f)["prompts"]
        
    math_prompts = get_category_prompts(prompts_data, "Mathematical")
    creative_prompts = get_category_prompts(prompts_data, "Creative")
    
    # We will sample representative layers spanning early, mid, late, and final.
    target_layers = [5, 10, 15, 20, 25, 29]
    hook = LogitLensHook(target_layers)
    hook.attach(model)
    
    # Extract unembedding matrix and final layernorm
    W_U = model.lm_head.weight # [vocab_size, d_model]
    norm_f = model.ln_f      # Final RMSNorm
    
    def process_prompts(prompts, desc):
        # Clear hooks
        for layer in target_layers:
            hook.captured_acts[layer] = []
            
        for prompt in tqdm(prompts, desc=desc):
            chat_input = format_chatml_prompt(prompt)
            input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
            
            # Generate 16 tokens to gather a representative sample of states
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=16, temperature=1.0)
                
        # Project collected states
        layer_vocab_ers = {}
        
        for layer in target_layers:
            # acts shape: [num_prompts * 16 steps, d_model]
            acts = np.concatenate(hook.captured_acts[layer], axis=0)
            acts_tensor = torch.tensor(acts, device=device, dtype=torch.float32)
            
            # Apply final norm 
            acts_normed = norm_f(acts_tensor)
            
            # Multiply by unembed to get logits
            logits = torch.matmul(acts_normed, W_U.T) # [N, vocab_size]
            
            # Softmax to get probs
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
            
            # Calculate ER per token, then mean
            ers = [calculate_shannon_er(p) for p in probs]
            layer_vocab_ers[layer] = np.mean(ers)
            
        return layer_vocab_ers

    print("\nExecuting Logit Lens over Math Prompts...")
    math_ers = process_prompts(math_prompts, "Math Domain")
    
    print("\nExecuting Logit Lens over Creative Prompts...")
    creative_ers = process_prompts(creative_prompts, "Creative Domain")
    
    hook.remove()
    
    print("\n--- Diagnostic Results (Vocabulary Effective Rank) ---")
    print("Layer\tMath Vocab ER\tCreative Vocab ER\tDiff (Creative - Math)")
    print("-" * 75)
    
    results_str = "L-Lens Vocab ER Strategy\n==========================\n"
    for layer in target_layers:
        m_er = math_ers[layer]
        c_er = creative_ers[layer]
        diff = c_er - m_er
        print(f"L{layer:02d}\t{m_er:8.1f}\t{c_er:12.1f}\t\t{diff:+8.1f}")
        results_str += f"L{layer:02d}: Math={m_er:.1f}, Creative={c_er:.1f}, Diff={diff:+.1f}\n"
        
    os.makedirs("measurements", exist_ok=True)
    with open("measurements/logit_lens_er.txt", "w") as f:
        f.write(results_str)
        
    print("\nSaved logit lens vocabulary ER metrics to measurements/logit_lens_er.txt")

if __name__ == "__main__":
    main()
