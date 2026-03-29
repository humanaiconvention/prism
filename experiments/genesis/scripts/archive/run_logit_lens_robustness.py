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
    """Calculate Shannon Effective Rank."""
    p = distribution[distribution > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

class LogitLensHook:
    def __init__(self, target_layers):
        self.target_layers = target_layers
        self.handles = []
        self.captured_acts = {layer: [] for layer in target_layers}

    def _make_hook(self, layer_idx):
        def hook_fn(module, inp):
            x = inp[0]
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

def main():
    print("Loading Genesis-152M Logit Lens Robustness Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    
    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_list = json.load(f)["prompts"]
        
    # Take 10 math and 10 creative prompts
    test_prompts = [p["text"] if isinstance(p, dict) else p for p in prompts_list[0:10] + prompts_list[10:20]]
    
    # We want to find the precise layer where ER drops below 40 words consistently
    # Let's test layers 10 through 19
    target_layers = list(range(10, 20))
    hook = LogitLensHook(target_layers)
    hook.attach(model)
    
    W_U = model.lm_head.weight
    norm_f = model.ln_f
    
    crossover_layers = []
    
    THRESHOLD = 40.0 # Define a crossover threshold (e.g. dropping into a committed vocabulary subspace)
    
    for prompt_idx, prompt in enumerate(tqdm(test_prompts, desc="Testing Prompts")):
        # Clear specific hook states for this prompt run
        for layer in target_layers:
            hook.captured_acts[layer] = []
            
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=16, temperature=1.0)
            
        # Analyze layers for this sequence
        prompt_crossover = None
        for layer in target_layers:
            acts = np.concatenate(hook.captured_acts[layer], axis=0) # [16 steps, d_model]
            acts_tensor = torch.tensor(acts, device=device, dtype=torch.float32)
            acts_normed = norm_f(acts_tensor)
            logits = torch.matmul(acts_normed, W_U.T)
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
            
            ers = [calculate_shannon_er(p) for p in probs]
            layer_er = np.mean(ers)
            
            if layer_er < THRESHOLD and prompt_crossover is None:
                prompt_crossover = layer
                break # Found crossover
                
        if prompt_crossover is not None:
            crossover_layers.append(prompt_crossover)
        else:
            crossover_layers.append(20) # Occurs late
            
    hook.remove()
    
    mean_cross = np.mean(crossover_layers)
    std_cross = np.std(crossover_layers)
    
    print(f"\n--- Logit Lens Robustness Results (N=20 prompts) ---")
    print(f"Mean Crossover Layer (vocab dropping below {THRESHOLD} words): {mean_cross:.1f} ± {std_cross:.1f}")
    
    for i in range(10, 21):
        count = crossover_layers.count(i)
        if count > 0:
            print(f"  Layer {i}: {count} prompts")
            
    with open("measurements/logit_lens_robustness.txt", "w") as f:
        f.write("Logit Lens Lexical Crossover Robustness (N=20)\n")
        f.write("==============================================\n")
        f.write(f"Mean Crossover Layer: L{mean_cross:.1f} ± {std_cross:.1f}\n\n")
        f.write("Distribution:\n")
        for i in range(10, 21):
            count = crossover_layers.count(i)
            if count > 0:
                f.write(f"  Layer {i}: {count} prompts\n")
                
if __name__ == "__main__":
    main()
