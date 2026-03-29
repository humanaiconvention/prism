"""Phase 7D: Pythia-160M Cross-Architecture ER Replication.

Applies the Layer Parity and Oscillatory testing to EleutherAI/pythia-160m.
If the 4x sub-block oscillation is a universal property of transformers, 
we should see oscillating Shannon ER. If it's specific to the Genesis
architecture (GLA/FoX block rotation), Pythia will NOT oscillate.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class WelfordCovariance:
    def __init__(self, d):
        self.d = d
        self.n = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros((d, d), dtype=np.float64)
    
    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def get_shannon_er(self):
        if self.n < 2:
            return 0.0
        cov = self.M2 / (self.n - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) == 0:
            return 0.0
        p = eigvals / eigvals.sum()
        H = -np.sum(p * np.log(p))
        return np.exp(H)

class PythiaHook:
    """Hooks Pythia layers to extract pre-attn (post-norm), post-attn, and post-ffn states."""
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.data = {}
        self.handles = []
        
    def _make_hook(self, layer_idx, point_name):
        def hook_fn(module, inp, output):
            x = output[0] if isinstance(output, tuple) else output
            if x is not None and x.dim() == 3:
                self.data[(layer_idx, point_name)] = x[:, -1, :].detach().float().cpu().numpy()
        return hook_fn
        
    def _make_pre_hook(self, layer_idx, point_name):
        def pre_hook_fn(module, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x is not None and x.dim() == 3:
                self.data[(layer_idx, point_name)] = x[:, -1, :].detach().float().cpu().numpy()
        return pre_hook_fn
        
    def attach(self, model):
        for i, layer in enumerate(model.gpt_neox.layers):
            # Input to the layer happens before any processing
            self.handles.append(layer.register_forward_pre_hook(self._make_pre_hook(i, 'block_input')))
            
            # Post-Attention (Pythia has parallel attention and FFN, so post_attn is the output of the attention block)
            self.handles.append(layer.attention.register_forward_hook(self._make_hook(i, 'post_attn')))
            
            # Post-FFN
            self.handles.append(layer.mlp.register_forward_hook(self._make_hook(i, 'post_ffn')))
            
            # Block output
            self.handles.append(layer.register_forward_hook(self._make_hook(i, 'block_output')))
            
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        
    def clear(self):
        self.data.clear()

def main():
    print("Loading EleutherAI/pythia-160m for Cross-Archive Replication...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").to(device)
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size # 768 for pythia-160m
    
    # Init Welford for 4 measurement points across all 12 layers
    points = ['block_input', 'post_attn', 'post_ffn', 'block_output']
    accum = {(i, pt): WelfordCovariance(d) for i in range(n_layers) for pt in points}
    
    with open("prompts/prompts_60.json", "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
        
    hooks = PythiaHook(n_layers)
    hooks.attach(model)
    
    max_tokens = 64
    total_samples = 0
    
    print("\nProcessing Pythia ER...")
    for pidx, prompt_data in enumerate(tqdm(prompts)):
        text = prompt_data.get("text", "") if isinstance(prompt_data, dict) else prompt_data
        
        # We don't use ChatML for raw pythia since it's a base model, we just feed it the prompt
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        current_ids = input_ids
        past_kv = None
        
        for step in range(max_tokens):
            hooks.clear()
            with torch.no_grad():
                if past_kv is not None:
                    out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    out = model(current_ids, use_cache=True)
                    
            past_kv = out.past_key_values
            
            for i in range(n_layers):
                for pt in points:
                    if (i, pt) in hooks.data:
                        accum[(i, pt)].update(hooks.data[(i, pt)].squeeze())
            
            total_samples += 1
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_id], dim=1)
            
            if next_id.item() == tokenizer.eos_token_id:
                break
                
    hooks.remove()
    
    print("\n" + "="*70)
    print("PHASE 7D: PYTHIA-160M CROSS-ARCHITECTURE REPLICATION")
    print("="*70)
    
    print(f"\n{'Layer':<8} | {'Block Out ER':>15} | {'Delta vs Prev':>15}")
    print("-" * 45)
    prev_er = 0
    for i in range(n_layers):
        er = accum[(i, 'block_output')].get_shannon_er()
        delta = er - prev_er if i > 0 else 0
        
        # Check if the delta swings wildly up/down like Genesis
        print(f"L{i:<7} | {er:>15.1f} | {delta:>15.1f}")
        prev_er = er
        
    # Test for Periodicity
    deltas = []
    for i in range(1, n_layers):
        er1 = accum[(i-1, 'block_output')].get_shannon_er()
        er2 = accum[(i, 'block_output')].get_shannon_er()
        deltas.append(er2 - er1)
        
    oscillations = 0
    for i in range(1, len(deltas)):
        if np.sign(deltas[i]) != np.sign(deltas[i-1]):
            oscillations += 1
            
    print(f"\nSign Reversals in Delta (Oscillation Metric): {oscillations} / {len(deltas)-1}")
    if oscillations > len(deltas) * 0.6:
        print("Conclusion: STRONG OSCILLATION DETECTED. The Genesis Phase 6 finding extends to standard Transformers.")
    else:
        print("Conclusion: NO OSCILLATION EXPECTED. The Genesis Phase 6 finding is architecturally induced by the 4x GLA/FoX blocks.")

    os.makedirs("logs", exist_ok=True)
    with open("logs/pythia_replication.txt", "w") as f:
        f.write(f"Pythia-160M Block Output Effective Ranks (N={total_samples}):\n")
        for i in range(n_layers):
            er = accum[(i, 'block_output')].get_shannon_er()
            f.write(f"L{i}: {er:.2f}\n")
        f.write(f"Oscillations: {oscillations}\n")
        
if __name__ == "__main__":
    main()
