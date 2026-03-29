import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

def get_shannon_er(cov):
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.0
    p = eigvals / eigvals.sum()
    H = -torch.sum(p * torch.log(p))
    return torch.exp(H).item()

class BlockOutputHook:
    def __init__(self, target_layers):
        self.target_layers = target_layers
        self.data = {l: None for l in target_layers}
        self.handles = []
    
    def _make_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            x = output[0] if isinstance(output, tuple) else output
            if x is not None and x.dim() == 3:
                self.data[layer_idx] = x.detach().float().cpu()
        return hook_fn
    
    def attach(self, model):
        for i, block in enumerate(model.blocks):
            if i in self.target_layers:
                h = block.register_forward_hook(self._make_hook(i))
                self.handles.append(h)
                
    def clear(self):
        for l in self.target_layers:
            self.data[l] = None
            
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

def generate_random_prompt(length):
    """Fallback since we don't have natural 1024-token prompts prepared."""
    vocab_size = 50257 # genesis vocab
    # exclude special tokens
    return torch.randint(100, vocab_size-100, (1, length))

def main():
    print("Loading Genesis-152M for Phase 7E: Long-Context GLA Saturation (T=1024)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    
    target_layers = [0, 14, 28] # L0 = entry, L14 = collapsed core, L28 = exit
    hooks = BlockOutputHook(target_layers)
    hooks.attach(model)
    
    max_tokens = 1024 # Testing beyond the 64-256 limit of Phase 4
    num_prompts = 20 # Can't do 100 on 1024 length without taking all day
    
    # We will compute the ER at specific time steps: T=64, 128, 256, 512, 1024
    time_steps = [1, 16, 64, 128, 256, 512, 1024]
    
    # Storage for states
    # Map: T -> Layer -> list_of_vectors (1 per prompt)
    states_at_T = {t: {l: [] for l in target_layers} for t in time_steps}
    
    print(f"\nEvaluating {num_prompts} random {max_tokens}-length sequences...")
    torch.manual_seed(42)
    
    for pidx in tqdm(range(num_prompts)):
        input_ids = generate_random_prompt(max_tokens).to(device)
        
        hooks.clear()
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
            
        for l in target_layers:
            # hooks.data[l] shape is (1, seq_len, dim)
            hidden_states = hooks.data[l][0] # (seq_len, dim)
            
            for t in time_steps:
                t_idx = min(t - 1, hidden_states.shape[0] - 1)
                states_at_T[t][l].append(hidden_states[t_idx])
                
    hooks.remove()
    
    print("\n" + "="*70)
    print(f"PHASE 7E: LONG-CONTEXT SATURATION (T={max_tokens})")
    print("="*70)
    
    os.makedirs("logs", exist_ok=True)
    out_csv = "logs/phase7_long_context_saturation.csv"
    
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TimeStep_T", "Layer_0_ER", "Layer_14_ER", "Layer_28_ER"])
        
        print(f"{'TimeStep':>10} | {'L0 ER':>10} | {'L14 ER':>10} | {'L28 ER':>10}")
        print("-" * 50)
        
        for t in time_steps:
            er_vals = []
            for l in target_layers:
                # Stack 20 states, shape (20, dim)
                X = torch.stack(states_at_T[t][l], dim=0)
                cov = torch.cov(X.T)
                er = get_shannon_er(cov)
                er_vals.append(er)
                
            writer.writerow([t, er_vals[0], er_vals[1], er_vals[2]])
            print(f"T={t:<8} | {er_vals[0]:>10.1f} | {er_vals[1]:>10.1f} | {er_vals[2]:>10.1f}")
            
    print(f"\nConclusion: Examine L14. Does it continue dropping, or settle into a steady-state limit cycle limit strictly above 0?")
    print(f"Data saved to: {out_csv}")

if __name__ == "__main__":
    main()
