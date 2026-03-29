import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

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

def get_shannon_er(cov):
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 0.0
    p = eigvals / eigvals.sum()
    H = -torch.sum(p * torch.log(p))
    return torch.exp(H).item()

def main():
    print("Loading Genesis-152M for Bootstrapped Prompt-Type ER Measurement (Phase 7C-CI)...")
    device = "cuda"
    model, tokenizer, config = load_genesis_model(device=device)
    
    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_data = json.load(f)["prompts"]
        
    # Phase 3 mapping
    # Math: indices 0-9, 60-89
    # Creative: indices 10-19, 90-126
    
    math_indices = list(range(0, 10)) + list(range(60, 80))  # 30 prompts
    creative_indices = list(range(10, 20)) + list(range(90, 110)) # 30 prompts
    
    math_prompts = [prompts_data[i]["text"] if isinstance(prompts_data[i], dict) else prompts_data[i] for i in math_indices]
    creative_prompts = [prompts_data[i]["text"] if isinstance(prompts_data[i], dict) else prompts_data[i] for i in creative_indices]
    
    max_tokens = 64
    target_layers = [15, 29]
    hooks = BlockOutputHook(target_layers)
    hooks.attach(model)
    
    def extract_states(prompts_list, category_name):
        print(f"\nExtracting {category_name} hidden states...")
        state_blocks = {l: [] for l in target_layers}
        
        for prompt in tqdm(prompts_list):
            inp = format_chatml_prompt(prompt)
            input_ids = torch.tensor([tokenizer.encode(inp)], device=device)
            current_ids = input_ids
            past_kv = None
            
            prompt_states = {l: [] for l in target_layers}
            
            for step in range(max_tokens):
                hooks.clear()
                with torch.no_grad():
                    if past_kv is not None:
                        out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                    else:
                        out = model(current_ids, use_cache=True)
                    logits = out[0]
                    past_kv = out[3]
                    
                for l in target_layers:
                    prompt_states[l].append(hooks.data[l][0, -1, :]) # (dim,)
                    
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                current_ids = torch.cat([current_ids, torch.tensor([[next_id]], device=device)], dim=1)
                
            for l in target_layers:
                # Store as (max_tokens, dim) block for THIS prompt
                state_blocks[l].append(torch.stack(prompt_states[l], dim=0))
                
        return state_blocks
        
    math_data = extract_states(math_prompts, "Mathematical")
    creative_data = extract_states(creative_prompts, "Creative")
    
    hooks.remove()
    
    # Compute observed (non-resampled) ERs once for reporting.
    observed_results = {}
    for l in target_layers:
        X_math_obs = torch.cat(math_data[l], dim=0)
        X_creative_obs = torch.cat(creative_data[l], dim=0)

        cov_math_obs = torch.cov(X_math_obs.T)
        cov_creative_obs = torch.cov(X_creative_obs.T)

        observed_math = get_shannon_er(cov_math_obs)
        observed_creative = get_shannon_er(cov_creative_obs)
        observed_results[l] = {
            'math': observed_math,
            'creative': observed_creative,
            'diff': observed_creative - observed_math,
        }

    # Bootstrap
    n_bootstraps = 1000
    n_prompts = 30
    print(f"\nRunning {n_bootstraps} cluster-bootstrap resamples (resampling N=30 prompts with replacement).")
    
    bootstrap_results = {l: {'math': [], 'creative': [], 'diff': []} for l in target_layers}
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # We aggregate and compute covariance using torch.cov
    for l in target_layers:
        print(f"  Bootstrapping Layer {l}...")
        for b in tqdm(range(n_bootstraps), leave=False):
            # Sample prompt indices with replacement
            idx_math = np.random.choice(n_prompts, size=n_prompts, replace=True)
            idx_creative = np.random.choice(n_prompts, size=n_prompts, replace=True)
            
            # Aggregate blocks (64*30 = 1920 tokens x dim)
            X_math = torch.cat([math_data[l][i] for i in idx_math], dim=0)
            X_creative = torch.cat([creative_data[l][i] for i in idx_creative], dim=0)
            
            # Covariance and ER
            cov_math = torch.cov(X_math.T)
            cov_creative = torch.cov(X_creative.T)
            
            er_math = get_shannon_er(cov_math)
            er_crea = get_shannon_er(cov_creative)
            
            bootstrap_results[l]['math'].append(er_math)
            bootstrap_results[l]['creative'].append(er_crea)
            bootstrap_results[l]['diff'].append(er_crea - er_math)
            
    print("\n" + "="*70)
    print("PHASE 7C-CI: 1,000-ITERATION BOOTSTRAP RESAMPLING RESULTS")
    print("="*70)
    
    for l in target_layers:
        diffs = np.array(bootstrap_results[l]['diff'])
        crea = np.array(bootstrap_results[l]['creative'])
        math = np.array(bootstrap_results[l]['math'])
        obs = observed_results[l]
        
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        p_directional_creative_le_math = np.mean(diffs <= 0)
        p_two_tailed = min(1.0, 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
        
        print(f"\n[Layer {l}]")
        print(f"  Observed Math ER:     {obs['math']:.1f}")
        print(f"  Observed Creative ER: {obs['creative']:.1f}")
        print(f"  Observed ER Difference (Creative - Math): {obs['diff']:.1f}")
        print(f"  Math ER:     {math.mean():.1f} ± {math.std():.1f}")
        print(f"  Creative ER: {crea.mean():.1f} ± {crea.std():.1f}")
        print(f"  ER Difference (Creative - Math): {diffs.mean():.1f}")
        print(f"  95% Confidence Interval for Difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  Directional Probability P(Creative <= Math): {p_directional_creative_le_math:.4e}")
        print(f"  Two-tailed Bootstrap P-value vs 0 gap:       {p_two_tailed:.4e}")
        
        if ci_lower > 0:
            print("  Conclusion: SIGNIFICANT gap confirmed (Creative ER > Math ER).")
        elif ci_upper < 0:
            print("  Conclusion: SIGNIFICANT gap confirmed (Math ER > Creative ER).")
        else:
            print("  Conclusion: NOT SIGNIFICANT at 95% confidence level.")

    os.makedirs("logs", exist_ok=True)
    with open("logs/bootstrap_ci_results.log", "w") as f:
        for l in target_layers:
            diffs = np.array(bootstrap_results[l]['diff'])
            obs = observed_results[l]
            p_directional_creative_le_math = np.mean(diffs <= 0)
            p_two_tailed = min(1.0, 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
            f.write(f"Layer {l} Observed Math ER: {obs['math']:.2f}\n")
            f.write(f"Layer {l} Observed Creative ER: {obs['creative']:.2f}\n")
            f.write(f"Layer {l} Observed Diff (Creative - Math): {obs['diff']:.2f}\n")
            f.write(f"Layer {l} Mean Bootstrap Diff: {diffs.mean():.2f}\n")
            f.write(f"Layer {l} 95% CI: [{np.percentile(diffs, 2.5):.2f}, {np.percentile(diffs, 97.5):.2f}]\n")
            f.write(f"Layer {l} Directional P(Creative <= Math): {p_directional_creative_le_math:.4e}\n")
            f.write(f"Layer {l} Two-tailed P-value vs 0 gap: {p_two_tailed:.4e}\n\n")

if __name__ == "__main__":
    main()
