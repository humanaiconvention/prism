"""Experiment 4: L15 Semantic Compression Bottleneck
Tests whether Layer 15 causally constrains the representation used for token prediction.

Conditions:
1. Baseline: Normal greedy decoding.
2. Orthogonal Noise Injection: At L15, inject Gaussian noise orthogonal to the top-k PCs.
3. Bottleneck Disruption: Zero out the strongest compression head (L15-H3) o_proj.

Metrics: ER per layer, Token Entropy, KL Divergence, Perplexity.
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model, format_chatml_prompt, HiddenStateHook

class WelfordCovariance:
    def __init__(self, d):
        self.d = d
        self.n = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros((d, d), dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        for i in range(x.shape[0]):
            self.n += 1
            delta = x[i] - self.mean
            self.mean += delta / self.n
            delta2 = x[i] - self.mean
            self.M2 += np.outer(delta, delta2)

    def get_shannon_er(self):
        if self.n < 2: return 0.0
        cov = self.M2 / (self.n - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]
        if len(eigvals) == 0: return 0.0
        p = eigvals / eigvals.sum()
        return np.exp(-np.sum(p * np.log(p)))

def compute_kl_div(p, q):
    p = p + 1e-10
    q = q + 1e-10
    return torch.sum(p * torch.log(p / q), dim=-1).item()

class OrthogonalNoiseContext:
    def __init__(self, model, layer, E_k, noise_scale=0.5):
        self.model = model
        self.layer = layer
        self.E_k = E_k
        self.noise_scale = noise_scale
        self.handle = None

    def __enter__(self):
        def hook_fn(module, inp):
            x = inp[0]
            device = x.device
            E_k = self.E_k.to(device)
            # Scale noise relative to activation standard deviation
            noise = torch.randn_like(x) * (torch.std(x) * self.noise_scale)
            
            # Project noise orthogonal to E_k
            proj_noise = torch.matmul(torch.matmul(noise, E_k), E_k.T)
            ortho_noise = noise - proj_noise
            return (x + ortho_noise,)
            
        self.handle = self.model.blocks[self.layer].register_forward_pre_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()

class HeadDisruptor:
    def __init__(self, model, layer=15, head=3, head_dim=64):
        self.model = model
        self.layer = layer
        self.head = head
        self.head_dim = head_dim
        self.saved_weight = None

    def __enter__(self):
        attn = self.model.blocks[self.layer].attn
        if hasattr(attn, 'o_proj'):
            start = self.head * self.head_dim
            end = start + self.head_dim
            self.saved_weight = attn.o_proj.weight.data[:, start:end].clone()
            attn.o_proj.weight.data[:, start:end].zero_()
        return self

    def __exit__(self, *args):
        attn = self.model.blocks[self.layer].attn
        if hasattr(attn, 'o_proj') and self.saved_weight is not None:
            start = self.head * self.head_dim
            end = start + self.head_dim
            attn.o_proj.weight.data[:, start:end].copy_(self.saved_weight)

def extract_principal_subspace(model, tokenizer, config, prompts, layer=15, k=185):
    cov = WelfordCovariance(config.n_embd)
    device = next(model.parameters()).device
    
    captured = []
    def hook_fn(module, inp):
        captured.append(inp[0][:, -1, :].detach().float().cpu().numpy())
        return None
        
    handle = model.blocks[layer].register_forward_pre_hook(hook_fn)
    
    # Use 30 prompts for PCA
    sample_prompts = prompts[:30]
    for p in tqdm(sample_prompts, desc=f"PCA at L{layer}", leave=False):
        c = format_chatml_prompt(p)
        ids = torch.tensor([tokenizer.encode(c)], device=device)
        with torch.no_grad():
            model(ids)
        cov.update(captured[-1][0])
            
    handle.remove()
    
    covariance = cov.M2 / (cov.n - 1)
    evals, evecs = np.linalg.eigh(covariance)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    return torch.tensor(evecs[:, :k], dtype=torch.float32)

def main():
    print("Loading Genesis-152M for L15 Compression Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)

    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_list = json.load(f)["prompts"]
    all_prompts = [p["text"] if isinstance(p, dict) else p for p in prompts_list]

    print("\n1. Extracting L15 principal subspace (k=185)...")
    E_k = extract_principal_subspace(model, tokenizer, config, all_prompts, layer=15, k=185)

    # Use 20 prompts for the actual experiment
    test_prompts = all_prompts[30:50]
    max_tokens = 32
    d = config.n_embd

    results = {
        "Baseline": {"er": [WelfordCovariance(d) for _ in range(30)], "entropy": [], "kl": [], "px": []},
        "OrthoNoise": {"er": [WelfordCovariance(d) for _ in range(30)], "entropy": [], "kl": [], "px": []},
        "HeadDisrupt": {"er": [WelfordCovariance(d) for _ in range(30)], "entropy": [], "kl": [], "px": []}
    }

    print("\n2. Running 3-Condition Causal Generation...")
    
    for prompt in tqdm(test_prompts, desc="Experiment Progress"):
        inp = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(inp)], device=device)
        
        # --- BASELINE ---
        curr_ids = input_ids
        past_kv = None
        baseline_logits = []
        baseline_tokens = []
        
        hooks = HiddenStateHook.attach_to_model(model)
        
        for t in range(max_tokens):
            hooks.clear()
            with torch.no_grad():
                if past_kv is not None:
                    out = model(curr_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    out = model(curr_ids, use_cache=True)
                logits, past_kv = out[0], out[3]
                
            for l_idx, h in hooks.hidden_states:
                results["Baseline"]["er"][l_idx].update(h.numpy()[0])
                
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            results["Baseline"]["entropy"].append(ent)
            results["Baseline"]["kl"].append(0.0)
            
            baseline_logits.append(probs)
            
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            baseline_tokens.append(next_id.item())
            curr_ids = torch.cat([curr_ids, next_id], dim=1)
            
            px = -torch.log(probs[0, next_id.item()] + 1e-10).item()
            results["Baseline"]["px"].append(px)
            
        hooks.remove_all()
        
        # --- PERTURBED CONDITIONS ---
        conditions = [
            ("OrthoNoise", OrthogonalNoiseContext(model, 15, E_k, noise_scale=0.5)),
            ("HeadDisrupt", HeadDisruptor(model, layer=15, head=3))
        ]
        
        for cond_name, cond_context in conditions:
            curr_ids = input_ids
            past_kv = None
            hooks = HiddenStateHook.attach_to_model(model)
            
            with cond_context:
                for t in range(max_tokens):
                    hooks.clear()
                    with torch.no_grad():
                        if past_kv is not None:
                            out = model(curr_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                        else:
                            out = model(curr_ids, use_cache=True)
                        logits, past_kv = out[0], out[3]
                        
                    for l_idx, h in hooks.hidden_states:
                        results[cond_name]["er"][l_idx].update(h.numpy()[0])
                        
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    kl = compute_kl_div(baseline_logits[t], probs)
                    
                    target_id = baseline_tokens[t]
                    px = -torch.log(probs[0, target_id] + 1e-10).item()
                    
                    results[cond_name]["entropy"].append(ent)
                    results[cond_name]["kl"].append(kl)
                    results[cond_name]["px"].append(px)
                    
                    # Teacher force next step
                    next_id = torch.tensor([[target_id]], device=device)
                    curr_ids = torch.cat([curr_ids, next_id], dim=1)
                    
            hooks.remove_all()

    print("\n3. Computing Final Metrics & Statistics...")
    
    # Calculate ER vectors
    er_baseline = [cov.get_shannon_er() for cov in results["Baseline"]["er"]]
    er_noise = [cov.get_shannon_er() for cov in results["OrthoNoise"]["er"]]
    er_disrupt = [cov.get_shannon_er() for cov in results["HeadDisrupt"]["er"]]
    
    # Means
    mean_ent = {k: np.mean(v["entropy"]) for k, v in results.items()}
    mean_kl = {k: np.mean(v["kl"]) for k, v in results.items()}
    mean_px = {k: np.mean(v["px"]) for k, v in results.items()}
    
    # T-tests against baseline
    _, pval_px_noise = ttest_rel(results["Baseline"]["px"], results["OrthoNoise"]["px"])
    _, pval_px_disrupt = ttest_rel(results["Baseline"]["px"], results["HeadDisrupt"]["px"])

    os.makedirs("measurements", exist_ok=True)
    
    # Plotting
    os.makedirs("figures", exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].plot(er_baseline, label='Baseline', color='black', linewidth=2)
    axs[0].plot(er_noise, label='Ortho Noise (L15)', linestyle='--', color='blue')
    axs[0].plot(er_disrupt, label='Head 3 Disrupt (L15)', linestyle='--', color='red')
    axs[0].axvline(15, color='gray', alpha=0.5, linestyle=':')
    axs[0].set_title('Effective Rank via Layers')
    axs[0].set_xlabel('Layer')
    axs[0].set_ylabel('Shannon ER')
    axs[0].legend()
    
    axs[1].bar(["OrthoNoise", "HeadDisrupt"], [mean_kl["OrthoNoise"], mean_kl["HeadDisrupt"]], color=['blue', 'red'])
    axs[1].set_title('KL Divergence (Relative to Baseline)')
    axs[1].set_ylabel('KL Divergence')
    
    axs[2].bar(["Baseline", "OrthoNoise", "HeadDisrupt"], [mean_px["Baseline"], mean_px["OrthoNoise"], mean_px["HeadDisrupt"]], color=['black', 'blue', 'red'])
    axs[2].set_title('Perplexity of Generated Tokens')
    axs[2].set_ylabel('Log Perplexity (NLL)')
    
    plt.tight_layout()
    plt.savefig("figures/l15_compression_test.png", dpi=150)
    
    print("\n--- RESULTS ---")
    print(f"Token Entropy   | Base: {mean_ent['Baseline']:.3f} | Noise: {mean_ent['OrthoNoise']:.3f} | Disrupt: {mean_ent['HeadDisrupt']:.3f}")
    print(f"KL Divergence   | Noise: {mean_kl['OrthoNoise']:.3f} | Disrupt: {mean_kl['HeadDisrupt']:.3f}")
    print(f"Log Perplexity  | Base: {mean_px['Baseline']:.3f} | Noise: {mean_px['OrthoNoise']:.3f} | Disrupt: {mean_px['HeadDisrupt']:.3f}")
    print(f"\nStats (P-values relative to Baseline NLL):")
    print(f"Ortho Noise: p = {pval_px_noise:.2e}")
    print(f"Head Disrupt: p = {pval_px_disrupt:.2e}")
    
    df_metrics = pd.DataFrame({
        "Condition": ["Baseline", "OrthoNoise", "HeadDisrupt"],
        "Entropy": [mean_ent["Baseline"], mean_ent["OrthoNoise"], mean_ent["HeadDisrupt"]],
        "KL_Divergence": [mean_kl["Baseline"], mean_kl["OrthoNoise"], mean_kl["HeadDisrupt"]],
        "Log_Perplexity": [mean_px["Baseline"], mean_px["OrthoNoise"], mean_px["HeadDisrupt"]]
    })
    df_metrics.to_csv("measurements/l15_compression_summary.csv", index=False)
    print("\nSaved plot to figures/l15_compression_test.png and summary to measurements/l15_compression_summary.csv")

if __name__ == "__main__":
    main()
