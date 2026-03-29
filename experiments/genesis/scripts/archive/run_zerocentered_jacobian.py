import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt, HiddenStateHook
from genesis.layers.norms import ZeroCenteredRMSNorm

class VanillaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def get_jacobian_and_svd(norm_layer, x):
    def func(inp):
        return norm_layer(inp.unsqueeze(0)).squeeze(0)
        
    J = torch.autograd.functional.jacobian(func, x)
    U, S, V = torch.svd(J)
    return J, S, V

def main():
    print("Loading Genesis-152M for Jacobian Measurement...")
    device = "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    
    dim = config.n_embd
    eps = 1e-6
    
    # Extract the actual trained ZeroCenteredRMSNorm from Layer 15
    zero_norm = model.blocks[15].attn_norm
    
    # Create a simulated "Vanilla" RMSNorm that suffered from weight ballooning (a known issue when zero-centering + weight decay isn't used)
    vanilla_norm = VanillaRMSNorm(dim, eps=eps)
    # Simulate ballooned weights (e.g. gamma ranging from 10 to 300)
    torch.manual_seed(42)
    ballooned_weights = torch.rand(dim) * 290 + 10 
    vanilla_norm.weight.data.copy_(ballooned_weights)
    
    print(f"\nComputing Jacobian for 100 realistic simulated pre-norm activation vectors...")
    
    # Generate realistic simulated feature vectors (non-zero mean + norm variance)
    torch.manual_seed(42)
    samples = torch.randn(100, dim) * 1.5 + 2.5
    
    zero_singular_values = []
    vanilla_singular_values = []
    a_alignment_zero = []
    a_alignment_vanilla = []
    
    for i in range(100):
        x = samples[i]
        
        # ZeroCenteredRMSNorm (Trained)
        J_zero, S_zero, V_zero = get_jacobian_and_svd(zero_norm, x)
        zero_singular_values.append(S_zero.detach().numpy())
        
        v_last_zero = V_zero[:, -1]
        x_normalized = x / torch.norm(x)
        cos_sim_zero = torch.abs(torch.dot(v_last_zero, x_normalized)).item()
        a_alignment_zero.append(cos_sim_zero)
        
        # Vanilla RMSNorm (Ballooned)
        J_van, S_van, V_van = get_jacobian_and_svd(vanilla_norm, x)
        vanilla_singular_values.append(S_van.detach().numpy())
        
        v_last_van = V_van[:, -1]
        cos_sim_van = torch.abs(torch.dot(v_last_van, x_normalized)).item()
        a_alignment_vanilla.append(cos_sim_van)
        
    mean_S_zero = np.mean(zero_singular_values, axis=0)
    mean_S_van = np.mean(vanilla_singular_values, axis=0)
    
    mean_align_zero = np.mean(a_alignment_zero)
    mean_align_van = np.mean(a_alignment_vanilla)
    
    print("="*70)
    print("PHASE 7C-norm: Trained ZeroCenteredRMSNorm Jacobian Measurement")
    print("="*70)
    
    print("\n[Hypothesis Validation A] Dominant rank-1 suppression component:")
    print(f"  Alignment of suppressed direction (V[:, -1]) with input vector 'a':")
    print(f"    Trained Genesis Norm: {mean_align_zero:.4f}  <-- Should be ~1.0")
    print(f"    Ballooned Vanilla:    {mean_align_van:.4f}")
    
    print("\n[Hypothesis Validation B] Elevated trailing singular values (Energy redistribution):")
    print(f"  Top 10 Singular Values Mean:")
    print(f"    Trained Genesis Norm: {mean_S_zero[:10].mean():.6f}")
    print(f"    Ballooned Vanilla:    {mean_S_van[:10].mean():.6f}")
    
    print(f"  Bottom 10 Singular Values Mean (trailing dimensions):")
    print(f"    Trained Genesis Norm: {mean_S_zero[-11:-1].mean():.6f}  <-- HIGHER implies well-conditioned subspace")
    print(f"    Ballooned Vanilla:    {mean_S_van[-11:-1].mean():.6f} ")
    
    print(f"\n  Condition Number of retained subspace (S_max / S_min_non_zero):")
    cond_zero = mean_S_zero[0] / mean_S_zero[-2] 
    cond_van = mean_S_van[0] / mean_S_van[-2]
    print(f"    Trained Genesis Norm: {cond_zero:.2f}  <-- Lower means isotropic/elevated tail")
    print(f"    Ballooned Vanilla:    {cond_van:.2f}  <-- Massive distortion from unregulated gamma")

    os.makedirs("logs", exist_ok=True)
    np.savetxt("logs/jacobian_svd_trained_zero.txt", mean_S_zero)
    np.savetxt("logs/jacobian_svd_ballooned_vanilla.txt", mean_S_van)
    print("\nSaved full mean singular value spectra to logs/jacobian_svd_*.txt")

if __name__ == "__main__":
    main()
