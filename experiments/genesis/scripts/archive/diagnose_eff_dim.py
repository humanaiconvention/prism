"""Diagnostic: Compare windowed vs corpus-level effective dimensionality.

Resolves whether our low effective dim (5-7 in 576-dim space) is a
measurement artifact from small window sizes or a genuine architectural property.

Literature method: compute effective rank from covariance of hidden states
across many tokens (hundreds/thousands), using Shannon entropy of normalized
singular values.
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt, HiddenStateHook

def effective_rank_shannon(X):
    """Compute effective rank via Shannon entropy of normalized singular values.
    
    This is the standard metric from Roy & Bhattacharyya (2007) used in
    the "Layer by Layer" paper and Pythia/GPT-2 spectral analyses.
    
    Args:
        X: (n_samples, n_dims) matrix of hidden states
    Returns:
        effective_rank (float), singular_values (array)
    """
    # Center the data
    X_centered = X - X.mean(dim=0, keepdim=True)
    
    # SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # Normalize singular values to get a probability distribution
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-12]  # Remove zeros
    
    # Shannon entropy
    entropy = -(S_norm * torch.log(S_norm)).sum().item()
    
    # Effective rank = exp(entropy)
    eff_rank = np.exp(entropy)
    
    return eff_rank, S.cpu().numpy()


def participation_ratio(X):
    """Compute participation ratio (our current metric).
    
    PR = (sum λ_i)^2 / sum(λ_i^2)
    This is what compute_spectral_metrics uses.
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = X_centered.T @ X_centered / max(X_centered.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    
    total = eigenvalues.sum()
    if total > 0:
        pr = (total ** 2) / ((eigenvalues ** 2).sum() + 1e-12)
    else:
        pr = 0.0
    
    return pr.item(), eigenvalues.cpu().numpy()


def main():
    print("=" * 60)
    print("DIAGNOSTIC: Effective Dimensionality Measurement")
    print("=" * 60)
    
    model, tokenizer, config = load_genesis_model()
    device = next(model.parameters()).device
    
    # Test prompts
    prompts = [
        "Prove that the square root of 2 is irrational.",
        "What is the halting problem and why is it undecidable?",
        "Translate 'the cat sat on the mat' into formal logic.",
        "Write a short poem about the ocean.",
        "Explain how a neural network learns through backpropagation.",
    ]
    
    all_hidden_states = []  # Collect across all prompts
    
    for pidx, prompt in enumerate(prompts):
        print(f"\n[{pidx+1}/{len(prompts)}] {prompt[:60]}...")
        
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        
        prompt_hidden = []
        current_ids = input_ids
        past_kv = None
        
        for step in range(64):
            hooks = HiddenStateHook.attach_to_model(model)
            
            with torch.no_grad():
                if past_kv is not None:
                    out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    out = model(current_ids, use_cache=True)
                
                logits = out[0]
                past_kv = out[3]
            
            hidden_states = hooks.get_hidden_states()
            hooks.remove_all()
            
            if hidden_states:
                h_last = hidden_states[-1].squeeze(0)  # (n_embd,)
                prompt_hidden.append(h_last.cpu())
                all_hidden_states.append(h_last.cpu())
            
            # Greedy next token
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            next_token = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break
        
        tokens_generated = len(prompt_hidden)
        print(f"  Generated {tokens_generated} tokens")
        
        # Per-prompt metrics at different window sizes
        H = torch.stack(prompt_hidden, dim=0)  # (T, 576)
        print(f"  Hidden states shape: {H.shape}")
        
        for w in [5, 10, 20, 40, tokens_generated]:
            w = min(w, tokens_generated)
            H_window = H[:w]
            pr, _ = participation_ratio(H_window)
            er, _ = effective_rank_shannon(H_window)
            print(f"  Window={w:3d}: PR={pr:6.1f}  EffRank(Shannon)={er:6.1f}  "
                  f"PR/dim={pr/576*100:.1f}%  ER/dim={er/576*100:.1f}%")
    
    # Corpus-level (all prompts stacked)
    print(f"\n{'='*60}")
    print(f"CORPUS-LEVEL (all prompts, {len(all_hidden_states)} total tokens)")
    print(f"{'='*60}")
    
    H_all = torch.stack(all_hidden_states, dim=0)  # (N, 576)
    print(f"Shape: {H_all.shape}")
    
    pr_all, eigs_pr = participation_ratio(H_all)
    er_all, svs = effective_rank_shannon(H_all)
    
    print(f"Participation Ratio:      {pr_all:.1f} / 576 = {pr_all/576*100:.1f}%")
    print(f"Effective Rank (Shannon): {er_all:.1f} / 576 = {er_all/576*100:.1f}%")
    
    # Top singular values
    print(f"\nTop 20 singular values:")
    for i, s in enumerate(svs[:20]):
        print(f"  σ_{i}: {s:.4f}  ({s/svs[0]*100:.1f}% of max)")
    
    # Decay profile
    cumvar = np.cumsum(svs**2) / np.sum(svs**2)
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        k = np.searchsorted(cumvar, threshold) + 1
        print(f"  {threshold*100:.0f}% variance explained by top {k} components ({k/576*100:.1f}% of dims)")


if __name__ == "__main__":
    main()
