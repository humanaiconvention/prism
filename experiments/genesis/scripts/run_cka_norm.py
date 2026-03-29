"""CKA Norm Diagnostic — Phase 3 Experiment 1.

Tests whether ZeroCenteredRMSNorm's massive ER inflation is REAL
diversification (CKA < 1.0) or just invertible rescaling (CKA ≈ 1.0).

Linear CKA (Kornblith et al. 2019):
    CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)

If CKA ≈ 1.0 → norm is an invertible linear transform (same representation)
If CKA < 1.0 → norm genuinely reorganizes the representation

Usage:
    python scripts/run_cka_norm.py --max-prompts 60 --max-tokens 64
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import (
    load_genesis_model, format_chatml_prompt,
    FOX_LAYER_INDICES, GLA_LAYER_INDICES,
)


class CKAAccumulator:
    """Accumulates paired activation matrices for CKA computation.
    
    Stores centered activation matrices X (pre-norm) and Y (post-norm)
    across all samples, then computes Linear CKA at the end.
    
    For memory efficiency with large N, we accumulate the Gram matrices
    (X^T X, Y^T Y, Y^T X) online instead of storing raw activations.
    """
    
    def __init__(self, d):
        self.d = d
        self.n = 0
        # Accumulate centered gram matrices: X^T X, Y^T Y, Y^T X
        self.XtX = np.zeros((d, d), dtype=np.float64)
        self.YtY = np.zeros((d, d), dtype=np.float64)
        self.YtX = np.zeros((d, d), dtype=np.float64)
        # Running means for centering
        self.mean_x = np.zeros(d, dtype=np.float64)
        self.mean_y = np.zeros(d, dtype=np.float64)
        # Raw sums for post-hoc centering
        self.sum_x = np.zeros(d, dtype=np.float64)
        self.sum_y = np.zeros(d, dtype=np.float64)
        self.sum_xxt = np.zeros((d, d), dtype=np.float64)
        self.sum_yyt = np.zeros((d, d), dtype=np.float64)
        self.sum_yxt = np.zeros((d, d), dtype=np.float64)
    
    def update(self, x, y):
        """Add a single paired sample (pre-norm x, post-norm y)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xxt += np.outer(x, x)
        self.sum_yyt += np.outer(y, y)
        self.sum_yxt += np.outer(y, x)
    
    def get_linear_cka(self):
        """Compute Linear CKA from accumulated data.
        
        Uses centered Gram matrices: X_c = X - mean(X), etc.
        CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F · ||Y_c^T Y_c||_F)
        """
        if self.n < 2:
            return 0.0
        
        # Centered cross-products: Σ(x-μ_x)(x-μ_x)^T = Σxx^T - n·μ_x·μ_x^T
        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        
        XtX = self.sum_xxt - self.n * np.outer(mean_x, mean_x)
        YtY = self.sum_yyt - self.n * np.outer(mean_y, mean_y)
        YtX = self.sum_yxt - self.n * np.outer(mean_y, mean_x)
        
        # CKA = ||YtX||_F^2 / (||XtX||_F * ||YtY||_F)
        numerator = np.linalg.norm(YtX, 'fro') ** 2
        denominator = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator


class PairedHook:
    """Captures paired pre-norm (block_input) and post-norm activations."""
    
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.data = {}  # (layer_idx, point_name) -> numpy array
        self.handles = []
    
    def _make_hook(self, layer_idx, point_name):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            if x is not None and x.dim() == 3:
                self.data[(layer_idx, point_name)] = x[:, -1, :].detach().float().cpu().numpy()
        return hook_fn
    
    def _make_input_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            if isinstance(inp, tuple):
                x_in = inp[0]
            else:
                x_in = inp
            if x_in is not None and x_in.dim() == 3:
                self.data[(layer_idx, 'block_input')] = x_in[:, -1, :].detach().float().cpu().numpy()
        return hook_fn
    
    @classmethod
    def attach(cls, model):
        inst = cls(len(model.blocks))
        for i, block in enumerate(model.blocks):
            # Block-level input capture
            h = block.register_forward_hook(inst._make_input_hook(i))
            inst.handles.append(h)
            # Post-norm capture
            if hasattr(block, 'attn_norm'):
                h = block.attn_norm.register_forward_hook(inst._make_hook(i, 'post_norm'))
                inst.handles.append(h)
        return inst
    
    def clear(self):
        self.data = {}
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_cka_diagnostic(model, tokenizer, config, prompts, max_new_tokens=64,
                       output_dir="logs/cka_norm"):
    """Run CKA diagnostic comparing pre-norm vs post-norm representations."""
    
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    d = config.n_embd
    
    # Initialize CKA accumulators for each layer
    cka_accum = {i: CKAAccumulator(d) for i in range(n_layers)}
    
    hooks = PairedHook.attach(model)
    total_samples = 0
    t_start = time.time()
    
    for pidx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
        
        print(f"\n[{pidx+1}/{len(prompts)}] {prompt_text[:60]}...")
        t0 = time.time()
        
        chat_input = format_chatml_prompt(prompt_text)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        
        current_ids = input_ids
        past_kv = None
        
        for step in range(max_new_tokens):
            hooks.clear()
            
            with torch.no_grad():
                if past_kv is not None:
                    out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    out = model(current_ids, use_cache=True)
                logits = out[0]
                past_kv = out[3]
            
            # Accumulate CKA pairs
            for i in range(n_layers):
                if (i, 'block_input') in hooks.data and (i, 'post_norm') in hooks.data:
                    x = hooks.data[(i, 'block_input')].squeeze()
                    y = hooks.data[(i, 'post_norm')].squeeze()
                    cka_accum[i].update(x, y)
            
            total_samples += 1
            
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            next_token = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break
        
        elapsed = time.time() - t0
        elapsed_total = time.time() - t_start
        remaining = len(prompts) - (pidx + 1)
        avg_per = elapsed_total / (pidx + 1)
        eta = remaining * avg_per / 60
        
        print(f"  {step+1} tokens | N={total_samples} | {elapsed:.0f}s | ETA: {eta:.0f} min")
        
        # Periodic CKA output
        if (pidx + 1) % 10 == 0 or pidx == len(prompts) - 1:
            print(f"\n  === Interim CKA (N={total_samples}) ===")
            for i in [0, 3, 7, 11, 15, 19, 23, 27, 29]:
                ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
                cka = cka_accum[i].get_linear_cka()
                print(f"  L{i:2d} ({ltype}): CKA = {cka:.4f}")
    
    hooks.remove()
    
    # Final results
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"CKA NORM DIAGNOSTIC RESULTS")
    print(f"  N = {total_samples}, d = {d}")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"{'='*70}")
    
    # Write CSV
    with open(f"{output_dir}/cka_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "type", "N", "linear_cka"])
        
        for i in range(n_layers):
            ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
            cka = cka_accum[i].get_linear_cka()
            writer.writerow([i, ltype, total_samples, f"{cka:.6f}"])
    
    # Print summary table
    print(f"\n{'Layer':<12} {'Type':<5} {'Linear CKA':>12}  Interpretation")
    print(f"{'-'*60}")
    
    summary_lines = []
    for i in range(n_layers):
        ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
        cka = cka_accum[i].get_linear_cka()
        
        if cka > 0.99:
            interp = "INVERTIBLE (just rescaling)"
        elif cka > 0.95:
            interp = "mostly invertible"
        elif cka > 0.80:
            interp = "PARTIAL diversification"
        else:
            interp = "GENUINE diversification"
        
        marker = " <<FoX" if ltype == "fox" else ""
        line = f"L{i:2d} ({ltype})  {cka:12.4f}  {interp}{marker}"
        print(line)
        summary_lines.append(line)
    
    # Summary file
    summary = f"""CKA NORM DIAGNOSTIC
====================
N = {total_samples} (d = {d})
Time: {total_time/60:.1f} min

Question: Is ZeroCenteredRMSNorm inflation real diversification or metric reweighting?
Method: Linear CKA(block_input, post_norm) per layer.
  CKA ≈ 1.0 → norm is invertible (same representation)
  CKA < 1.0 → genuine reorganization

Results:
""" + "\n".join(summary_lines) + "\n"
    
    with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="CKA Norm Diagnostic (Phase 3)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="logs/cka_norm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    n_samples = len(prompts) * args.max_tokens
    print(f"\nPlan: {len(prompts)} prompts × {args.max_tokens} tokens = {n_samples} samples")
    print(f"Method: Linear CKA(block_input, post_norm) per layer")
    
    run_cka_diagnostic(model, tokenizer, config, prompts,
                       max_new_tokens=args.max_tokens,
                       output_dir=args.output_dir)


if __name__ == "__main__":
    main()
