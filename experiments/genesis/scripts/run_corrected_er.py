"""Corrected Effective Rank Measurement — Overnight Run.

PROBLEM: Prior measurements used SVD of a (T×d) matrix with T=64, d=576.
Shannon ER is capped at min(T,d)=64, so ER/576 was capped at 11.1%.
The reported "8.3% utilization" was near the measurement ceiling.

FIX: Use Welford's online covariance estimation to accumulate a (d×d)
covariance matrix over N >> d hidden states, stacking across ALL prompts
AND all token positions. Then compute eigendecomposition of the full
576×576 covariance and report proper Shannon ER.

Also performs sub-block measurement with proper sample counts.

Expected runtime: 60 prompts × 64 tokens × ~3.5s/tok ≈ 3.7 hours

Usage:
    python scripts/run_corrected_er.py --max-prompts 60 --max-tokens 64
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


class WelfordCovariance:
    """Online covariance estimator using Welford's algorithm.
    
    Accumulates mean and covariance of d-dimensional vectors
    one sample at a time, using O(d²) memory regardless of N.
    """
    
    def __init__(self, d):
        self.d = d
        self.n = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros((d, d), dtype=np.float64)  # sum of (x-mean_old)(x-mean_new)^T
    
    def update(self, x):
        """Add a single d-dimensional sample."""
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def update_batch(self, X):
        """Add a batch of samples (N×d matrix)."""
        for x in X:
            self.update(x)
    
    def get_covariance(self):
        """Returns the sample covariance matrix (d×d)."""
        if self.n < 2:
            return np.zeros((self.d, self.d))
        return self.M2 / (self.n - 1)
    
    def get_shannon_er(self):
        """Compute Shannon effective rank from covariance eigenvalues."""
        cov = self.get_covariance()
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-10]  # filter numerical zeros
        if len(eigvals) == 0:
            return 0.0
        # Normalize to probability distribution
        p = eigvals / eigvals.sum()
        # Shannon entropy
        H = -np.sum(p * np.log(p))
        return np.exp(H)


class MultiPointHook:
    """Captures hidden states at 4 sub-block points + block input/output."""
    
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.data = {}  # (layer_idx, point_name) -> tensor
        self.handles = []
    
    def _make_subblock_hook(self, layer_idx, point_name):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            if x is not None and x.dim() == 3:
                self.data[(layer_idx, point_name)] = x[:, -1, :].detach().float().cpu().numpy()
        return hook_fn
    
    def _make_block_io_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            # Capture input
            if isinstance(inp, tuple):
                x_in = inp[0]
            else:
                x_in = inp
            if x_in is not None and x_in.dim() == 3:
                self.data[(layer_idx, 'block_input')] = x_in[:, -1, :].detach().float().cpu().numpy()
            
            # Capture output
            if isinstance(output, tuple):
                x_out = output[0]
            else:
                x_out = output
            if x_out is not None and x_out.dim() == 3:
                self.data[(layer_idx, 'block_output')] = x_out[:, -1, :].detach().float().cpu().numpy()
        return hook_fn
    
    @classmethod
    def attach(cls, model):
        inst = cls(len(model.blocks))
        for i, block in enumerate(model.blocks):
            # Block-level input/output
            h = block.register_forward_hook(inst._make_block_io_hook(i))
            inst.handles.append(h)
            
            # Sub-block hooks
            if hasattr(block, 'attn_norm'):
                h = block.attn_norm.register_forward_hook(inst._make_subblock_hook(i, 'post_norm'))
                inst.handles.append(h)
            if hasattr(block, 'attn'):
                h = block.attn.register_forward_hook(inst._make_subblock_hook(i, 'post_mixer'))
                inst.handles.append(h)
            if hasattr(block, 'ffn'):
                h = block.ffn.register_forward_hook(inst._make_subblock_hook(i, 'post_ffn'))
                inst.handles.append(h)
        return inst
    
    def clear(self):
        self.data = {}
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_corrected_measurement(model, tokenizer, config, prompts, max_new_tokens=64,
                               output_dir="logs/corrected"):
    """Corrected ER measurement with Welford's covariance across all prompts."""
    
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    d = config.n_embd  # 576
    
    # Initialize Welford accumulators for each measurement point
    points = ['block_output', 'post_norm', 'post_mixer', 'post_ffn', 'block_input']
    accum = {}
    for i in range(n_layers):
        for pt in points:
            accum[(i, pt)] = WelfordCovariance(d)
    
    # Also track energy ratios
    energy_data = {i: {'x_norms': [], 'y_norms': [], 'sum_norms': [], 'cos_sims': []}
                   for i in range(n_layers)}
    
    hooks = MultiPointHook.attach(model)
    total_samples = 0
    t_start = time.time()
    
    for pidx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
        category = prompt_data.get("category", "unknown") if isinstance(prompt_data, dict) else "unknown"
        
        print(f"\n[{pidx+1}/{len(prompts)}] {category}: {prompt_text[:60]}...")
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
            
            # Accumulate into Welford estimators
            for i in range(n_layers):
                for pt in points:
                    if (i, pt) in hooks.data:
                        vec = hooks.data[(i, pt)].squeeze()
                        accum[(i, pt)].update(vec)
                
                # Energy ratio tracking
                if (i, 'block_input') in hooks.data and (i, 'block_output') in hooks.data:
                    x = hooks.data[(i, 'block_input')].squeeze()
                    x_out = hooks.data[(i, 'block_output')].squeeze()
                    y = x_out - x
                    
                    x_norm = np.linalg.norm(x)
                    y_norm = np.linalg.norm(y)
                    sum_norm = np.linalg.norm(x_out)
                    cos = np.dot(x, y) / (x_norm * y_norm + 1e-10)
                    
                    energy_data[i]['x_norms'].append(x_norm)
                    energy_data[i]['y_norms'].append(y_norm)
                    energy_data[i]['sum_norms'].append(sum_norm)
                    energy_data[i]['cos_sims'].append(cos)
            
            total_samples += 1
            
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            next_token = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break
        
        elapsed = time.time() - t0
        elapsed_total = time.time() - t_start
        remaining_prompts = len(prompts) - (pidx + 1)
        avg_per_prompt = elapsed_total / (pidx + 1)
        eta_min = remaining_prompts * avg_per_prompt / 60
        
        print(f"  {step+1} tokens | N={total_samples} samples | {elapsed:.0f}s | ETA: {eta_min:.0f} min")
        
        # Periodic ER computation (every 5 prompts)
        if (pidx + 1) % 5 == 0 or pidx == len(prompts) - 1:
            print(f"\n  === Interim ER (N={total_samples}, d={d}) ===")
            print(f"  {'Layer':<12} {'block_out ER':>12} {'post_norm ER':>13} {'post_mix ER':>12} {'post_ffn ER':>12}")
            print(f"  {'-'*62}")
            for i in [0, 3, 7, 11, 15, 19, 23, 27, 29]:
                ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
                er_out = accum[(i, 'block_output')].get_shannon_er()
                er_norm = accum[(i, 'post_norm')].get_shannon_er()
                er_mix = accum[(i, 'post_mixer')].get_shannon_er()
                er_ffn = accum[(i, 'post_ffn')].get_shannon_er()
                marker = " <<FoX" if ltype == "fox" else ""
                print(f"  L{i:2d} ({ltype})  {er_out:12.1f} {er_norm:13.1f} {er_mix:12.1f} {er_ffn:12.1f}{marker}")
            print(f"  Overall utilization: {accum[(29, 'block_output')].get_shannon_er():.1f}/{d} ({accum[(29, 'block_output')].get_shannon_er()/d*100:.1f}%)")
    
    hooks.remove()
    
    # Final results
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"CORRECTED EFFECTIVE RANK MEASUREMENT")
    print(f"  Total samples: N = {total_samples} (vs d = {d})")
    print(f"  N/d ratio: {total_samples/d:.1f}x (need >>1 for reliable ER)")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'='*70}")
    
    # Write per-layer results
    with open(f"{output_dir}/corrected_er.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "type", "N", "d",
                        "block_output_er", "post_norm_er", "post_mixer_er", "post_ffn_er",
                        "block_input_er",
                        "mean_x_norm", "mean_y_norm", "mean_ratio", "mean_cos_sim",
                        "block_output_er_pct", "post_norm_er_pct", "post_mixer_er_pct", "post_ffn_er_pct"])
        
        print(f"\n{'Layer':<12} {'Type':<5} {'Out ER':>8} {'Norm ER':>8} {'Mix ER':>8} {'FFN ER':>8} {'In ER':>8}  {'||y/x||':>8} {'cos':>6}")
        print(f"{'-'*82}")
        
        for i in range(n_layers):
            ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
            er_out = accum[(i, 'block_output')].get_shannon_er()
            er_norm = accum[(i, 'post_norm')].get_shannon_er()
            er_mix = accum[(i, 'post_mixer')].get_shannon_er()
            er_ffn = accum[(i, 'post_ffn')].get_shannon_er()
            er_in = accum[(i, 'block_input')].get_shannon_er()
            
            ed = energy_data[i]
            mean_ratio = np.mean(ed['y_norms']) / max(np.mean(ed['x_norms']), 1e-8) if ed['x_norms'] else 0
            mean_cos = np.mean(ed['cos_sims']) if ed['cos_sims'] else 0
            mean_x = np.mean(ed['x_norms']) if ed['x_norms'] else 0
            mean_y = np.mean(ed['y_norms']) if ed['y_norms'] else 0
            
            marker = " <<FoX" if ltype == "fox" else ""
            print(f"L{i:2d} ({ltype})  {er_out:8.1f} {er_norm:8.1f} {er_mix:8.1f} {er_ffn:8.1f} {er_in:8.1f}  {mean_ratio:8.3f} {mean_cos:+6.3f}{marker}")
            
            writer.writerow([i, ltype, total_samples, d,
                            f"{er_out:.2f}", f"{er_norm:.2f}", f"{er_mix:.2f}", f"{er_ffn:.2f}",
                            f"{er_in:.2f}",
                            f"{mean_x:.1f}", f"{mean_y:.1f}", f"{mean_ratio:.4f}", f"{mean_cos:.4f}",
                            f"{er_out/d*100:.2f}", f"{er_norm/d*100:.2f}",
                            f"{er_mix/d*100:.2f}", f"{er_ffn/d*100:.2f}"])
    
    # Summary
    er_final = accum[(n_layers-1, 'block_output')].get_shannon_er()
    er_mid = accum[(n_layers//2, 'block_output')].get_shannon_er()
    er_first = accum[(0, 'block_output')].get_shannon_er()
    
    summary = f"""
CORRECTED MEASUREMENT SUMMARY
==============================
Samples: N = {total_samples} (d = {d}, N/d = {total_samples/d:.1f}x)
Method: Welford online covariance → eigendecomposition of {d}×{d} matrix

Shannon Effective Rank (from covariance eigenvalues):
  L0  (first block):  {er_first:.1f}/{d}  ({er_first/d*100:.1f}%)
  L{n_layers//2} (mid block):    {er_mid:.1f}/{d}  ({er_mid/d*100:.1f}%)
  L{n_layers-1} (last block):  {er_final:.1f}/{d}  ({er_final/d*100:.1f}%)

Comparison:
  Prior measurement (T=64, SVD of data matrix):  48/576 (8.3%) — CAPPED at min(T,d)=64
  Corrected (N={total_samples}, covariance eigendecomp): {er_final:.1f}/576 ({er_final/d*100:.1f}%)
  Max possible ER: {d} (100%)

Sub-block rank flow (L15, mid-network):
  post_norm:   {accum[(15, 'post_norm')].get_shannon_er():.1f}/{d}
  post_mixer:  {accum[(15, 'post_mixer')].get_shannon_er():.1f}/{d}
  post_ffn:    {accum[(15, 'post_ffn')].get_shannon_er():.1f}/{d}
  block_output:{accum[(15, 'block_output')].get_shannon_er():.1f}/{d}
"""
    print(summary)
    
    with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  corrected_er.csv — per-layer ER at all measurement points")
    print(f"  summary.txt — key numbers")


def disable_ttt(model):
    """Zero out TTT/metacognition contribution."""
    found = False
    for name, mod in model.named_modules():
        if 'metacognition' in name and 'output_gate' in name:
            with torch.no_grad():
                if hasattr(mod, 'weight') and mod.weight is not None:
                    mod.weight.zero_()
                if hasattr(mod, 'bias') and mod.bias is not None:
                    mod.bias.fill_(-100.0)
            found = True
            print(f"[TTT Control] Disabled: {name}")
    if not found:
        print("[TTT Control] WARNING: Could not find metacognition output gate")
        for name, mod in model.named_modules():
            if 'metacognition' in name and 'self_model' in name:
                with torch.no_grad():
                    if hasattr(mod, 'weight') and mod.weight is not None:
                        mod.weight.zero_()
                    if hasattr(mod, 'bias') and mod.bias is not None:
                        mod.bias.zero_()
                print(f"[TTT Control] Zeroed: {name}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Corrected ER Measurement (Overnight)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="logs/corrected")
    parser.add_argument("--disable-ttt", action="store_true", help="Zero TTT metacognition output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)
    
    if args.disable_ttt:
        print("\n*** TTT CONTROL: Disabling metacognition ***")
        model = disable_ttt(model)
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    n_samples = len(prompts) * args.max_tokens
    mode_str = "TTT-DISABLED" if args.disable_ttt else "NORMAL"
    print(f"\nPlan: {len(prompts)} prompts × {args.max_tokens} tokens = {n_samples} samples (d={config.n_embd})")
    print(f"Mode: {mode_str}")
    print(f"N/d ratio: {n_samples/config.n_embd:.1f}x {'(GOOD: >>1)' if n_samples > config.n_embd * 2 else '(WARNING: need more samples)'}")
    print(f"Method: Welford online covariance → eigendecomposition of {config.n_embd}×{config.n_embd} matrix")
    
    run_corrected_measurement(model, tokenizer, config, prompts,
                               max_new_tokens=args.max_tokens,
                               output_dir=args.output_dir)


if __name__ == "__main__":
    main()
