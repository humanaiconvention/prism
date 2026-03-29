# Fix Windows console encoding for genesis-llm Unicode chars (DeltaNet repr)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""Pre-W_o vs Post-W_o Rank Diagnostic.

PURPOSE: Determine how much of the mixer's rank compression comes from W_o
(the output projection) vs from the attention computation itself (head overlap).

This answers the key question from external review:
- If ER drops mainly ACROSS W_o -> W_o is the compressor, increasing KV heads won't help
- If ER is already low BEFORE W_o -> attention/GQA are the compressors

METHOD: Hook both block.attn.o_proj INPUT (pre-W_o) and block.attn OUTPUT (post-mixer).
The pre-W_o representation is the concatenated per-head output before projection.
"""
import argparse
import csv
import os
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
    """Online covariance estimator using Welford's algorithm."""
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


class WoProbeHook:
    """Hooks to capture pre-W_o and post-W_o hidden states."""
    
    def __init__(self, target_layers):
        self.target_layers = target_layers
        self.handles = []
        self.data = {}  # (layer_idx, 'pre_wo' or 'post_mixer') -> numpy vector
    
    def attach(self, model):
        for layer_idx in self.target_layers:
            block = model.blocks[layer_idx]
            
            # Hook on o_proj INPUT = pre-W_o (concatenated head output)
            if hasattr(block.attn, 'o_proj'):
                h = block.attn.o_proj.register_forward_hook(
                    self._make_hook(layer_idx, 'pre_wo', capture_input=True)
                )
                self.handles.append(h)
            
            # Hook on full attn module OUTPUT = post-W_o (post-mixer)
            h = block.attn.register_forward_hook(
                self._make_hook(layer_idx, 'post_mixer', capture_input=False)
            )
            self.handles.append(h)
            
            # Also hook attn_norm OUTPUT = post-norm (for reference)
            if hasattr(block, 'attn_norm'):
                h = block.attn_norm.register_forward_hook(
                    self._make_hook(layer_idx, 'post_norm', capture_input=False)
                )
                self.handles.append(h)
        
        return self
    
    def _make_hook(self, layer_idx, point_name, capture_input=False):
        def hook_fn(module, inp, output):
            if capture_input:
                # For o_proj, we want the INPUT (pre-W_o)
                x = inp[0] if isinstance(inp, tuple) else inp
            else:
                x = output[0] if isinstance(output, tuple) else output
            if x is not None and x.dim() >= 2:
                # Take last token position
                vec = x[:, -1, :] if x.dim() == 3 else x[-1, :]
                self.data[(layer_idx, point_name)] = vec.detach().float().cpu().numpy()
        return hook_fn
    
    def clear(self):
        self.data = {}
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_wo_probe(model, tokenizer, config, prompts, max_new_tokens=64,
                 target_layers=None, output_dir="logs/wo_probe"):
    """Measure pre-W_o vs post-W_o effective rank."""
    
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)
    d = config.n_embd  # 576
    
    if target_layers is None:
        # Measure all FoX layers + first and last GLA
        target_layers = sorted(FOX_LAYER_INDICES + [0, 29])
    
    # Set up Welford accumulators for each measurement point
    accumulators = {}
    for layer_idx in target_layers:
        for point in ['post_norm', 'pre_wo', 'post_mixer']:
            accumulators[(layer_idx, point)] = WelfordCovariance(d)
    
    # Attach hooks
    hook = WoProbeHook(target_layers)
    hook.attach(model)
    
    total_samples = 0
    start_time = time.time()
    
    print(f"W_o Probe Diagnostic")
    print(f"  Target layers: {target_layers}")
    print(f"  Prompts: {len(prompts)}, Tokens per prompt: {max_new_tokens}")
    print(f"  Expected samples: {len(prompts) * max_new_tokens}")
    print()
    
    for prompt_idx, prompt_text in enumerate(prompts):
        if isinstance(prompt_text, dict):
            prompt_text = prompt_text.get("text", prompt_text.get("prompt", ""))
        
        formatted = format_chatml_prompt(prompt_text)
        input_ids = torch.tensor([tokenizer.encode(formatted)], device=device)
        
        current_ids = input_ids
        past_kv = None
        
        # Greedy AR generation with KV-cache
        with torch.no_grad():
            for step in range(max_new_tokens):
                hook.clear()
                
                if past_kv is not None:
                    out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    out = model(current_ids, use_cache=True)
                
                logits = out[0]
                past_kv = out[3]
                
                next_id = logits[:, -1, :].argmax(dim=-1).item()
                next_token = torch.tensor([[next_id]], device=device)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Accumulate into Welford estimators
                for (layer_idx, point), vec in hook.data.items():
                    key = (layer_idx, point)
                    if key in accumulators:
                        v = vec.flatten()
                        # pre_wo may have different dim if there's gating
                        if v.shape[0] == d:
                            accumulators[key].update(v)
                
                total_samples += 1
        
        elapsed = time.time() - start_time
        rate = total_samples / elapsed if elapsed > 0 else 0
        print(f"  [{prompt_idx+1}/{len(prompts)}] N={total_samples}, "
              f"{rate:.1f} samples/s, {elapsed:.0f}s elapsed")
    
    # Compute results
    print(f"\n{'='*70}")
    print(f"RESULTS: Pre-W_o vs Post-W_o Rank (N={total_samples}, N/d={total_samples/d:.1f}x)")
    print(f"{'='*70}")
    
    results = []
    for layer_idx in target_layers:
        layer_type = "FoX" if layer_idx in FOX_LAYER_INDICES else "GLA"
        
        norm_er = accumulators.get((layer_idx, 'post_norm'), None)
        pre_wo_er = accumulators.get((layer_idx, 'pre_wo'), None)
        post_mixer_er = accumulators.get((layer_idx, 'post_mixer'), None)
        
        norm_val = norm_er.get_shannon_er() if norm_er and norm_er.n > 1 else 0
        pre_val = pre_wo_er.get_shannon_er() if pre_wo_er and pre_wo_er.n > 1 else 0
        post_val = post_mixer_er.get_shannon_er() if post_mixer_er and post_mixer_er.n > 1 else 0
        
        row = {
            'layer': layer_idx,
            'type': layer_type,
            'post_norm_er': round(norm_val, 1),
            'pre_wo_er': round(pre_val, 1),
            'post_mixer_er': round(post_val, 1),
            'wo_compression': round(pre_val - post_val, 1) if pre_val > 0 else 'N/A',
            'attn_compression': round(norm_val - pre_val, 1) if norm_val > 0 and pre_val > 0 else 'N/A',
            'pre_wo_n': pre_wo_er.n if pre_wo_er else 0,
        }
        results.append(row)
        
        print(f"\n  L{layer_idx} ({layer_type}):")
        print(f"    post_norm:    {norm_val:6.1f}/576  ({100*norm_val/576:.1f}%)")
        if pre_val > 0:
            print(f"    pre_W_o:      {pre_val:6.1f}/576  ({100*pre_val/576:.1f}%)  "
                  f"<- attention compresses by {norm_val - pre_val:.1f}")
            print(f"    post_mixer:   {post_val:6.1f}/576  ({100*post_val/576:.1f}%)  "
                  f"<- W_o compresses by {pre_val - post_val:.1f}")
        else:
            print(f"    pre_W_o:      [not captured - dim mismatch?]")
            print(f"    post_mixer:   {post_val:6.1f}/576  ({100*post_val/576:.1f}%)")
    
    # Save CSV
    csv_path = os.path.join(output_dir, "wo_probe.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"W_o PROBE DIAGNOSTIC\n")
        f.write(f"N = {total_samples} (d = {d}, N/d = {total_samples/d:.1f}x)\n")
        f.write(f"Time: {(time.time()-start_time)/60:.1f} min\n\n")
        for r in results:
            f.write(f"L{r['layer']} ({r['type']}): "
                    f"norm={r['post_norm_er']}, pre_wo={r['pre_wo_er']}, "
                    f"post_mixer={r['post_mixer_er']}, "
                    f"wo_delta={r['wo_compression']}, attn_delta={r['attn_compression']}\n")
    
    print(f"\n  Results saved to {csv_path}")
    print(f"  Summary saved to {summary_path}")
    
    hook.remove()
    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-W_o vs Post-W_o rank probe")
    parser.add_argument("--max-prompts", type=int, default=20,
                        help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Tokens per prompt")
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json",
                        help="Path to prompts JSON file")
    parser.add_argument("--output-dir", type=str, default="logs/wo_probe",
                        help="Output directory")
    args = parser.parse_args()
    
    print("Loading Genesis-152M...")
    model, tokenizer, config = load_genesis_model()
    
    # Load prompts
    import json
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        all_prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    prompts = all_prompts[:args.max_prompts]
    
    run_wo_probe(model, tokenizer, config, prompts,
                 max_new_tokens=args.max_tokens,
                 output_dir=args.output_dir)


if __name__ == "__main__":
    main()
