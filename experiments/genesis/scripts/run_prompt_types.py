"""Prompt-Type ER Re-measurement — Phase 3 Experiment 3.

Partitions prompts_200.json by cognitive category and measures
Shannon ER independently per category using Welford covariance.

Categories (from prompts_200.json):
  mathematical:      indices 0-9, 60-89       (40 prompts)
  creative:          indices 10-19, 90-126     (37 prompts)
  analytical:        indices 20-29, 127-157    (41 prompts)
  metacognitive:     indices 30-39, 158-178    (31 prompts)
  spatial_physical:  indices 40-49, 179-199    (31 prompts)
  social_emotional:  indices 50-59, 200-209    (20 prompts)

Usage:
    python scripts/run_prompt_types.py --max-prompts-per-cat 30 --max-tokens 64
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
    """Online covariance estimator (copied from run_corrected_er.py)."""
    
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


# Category definitions (prompt indices in prompts_200.json)
CATEGORIES = {
    "mathematical":     list(range(0, 10)) + list(range(64, 96)),
    "creative":         list(range(10, 20)) + list(range(96, 127)),
    "analytical":       list(range(20, 30)) + list(range(127, 158)),
    "metacognitive":    list(range(30, 40)) + list(range(158, 179)),
    "spatial_physical": list(range(40, 50)) + list(range(179, 200)),
    "social_emotional": list(range(50, 60)) + list(range(200, 209)),
}


class BlockOutputHook:
    """Captures block output hidden states."""
    
    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.data = {}
        self.handles = []
    
    def _make_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            if x is not None and x.dim() == 3:
                self.data[layer_idx] = x[:, -1, :].detach().float().cpu().numpy()
        return hook_fn
    
    @classmethod
    def attach(cls, model):
        inst = cls(len(model.blocks))
        for i, block in enumerate(model.blocks):
            h = block.register_forward_hook(inst._make_hook(i))
            inst.handles.append(h)
        return inst
    
    def clear(self):
        self.data = {}
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_prompt_type_measurement(model, tokenizer, config, all_prompts,
                                 max_prompts_per_cat=30, max_new_tokens=64,
                                 output_dir="logs/prompt_types"):
    """Measure ER per cognitive category."""
    
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    d = config.n_embd
    
    # Partition prompts by category
    cat_prompts = {}
    for cat_name, indices in CATEGORIES.items():
        valid_indices = [i for i in indices if i < len(all_prompts)]
        cat_prompts[cat_name] = [all_prompts[i] for i in valid_indices[:max_prompts_per_cat]]
        print(f"  {cat_name}: {len(cat_prompts[cat_name])} prompts (from {len(valid_indices)} available)")
    
    # Initialize per-category Welford accumulators
    # accum[category][layer_idx] = WelfordCovariance
    accum = {}
    for cat_name in CATEGORIES:
        accum[cat_name] = {i: WelfordCovariance(d) for i in range(n_layers)}
    
    hooks = BlockOutputHook.attach(model)
    t_start = time.time()
    total_run = 0
    
    for cat_name, prompts in cat_prompts.items():
        print(f"\n{'='*60}")
        print(f"CATEGORY: {cat_name} ({len(prompts)} prompts × {max_new_tokens} tokens)")
        print(f"{'='*60}")
        
        cat_samples = 0
        
        for pidx, prompt_text in enumerate(prompts):
            if isinstance(prompt_text, dict):
                prompt_text = prompt_text.get("text", prompt_text.get("prompt", str(prompt_text)))
            
            print(f"  [{pidx+1}/{len(prompts)}] {prompt_text[:50]}...")
            
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
                
                for i in range(n_layers):
                    if i in hooks.data:
                        vec = hooks.data[i].squeeze()
                        accum[cat_name][i].update(vec)
                
                cat_samples += 1
                
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                next_token = torch.tensor([[next_id]], device=device)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                eos_id = getattr(tokenizer, 'eos_token_id', None)
                if eos_id is not None and next_id == eos_id:
                    break
            
            total_run += 1
        
        # Category interim report
        er_final = accum[cat_name][n_layers - 1].get_shannon_er()
        er_mid = accum[cat_name][15].get_shannon_er()
        print(f"\n  {cat_name} interim: L15 ER={er_mid:.1f}, L29 ER={er_final:.1f} (N={cat_samples})")
    
    hooks.remove()
    total_time = time.time() - t_start
    
    # Final comparison table
    print(f"\n{'='*70}")
    print(f"PROMPT-TYPE ER COMPARISON")
    print(f"{'='*70}")
    
    # Header
    layers_to_show = [0, 3, 7, 15, 23, 27, 29]
    header = f"{'Category':<20}"
    for li in layers_to_show:
        header += f" {'L'+str(li):>6}"
    header += f" {'N':>6}"
    print(header)
    print("-" * len(header))
    
    # Write CSV
    with open(f"{output_dir}/prompt_type_er.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        csv_header = ["category", "N"] + [f"L{li}_er" for li in range(n_layers)]
        writer.writerow(csv_header)
        
        summary_lines = []
        for cat_name in CATEGORIES:
            n_samples = accum[cat_name][0].n
            row = f"{cat_name:<20}"
            csv_row = [cat_name, n_samples]
            
            for li in layers_to_show:
                er = accum[cat_name][li].get_shannon_er()
                row += f" {er:6.1f}"
            
            for li in range(n_layers):
                csv_row.append(f"{accum[cat_name][li].get_shannon_er():.2f}")
            
            row += f" {n_samples:6d}"
            print(row)
            summary_lines.append(row)
            writer.writerow(csv_row)
    
    # Summary file
    summary = f"""PROMPT-TYPE ER RE-MEASUREMENT
==============================
Method: Welford covariance per category, matched prompts per category
Max tokens per prompt: {max_new_tokens}
Time: {total_time/60:.1f} min

{header}
{'-' * len(header)}
""" + "\n".join(summary_lines) + "\n"
    
    with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Prompt-Type ER Re-measurement (Phase 3)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--max-prompts-per-cat", type=int, default=30,
                        help="Max prompts per category")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="logs/prompt_types")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        all_prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    
    print(f"\nLoaded {len(all_prompts)} prompts from {args.prompts}")
    print(f"Plan: {args.max_prompts_per_cat} prompts/category × {args.max_tokens} tokens")
    print(f"Categories: {len(CATEGORIES)}")
    
    run_prompt_type_measurement(model, tokenizer, config, all_prompts,
                                max_prompts_per_cat=args.max_prompts_per_cat,
                                max_new_tokens=args.max_tokens,
                                output_dir=args.output_dir)


if __name__ == "__main__":
    main()
