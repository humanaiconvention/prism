"""Per-Prompt ER Variance — Phase 4 Experiment D.

For each prompt individually, computes an independent ER from its
generated tokens (64 samples per prompt). Reports distribution
statistics and identifies outlier prompts.

Usage:
    python scripts/run_per_prompt_er.py --max-prompts 60 --max-tokens 64
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


class BlockOutputHook:
    def __init__(self):
        self.data = {}
        self.handles = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            x = output[0] if isinstance(output, tuple) else output
            if x is not None and x.dim() == 3:
                self.data[layer_idx] = x[:, -1, :].detach().float().cpu().numpy()
        return hook_fn

    @classmethod
    def attach(cls, model):
        inst = cls()
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


def compute_er_from_matrix(X):
    """Compute Shannon ER from a (N x d) activation matrix via SVD."""
    if X.shape[0] < 2:
        return 0.0
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    s2 = s2[s2 > 1e-10]
    if len(s2) == 0:
        return 0.0
    p = s2 / s2.sum()
    H = -np.sum(p * np.log(p))
    return np.exp(H)


def run_per_prompt_er(model, tokenizer, config, prompts, max_new_tokens=64,
                      output_dir="logs/phase4/per_prompt_variance"):
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    d = config.n_embd
    key_layers = [0, 7, 15, 23, 29]

    hooks = BlockOutputHook.attach(model)
    t_start = time.time()

    # per_prompt_er[pidx][layer] = ER
    per_prompt_er = []

    for pidx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
        category = prompt_data.get("category", "unknown") if isinstance(prompt_data, dict) else "unknown"

        print(f"[{pidx+1}/{len(prompts)}] {category}: {prompt_text[:50]}...", end="")

        chat_input = format_chatml_prompt(prompt_text)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        current_ids = input_ids
        past_kv = None

        # Collect activations for this single prompt
        layer_activations = {i: [] for i in range(n_layers)}

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
                    layer_activations[i].append(hooks.data[i].squeeze())

            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            next_token = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break

        # Compute per-prompt ER for each layer
        prompt_results = {}
        for i in range(n_layers):
            if layer_activations[i]:
                X = np.array(layer_activations[i])
                prompt_results[i] = compute_er_from_matrix(X)
            else:
                prompt_results[i] = 0.0

        per_prompt_er.append({
            'prompt_idx': pidx,
            'text': prompt_text[:80],
            'category': category,
            'n_tokens': len(layer_activations[0]),
            'er': prompt_results,
        })

        print(f" | N={len(layer_activations[0])} | L29 ER={prompt_results[29]:.1f}")

    hooks.remove()
    total_time = time.time() - t_start

    # Write full CSV
    with open(f"{output_dir}/per_prompt_er.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["prompt_idx", "category", "n_tokens", "prompt_text"]
        header += [f"L{i}_er" for i in range(n_layers)]
        writer.writerow(header)

        for p in per_prompt_er:
            row = [p['prompt_idx'], p['category'], p['n_tokens'], p['text']]
            row += [f"{p['er'][i]:.2f}" for i in range(n_layers)]
            writer.writerow(row)

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"PER-PROMPT ER VARIANCE ANALYSIS (N={len(prompts)} prompts)")
    print(f"{'='*70}")

    summary_lines = [f"Per-prompt ER (each prompt = {max_new_tokens} tokens, SVD-based ER)\n"]
    header = f"{'Layer':<8} {'Mean':>7} {'Std':>7} {'Min':>7} {'Q25':>7} {'Median':>7} {'Q75':>7} {'Max':>7} {'CV%':>6}"
    print(header)
    print("-" * len(header))
    summary_lines.append(header)
    summary_lines.append("-" * len(header))

    for li in key_layers:
        vals = np.array([p['er'][li] for p in per_prompt_er])
        mean = np.mean(vals)
        std = np.std(vals)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        cv = 100 * std / mean if mean > 0 else 0

        line = f"L{li:<6d} {mean:7.1f} {std:7.1f} {np.min(vals):7.1f} {q25:7.1f} {med:7.1f} {q75:7.1f} {np.max(vals):7.1f} {cv:5.1f}%"
        print(line)
        summary_lines.append(line)

    # Identify outlier prompts at L29
    vals_29 = np.array([p['er'][29] for p in per_prompt_er])
    mean_29 = np.mean(vals_29)
    std_29 = np.std(vals_29)
    threshold_high = mean_29 + 2 * std_29
    threshold_low = mean_29 - 2 * std_29

    outliers_high = [(p['prompt_idx'], p['category'], p['er'][29], p['text'][:50])
                     for p in per_prompt_er if p['er'][29] > threshold_high]
    outliers_low = [(p['prompt_idx'], p['category'], p['er'][29], p['text'][:50])
                    for p in per_prompt_er if p['er'][29] < threshold_low]

    summary_lines.append(f"\nOutlier prompts at L29 (±2σ from mean {mean_29:.1f}):")
    if outliers_high:
        summary_lines.append("  HIGH ER:")
        for idx, cat, er, text in outliers_high:
            summary_lines.append(f"    #{idx} ({cat}): ER={er:.1f} — {text}")
    if outliers_low:
        summary_lines.append("  LOW ER:")
        for idx, cat, er, text in outliers_low:
            summary_lines.append(f"    #{idx} ({cat}): ER={er:.1f} — {text}")
    if not outliers_high and not outliers_low:
        summary_lines.append("  No outliers detected.")

    summary = f"""PER-PROMPT ER VARIANCE ANALYSIS
================================
Prompts: {len(prompts)}
Tokens per prompt: {max_new_tokens}
Time: {total_time/60:.1f} min

NOTE: Per-prompt ER is computed via SVD of a ({max_new_tokens} x {d}) matrix,
so it is capped at min(T,d)={min(max_new_tokens, d)}. This is fine for
*relative* comparisons between prompts, but absolute values are not
directly comparable to the Welford-based global ER.

""" + "\n".join(summary_lines) + "\n"

    with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Per-Prompt ER Variance (Phase 4)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="logs/phase4/per_prompt_variance")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    prompts = prompts[:args.max_prompts]

    print(f"\nPlan: Per-prompt ER for {len(prompts)} prompts × {args.max_tokens} tokens each")

    run_per_prompt_er(model, tokenizer, config, prompts,
                      max_new_tokens=args.max_tokens, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
