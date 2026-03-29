"""Bootstrap Confidence Intervals — Phase 4 Experiment B.

Runs 10 bootstrap replicates of the Welford ER measurement at N=3840
(60 prompts × 64 tokens) with different random seeds and prompt orderings.
Reports mean, std, and 95% CI for key layer ER values.

Usage:
    python scripts/run_bootstrap_er.py --n-replicates 10 --max-prompts 60 --max-tokens 64
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import argparse
import csv
import json
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import (
    load_genesis_model, format_chatml_prompt,
    FOX_LAYER_INDICES, GLA_LAYER_INDICES,
)


class WelfordCovariance:
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


def run_single_replicate(model, tokenizer, config, prompts, max_new_tokens, seed):
    """Run a single ER measurement with a specific seed/prompt ordering."""
    device = next(model.parameters()).device
    n_layers = config.n_layer
    d = config.n_embd

    rng = random.Random(seed)
    shuffled = list(prompts)
    rng.shuffle(shuffled)

    accum = {i: WelfordCovariance(d) for i in range(n_layers)}
    hooks = BlockOutputHook.attach(model)
    total_samples = 0

    for pidx, prompt_data in enumerate(shuffled):
        prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))

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
                    accum[i].update(hooks.data[i].squeeze())
            total_samples += 1

            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            next_token = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break

    hooks.remove()

    results = {}
    for i in range(n_layers):
        results[i] = accum[i].get_shannon_er()
    return results, total_samples


def main():
    parser = argparse.ArgumentParser(description="Bootstrap ER Confidence Intervals (Phase 4)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="logs/phase4/bootstrap_ci")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.base_seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    prompts = prompts[:args.max_prompts]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    d = config.n_embd
    key_layers = [0, 3, 7, 15, 23, 27, 29]

    print(f"\nBootstrap ER: {args.n_replicates} replicates × {len(prompts)} prompts × {args.max_tokens} tokens")
    print(f"Key layers: {key_layers}")

    all_results = {i: [] for i in range(n_layers)}
    t_start = time.time()

    for rep in range(args.n_replicates):
        seed = args.base_seed + rep * 7
        print(f"\n--- Replicate {rep+1}/{args.n_replicates} (seed={seed}) ---")
        t0 = time.time()

        results, n_samples = run_single_replicate(model, tokenizer, config, prompts, args.max_tokens, seed)

        for i in range(n_layers):
            all_results[i].append(results[i])

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = (args.n_replicates - rep - 1) * total_elapsed / (rep + 1)
        print(f"  N={n_samples} | L29 ER={results[29]:.1f} | {elapsed:.0f}s | ETA: {eta/60:.0f} min")

        # Interim stats
        if rep >= 1:
            for li in key_layers:
                vals = all_results[li]
                print(f"    L{li}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

    total_time = time.time() - t_start

    # Final results
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"  {args.n_replicates} replicates, N={len(prompts)*args.max_tokens} each")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'='*70}")

    with open(f"{args.output_dir}/bootstrap_er.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["layer", "type", "mean_er", "std_er", "ci95_low", "ci95_high", "min_er", "max_er"]
        header += [f"rep_{i}" for i in range(args.n_replicates)]
        writer.writerow(header)

        summary_lines = []
        for i in range(n_layers):
            ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
            vals = np.array(all_results[i])
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            ci_low = mean - 1.96 * std / np.sqrt(len(vals))
            ci_high = mean + 1.96 * std / np.sqrt(len(vals))

            row = [i, ltype, f"{mean:.2f}", f"{std:.2f}", f"{ci_low:.2f}", f"{ci_high:.2f}",
                   f"{np.min(vals):.2f}", f"{np.max(vals):.2f}"]
            row += [f"{v:.2f}" for v in vals]
            writer.writerow(row)

            if i in key_layers:
                line = f"L{i:2d} ({ltype}): {mean:6.1f} ± {std:4.1f}  95%CI [{ci_low:.1f}, {ci_high:.1f}]  range [{np.min(vals):.1f}, {np.max(vals):.1f}]"
                print(line)
                summary_lines.append(line)

    summary = f"""BOOTSTRAP CONFIDENCE INTERVALS
==============================
Replicates: {args.n_replicates}
N per replicate: {len(prompts)*args.max_tokens}
Time: {total_time/60:.1f} min

Key Layer Results (mean ± std, 95% CI):
""" + "\n".join(summary_lines) + "\n"

    with open(f"{args.output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
