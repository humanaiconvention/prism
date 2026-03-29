"""GLA vs FoX Layer Comparison — Phase 4 Experiment C.

Full sub-block ER profile for ALL 30 layers (not just the 7 FoX).
Compares norm inflation, mixer compression, FFN recovery,
and residual compression across GLA vs FoX layer types.

Usage:
    python scripts/run_gla_vs_fox.py --max-prompts 60 --max-tokens 64
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


class FullSubBlockHook:
    """Captures all sub-block points for ALL 30 layers."""

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.data = {}
        self.handles = []

    def _make_hook(self, layer_idx, point_name):
        def hook_fn(module, inp, output):
            x = output[0] if isinstance(output, tuple) else output
            if x is not None and x.dim() == 3:
                self.data[(layer_idx, point_name)] = x[:, -1, :].detach().float().cpu().numpy()
        return hook_fn

    def _make_io_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            x_in = inp[0] if isinstance(inp, tuple) else inp
            if x_in is not None and x_in.dim() == 3:
                self.data[(layer_idx, 'block_input')] = x_in[:, -1, :].detach().float().cpu().numpy()
            x_out = output[0] if isinstance(output, tuple) else output
            if x_out is not None and x_out.dim() == 3:
                self.data[(layer_idx, 'block_output')] = x_out[:, -1, :].detach().float().cpu().numpy()
        return hook_fn

    @classmethod
    def attach(cls, model):
        inst = cls(len(model.blocks))
        for i, block in enumerate(model.blocks):
            h = block.register_forward_hook(inst._make_io_hook(i))
            inst.handles.append(h)
            if hasattr(block, 'attn_norm'):
                h = block.attn_norm.register_forward_hook(inst._make_hook(i, 'post_norm'))
                inst.handles.append(h)
            if hasattr(block, 'attn'):
                h = block.attn.register_forward_hook(inst._make_hook(i, 'post_mixer'))
                inst.handles.append(h)
            if hasattr(block, 'ffn'):
                h = block.ffn.register_forward_hook(inst._make_hook(i, 'post_ffn'))
                inst.handles.append(h)
        return inst

    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_gla_vs_fox(model, tokenizer, config, prompts, max_new_tokens=64,
                   output_dir="logs/phase4/gla_vs_fox"):
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    d = config.n_embd

    points = ['block_input', 'post_norm', 'post_mixer', 'post_ffn', 'block_output']
    accum = {}
    for i in range(n_layers):
        for pt in points:
            accum[(i, pt)] = WelfordCovariance(d)

    hooks = FullSubBlockHook.attach(model)
    total_samples = 0
    t_start = time.time()

    for pidx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
        print(f"\n[{pidx+1}/{len(prompts)}] {prompt_text[:50]}...")

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
                for pt in points:
                    if (i, pt) in hooks.data:
                        accum[(i, pt)].update(hooks.data[(i, pt)].squeeze())
            total_samples += 1

            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            next_token = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break

        elapsed_total = time.time() - t_start
        remaining = len(prompts) - (pidx + 1)
        eta = remaining * elapsed_total / (pidx + 1) / 60
        print(f"  {step+1} tokens | N={total_samples} | ETA: {eta:.0f} min")

    hooks.remove()
    total_time = time.time() - t_start

    # Write full CSV
    with open(f"{output_dir}/gla_vs_fox_full.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "type", "N",
                         "input_er", "norm_er", "mixer_er", "ffn_er", "output_er",
                         "norm_inflation", "mixer_compression", "ffn_recovery", "residual_compression"])

        for i in range(n_layers):
            ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
            er = {pt: accum[(i, pt)].get_shannon_er() for pt in points}
            norm_infl = er['post_norm'] - er['block_input']
            mix_comp = er['post_mixer'] - er['post_norm']
            ffn_rec = er['post_ffn'] - er['post_mixer']
            res_comp = er['block_output'] - er['post_ffn']

            writer.writerow([i, ltype, total_samples,
                             f"{er['block_input']:.2f}", f"{er['post_norm']:.2f}",
                             f"{er['post_mixer']:.2f}", f"{er['post_ffn']:.2f}",
                             f"{er['block_output']:.2f}",
                             f"{norm_infl:.2f}", f"{mix_comp:.2f}",
                             f"{ffn_rec:.2f}", f"{res_comp:.2f}"])

    # Summary: GLA vs FoX comparison
    print(f"\n{'='*90}")
    print(f"GLA vs FoX FULL SUB-BLOCK COMPARISON (N={total_samples})")
    print(f"{'='*90}")

    header = f"{'Layer':<10} {'Type':<5} {'Input':>7} {'Norm':>7} {'Mixer':>7} {'FFN':>7} {'Output':>7} | {'ΔNorm':>7} {'ΔMix':>7} {'ΔFFN':>7} {'ΔRes':>7}"
    print(header)
    print("-" * len(header))

    gla_stats = {'norm': [], 'mix': [], 'ffn': [], 'res': []}
    fox_stats = {'norm': [], 'mix': [], 'ffn': [], 'res': []}
    summary_lines = [header, "-" * len(header)]

    for i in range(n_layers):
        ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
        er = {pt: accum[(i, pt)].get_shannon_er() for pt in points}
        norm_infl = er['post_norm'] - er['block_input']
        mix_comp = er['post_mixer'] - er['post_norm']
        ffn_rec = er['post_ffn'] - er['post_mixer']
        res_comp = er['block_output'] - er['post_ffn']

        stats = fox_stats if ltype == "fox" else gla_stats
        stats['norm'].append(norm_infl)
        stats['mix'].append(mix_comp)
        stats['ffn'].append(ffn_rec)
        stats['res'].append(res_comp)

        marker = " <<FoX" if ltype == "fox" else ""
        line = f"L{i:2d}       {ltype:<5} {er['block_input']:7.1f} {er['post_norm']:7.1f} {er['post_mixer']:7.1f} {er['post_ffn']:7.1f} {er['block_output']:7.1f} | {norm_infl:+7.1f} {mix_comp:+7.1f} {ffn_rec:+7.1f} {res_comp:+7.1f}{marker}"
        print(line)
        summary_lines.append(line)

    # Aggregate comparison
    agg_lines = ["\n--- AGGREGATE COMPARISON ---"]
    for label, key in [("Norm inflation", "norm"), ("Mixer compression", "mix"),
                       ("FFN recovery", "ffn"), ("Residual compression", "res")]:
        gla_mean = np.mean(gla_stats[key])
        fox_mean = np.mean(fox_stats[key])
        line = f"  {label:<25}: GLA avg = {gla_mean:+7.1f} | FoX avg = {fox_mean:+7.1f} | Δ = {fox_mean-gla_mean:+7.1f}"
        print(line)
        agg_lines.append(line)

    summary = f"""GLA vs FoX FULL SUB-BLOCK COMPARISON
======================================
N = {total_samples} (d = {d})
Time: {total_time/60:.1f} min
Layers: {len(GLA_LAYER_INDICES)} GLA + {len(FOX_LAYER_INDICES)} FoX

""" + "\n".join(summary_lines) + "\n" + "\n".join(agg_lines) + "\n"

    with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="GLA vs FoX Spectral Comparison (Phase 4)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="logs/phase4/gla_vs_fox")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    prompts = prompts[:args.max_prompts]

    print(f"\nPlan: Full sub-block profile for ALL {config.n_layer} layers")
    print(f"  {len(prompts)} prompts × {args.max_tokens} tokens = {len(prompts)*args.max_tokens} samples")

    run_gla_vs_fox(model, tokenizer, config, prompts,
                   max_new_tokens=args.max_tokens, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
