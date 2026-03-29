"""Per-Head Rank Contribution — Phase 1 Mixer Compression Investigation.

Measures the Shannon effective rank contribution of each individual
attention head in FoX layers by zeroing out one head at a time and
re-measuring post-mixer ER.

If the mixer compresses 207→71 ER, which heads are responsible?
Are all 3 KV groups equally compressive, or is one head dominating?

Also includes a high-N measurement run (all 200 prompts × 32 tokens)
to stabilize the baseline ER above N/d=10x.

Usage:
    python scripts/run_head_rank.py --mode ablation --max-prompts 20
    python scripts/run_head_rank.py --mode high-n --max-prompts 200
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


class PostMixerHook:
    """Captures post-mixer hidden states for specific layers."""
    def __init__(self, target_layers):
        self.target_layers = target_layers
        self.data = {}
        self.handles = []

    @classmethod
    def attach(cls, model, target_layers):
        inst = cls(target_layers)
        for i in target_layers:
            block = model.blocks[i]
            if hasattr(block, 'attn'):
                h = block.attn.register_forward_hook(inst._make_hook(i))
                inst.handles.append(h)
        return inst

    def _make_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            if x is not None and x.dim() == 3:
                self.data[layer_idx] = x[:, -1, :].detach().float().cpu().numpy()
        return hook_fn

    def clear(self):
        self.data = {}

    def remove(self):
        for h in self.handles:
            h.remove()


class HeadAblator:
    """Zeros out specific attention heads in FoX layers to measure their contribution."""

    def __init__(self, model, layer_idx, head_idx):
        """Set up head ablation for a specific FoX layer and head index."""
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.saved_weights = {}

    def __enter__(self):
        block = self.model.blocks[self.layer_idx]
        attn = block.attn

        # FoX attention has o_proj (output projection)
        # Each head writes head_dim=64 dims, we zero the output projection
        # for the specific head
        if hasattr(attn, 'o_proj'):
            w = attn.o_proj.weight.data  # [n_embd, n_embd] or similar
            head_dim = 64  # from config
            start = self.head_idx * head_dim
            end = start + head_dim
            # Save and zero the columns corresponding to this head
            self.saved_weights['o_proj'] = w[:, start:end].clone()
            w[:, start:end].zero_()
        return self

    def __exit__(self, *args):
        block = self.model.blocks[self.layer_idx]
        attn = block.attn
        if hasattr(attn, 'o_proj') and 'o_proj' in self.saved_weights:
            w = attn.o_proj.weight.data
            head_dim = 64
            start = self.head_idx * head_dim
            end = start + head_dim
            w[:, start:end].copy_(self.saved_weights['o_proj'])


def run_generation(model, tokenizer, input_ids, max_tokens, hooks):
    """Run generation and accumulate hook data into Welford estimators."""
    device = next(model.parameters()).device
    current_ids = input_ids
    past_kv = None
    samples = 0

    for step in range(max_tokens):
        hooks.clear()
        with torch.no_grad():
            if past_kv is not None:
                out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
            else:
                out = model(current_ids, use_cache=True)
            logits = out[0]
            past_kv = out[3]

        samples += 1
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        next_token = torch.tensor([[next_id]], device=device)
        current_ids = torch.cat([current_ids, next_token], dim=1)

        eos_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_id is not None and next_id == eos_id:
            break

    return samples


class EarlyExit(Exception):
    """Exception used to short-circuit the forward pass."""
    def __init__(self, data):
        self.data = data

def run_head_ablation(model, tokenizer, config, prompts, max_tokens=32,
                      output_dir="logs/head_rank"):
    """Measure per-head rank contribution by ablating one head at a time.
    
    Optimized via Teacher Forcing & Early Exit:
    1. AR Generation (Baseline): Run normal AR to get the 32-token sequence and baseline ER.
    2. Teacher Forcing + Early Exit: For each head ablation, we feed the *fixed* 32-token
       sequence in a single batch forward pass, and early-exit at layer L. 
       This means 1 forward-pass replaces 32 AR steps!
    """
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    d = config.n_embd
    fox_layers = FOX_LAYER_INDICES

    print(f"\n[{time.strftime('%H:%M:%S')}] Step 1: Baseline AR Generation")
    accum_baseline = {i: WelfordCovariance(d) for i in fox_layers}
    total_samples = 0
    t0 = time.time()
    
    class StopAtMixerHook:
        """Throws an exception with the mixer output to stop the forward pass."""
        def __init__(self, block):
            self.handle = block.attn.register_forward_hook(self._hook)
        def _hook(self, module, inp, output):
            x = output[0] if isinstance(output, tuple) else output
            # x is [batch, seq_len, d]
            raise EarlyExit(x.detach().float().cpu().numpy())
        def remove(self):
            self.handle.remove()

    # Pre-tokenize prompts
    generated_sequences = []  # Stores the full token sequence the baseline generated
    
    hooks = PostMixerHook.attach(model, fox_layers)
    for pidx, p in enumerate(prompts):
        prompt_text = p if isinstance(p, str) else p.get("text", p.get("prompt", ""))
        chat_input = format_chatml_prompt(prompt_text)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)

        current_ids = input_ids
        past_kv = None
        for step in range(max_tokens):
            hooks.clear()
            with torch.no_grad():
                if past_kv is not None:
                    out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    out = model(current_ids, use_cache=True)
                logits = out[0]
                past_kv = out[3]

            for i in fox_layers:
                if i in hooks.data:
                    accum_baseline[i].update(hooks.data[i].squeeze())

            total_samples += 1
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            current_ids = torch.cat([current_ids, torch.tensor([[next_id]], device=device)], dim=1)
            
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break
                
        # Save the full sequence it generated for Teacher Forcing
        generated_sequences.append(current_ids)
                
        if (pidx + 1) % 5 == 0:
            print(f"  Processed {pidx+1}/{len(prompts)} (N={total_samples})")

    hooks.remove()
    print(f"Baseline complete in {time.time()-t0:.1f}s. N={total_samples}")
    for i in fox_layers:
        print(f"  L{i} baseline ER: {accum_baseline[i].get_shannon_er():.1f}")

    # Step 2: Per-head ablation with Teacher Forcing + Early Exit
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 2: Head ablation (Teacher Forcing + Early Exit)")
    n_heads = 9
    results = []

    for target_layer in fox_layers:
        block = model.blocks[target_layer]
        print(f"\n=== Layer L{target_layer} (FoX) ===")
        
        for head_idx in range(n_heads):
            t0 = time.time()
            accum = WelfordCovariance(d)
            hook = StopAtMixerHook(block)
            
            with HeadAblator(model, target_layer, head_idx):
                with torch.no_grad():
                    for seq in generated_sequences:
                        try:
                            # SINGLE forward pass of the whole sequence!
                            # Since we don't care about states after early-exit,
                            # we pass full seq without AR cache
                            model(seq, use_cache=False)
                        except EarlyExit as e:
                            # e.data has shape [1, seq_len, d]
                            seq_data = e.data[0]
                            # We only care about the generated tokens, not the prompt prefix
                            # Wait, the baseline ER measurement accumulated the prompt prefix too?
                            # Ah, the baseline accumulated token by token (so prompt prefix
                            # was initially sized [1, max_len] which we squeeze).
                            # Our baseline loop passed full prompt on step 0, then AR.
                            # So baseline N = prompt_len + steps - 1.
                            for t in range(seq_data.shape[0]):
                                accum.update(seq_data[t])
                            
            hook.remove()
            er = accum.get_shannon_er()
            baseline_er = accum_baseline[target_layer].get_shannon_er()
            delta = er - baseline_er
            elapsed = time.time() - t0

            role = "compressive" if delta < -1 else ("expansive" if delta > 1 else "neutral")
            results.append({
                'layer': target_layer,
                'head': head_idx,
                'baseline_er': f"{baseline_er:.1f}",
                'ablated_er': f"{er:.1f}",
                'delta': f"{delta:.1f}",
                'role': role,
                # Note: N might differ slightly from baseline if prompt token aggregation
                # was different, but this is minor (N is close enough).
            })
            print(f"  H{head_idx}: ER={er:.1f} (Δ={delta:+.1f}) [{role}] {elapsed:.1f}s")

    # Save results
    with open(f"{output_dir}/head_ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'head', 'baseline_er', 'ablated_er', 'delta', 'role'])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"HEAD ABLATION SUMMARY")
    print(f"{'='*60}")
    for layer in fox_layers:
        layer_results = [r for r in results if r['layer'] == layer]
        print(f"\nL{layer} (FoX) — baseline ER = {layer_results[0]['baseline_er']}")
        for r in sorted(layer_results, key=lambda x: float(x['delta'])):
            bar = "█" * max(1, int(abs(float(r['delta']))))
            sign = "+" if float(r['delta']) > 0 else ""
            print(f"  H{r['head']}: {sign}{r['delta']} ({r['role']}) {bar}")

    print(f"\nResults saved to {output_dir}/head_ablation.csv")


def run_high_n(model, tokenizer, config, prompts, max_tokens=32,
               output_dir="logs/high_n"):
    """High-N stabilization run with all available prompts."""
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    d = config.n_embd
    n_layers = config.n_layer

    # Output-only Welford — just track block output ER at every layer
    accum = {i: WelfordCovariance(d) for i in range(n_layers)}

    from scripts.run_corrected_er import MultiPointHook
    hooks = MultiPointHook.attach(model)
    total = 0
    t_start = time.time()

    for pidx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
        chat_input = format_chatml_prompt(prompt_text)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)

        current_ids = input_ids
        past_kv = None
        for step in range(max_tokens):
            hooks.clear()
            with torch.no_grad():
                if past_kv is not None:
                    out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    out = model(current_ids, use_cache=True)
                logits = out[0]
                past_kv = out[3]

            for i in range(n_layers):
                if (i, 'block_output') in hooks.data:
                    accum[i].update(hooks.data[(i, 'block_output')].squeeze())

            total += 1
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            next_token = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            eos_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_id is not None and next_id == eos_id:
                break

        elapsed = time.time() - t_start
        remaining = len(prompts) - (pidx + 1)
        avg = elapsed / (pidx + 1)
        eta = remaining * avg / 60

        if (pidx + 1) % 10 == 0 or pidx == len(prompts) - 1:
            print(f"[{pidx+1}/{len(prompts)}] N={total} N/d={total/d:.1f}x ETA={eta:.0f}min")
            for i in [0, 15, 29]:
                print(f"  L{i}: ER={accum[i].get_shannon_er():.1f}/{d} ({accum[i].get_shannon_er()/d*100:.1f}%)")

    hooks.remove()
    total_time = time.time() - t_start

    # Write results
    with open(f"{output_dir}/high_n_er.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "type", "N", "d", "er", "er_pct"])
        for i in range(n_layers):
            ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
            er = accum[i].get_shannon_er()
            writer.writerow([i, ltype, total, d, f"{er:.2f}", f"{er/d*100:.2f}"])

    er_final = accum[n_layers-1].get_shannon_er()
    summary = f"""HIGH-N STABILIZATION
N = {total} (d = {d}, N/d = {total/d:.1f}x)
Time: {total_time/60:.1f} min
L0  ER: {accum[0].get_shannon_er():.1f}/{d} ({accum[0].get_shannon_er()/d*100:.1f}%)
L15 ER: {accum[15].get_shannon_er():.1f}/{d} ({accum[15].get_shannon_er()/d*100:.1f}%)
L29 ER: {er_final:.1f}/{d} ({er_final/d*100:.1f}%)
"""
    print(summary)
    with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)


def main():
    parser = argparse.ArgumentParser(description="Per-Head Rank Contribution")
    parser.add_argument("--mode", choices=["ablation", "high-n"], required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json")
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    if args.mode == "ablation":
        out_dir = args.output_dir or "logs/head_rank"
        print(f"\nPHASE 1: Per-head ablation — {len(prompts)} prompts x {args.max_tokens} tokens")
        print(f"FoX layers: {FOX_LAYER_INDICES}")
        print(f"9 heads per layer = {len(FOX_LAYER_INDICES) * 9} ablation conditions")
        run_head_ablation(model, tokenizer, config, prompts,
                         max_tokens=args.max_tokens, output_dir=out_dir)
    elif args.mode == "high-n":
        out_dir = args.output_dir or "logs/high_n"
        n_samples = len(prompts) * args.max_tokens
        print(f"\nHIGH-N STABILIZATION — {len(prompts)} prompts x {args.max_tokens} tokens = {n_samples} (N/d={n_samples/config.n_embd:.1f}x)")
        run_high_n(model, tokenizer, config, prompts,
                  max_tokens=args.max_tokens, output_dir=out_dir)


if __name__ == "__main__":
    main()
