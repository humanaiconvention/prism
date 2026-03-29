"""Energy Ratio Test: Verify the "Spectral Eclipse" mechanism.

Measures ||x_residual||, ||block_output||, and ||x_new|| at every block
to test whether the -5 ER drop at the residual add is caused by
||block_output|| >> ||x_residual|| at deeper layers.

If ratio ||y||/||x|| >> 1 in deep layers, the block output dominates
the sum and the skip-path features are "eclipsed."

References:
  - Peri-LN: "massive activations" in Pre-LN, norm growth pathology
  - "Residual Connections Harm Generative Representation Learning"

Usage:
    python scripts/run_energy_ratio.py --max-prompts 5 --max-tokens 64
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
from prism.analysis import compute_shannon_effective_rank


class EnergyRatioHook:
    """Captures block input AND output to compute energy ratios.
    
    For each block, captures:
      - x_in: input to the block (the residual stream before this block)
      - x_out: output of the block (residual stream after this block)
    
    Then computes:
      - ||x|| = norm of input (skip path)
      - ||y|| = norm of (x_out - x_in) = block's contribution
      - ||x+y|| = norm of output
      - ratio = ||y|| / ||x||
    """
    
    def __init__(self):
        self.inputs = {}   # layer_idx -> tensor
        self.outputs = {}  # layer_idx -> tensor
        self.handles = []
    
    def _make_hook(self, layer_idx):
        def hook_fn(module, inp, output):
            # Input
            if isinstance(inp, tuple):
                x_in = inp[0]
            else:
                x_in = inp
            if x_in.dim() == 3:
                self.inputs[layer_idx] = x_in[:, -1, :].detach().float().cpu()
            
            # Output
            if isinstance(output, tuple):
                x_out = output[0]
            else:
                x_out = output
            if x_out.dim() == 3:
                self.outputs[layer_idx] = x_out[:, -1, :].detach().float().cpu()
        return hook_fn
    
    @classmethod
    def attach(cls, model):
        inst = cls()
        for i, block in enumerate(model.blocks):
            h = block.register_forward_hook(inst._make_hook(i))
            inst.handles.append(h)
        return inst
    
    def get_energy_ratios(self):
        """Returns dict of layer_idx -> {x_norm, y_norm, sum_norm, ratio, er_x, er_y, er_sum}"""
        results = {}
        for i in sorted(self.inputs.keys()):
            if i in self.outputs:
                x = self.inputs[i].squeeze(0)
                x_out = self.outputs[i].squeeze(0)
                y = x_out - x  # block contribution
                
                x_norm = x.norm().item()
                y_norm = y.norm().item()
                sum_norm = x_out.norm().item()
                ratio = y_norm / max(x_norm, 1e-8)
                
                # Cosine similarity between skip and block output
                cos_sim = torch.nn.functional.cosine_similarity(
                    x.unsqueeze(0), y.unsqueeze(0)
                ).item()
                
                results[i] = {
                    'x_norm': x_norm,
                    'y_norm': y_norm,
                    'sum_norm': sum_norm,
                    'ratio': ratio,
                    'cos_sim': cos_sim,
                }
        return results
    
    def clear(self):
        self.inputs = {}
        self.outputs = {}
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_energy_ratio(model, tokenizer, config, prompts, max_new_tokens=64,
                     output_csv="logs/energy_ratio.csv"):
    """Measure energy ratios across depth for each prompt."""
    
    device = next(model.parameters()).device
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    
    fieldnames = ["prompt_idx", "category", "prompt_short", "tokens"]
    for i in range(n_layers):
        ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
        fieldnames.extend([
            f"L{i}_{ltype}_x_norm",
            f"L{i}_{ltype}_y_norm",
            f"L{i}_{ltype}_ratio",
            f"L{i}_{ltype}_cos_sim",
        ])
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pidx, prompt_data in enumerate(prompts):
            prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
            category = prompt_data.get("category", "unknown") if isinstance(prompt_data, dict) else "unknown"
            
            print(f"\n[{pidx+1}/{len(prompts)}] {category}: {prompt_text[:60]}...")
            t0 = time.time()
            
            chat_input = format_chatml_prompt(prompt_text)
            input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
            
            # Accumulate energy ratios across steps
            layer_ratios = {i: {'x_norms': [], 'y_norms': [], 'ratios': [], 'cos_sims': []}
                          for i in range(n_layers)}
            
            current_ids = input_ids
            past_kv = None
            hooks = EnergyRatioHook.attach(model)
            
            for step in range(max_new_tokens):
                hooks.clear()
                
                with torch.no_grad():
                    if past_kv is not None:
                        out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                    else:
                        out = model(current_ids, use_cache=True)
                    logits = out[0]
                    past_kv = out[3]
                
                ratios = hooks.get_energy_ratios()
                for i, data in ratios.items():
                    layer_ratios[i]['x_norms'].append(data['x_norm'])
                    layer_ratios[i]['y_norms'].append(data['y_norm'])
                    layer_ratios[i]['ratios'].append(data['ratio'])
                    layer_ratios[i]['cos_sims'].append(data['cos_sim'])
                
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                next_token = torch.tensor([[next_id]], device=device)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                eos_id = getattr(tokenizer, 'eos_token_id', None)
                if eos_id is not None and next_id == eos_id:
                    break
            
            hooks.remove()
            tokens_gen = step + 1
            elapsed = time.time() - t0
            print(f"  {tokens_gen} tokens in {elapsed:.1f}s")
            
            # Write row with mean values across steps
            row = {
                "prompt_idx": pidx, "category": category,
                "prompt_short": prompt_text[:80], "tokens": tokens_gen,
            }
            
            for i in range(n_layers):
                ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
                lr = layer_ratios[i]
                row[f"L{i}_{ltype}_x_norm"] = f"{np.mean(lr['x_norms']):.1f}" if lr['x_norms'] else "0"
                row[f"L{i}_{ltype}_y_norm"] = f"{np.mean(lr['y_norms']):.1f}" if lr['y_norms'] else "0"
                row[f"L{i}_{ltype}_ratio"] = f"{np.mean(lr['ratios']):.3f}" if lr['ratios'] else "0"
                row[f"L{i}_{ltype}_cos_sim"] = f"{np.mean(lr['cos_sims']):.3f}" if lr['cos_sims'] else "0"
            
            writer.writerow(row)
            
            # Print summary
            print(f"\n  {'Layer':<10} {'||x||':>8} {'||y||':>8} {'||y||/||x||':>11} {'cos(x,y)':>9}")
            print(f"  {'-'*50}")
            for i in range(n_layers):
                ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
                lr = layer_ratios[i]
                if lr['ratios']:
                    x_m = np.mean(lr['x_norms'])
                    y_m = np.mean(lr['y_norms'])
                    r_m = np.mean(lr['ratios'])
                    c_m = np.mean(lr['cos_sims'])
                    marker = " *** FoX" if ltype == "fox" else ""
                    bar = "█" * min(int(r_m * 20), 40)
                    print(f"  L{i:2d} ({ltype}) {x_m:8.0f} {y_m:8.0f} {r_m:11.3f} {c_m:+9.3f}  {bar}{marker}")
    
    print(f"\n{'='*60}")
    print(f"Energy ratio audit saved to: {output_csv}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Energy Ratio Test")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json")
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=str, default="logs/energy_ratio.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"\nLoaded {len(prompts)} prompts")
    run_energy_ratio(model, tokenizer, config, prompts,
                     max_new_tokens=args.max_tokens, output_csv=args.output)


if __name__ == "__main__":
    main()
