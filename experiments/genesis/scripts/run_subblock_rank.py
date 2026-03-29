"""Sub-block rank localization: where does effective rank drop?

Hooks into EACH sub-component within every Genesis block:
  1. Post-attn_norm (before mixer) 
  2. Post-mixer (after GLA/FoX attention, before residual add)
  3. Post-block (after FFN + residual, = block output)

Computes Shannon effective rank at each measurement point to answer:
  Q: Is the rank bottleneck in the mixer, the MLP, or the normalization?

This is the decisive experiment recommended by Perplexity to identify
why Genesis shows 8.3% effective rank vs Pythia's 86%.

Usage:
    python scripts/run_subblock_rank.py --max-prompts 5 --max-tokens 64
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
from spectral_microscope.analysis import compute_shannon_effective_rank


class SubBlockHook:
    """Hooks into sub-components within each GenesisBlock.
    
    For each block, attaches hooks to:
      - attn_norm (post-normalization, pre-mixer input)
      - attn (post-mixer output)  
      - ffn (post-MLP output, = post-block after residual)
    """
    
    def __init__(self):
        self.states = {}  # {(layer_idx, component): tensor}
        self.handles = []
    
    def _make_hook(self, layer_idx, component):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            if x.dim() == 3:  # (batch, seq, dim)
                self.states[(layer_idx, component)] = x[:, -1, :].detach().float().cpu()
            elif x.dim() == 2:  # (batch, dim)
                self.states[(layer_idx, component)] = x.detach().float().cpu()
        return hook_fn
    
    @classmethod
    def attach(cls, model):
        inst = cls()
        for i, block in enumerate(model.blocks):
            # Hook the normalization before attention
            if hasattr(block, 'attn_norm'):
                h = block.attn_norm.register_forward_hook(inst._make_hook(i, 'post_norm'))
                inst.handles.append(h)
            
            # Hook the attention/mixer output
            if hasattr(block, 'attn'):
                h = block.attn.register_forward_hook(inst._make_hook(i, 'post_mixer'))
                inst.handles.append(h)
            
            # Hook the FFN output
            if hasattr(block, 'ffn'):
                h = block.ffn.register_forward_hook(inst._make_hook(i, 'post_ffn'))
                inst.handles.append(h)
            
            # Hook the full block (post residual)
            h = block.register_forward_hook(inst._make_hook(i, 'post_block'))
            inst.handles.append(h)
        
        return inst
    
    def get_states(self):
        return dict(self.states)
    
    def clear(self):
        self.states = {}
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_subblock_rank(model, tokenizer, config, prompts, max_new_tokens=64,
                      output_csv="logs/subblock_rank.csv"):
    """Compute effective rank at each sub-block measurement point."""
    
    device = next(model.parameters()).device
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    
    components = ['post_norm', 'post_mixer', 'post_ffn', 'post_block']
    
    # CSV: one row per prompt, columns for each layer×component effective rank
    fieldnames = ["prompt_idx", "category", "prompt_short", "tokens"]
    for i in range(n_layers):
        ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
        for comp in components:
            fieldnames.append(f"L{i}_{ltype}_{comp}_er")
    # Aggregate columns
    for comp in components:
        fieldnames.extend([f"gla_mean_{comp}_er", f"fox_mean_{comp}_er"])
    
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
            
            # Collect per-component hidden states across generation
            comp_hiddens = {(i, c): [] for i in range(n_layers) for c in components}
            
            current_ids = input_ids
            past_kv = None
            hooks = SubBlockHook.attach(model)
            
            for step in range(max_new_tokens):
                hooks.clear()
                
                with torch.no_grad():
                    if past_kv is not None:
                        out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                    else:
                        out = model(current_ids, use_cache=True)
                    logits = out[0]
                    past_kv = out[3]
                
                states = hooks.get_states()
                for (layer_idx, comp), h in states.items():
                    comp_hiddens[(layer_idx, comp)].append(h.squeeze(0))
                
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
            
            # Compute effective rank for each layer × component
            row = {
                "prompt_idx": pidx, "category": category,
                "prompt_short": prompt_text[:80], "tokens": tokens_gen,
            }
            
            comp_aggregates = {c: {"gla": [], "fox": []} for c in components}
            
            for i in range(n_layers):
                ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
                for comp in components:
                    key = (i, comp)
                    if comp_hiddens[key]:
                        H = torch.stack(comp_hiddens[key], dim=0)
                        er = compute_shannon_effective_rank(H)
                    else:
                        er = 0.0
                    
                    row[f"L{i}_{ltype}_{comp}_er"] = f"{er:.2f}"
                    comp_aggregates[comp][ltype].append(er)
            
            # Aggregate columns
            for comp in components:
                for ltype in ["gla", "fox"]:
                    vals = comp_aggregates[comp][ltype]
                    mean_val = np.mean(vals) if vals else 0.0
                    row[f"{ltype}_mean_{comp}_er"] = f"{mean_val:.2f}"
            
            writer.writerow(row)
            
            # Print summary table
            print(f"  {'Component':<15} {'GLA mean ER':>12} {'FoX mean ER':>12} {'Δ(FoX-GLA)':>12}")
            print(f"  {'-'*51}")
            for comp in components:
                gla_m = np.mean(comp_aggregates[comp]["gla"]) if comp_aggregates[comp]["gla"] else 0
                fox_m = np.mean(comp_aggregates[comp]["fox"]) if comp_aggregates[comp]["fox"] else 0
                delta = fox_m - gla_m
                print(f"  {comp:<15} {gla_m:>10.1f}/576 {fox_m:>10.1f}/576 {delta:>+10.1f}")
    
    print(f"\n{'='*60}")
    print(f"Sub-block rank audit saved to: {output_csv}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Sub-Block Rank Localization")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json")
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=str, default="logs/subblock_rank.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"\nLoaded {len(prompts)} prompts")
    run_subblock_rank(model, tokenizer, config, prompts,
                      max_new_tokens=args.max_tokens, output_csv=args.output)


if __name__ == "__main__":
    main()
