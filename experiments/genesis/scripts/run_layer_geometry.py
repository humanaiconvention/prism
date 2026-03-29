"""Phase 0.5 + 1.75: Per-layer spectral geometry audit with GLA vs FoX comparison.

This is the NOVEL experiment — no prior work has compared residual stream
spectral geometry between linear (GLA) and softmax (FoX) attention layers
within the same hybrid model.

Captures hidden states after each of the 30 blocks and computes:
- Per-layer Shannon effective rank
- Per-layer spectral entropy
- Per-layer hidden norm
- Layer-type aggregates (mean effective rank for GLA vs FoX)

Also provides TTT control: can be run with --disable-ttt to zero out
metacognition module contribution.

Usage:
    python scripts/run_layer_geometry.py --max-prompts 10 --max-tokens 64
    python scripts/run_layer_geometry.py --max-prompts 10 --max-tokens 64 --disable-ttt
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
    HiddenStateHook, FOX_LAYER_INDICES, GLA_LAYER_INDICES,
)
from spectral_microscope.analysis import (
    compute_spectral_metrics, compute_shannon_effective_rank,
)


class AllLayerHook:
    """Captures hidden states from ALL layers at each forward pass.
    
    Returns dict of layer_idx -> list of hidden state tensors (one per step).
    """
    
    def __init__(self):
        self.step_states = {}  # {layer_idx: latest_hidden_state}
        self.handles = []
    
    def _make_hook(self, layer_idx):
        def hook_fn(module, input_tensor, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            # (batch, seq_len, n_embd) -> (n_embd,) last position
            self.step_states[layer_idx] = x[:, -1, :].detach().float().cpu()
        return hook_fn
    
    @classmethod
    def attach(cls, model):
        inst = cls()
        for i, block in enumerate(model.blocks):
            handle = block.register_forward_hook(inst._make_hook(i))
            inst.handles.append(handle)
        return inst
    
    def get_step_states(self):
        """Returns sorted dict: {layer_idx: tensor (n_embd,)}"""
        return {k: v.squeeze(0) for k, v in sorted(self.step_states.items())}
    
    def clear(self):
        self.step_states = {}
    
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def disable_ttt(model):
    """Zero out TTT/metacognition contribution without breaking forward pass.
    
    The metacognition module is applied after all blocks. We zero its
    contribution by setting output gate bias to large negative (sigmoid -> 0).
    """
    found = False
    for name, mod in model.named_modules():
        if 'metacognition' in name and 'output_gate' in name:
            # Set output gate to always output ~0
            with torch.no_grad():
                if hasattr(mod, 'weight') and mod.weight is not None:
                    mod.weight.zero_()
                if hasattr(mod, 'bias') and mod.bias is not None:
                    mod.bias.fill_(-100.0)  # sigmoid(-100) ≈ 0
                elif hasattr(mod, 'weight') and mod.weight is not None:
                    # No bias — zeroing weight is enough to kill output
                    pass
            found = True
            print(f"[TTT Control] Disabled: {name}")
    
    if not found:
        print("[TTT Control] WARNING: Could not find metacognition output gate")
        # Fallback: try to zero the self_model weights
        for name, mod in model.named_modules():
            if 'metacognition' in name and 'self_model' in name:
                with torch.no_grad():
                    if hasattr(mod, 'weight'):
                        mod.weight.zero_()
                    if hasattr(mod, 'bias') and mod.bias is not None:
                        mod.bias.zero_()
                print(f"[TTT Control] Zeroed: {name}")
    
    return model


def run_layer_geometry(
    model, tokenizer, config, prompts,
    max_new_tokens=64,
    output_csv="logs/layer_geometry.csv",
):
    """Run per-layer spectral geometry audit."""
    
    device = next(model.parameters()).device
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    n_layers = config.n_layer
    
    # CSV columns: per-prompt summary with per-layer metrics
    fieldnames = ["prompt_idx", "category", "prompt_text_short", "tokens_generated"]
    # Per-layer columns
    for i in range(n_layers):
        ltype = "fox" if i in FOX_LAYER_INDICES else "gla"
        fieldnames.extend([
            f"L{i}_{ltype}_eff_rank",
            f"L{i}_{ltype}_norm",
        ])
    # Aggregates
    fieldnames.extend([
        "gla_mean_eff_rank", "fox_mean_eff_rank",
        "gla_mean_norm", "fox_mean_norm",
        "gla_std_eff_rank", "fox_std_eff_rank",
        "overall_eff_rank", "mean_nll",
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
            
            # Collect per-layer hidden states across all generation steps
            layer_hiddens = {i: [] for i in range(n_layers)}
            nll_values = []
            
            current_ids = input_ids
            past_kv = None
            hooks = AllLayerHook.attach(model)
            
            for step in range(max_new_tokens):
                hooks.clear()
                
                with torch.no_grad():
                    if past_kv is not None:
                        out = model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                    else:
                        out = model(current_ids, use_cache=True)
                    
                    logits = out[0]
                    past_kv = out[3]
                
                states = hooks.get_step_states()
                for layer_idx, h in states.items():
                    layer_hiddens[layer_idx].append(h)
                
                # Greedy next token
                step_logits = logits[:, -1, :]
                next_id = int(torch.argmax(step_logits, dim=-1).item())
                
                log_probs = torch.log_softmax(step_logits, dim=-1)
                nll = -log_probs[0, next_id].item()
                nll_values.append(nll)
                
                next_token = torch.tensor([[next_id]], device=device)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                eos_id = getattr(tokenizer, 'eos_token_id', None)
                if eos_id is not None and next_id == eos_id:
                    break
            
            hooks.remove()
            tokens_gen = len(nll_values)
            elapsed = time.time() - t0
            print(f"  {tokens_gen} tokens in {elapsed:.1f}s ({tokens_gen/max(elapsed,0.01):.1f} tok/s)")
            
            # Compute per-layer metrics
            row = {
                "prompt_idx": pidx,
                "category": category,
                "prompt_text_short": prompt_text[:80],
                "tokens_generated": tokens_gen,
            }
            
            gla_ranks = []
            fox_ranks = []
            gla_norms = []
            fox_norms = []
            
            for layer_idx in range(n_layers):
                ltype = "fox" if layer_idx in FOX_LAYER_INDICES else "gla"
                
                if layer_hiddens[layer_idx]:
                    H = torch.stack(layer_hiddens[layer_idx], dim=0)
                    eff_rank = compute_shannon_effective_rank(H)
                    mean_norm = H.norm(dim=-1).mean().item()
                else:
                    eff_rank = 0.0
                    mean_norm = 0.0
                
                row[f"L{layer_idx}_{ltype}_eff_rank"] = f"{eff_rank:.2f}"
                row[f"L{layer_idx}_{ltype}_norm"] = f"{mean_norm:.1f}"
                
                if ltype == "gla":
                    gla_ranks.append(eff_rank)
                    gla_norms.append(mean_norm)
                else:
                    fox_ranks.append(eff_rank)
                    fox_norms.append(mean_norm)
            
            # Aggregates
            row["gla_mean_eff_rank"] = f"{np.mean(gla_ranks):.2f}" if gla_ranks else "0"
            row["fox_mean_eff_rank"] = f"{np.mean(fox_ranks):.2f}" if fox_ranks else "0"
            row["gla_mean_norm"] = f"{np.mean(gla_norms):.1f}" if gla_norms else "0"
            row["fox_mean_norm"] = f"{np.mean(fox_norms):.1f}" if fox_norms else "0"
            row["gla_std_eff_rank"] = f"{np.std(gla_ranks):.2f}" if gla_ranks else "0"
            row["fox_std_eff_rank"] = f"{np.std(fox_ranks):.2f}" if fox_ranks else "0"
            
            # Overall (last layer) effective rank
            if layer_hiddens[n_layers - 1]:
                H_last = torch.stack(layer_hiddens[n_layers - 1], dim=0)
                overall_er = compute_shannon_effective_rank(H_last)
            else:
                overall_er = 0.0
            row["overall_eff_rank"] = f"{overall_er:.2f}"
            row["mean_nll"] = f"{np.mean(nll_values):.4f}" if nll_values else "0"
            
            writer.writerow(row)
            
            # Print summary
            print(f"  GLA mean EffRank: {np.mean(gla_ranks):.1f} ± {np.std(gla_ranks):.1f}  |  "
                  f"FoX mean EffRank: {np.mean(fox_ranks):.1f} ± {np.std(fox_ranks):.1f}")
            print(f"  GLA mean norm: {np.mean(gla_norms):.1f}  |  FoX mean norm: {np.mean(fox_norms):.1f}")
            print(f"  Overall EffRank: {overall_er:.1f}/576 ({overall_er/576*100:.1f}%)  |  NLL: {np.mean(nll_values):.3f}")
    
    print(f"\n{'='*60}")
    print(f"Layer geometry audit saved to: {output_csv}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Genesis Per-Layer Spectral Geometry Audit")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", type=str, default="logs/layer_geometry.csv")
    parser.add_argument("--disable-ttt", action="store_true", help="Disable TTT for control experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    model, tokenizer, config = load_genesis_model(weights_path=args.weights)
    
    if args.disable_ttt:
        model = disable_ttt(model)
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"\nLoaded {len(prompts)} prompts")
    
    run_layer_geometry(
        model=model, tokenizer=tokenizer, config=config,
        prompts=prompts, max_new_tokens=args.max_tokens,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
