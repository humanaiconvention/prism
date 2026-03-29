"""Phase 1: FoX layer head causal ablation sweep for Genesis-152M.

Ablates each of the 9 query heads (3 KV groups x 3:1 GQA) in each
of the 7 FoX (Forgetting Attention / softmax) layers.

Measures ΔNLL (teacher-forced) to map head roles:
- Interfering: removal improves NLL (ΔNLL < 0)
- Load-bearing: removal degrades NLL (ΔNLL > 0)
- Redundant: removal has no significant effect

This script targets FoX layers only — GLA (DeltaNet) layers use
linear attention with recurrent state, where individual head ablation
has different mechanistic meaning.

Usage:
    python scripts/run_head_sweep.py \
        --weights weights/genesis_152m_instruct.safetensors \
        --prompts prompts/prompts_60.json \
        --max-tokens 256 \
        --output logs/phase1_head_sweep.csv
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")  # Must be before genesis import

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import (
    load_genesis_model,
    format_chatml_prompt,
    get_layer_info,
    inspect_model_architecture,
    FOX_LAYER_INDICES,
)


class HeadAblationHook:
    """Forward hook that zeros out specific attention heads at the output projection.
    
    For GQA models: Genesis has 9 query heads and 3 KV heads (3:1 ratio).
    Each KV head serves 3 query heads (KV group).
    head_dim = 64, so total o_proj input = 9 * 64 = 576.
    
    Ablating query head H means zeroing columns [H*64 : (H+1)*64] of the
    output projection's input activation.
    """
    
    def __init__(self, heads_to_ablate: List[int], head_dim: int = 64, mode: str = "zero"):
        """
        Args:
            heads_to_ablate: List of query head indices to ablate (0-8 for Genesis).
            head_dim: Dimension per head (64 for Genesis).
            mode: 'zero' (hard ablation) or 'scale' (soft dampening).
        """
        self.heads_to_ablate = heads_to_ablate
        self.head_dim = head_dim
        self.mode = mode
        self.handle = None
    
    def __call__(self, module, input_tensor, output):
        """Hook function — modifies the output of the attention output projection."""
        # output shape: (batch, seq_len, hidden_dim)
        # For GQA: hidden_dim = n_query_heads * head_dim = 9 * 64 = 576
        modified = output.clone()
        for h in self.heads_to_ablate:
            start = h * self.head_dim
            end = start + self.head_dim
            if self.mode == "zero":
                modified[:, :, start:end] = 0.0
            elif self.mode == "scale":
                modified[:, :, start:end] *= 0.5
        return modified
    
    def register(self, module: nn.Module):
        """Register this hook on a module."""
        self.handle = module.register_forward_hook(self)
        return self
    
    def remove(self):
        """Remove this hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def find_o_proj_modules(model, fox_layer_indices: List[int]):
    """Find the output projection modules for FoX attention layers.
    
    Returns a dict mapping layer_idx -> o_proj module.
    This needs adaptation based on Genesis's actual module naming.
    """
    o_proj_map = {}
    
    # Try common naming patterns
    for name, module in model.named_modules():
        for layer_idx in fox_layer_indices:
            patterns = [
                f"layers.{layer_idx}.attn.o_proj",
                f"layers.{layer_idx}.self_attn.o_proj",
                f"layers.{layer_idx}.attention.o_proj",
                f"layers.{layer_idx}.fox.o_proj",
                f"blocks.{layer_idx}.attn.o_proj",
                f"transformer.layers.{layer_idx}.attn.o_proj",
            ]
            for pattern in patterns:
                if name.endswith(pattern) or name == pattern:
                    o_proj_map[layer_idx] = module
                    break
    
    if not o_proj_map:
        print("[WARNING] Could not auto-detect o_proj modules. Dumping attention-related modules:")
        for name, module in model.named_modules():
            if any(kw in name.lower() for kw in ['attn', 'attention', 'fox', 'o_proj', 'out_proj']):
                print(f"  {name}: {type(module).__name__} shape={getattr(module, 'weight', torch.tensor([])).shape if hasattr(module, 'weight') else 'N/A'}")
    
    return o_proj_map


def compute_sequence_nll(model, input_ids, device):
    """Compute mean NLL for a sequence (teacher-forced).
    
    Genesis forward(idx, targets) returns (logits, loss, metrics).
    When targets are provided, loss is computed internally.
    """
    targets = input_ids[:, 1:]  # Shift right for next-token prediction
    source = input_ids[:, :-1]  # All but last token as input
    
    with torch.no_grad():
        try:
            logits, loss, metrics = model(source, targets=targets)
        except Exception as e:
            print(f"  [ERROR] Forward pass failed: {e}")
            return float('nan')
    
    if loss is not None:
        return loss.item()
    return float('nan')


def run_head_sweep(
    model,
    tokenizer,
    config,
    prompts: list,
    fox_layers: List[int],
    n_query_heads: int = 9,
    head_dim: int = 64,
    max_tokens: int = 256,
    output_csv: str = "logs/phase1_head_sweep.csv",
):
    """Run single-head ablation sweep on all FoX layers.
    
    For each FoX layer × each query head: measure baseline NLL, ablated NLL, compute ΔNLL.
    """
    device = next(model.parameters()).device
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Find o_proj modules for hooking
    o_proj_map = find_o_proj_modules(model, fox_layers)
    
    if not o_proj_map:
        print("[ERROR] No o_proj modules found. Cannot proceed with head sweep.")
        print("       Run --inspect-layers on genesis_loader.py first to identify hook points.")
        return
    
    print(f"\nFound o_proj modules for layers: {sorted(o_proj_map.keys())}")
    
    fieldnames = [
        "prompt_idx", "prompt_text_short", "category",
        "layer_idx", "head_idx", "condition",
        "baseline_nll", "ablated_nll", "delta_nll",
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        total_conditions = len(fox_layers) * n_query_heads
        condition_count = 0
        
        for layer_idx in sorted(o_proj_map.keys()):
            for head_idx in range(n_query_heads):
                condition_count += 1
                condition_name = f"L{fox_layers.index(layer_idx)}_H{head_idx}_ablate"
                
                print(f"\n[{condition_count}/{total_conditions}] {condition_name} "
                      f"(model layer {layer_idx}, head {head_idx})")
                
                for pidx, prompt_data in enumerate(prompts):
                    prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
                    category = prompt_data.get("category", "unknown") if isinstance(prompt_data, dict) else "unknown"
                    
                    # Tokenize prompt + expected continuation
                    chat_input = format_chatml_prompt(prompt_text)
                    input_ids = torch.tensor(
                        [tokenizer.encode(chat_input)],
                        device=device
                    )
                    
                    # Truncate to max_tokens
                    if input_ids.shape[1] > max_tokens:
                        input_ids = input_ids[:, :max_tokens]
                    
                    # Baseline NLL (no ablation)
                    baseline_nll = compute_sequence_nll(model, input_ids, device)
                    
                    # Ablated NLL
                    hook = HeadAblationHook(
                        heads_to_ablate=[head_idx],
                        head_dim=head_dim,
                        mode="zero",
                    )
                    hook.register(o_proj_map[layer_idx])
                    
                    ablated_nll = compute_sequence_nll(model, input_ids, device)
                    
                    hook.remove()
                    
                    delta_nll = ablated_nll - baseline_nll
                    
                    writer.writerow({
                        "prompt_idx": pidx,
                        "prompt_text_short": prompt_text[:80],
                        "category": category,
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "condition": condition_name,
                        "baseline_nll": f"{baseline_nll:.6f}",
                        "ablated_nll": f"{ablated_nll:.6f}",
                        "delta_nll": f"{delta_nll:.6f}",
                    })
                
                # Print summary for this condition
                print(f"  Completed {len(prompts)} prompts for {condition_name}")
    
    print(f"\n{'='*60}")
    print(f"Head sweep saved to: {output_csv}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Genesis-152M FoX Head Causal Ablation (Phase 1)")
    parser.add_argument("--weights", type=str, default=None, help="Path to safetensors weights")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json", help="Prompt file")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max sequence length for NLL")
    parser.add_argument("--output", type=str, default="logs/phase1_head_sweep.csv", help="Output CSV")
    parser.add_argument("--inspect", action="store_true", help="Inspect model architecture and exit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    model, tokenizer, config = load_genesis_model(
        weights_path=args.weights,
        device=args.device,
    )
    
    # Get architecture layout
    layer_info = get_layer_info()
    fox_layers = layer_info["fox_layers"]
    
    if args.inspect:
        inspect_model_architecture(model, config)
        return
    
    # Determine head params
    n_query_heads = getattr(config, 'n_head', 9)
    head_dim = getattr(config, 'head_dim', 64)
    
    print(f"\nFoX layers for ablation: {fox_layers}")
    print(f"Query heads per layer: {n_query_heads}")
    print(f"Head dim: {head_dim}")
    
    # Load prompts
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"Loaded {len(prompts)} prompts")
    print(f"Total conditions: {len(fox_layers)} layers × {n_query_heads} heads = {len(fox_layers) * n_query_heads}")
    print(f"Total forward passes: {len(fox_layers) * n_query_heads * len(prompts) * 2}")
    
    run_head_sweep(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompts=prompts,
        fox_layers=fox_layers,
        n_query_heads=n_query_heads,
        head_dim=head_dim,
        max_tokens=args.max_tokens,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
