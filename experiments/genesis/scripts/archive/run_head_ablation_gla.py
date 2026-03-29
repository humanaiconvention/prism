"""Phase 7C: GLA Layer Head Causal Ablation Sweep.

Explicitly targets the 36 conditions corresponding to the 9 heads 
of the 4 key GLA layers: L0, L14, L22, L28.
Maps out any compressor nodes in the linear layers comparable to L15-H3.
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import (
    load_genesis_model,
    format_chatml_prompt,
)

class HeadAblationHook:
    def __init__(self, heads_to_ablate: List[int], head_dim: int = 64):
        self.heads_to_ablate = heads_to_ablate
        self.head_dim = head_dim
        self.handle = None
    
    def __call__(self, module, input_tensor, output):
        modified = output.clone()
        for h in self.heads_to_ablate:
            start = h * self.head_dim
            end = start + self.head_dim
            modified[:, :, start:end] = 0.0
        return modified
    
    def register(self, module: nn.Module):
        self.handle = module.register_forward_hook(self)
        return self
    
    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

def find_attn_o_proj(model, target_layers: List[int]):
    o_proj_map = {}
    for name, module in model.named_modules():
        for layer_idx in target_layers:
            patterns = [
                f"layers.{layer_idx}.attn.o_proj",
                f"layers.{layer_idx}.self_attn.o_proj",
                f"layers.{layer_idx}.attention.o_proj",
                f"blocks.{layer_idx}.attn.o_proj",
            ]
            for pattern in patterns:
                if name.endswith(pattern) or name == pattern:
                    o_proj_map[layer_idx] = module
                    break
    return o_proj_map

def compute_sequence_nll(model, input_ids, device):
    targets = input_ids[:, 1:]
    source = input_ids[:, :-1]
    
    with torch.no_grad():
        try:
            logits, loss, metrics = model(source, targets=targets)
        except Exception as e:
            print(f"  [ERROR] Forward pass failed: {e}")
            return float('nan')
            
    return loss.item() if loss is not None else float('nan')

def run_gla_ablation(
    model, tokenizer, prompts: list, target_layers: List[int],
    n_query_heads: int = 9, head_dim: int = 64, max_tokens: int = 256,
    output_dir: str = "logs"
):
    device = next(model.parameters()).device
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_csv = f"{output_dir}/phase7_gla_ablation.csv"
    
    o_proj_map = find_attn_o_proj(model, target_layers)
    if not o_proj_map:
        print("[ERROR] No o_proj modules found.")
        return
        
    print(f"\nFound o_proj modules for GLA layers: {sorted(o_proj_map.keys())}")
    
    fieldnames = [
        "prompt_idx", "category",
        "layer_idx", "head_idx", "condition",
        "baseline_nll", "ablated_nll", "delta_nll",
    ]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        condition_count = 0
        total_conditions = len(target_layers) * n_query_heads
        
        for layer_idx in sorted(o_proj_map.keys()):
            for head_idx in range(n_query_heads):
                condition_count += 1
                cond_name = f"L{layer_idx}_H{head_idx}_ablate"
                print(f"\n[{condition_count}/{total_conditions}] {cond_name}")
                
                # We use a random sample of 20 prompts to save time (since we need to test 36 conditions)
                import random
                random.seed(42)
                test_prompts = random.sample(prompts, min(20, len(prompts)))
                
                for pidx, prompt_data in enumerate(test_prompts):
                    prompt_text = prompt_data["text"] if isinstance(prompt_data, dict) else str(prompt_data)
                    category = prompt_data.get("category", "unknown") if isinstance(prompt_data, dict) else "unknown"
                    
                    input_ids = torch.tensor([tokenizer.encode(format_chatml_prompt(prompt_text))], device=device)[:, :max_tokens]
                    
                    baseline_nll = compute_sequence_nll(model, input_ids, device)
                    
                    hook = HeadAblationHook(heads_to_ablate=[head_idx], head_dim=head_dim)
                    hook.register(o_proj_map[layer_idx])
                    
                    ablated_nll = compute_sequence_nll(model, input_ids, device)
                    hook.remove()
                    
                    writer.writerow({
                        "prompt_idx": pidx,
                        "category": category,
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "condition": cond_name,
                        "baseline_nll": f"{baseline_nll:.6f}",
                        "ablated_nll": f"{ablated_nll:.6f}",
                        "delta_nll": f"{ablated_nll - baseline_nll:.6f}",
                    })

def main():
    print("Loading Genesis-152M for Phase 7C-GLA Head Ablation...")
    device = "cuda"
    model, tokenizer, config = load_genesis_model(device=device)
    
    with open("prompts/prompts_60.json", "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        prompts = prompts_data["prompts"] if isinstance(prompts_data, dict) and "prompts" in prompts_data else prompts_data
        
    gla_targets = [0, 14, 22, 28]
    n_heads = getattr(config, 'n_head', 9)
    h_dim = getattr(config, 'head_dim', 64)
    
    run_gla_ablation(
        model=model, tokenizer=tokenizer, prompts=prompts,
        target_layers=gla_targets, n_query_heads=n_heads, head_dim=h_dim
    )
    print("\nGLA Ablation sweep complete: logs/phase7_gla_ablation.csv")

if __name__ == "__main__":
    main()
