"""Phase 9A: Semantic Extraction.

Extracts hidden state activations and computes background covariance 
for semantic mapping and steering.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Ensure spectral_microscope is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

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
    
    def get_covariance(self):
        if self.n < 2:
            return np.zeros((self.d, self.d))
        return self.M2 / (self.n - 1)

class MultiLayerCaptureHook:
    """Hook to capture activations from multiple layers simultaneously."""
    def __init__(self, layers):
        self.layers = layers
        self.handles = []
        self.captured_acts = {l: [] for l in layers}

    def _make_hook(self, layer_idx):
        def hook_fn(module, inp):
            # Input to the mixer block (pre-mixer)
            x = inp[0]
            self.captured_acts[layer_idx].append(x[:, -1, :].detach().float().cpu().numpy())
            return None
        return hook_fn

    def attach(self, model):
        for l in self.layers:
            h = model.blocks[l].attn.register_forward_pre_hook(self._make_hook(l))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        for l in self.layers:
            self.captured_acts[l] = []

def get_category_indices(category):
    if category == "Mathematical":
        return list(range(0, 10)) + list(range(60, 90))
    elif category == "Creative":
        return list(range(10, 20)) + list(range(90, 120))
    return []

def main():
    parser = argparse.ArgumentParser(description="Phase 9A: Semantic Extraction")
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layers", type=str, default=",".join(str(i) for i in range(30)), help="Comma-separated layer indices")
    parser.add_argument("--output-dir", type=str, default="logs/phase9/data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    target_layers = [int(l) for l in args.layers.split(",")]

    print(f"Loading Genesis-152M for Extraction...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)["prompts"]

    hook = MultiLayerCaptureHook(target_layers)
    hook.attach(model)

    # Initialize Welford for each layer
    covs = {l: WelfordCovariance(config.n_embd) for l in target_layers}
    category_sums = {l: {"Mathematical": np.zeros(config.n_embd), "Creative": np.zeros(config.n_embd)} for l in target_layers}
    category_counts = {"Mathematical": 0, "Creative": 0}

    math_indices = set(get_category_indices("Mathematical"))
    creative_indices = set(get_category_indices("Creative"))

    print(f"Extracting activations from {len(prompts_data[:args.max_prompts])} prompts...")
    
    for i, prompt_entry in enumerate(tqdm(prompts_data[:args.max_prompts], desc="Processing Prompts")):
        text = prompt_entry["text"] if isinstance(prompt_entry, dict) else prompt_entry
        chat_input = format_chatml_prompt(text)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)

        with torch.no_grad():
            model(input_ids)

        is_math = i in math_indices
        is_creative = i in creative_indices
        
        if is_math: category_counts["Mathematical"] += 1
        if is_creative: category_counts["Creative"] += 1

        for l in target_layers:
            act = hook.captured_acts[l][-1][0] # (d,)
            covs[l].update(act)
            if is_math: category_sums[l]["Mathematical"] += act
            if is_creative: category_sums[l]["Creative"] += act
        
        hook.clear()

    hook.remove()

    print(f"\nSaving results for layers {target_layers}...")
    for l in target_layers:
        data = {
            "mean": covs[l].mean,
            "covariance": covs[l].get_covariance(),
            "math_centroid": category_sums[l]["Mathematical"] / max(1, category_counts["Mathematical"]),
            "creative_centroid": category_sums[l]["Creative"] / max(1, category_counts["Creative"]),
            "n_samples": covs[l].n
        }
        np.savez(f"{args.output_dir}/layer_{l}_stats.npz", **data)
        print(f"  Layer {l}: Mean and Covariance saved.")

    # Save metadata
    metadata = {
        "layers": target_layers,
        "n_samples": covs[target_layers[0]].n,
        "math_count": category_counts["Mathematical"],
        "creative_count": category_counts["Creative"]
    }
    with open(f"{args.output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete. Data saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
