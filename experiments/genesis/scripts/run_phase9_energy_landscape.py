"""Phase 9J: Residual-stream energy landscape mapping.

Maps a 2D local residual plane around the semantic direction and measures how
next-token behavior changes along the semantic axis versus an orthogonal axis.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase9_semantic_utils import infer_summary_csv, load_semantic_direction, parse_int_list
from scripts.run_phase9_activation_patching import encode_choice_token, encode_prompt, resolve_position
from scripts.run_phase9_semantic_steering import load_eval_items


class ResidualPlaneHook:
    def __init__(self, semantic_vec, orth_vec, alpha=0.0, beta=0.0, position=-1):
        self.semantic_vec = semantic_vec
        self.orth_vec = orth_vec
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.position = int(position)
        self.handle = None

    def _make_hook(self):
        def hook_fn(module, args):
            x = args[0]
            idx = resolve_position(x.shape[1], self.position)
            x_mod = x.clone()
            x_mod[:, idx, :] = x_mod[:, idx, :] + (self.alpha * self.semantic_vec) + (self.beta * self.orth_vec)
            return (x_mod, *args[1:])
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def make_orthogonal_direction(semantic_vec, seed):
    g = torch.Generator(device=semantic_vec.device)
    g.manual_seed(int(seed))
    ortho = torch.randn(semantic_vec.shape, generator=g, device=semantic_vec.device, dtype=semantic_vec.dtype)
    ortho = ortho - torch.dot(ortho, semantic_vec) * semantic_vec
    norm = torch.norm(ortho)
    if norm < 1e-8:
        raise ValueError("Orthogonal control direction collapsed after Gram-Schmidt.")
    return ortho / norm


def score_style_item(model, tokenizer, item):
    device = next(model.parameters()).device
    prompt_ids = encode_prompt(tokenizer, item["prompt"], device)
    math_letter = item["math_option"].strip().upper()
    creative_letter = "B" if math_letter == "A" else "A"
    math_token_id = encode_choice_token(tokenizer, math_letter)
    creative_token_id = encode_choice_token(tokenizer, creative_letter)

    with torch.inference_mode():
        logits = model(prompt_ids)
    logits = logits[0] if isinstance(logits, tuple) else logits
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

    math_lp = float(log_probs[0, math_token_id].item())
    creative_lp = float(log_probs[0, creative_token_id].item())
    pairwise_denom = np.exp(math_lp) + np.exp(creative_lp)
    pairwise_math_prob = float(np.exp(math_lp) / max(pairwise_denom, 1e-12))
    label_is_math = item["label"].strip().lower() == "math"
    label_prob = pairwise_math_prob if label_is_math else (1.0 - pairwise_math_prob)
    signed_margin = (math_lp - creative_lp) * (1.0 if label_is_math else -1.0)

    return {
        "signed_label_margin": signed_margin,
        "label_probability": label_prob,
        "label_correct": int(signed_margin >= 0.0),
    }


def summarize_landscape(results_df):
    rows = []
    for layer, layer_df in results_df.groupby("layer"):
        beta_zero = sorted(layer_df["beta"].unique(), key=lambda x: abs(x))[0]
        alpha_zero = sorted(layer_df["alpha"].unique(), key=lambda x: abs(x))[0]
        directional_df = layer_df[np.isclose(layer_df["beta"], beta_zero)].sort_values("alpha")
        orthogonal_df = layer_df[np.isclose(layer_df["alpha"], alpha_zero)].sort_values("beta")

        directional_gradient = float(np.polyfit(directional_df["alpha"], directional_df["mean_logit_margin"], 1)[0])
        orthogonal_variance = float(np.var(orthogonal_df["mean_logit_margin"]))
        ridge_strength = float(
            np.var(directional_df["mean_logit_margin"]) / max(orthogonal_variance, 1e-8)
        )
        rows.append({
            "layer": int(layer),
            "ridge_strength": ridge_strength,
            "directional_gradient": directional_gradient,
            "orthogonal_variance": orthogonal_variance,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 9J: Residual stream energy landscape mapping")
    parser.add_argument("--layers", type=str, default="15")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/semantic_directions.json")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--grid-radius", type=float, default=4.0)
    parser.add_argument("--grid-resolution", type=int, default=21)
    parser.add_argument("--patch-token-position", type=int, default=-1)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/energy_landscape_results.csv")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    summary_csv = infer_summary_csv(args.output_csv)
    layers = parse_int_list(args.layers)
    grid_values = np.linspace(-args.grid_radius, args.grid_radius, args.grid_resolution)
    items = load_eval_items(args.eval_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)

    print("\n=== PHASE 9J: ENERGY LANDSCAPE ===")
    print(f"Layers: {layers}")
    print(f"Grid radius: {args.grid_radius}")
    print(f"Grid resolution: {args.grid_resolution}")
    print(f"Eval items: {len(items)}")

    rows = []
    for layer in layers:
        semantic_vec = torch.tensor(
            load_semantic_direction(args.semantic_directions, layer, vector_key=args.vector_key),
            device=device,
            dtype=torch.float32,
        )
        orth_vec = make_orthogonal_direction(semantic_vec, seed=args.seed + layer)

        for alpha in tqdm(grid_values, desc=f"Landscape L{layer}", leave=False):
            for beta in grid_values:
                margins = []
                probs = []
                accs = []
                for item in items:
                    hook = ResidualPlaneHook(
                        semantic_vec=semantic_vec,
                        orth_vec=orth_vec,
                        alpha=float(alpha),
                        beta=float(beta),
                        position=args.patch_token_position,
                    )
                    hook.attach(model, layer)
                    try:
                        metrics = score_style_item(model, tokenizer, item)
                    finally:
                        hook.remove()
                    margins.append(metrics["signed_label_margin"])
                    probs.append(metrics["label_probability"])
                    accs.append(metrics["label_correct"])

                rows.append({
                    "layer": layer,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "mean_logit_margin": float(np.mean(margins)),
                    "mean_target_prob": float(np.mean(probs)),
                    "accuracy": float(np.mean(accs)),
                })

    results_df = pd.DataFrame(rows)
    summary_df = summarize_landscape(results_df)

    results_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- ENERGY LANDSCAPE SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()