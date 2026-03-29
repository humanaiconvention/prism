"""Phase 9N: directional patching diagnostic.

Decomposes the clean-versus-corrupt residual delta into:
- full residual delta
- semantic-axis component (projection onto the candidate direction)
- orthogonal remainder

Each component is patched into the corrupt prompt to test whether the behaviorally
relevant signal is concentrated on the semantic axis or in non-semantic residual
content.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase9_semantic_utils import infer_detail_csv, infer_summary_csv, load_semantic_direction, parse_float_list, parse_int_list
from scripts.run_phase9_activation_patching import (
    capture_residual,
    load_swap_items,
    prepare_items,
    resolve_position,
    score_binary_choice,
)


class ResidualDeltaPatchHook:
    def __init__(self, delta_vector, alpha=1.0, position=-1):
        self.delta_vector = delta_vector
        self.alpha = float(alpha)
        self.position = int(position)
        self.handle = None

    def _make_hook(self):
        def hook_fn(module, args):
            if self.alpha == 0.0:
                return None
            x = args[0]
            idx = resolve_position(x.shape[1], self.position)
            x_mod = x.clone()
            x_mod[:, idx, :] = x_mod[:, idx, :] + self.alpha * self.delta_vector.unsqueeze(0)
            return (x_mod, *args[1:])
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def decompose_delta(clean_vec, corrupt_vec, direction):
    full_delta = clean_vec - corrupt_vec
    semantic_coeff = torch.dot(full_delta, direction)
    semantic_delta = semantic_coeff * direction
    orthogonal_delta = full_delta - semantic_delta
    full_norm = float(torch.norm(full_delta).item())
    semantic_norm = float(torch.norm(semantic_delta).item())
    orthogonal_norm = float(torch.norm(orthogonal_delta).item())
    semantic_fraction = np.nan if full_norm <= 1e-8 else semantic_norm / full_norm
    orthogonal_fraction = np.nan if full_norm <= 1e-8 else orthogonal_norm / full_norm
    signed_alignment = np.nan if full_norm <= 1e-8 else float(semantic_coeff.item() / full_norm)
    return {
        "full": full_delta,
        "semantic": semantic_delta,
        "orthogonal": orthogonal_delta,
        "full_norm": full_norm,
        "semantic_norm": semantic_norm,
        "orthogonal_norm": orthogonal_norm,
        "semantic_fraction": semantic_fraction,
        "orthogonal_fraction": orthogonal_fraction,
        "signed_alignment": signed_alignment,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 9N: directional patching diagnostic")
    parser.add_argument("--layers", type=str, default="15,16,17")
    parser.add_argument("--alpha-sweep", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--components", type=str, default="full,semantic,orthogonal")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/semantic_directions.json")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--patch-token-position", type=int, default=-1)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/directional_patching_results.csv")
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = infer_detail_csv(args.output_csv)
    summary_csv = infer_summary_csv(args.output_csv)
    layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alpha_sweep)
    components = [c.strip().lower() for c in args.components.split(",") if c.strip()]
    for component in components:
        if component not in {"full", "semantic", "orthogonal"}:
            raise ValueError(f"Unsupported component: {component}")

    items = load_swap_items(args.eval_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_items = prepare_items(items, tokenizer, device)

    print("\n=== PHASE 9N: DIRECTIONAL PATCHING ===")
    print(f"Layers: {layers}")
    print(f"Alphas: {alphas}")
    print(f"Components: {components}")
    print(f"Eval items: {len(prepared_items)}")

    baseline_cache = {}
    for layer in layers:
        direction = torch.tensor(
            load_semantic_direction(args.semantic_directions, layer, vector_key=args.vector_key),
            device=device,
            dtype=torch.float32,
        )
        layer_cache = []
        for item in prepared_items:
            clean_metrics = score_binary_choice(model, item["clean_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
            corrupt_metrics = score_binary_choice(model, item["corrupt_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
            clean_vec = capture_residual(model, item["clean_prompt_ids"], layer, args.patch_token_position)
            corrupt_vec = capture_residual(model, item["corrupt_prompt_ids"], layer, args.patch_token_position)
            layer_cache.append({
                "item": item,
                "clean_metrics": clean_metrics,
                "corrupt_metrics": corrupt_metrics,
                **decompose_delta(clean_vec, corrupt_vec, direction),
            })
        baseline_cache[layer] = layer_cache

    detail_rows = []
    for layer in layers:
        for component in components:
            for alpha in tqdm(alphas, desc=f"Directional patch L{layer} | {component}", leave=False):
                for cached in baseline_cache[layer]:
                    patch_hook = ResidualDeltaPatchHook(
                        delta_vector=cached[component],
                        alpha=alpha,
                        position=args.patch_token_position,
                    )
                    patch_hook.attach(model, layer)
                    try:
                        patched_metrics = score_binary_choice(
                            model,
                            cached["item"]["corrupt_prompt_ids"],
                            cached["item"]["clean_token_id"],
                            cached["item"]["corrupt_token_id"],
                        )
                    finally:
                        patch_hook.remove()

                    clean_margin = cached["clean_metrics"]["clean_minus_corrupt_logprob"]
                    corrupt_margin = cached["corrupt_metrics"]["clean_minus_corrupt_logprob"]
                    patched_margin = patched_metrics["clean_minus_corrupt_logprob"]
                    denom = clean_margin - corrupt_margin
                    contrast_valid = int(denom > 1e-8)
                    restoration_fraction = np.nan if not contrast_valid else (patched_margin - corrupt_margin) / denom

                    component_norm = cached[f"{component}_norm"]
                    component_fraction = np.nan if cached["full_norm"] <= 1e-8 else component_norm / cached["full_norm"]
                    detail_rows.append({
                        "layer": layer,
                        "component": component,
                        "alpha": alpha,
                        "item_name": cached["item"]["name"],
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patched_margin": patched_margin,
                        "patch_effect": patched_margin - corrupt_margin,
                        "restoration_fraction": restoration_fraction,
                        "contrast_valid": contrast_valid,
                        "patched_pairwise_prob": patched_metrics["pairwise_clean_prob"],
                        "patched_predicts_clean": patched_metrics["predicts_clean_option"],
                        "full_delta_norm": cached["full_norm"],
                        "component_norm": component_norm,
                        "component_norm_fraction": component_fraction,
                        "semantic_fraction": cached["semantic_fraction"],
                        "orthogonal_fraction": cached["orthogonal_fraction"],
                        "signed_semantic_alignment": cached["signed_alignment"],
                    })

    detail_df = pd.DataFrame(detail_rows)
    results_df = (
        detail_df.groupby(["layer", "component", "alpha"], as_index=False)
        .agg(
            mean_clean_margin=("clean_margin", "mean"),
            mean_corrupt_margin=("corrupt_margin", "mean"),
            mean_patched_margin=("patched_margin", "mean"),
            mean_patch_effect=("patch_effect", "mean"),
            mean_restoration_fraction=("restoration_fraction", "mean"),
            contrast_valid_rate=("contrast_valid", "mean"),
            patched_clean_choice_rate=("patched_predicts_clean", "mean"),
            mean_component_norm_fraction=("component_norm_fraction", "mean"),
            mean_semantic_fraction=("semantic_fraction", "mean"),
            mean_orthogonal_fraction=("orthogonal_fraction", "mean"),
            mean_signed_semantic_alignment=("signed_semantic_alignment", "mean"),
            n_items=("item_name", "count"),
        )
    )

    best_idx = results_df["mean_patch_effect"].idxmax()
    best_row = results_df.loc[best_idx]
    reference_alpha = 1.0 if any(abs(alpha - 1.0) < 1e-8 for alpha in alphas) else float(alphas[0])
    reference_df = results_df[np.isclose(results_df["alpha"], reference_alpha)]
    reference_pivot = reference_df.pivot(index="layer", columns="component", values="mean_patch_effect")

    semantic_beats_orthogonal_rate = np.nan
    semantic_beats_full_rate = np.nan
    mean_semantic_patch_effect = np.nan
    mean_orthogonal_patch_effect = np.nan
    mean_full_patch_effect = np.nan
    if not reference_pivot.empty:
        if {"semantic", "orthogonal"}.issubset(reference_pivot.columns):
            semantic_beats_orthogonal_rate = float((reference_pivot["semantic"] > reference_pivot["orthogonal"]).mean())
        if {"semantic", "full"}.issubset(reference_pivot.columns):
            semantic_beats_full_rate = float((reference_pivot["semantic"] > reference_pivot["full"]).mean())
        if "semantic" in reference_pivot.columns:
            mean_semantic_patch_effect = float(reference_pivot["semantic"].mean())
        if "orthogonal" in reference_pivot.columns:
            mean_orthogonal_patch_effect = float(reference_pivot["orthogonal"].mean())
        if "full" in reference_pivot.columns:
            mean_full_patch_effect = float(reference_pivot["full"].mean())

    if float(best_row["mean_patch_effect"]) > 0.0 and best_row["component"] == "semantic":
        verdict = "semantic_component_restores_behavior"
    elif np.isfinite(mean_semantic_patch_effect) and np.isfinite(mean_orthogonal_patch_effect) and mean_semantic_patch_effect > mean_orthogonal_patch_effect:
        verdict = "semantic_component_less_negative_than_orthogonal"
    else:
        verdict = "no_directional_behavioral_rescue"

    summary_df = pd.DataFrame([
        {
            "best_layer": int(best_row["layer"]),
            "best_component": best_row["component"],
            "best_alpha": float(best_row["alpha"]),
            "max_patch_effect": float(best_row["mean_patch_effect"]),
            "reference_alpha": reference_alpha,
            "mean_semantic_patch_effect_at_reference": mean_semantic_patch_effect,
            "mean_orthogonal_patch_effect_at_reference": mean_orthogonal_patch_effect,
            "mean_full_patch_effect_at_reference": mean_full_patch_effect,
            "semantic_beats_orthogonal_rate": semantic_beats_orthogonal_rate,
            "semantic_beats_full_rate": semantic_beats_full_rate,
            "directional_patching_verdict": verdict,
        }
    ])

    results_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- DIRECTIONAL PATCHING SUMMARY ---")
    print(results_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()