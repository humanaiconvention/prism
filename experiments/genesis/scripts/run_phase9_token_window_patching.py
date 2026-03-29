"""Phase 9M: Multi-token window patching diagnostic.

Tests whether the causal signal is distributed across a short token window near
the answer boundary rather than concentrated at a single token position.
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
from scripts.phase9_semantic_utils import infer_detail_csv, infer_summary_csv, parse_int_list
from scripts.run_phase9_activation_patching import (
    ResidualWindowPatchHook,
    capture_residual_window,
    load_swap_items,
    prepare_items,
    resolve_window_positions,
    score_binary_choice,
)


def build_summary(results_df):
    best_row = results_df.sort_values("mean_patch_effect", ascending=False).iloc[0]
    slope_rows = []
    for layer, layer_df in results_df.groupby("layer"):
        ordered = layer_df.sort_values("window_size")
        slope_rows.append({
            "layer": int(layer),
            "window_patch_slope": float(np.polyfit(ordered["window_size"], ordered["mean_patch_effect"], 1)[0]),
            "best_window_size_layer": int(ordered.loc[ordered["mean_patch_effect"].idxmax(), "window_size"]),
        })
    summary_row = {
        "best_layer": int(best_row["layer"]),
        "best_window_size": int(best_row["window_size"]),
        "max_patch_effect": float(best_row["mean_patch_effect"]),
        "max_patch_prob_effect": float(best_row["mean_patch_prob_effect"]),
        "global_window_patch_slope": float(np.polyfit(results_df["window_size"], results_df["mean_patch_effect"], 1)[0]),
    }
    return pd.DataFrame([summary_row]), pd.DataFrame(slope_rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 9M: token-window activation patching")
    parser.add_argument("--layers", type=str, default="13,14,15,16,17")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--window-sizes", type=str, default="1,2,3,4")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/token_window_patching_results.csv")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = infer_detail_csv(args.output_csv)
    summary_csv = infer_summary_csv(args.output_csv)
    slope_csv = str(Path(summary_csv).with_name(f"{Path(summary_csv).stem}_by_layer{Path(summary_csv).suffix}"))
    layers = parse_int_list(args.layers)
    window_sizes = parse_int_list(args.window_sizes)
    items = load_swap_items(args.eval_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_items = prepare_items(items, tokenizer, device)

    print("\n=== PHASE 9M: TOKEN-WINDOW PATCHING ===")
    print(f"Layers: {layers}")
    print(f"Window sizes: {window_sizes}")
    print(f"Eval items: {len(prepared_items)}")

    detail_rows = []
    for layer in layers:
        for window_size in tqdm(window_sizes, desc=f"Window patch L{layer}", leave=False):
            for item_idx, item in enumerate(prepared_items):
                clean_metrics = score_binary_choice(model, item["clean_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
                corrupt_metrics = score_binary_choice(model, item["corrupt_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])

                clean_len = item["clean_prompt_ids"].shape[1]
                corrupt_len = item["corrupt_prompt_ids"].shape[1]
                effective_window = min(int(window_size), int(clean_len), int(corrupt_len))
                clean_positions = resolve_window_positions(clean_len, effective_window, end_position=-1)
                corrupt_positions = resolve_window_positions(corrupt_len, effective_window, end_position=-1)
                source_window = capture_residual_window(model, item["clean_prompt_ids"], layer, clean_positions)

                patch_hook = ResidualWindowPatchHook(
                    source_window=source_window,
                    alpha=1.0,
                    positions=corrupt_positions,
                    control="clean",
                    seed=args.seed + (1000 * layer) + (100 * window_size) + item_idx,
                )
                patch_hook.attach(model, layer)
                try:
                    patched_metrics = score_binary_choice(
                        model,
                        item["corrupt_prompt_ids"],
                        item["clean_token_id"],
                        item["corrupt_token_id"],
                    )
                finally:
                    patch_hook.remove()

                clean_margin = clean_metrics["clean_minus_corrupt_logprob"]
                corrupt_margin = corrupt_metrics["clean_minus_corrupt_logprob"]
                patched_margin = patched_metrics["clean_minus_corrupt_logprob"]
                denom = clean_margin - corrupt_margin
                contrast_valid = int(denom > 1e-8)
                restoration_fraction = np.nan if not contrast_valid else (patched_margin - corrupt_margin) / denom

                detail_rows.append({
                    "layer": layer,
                    "window_size": window_size,
                    "effective_window_size": effective_window,
                    "item_name": item["name"],
                    "clean_margin": clean_margin,
                    "corrupt_margin": corrupt_margin,
                    "patched_margin": patched_margin,
                    "patch_effect": patched_margin - corrupt_margin,
                    "patch_prob_effect": patched_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"],
                    "restoration_fraction": restoration_fraction,
                    "contrast_valid": contrast_valid,
                    "accuracy": patched_metrics["predicts_clean_option"],
                    "baseline_accuracy": corrupt_metrics["predicts_clean_option"],
                })

    detail_df = pd.DataFrame(detail_rows)
    results_df = (
        detail_df.groupby(["layer", "window_size"], as_index=False)
        .agg(
            mean_patch_effect=("patch_effect", "mean"),
            mean_patch_prob_effect=("patch_prob_effect", "mean"),
            mean_restoration_fraction=("restoration_fraction", "mean"),
            contrast_valid_rate=("contrast_valid", "mean"),
            accuracy=("accuracy", "mean"),
            baseline_accuracy=("baseline_accuracy", "mean"),
            n_items=("item_name", "count"),
        )
    )
    results_df["delta_accuracy"] = results_df["accuracy"] - results_df["baseline_accuracy"]
    summary_df, slope_df = build_summary(results_df)

    results_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)

    print("\n--- TOKEN WINDOW PATCHING SUMMARY ---")
    print(results_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")
    print(f"Summary saved to {summary_csv}")
    print(f"Per-layer slope summary saved to {slope_csv}")


if __name__ == "__main__":
    main()