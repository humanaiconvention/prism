"""Phase 9O: sign-aware readout refinement for the L17 semantic patch.

Re-runs the strongest directional-patching candidate (semantic component at a
single layer) and reports metrics that distinguish raw margin shifts from true
movement toward the clean trajectory.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase9_semantic_utils import infer_detail_csv, infer_summary_csv, load_semantic_direction, parse_float_list
from scripts.run_phase9_activation_patching import capture_residual, load_swap_items, prepare_items, score_binary_choice
from scripts.run_phase9_directional_patching import ResidualDeltaPatchHook, decompose_delta


def subgroup_name(cached, alignment_threshold):
    margin_group = "clean_margin_gt_corrupt" if cached["target_margin_delta"] > 1e-8 else "clean_margin_le_corrupt"
    alignment_group = "high_alignment" if cached["abs_signed_alignment"] >= alignment_threshold else "low_alignment"
    return margin_group, alignment_group


def safe_signed_ratio(numer, denom, eps=1e-8):
    return np.nan if abs(float(denom)) <= eps else float(numer / denom)


def safe_weighted_fraction(signed_shift_series, target_delta_series, eps=1e-8):
    denom = float(target_delta_series.abs().sum())
    if denom <= eps:
        return np.nan
    return float(signed_shift_series.sum() / denom)


def main():
    parser = argparse.ArgumentParser(description="Phase 9O: sign-aware semantic patch readout refinement")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--alpha-sweep", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/semantic_directions.json")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--patch-token-position", type=int, default=-1)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/semantic_readout_refinement_results.csv")
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = infer_detail_csv(args.output_csv)
    summary_csv = infer_summary_csv(args.output_csv)
    subgroup_csv = str(Path(args.output_csv).with_name("semantic_readout_refinement_subgroups.csv"))
    alphas = parse_float_list(args.alpha_sweep)

    items = load_swap_items(args.eval_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_items = prepare_items(items, tokenizer, device)

    direction = torch.tensor(
        load_semantic_direction(args.semantic_directions, args.layer, vector_key=args.vector_key),
        device=device,
        dtype=torch.float32,
    )

    print("\n=== PHASE 9O: SEMANTIC READOUT REFINEMENT ===")
    print(f"Layer: {args.layer}")
    print(f"Alphas: {alphas}")
    print(f"Eval items: {len(prepared_items)}")

    baseline_cache = []
    for item in prepared_items:
        clean_metrics = score_binary_choice(model, item["clean_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
        corrupt_metrics = score_binary_choice(model, item["corrupt_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
        clean_vec = capture_residual(model, item["clean_prompt_ids"], args.layer, args.patch_token_position)
        corrupt_vec = capture_residual(model, item["corrupt_prompt_ids"], args.layer, args.patch_token_position)
        delta = decompose_delta(clean_vec, corrupt_vec, direction)
        target_margin_delta = clean_metrics["clean_minus_corrupt_logprob"] - corrupt_metrics["clean_minus_corrupt_logprob"]
        target_prob_delta = clean_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"]
        target_choice_delta = clean_metrics["predicts_clean_option"] - corrupt_metrics["predicts_clean_option"]
        baseline_cache.append({
            "item": item,
            "clean_metrics": clean_metrics,
            "corrupt_metrics": corrupt_metrics,
            "semantic_delta": delta["semantic"],
            "semantic_norm": delta["semantic_norm"],
            "semantic_fraction": delta["semantic_fraction"],
            "signed_alignment": delta["signed_alignment"],
            "abs_signed_alignment": abs(delta["signed_alignment"]),
            "target_margin_delta": target_margin_delta,
            "target_prob_delta": target_prob_delta,
            "target_choice_delta": target_choice_delta,
        })

    alignment_threshold = float(np.nanmedian([row["abs_signed_alignment"] for row in baseline_cache]))
    detail_rows = []
    for alpha in tqdm(alphas, desc=f"Readout refinement L{args.layer}", leave=False):
        for cached in baseline_cache:
            patch_hook = ResidualDeltaPatchHook(
                delta_vector=cached["semantic_delta"],
                alpha=alpha,
                position=args.patch_token_position,
            )
            patch_hook.attach(model, args.layer)
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
            raw_patch_effect = patched_margin - corrupt_margin
            margin_sign = float(np.sign(cached["target_margin_delta"]))
            signed_margin_shift = np.nan if margin_sign == 0.0 else margin_sign * raw_patch_effect
            signed_margin_restoration = safe_signed_ratio(raw_patch_effect, cached["target_margin_delta"])

            clean_prob = cached["clean_metrics"]["pairwise_clean_prob"]
            corrupt_prob = cached["corrupt_metrics"]["pairwise_clean_prob"]
            patched_prob = patched_metrics["pairwise_clean_prob"]
            raw_prob_shift = patched_prob - corrupt_prob
            prob_sign = float(np.sign(cached["target_prob_delta"]))
            signed_prob_shift = np.nan if prob_sign == 0.0 else prob_sign * raw_prob_shift
            signed_prob_restoration = safe_signed_ratio(raw_prob_shift, cached["target_prob_delta"])

            raw_choice_shift = patched_metrics["predicts_clean_option"] - cached["corrupt_metrics"]["predicts_clean_option"]
            choice_sign = float(np.sign(cached["target_choice_delta"]))
            signed_choice_shift = np.nan if choice_sign == 0.0 else choice_sign * raw_choice_shift
            choice_restoration = safe_signed_ratio(raw_choice_shift, cached["target_choice_delta"])
            margin_group, alignment_group = subgroup_name(cached, alignment_threshold)

            detail_rows.append({
                "layer": args.layer,
                "alpha": alpha,
                "item_name": cached["item"]["name"],
                "clean_option": cached["item"]["clean_option"],
                "corrupt_option": cached["item"]["corrupt_option"],
                "clean_margin": clean_margin,
                "corrupt_margin": corrupt_margin,
                "patched_margin": patched_margin,
                "raw_patch_effect": raw_patch_effect,
                "target_margin_delta": cached["target_margin_delta"],
                "abs_target_margin_delta": abs(cached["target_margin_delta"]),
                "signed_margin_shift": signed_margin_shift,
                "signed_margin_restoration": signed_margin_restoration,
                "clean_pairwise_prob": clean_prob,
                "corrupt_pairwise_prob": corrupt_prob,
                "patched_pairwise_prob": patched_prob,
                "raw_prob_shift": raw_prob_shift,
                "target_prob_delta": cached["target_prob_delta"],
                "abs_target_prob_delta": abs(cached["target_prob_delta"]),
                "signed_prob_shift": signed_prob_shift,
                "signed_prob_restoration": signed_prob_restoration,
                "clean_predicts_clean": cached["clean_metrics"]["predicts_clean_option"],
                "corrupt_predicts_clean": cached["corrupt_metrics"]["predicts_clean_option"],
                "patched_predicts_clean": patched_metrics["predicts_clean_option"],
                "raw_choice_shift": raw_choice_shift,
                "target_choice_delta": cached["target_choice_delta"],
                "signed_choice_shift": signed_choice_shift,
                "choice_restoration": choice_restoration,
                "patched_matches_clean_prediction": int(
                    patched_metrics["predicts_clean_option"] == cached["clean_metrics"]["predicts_clean_option"]
                ),
                "semantic_norm": cached["semantic_norm"],
                "semantic_fraction": cached["semantic_fraction"],
                "signed_semantic_alignment": cached["signed_alignment"],
                "abs_signed_alignment": cached["abs_signed_alignment"],
                "margin_group": margin_group,
                "alignment_group": alignment_group,
                "margin_contrast_valid": int(abs(cached["target_margin_delta"]) > 1e-8),
                "prob_contrast_valid": int(abs(cached["target_prob_delta"]) > 1e-8),
                "choice_contrast_valid": int(abs(cached["target_choice_delta"]) > 0.0),
                "toward_clean_margin": np.nan if np.isnan(signed_margin_shift) else float(signed_margin_shift > 1e-8),
                "toward_clean_prob": np.nan if np.isnan(signed_prob_shift) else float(signed_prob_shift > 1e-8),
                "toward_clean_choice": np.nan if np.isnan(signed_choice_shift) else float(signed_choice_shift > 0.0),
            })

    detail_df = pd.DataFrame(detail_rows)
    results_df = (
        detail_df.groupby(["layer", "alpha"], as_index=False)
        .agg(
            mean_raw_patch_effect=("raw_patch_effect", "mean"),
            mean_signed_margin_shift=("signed_margin_shift", "mean"),
            mean_signed_margin_restoration=("signed_margin_restoration", "mean"),
            mean_signed_prob_shift=("signed_prob_shift", "mean"),
            mean_signed_prob_restoration=("signed_prob_restoration", "mean"),
            mean_signed_choice_shift=("signed_choice_shift", "mean"),
            mean_choice_restoration=("choice_restoration", "mean"),
            margin_toward_clean_rate=("toward_clean_margin", "mean"),
            prob_toward_clean_rate=("toward_clean_prob", "mean"),
            choice_toward_clean_rate=("toward_clean_choice", "mean"),
            corrupt_clean_choice_rate=("corrupt_predicts_clean", "mean"),
            patched_clean_choice_rate=("patched_predicts_clean", "mean"),
            patched_matches_clean_prediction_rate=("patched_matches_clean_prediction", "mean"),
            margin_contrast_valid_rate=("margin_contrast_valid", "mean"),
            prob_contrast_valid_rate=("prob_contrast_valid", "mean"),
            choice_contrast_valid_rate=("choice_contrast_valid", "mean"),
            mean_semantic_fraction=("semantic_fraction", "mean"),
            mean_abs_signed_alignment=("abs_signed_alignment", "mean"),
            n_items=("item_name", "count"),
        )
    )
    results_df["net_signed_margin_fraction"] = results_df["alpha"].map(
        lambda alpha: safe_weighted_fraction(
            detail_df.loc[np.isclose(detail_df["alpha"], alpha), "signed_margin_shift"],
            detail_df.loc[np.isclose(detail_df["alpha"], alpha), "abs_target_margin_delta"],
        )
    )
    results_df["net_signed_prob_fraction"] = results_df["alpha"].map(
        lambda alpha: safe_weighted_fraction(
            detail_df.loc[np.isclose(detail_df["alpha"], alpha), "signed_prob_shift"],
            detail_df.loc[np.isclose(detail_df["alpha"], alpha), "abs_target_prob_delta"],
        )
    )

    subgroup_rows = []
    for alpha, alpha_df in detail_df.groupby("alpha"):
        subgroup_rows.append({
            "alpha": alpha,
            "subgroup": "all_items",
            "mean_signed_margin_shift": alpha_df["signed_margin_shift"].mean(),
            "mean_signed_margin_restoration": alpha_df["signed_margin_restoration"].mean(),
            "net_signed_margin_fraction": safe_weighted_fraction(alpha_df["signed_margin_shift"], alpha_df["abs_target_margin_delta"]),
            "margin_toward_clean_rate": alpha_df["toward_clean_margin"].mean(),
            "patched_matches_clean_prediction_rate": alpha_df["patched_matches_clean_prediction"].mean(),
            "n_items": int(len(alpha_df)),
        })
        for column in ["margin_group", "alignment_group"]:
            for subgroup, subgroup_df in alpha_df.groupby(column):
                subgroup_rows.append({
                    "alpha": alpha,
                    "subgroup": subgroup,
                    "mean_signed_margin_shift": subgroup_df["signed_margin_shift"].mean(),
                    "mean_signed_margin_restoration": subgroup_df["signed_margin_restoration"].mean(),
                    "net_signed_margin_fraction": safe_weighted_fraction(
                        subgroup_df["signed_margin_shift"],
                        subgroup_df["abs_target_margin_delta"],
                    ),
                    "margin_toward_clean_rate": subgroup_df["toward_clean_margin"].mean(),
                    "patched_matches_clean_prediction_rate": subgroup_df["patched_matches_clean_prediction"].mean(),
                    "n_items": int(len(subgroup_df)),
                })
    subgroup_df = pd.DataFrame(subgroup_rows)

    best_idx = results_df["net_signed_margin_fraction"].idxmax()
    best_row = results_df.loc[best_idx]
    reference_alpha = 1.0 if any(abs(alpha - 1.0) < 1e-8 for alpha in alphas) else float(alphas[0])
    reference_row = results_df[np.isclose(results_df["alpha"], reference_alpha)].iloc[0]

    subgroup_pivot = subgroup_df.pivot(index="alpha", columns="subgroup", values="net_signed_margin_fraction")
    reference_groups = subgroup_pivot.loc[reference_alpha] if reference_alpha in subgroup_pivot.index else pd.Series(dtype=float)
    reference_high_alignment = float(reference_groups.get("high_alignment", np.nan))
    reference_low_alignment = float(reference_groups.get("low_alignment", np.nan))
    reference_clean_better = float(reference_groups.get("clean_margin_gt_corrupt", np.nan))
    reference_clean_worse = float(reference_groups.get("clean_margin_le_corrupt", np.nan))

    max_raw_patch_effect = float(results_df["mean_raw_patch_effect"].max())
    max_signed_margin_shift = float(results_df["mean_signed_margin_shift"].max())
    max_net_signed_margin_fraction = float(best_row["net_signed_margin_fraction"])
    if max_net_signed_margin_fraction > 0.0:
        verdict = "sign_aware_readout_confirms_toward_clean_shift"
    elif max_raw_patch_effect > 0.0:
        verdict = "raw_positive_effect_collapses_under_sign_aware_readout"
    else:
        verdict = "no_readout_rescue_detected"

    summary_df = pd.DataFrame([
        {
            "layer": int(args.layer),
            "best_alpha_by_net_signed_margin": float(best_row["alpha"]),
            "max_mean_raw_patch_effect": max_raw_patch_effect,
            "max_mean_signed_margin_shift": max_signed_margin_shift,
            "max_net_signed_margin_fraction": max_net_signed_margin_fraction,
            "reference_alpha": reference_alpha,
            "reference_mean_signed_margin_shift": float(reference_row["mean_signed_margin_shift"]),
            "reference_mean_signed_margin_restoration": float(reference_row["mean_signed_margin_restoration"]),
            "reference_net_signed_margin_fraction": float(reference_row["net_signed_margin_fraction"]),
            "reference_mean_signed_prob_shift": float(reference_row["mean_signed_prob_shift"]),
            "reference_net_signed_prob_fraction": float(reference_row["net_signed_prob_fraction"]),
            "reference_margin_toward_clean_rate": float(reference_row["margin_toward_clean_rate"]),
            "reference_patched_matches_clean_prediction_rate": float(reference_row["patched_matches_clean_prediction_rate"]),
            "reference_high_alignment_net_signed_fraction": reference_high_alignment,
            "reference_low_alignment_net_signed_fraction": reference_low_alignment,
            "reference_clean_margin_gt_corrupt_net_signed_fraction": reference_clean_better,
            "reference_clean_margin_le_corrupt_net_signed_fraction": reference_clean_worse,
            "readout_refinement_verdict": verdict,
        }
    ])

    results_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    subgroup_df.to_csv(subgroup_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- SEMANTIC READOUT REFINEMENT SUMMARY ---")
    print(results_df.to_string(index=False))
    print(subgroup_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")
    print(f"Subgroups saved to {subgroup_csv}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()