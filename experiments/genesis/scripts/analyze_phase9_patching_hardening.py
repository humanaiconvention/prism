"""Summarize a small Phase 9 patching hardening sweep across candidate layers."""

import argparse
import os

import numpy as np
import pandas as pd


def infer_path(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext or '.csv'}"


def parse_steering_item(item_name):
    if item_name.endswith("__math"):
        return item_name[: -len("__math")], "math"
    if item_name.endswith("__creative"):
        return item_name[: -len("__creative")], "creative"
    raise ValueError(f"Unrecognized steering item name: {item_name}")


def parse_patching_item(item_name):
    if item_name.endswith("__math_to_creative"):
        return item_name[: -len("__math_to_creative")], "math"
    if item_name.endswith("__creative_to_math"):
        return item_name[: -len("__creative_to_math")], "creative"
    raise ValueError(f"Unrecognized patching item name: {item_name}")


def safe_corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2 or np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def sign_agreement(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys) & (np.abs(xs) > 1e-12) & (np.abs(ys) > 1e-12)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) == 0:
        return np.nan
    return float(np.mean(np.sign(xs) == np.sign(ys)))


def build_steering_effects(steering_df, steering_mode, steering_alpha):
    steering_df = steering_df[steering_df["mode"] == steering_mode].copy()
    steering_df[["pair_id", "pair_label"]] = steering_df["item_name"].apply(parse_steering_item).apply(pd.Series)
    steering_df["label_pairwise_prob"] = np.where(
        steering_df["label"] == "math",
        steering_df["pairwise_math_prob"],
        steering_df["pairwise_creative_prob"],
    )
    baseline_df = steering_df[steering_df["alpha"] == 0.0][
        ["layer", "control", "mode", "pair_id", "pair_label", "label_pairwise_prob", "signed_label_margin"]
    ].rename(columns={
        "control": "steering_control",
        "label_pairwise_prob": "baseline_label_pairwise_prob",
        "signed_label_margin": "baseline_signed_label_margin",
    })
    effect_df = steering_df[steering_df["alpha"] == steering_alpha][
        ["layer", "control", "mode", "alpha", "pair_id", "pair_label", "item_name", "label_pairwise_prob", "signed_label_margin"]
    ].rename(columns={"control": "steering_control", "alpha": "steering_alpha"})
    effect_df = effect_df.merge(
        baseline_df,
        on=["layer", "steering_control", "mode", "pair_id", "pair_label"],
        how="left",
    )
    effect_df["steering_label_prob_effect"] = effect_df["label_pairwise_prob"] - effect_df["baseline_label_pairwise_prob"]
    effect_df["steering_signed_margin_effect"] = effect_df["signed_label_margin"] - effect_df["baseline_signed_label_margin"]
    return effect_df


def main():
    parser = argparse.ArgumentParser(description="Analyze a small layer sweep for Phase 9 patching hardening")
    parser.add_argument("--steering-detail-csv", type=str, default="logs/phase9/steering_hardening_results_detail.csv")
    parser.add_argument("--patching-detail-csv", type=str, default="logs/phase9/activation_patching_hardening_results_detail.csv")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/patching_hardening_summary.csv")
    parser.add_argument("--steering-mode", type=str, default="add")
    parser.add_argument("--steering-alpha", type=float, default=12.5)
    parser.add_argument("--patch-alpha", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    consistency_csv = infer_path(args.output_csv, "_consistency")
    by_label_csv = infer_path(args.output_csv, "_by_label")
    candidate_csv = infer_path(args.output_csv, "_semantic_clean")

    steering_effect_df = build_steering_effects(pd.read_csv(args.steering_detail_csv), args.steering_mode, args.steering_alpha)
    patching_df = pd.read_csv(args.patching_detail_csv)
    patching_df = patching_df[patching_df["alpha"] == args.patch_alpha].copy()
    patching_df[["pair_id", "pair_label"]] = patching_df["item_name"].apply(parse_patching_item).apply(pd.Series)
    patching_df["patch_clean_prob_effect"] = patching_df["patched_pairwise_prob"] - patching_df["corrupt_pairwise_prob"]
    patching_df = patching_df.rename(columns={"control": "patch_control", "alpha": "patch_alpha", "patch_effect": "patch_margin_effect"})

    patch_summary_df = (
        patching_df.groupby(["layer", "patch_control"], as_index=False)
        .agg(
            n_items=("item_name", "count"),
            mean_patch_clean_prob_effect=("patch_clean_prob_effect", "mean"),
            mean_patch_margin_effect=("patch_margin_effect", "mean"),
            mean_restoration_fraction=("restoration_fraction", "mean"),
            contrast_valid_rate=("contrast_valid", "mean"),
            baseline_clean_choice_rate=("corrupt_predicts_clean", "mean"),
            patched_clean_choice_rate=("patched_predicts_clean", "mean"),
            mean_corrupt_entropy=("corrupt_entropy", "mean"),
            mean_patched_entropy=("patched_entropy", "mean"),
        )
    )
    patch_summary_df["delta_clean_choice_rate"] = (
        patch_summary_df["patched_clean_choice_rate"] - patch_summary_df["baseline_clean_choice_rate"]
    )

    merged_df = steering_effect_df.merge(
        patching_df[
            [
                "layer",
                "patch_control",
                "patch_alpha",
                "pair_id",
                "pair_label",
                "patch_clean_prob_effect",
                "patch_margin_effect",
                "contrast_valid",
                "restoration_fraction",
            ]
        ],
        on=["layer", "pair_id", "pair_label"],
        how="inner",
    )

    consistency_rows = []
    for (layer, steering_control, patch_control), rows in merged_df.groupby(["layer", "steering_control", "patch_control"]):
        consistency_rows.append({
            "layer": int(layer),
            "steering_control": steering_control,
            "patch_control": patch_control,
            "n_items": int(len(rows)),
            "corr_label_prob_vs_clean_prob": safe_corr(rows["steering_label_prob_effect"], rows["patch_clean_prob_effect"]),
            "corr_signed_margin_vs_patch_margin": safe_corr(rows["steering_signed_margin_effect"], rows["patch_margin_effect"]),
            "corr_abs_label_prob_vs_abs_clean_prob": safe_corr(np.abs(rows["steering_label_prob_effect"]), np.abs(rows["patch_clean_prob_effect"])),
            "corr_abs_margin_vs_abs_patch_margin": safe_corr(np.abs(rows["steering_signed_margin_effect"]), np.abs(rows["patch_margin_effect"])),
            "sign_agreement_signed_margin_vs_patch_margin": sign_agreement(rows["steering_signed_margin_effect"], rows["patch_margin_effect"]),
            "patch_contrast_valid_rate": float(rows["contrast_valid"].mean()),
        })
    consistency_df = pd.DataFrame(consistency_rows).sort_values(["layer", "steering_control", "patch_control"])

    by_label_df = (
        merged_df.groupby(["layer", "steering_control", "patch_control", "pair_label"], as_index=False)
        .agg(
            n_items=("pair_id", "count"),
            mean_steering_label_prob_effect=("steering_label_prob_effect", "mean"),
            mean_patch_clean_prob_effect=("patch_clean_prob_effect", "mean"),
            mean_patch_margin_effect=("patch_margin_effect", "mean"),
        )
    )

    semantic_clean_df = patch_summary_df[patch_summary_df["patch_control"] == "clean"].merge(
        consistency_df[
            (consistency_df["steering_control"] == "semantic")
            & (consistency_df["patch_control"] == "clean")
        ],
        on="layer",
        how="left",
        suffixes=("", "_consistency"),
    ).sort_values(
        ["mean_patch_clean_prob_effect", "delta_clean_choice_rate", "corr_abs_margin_vs_abs_patch_margin"],
        ascending=[False, False, False],
    )

    patch_summary_df.to_csv(args.output_csv, index=False)
    consistency_df.to_csv(consistency_csv, index=False)
    by_label_df.to_csv(by_label_csv, index=False)
    semantic_clean_df.to_csv(candidate_csv, index=False)

    print("\n--- PATCHING HARDENING: PATCH-ONLY SUMMARY ---")
    print(patch_summary_df.to_string(index=False))
    print("\n--- PATCHING HARDENING: CONSISTENCY SUMMARY ---")
    print(consistency_df.to_string(index=False))
    print("\n--- PATCHING HARDENING: SEMANTIC VS CLEAN CANDIDATES ---")
    print(semantic_clean_df.to_string(index=False))
    print(f"\nPatch summary saved to {args.output_csv}")
    print(f"Consistency summary saved to {consistency_csv}")
    print(f"By-label summary saved to {by_label_csv}")
    print(f"Semantic/clean candidate table saved to {candidate_csv}")


if __name__ == "__main__":
    main()