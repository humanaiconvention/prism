"""Analyze item-level consistency between Phase 9 steering and activation patching."""

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


def summarize_pair(merged_df, steering_control, patch_control):
    rows = merged_df[
        (merged_df["steering_control"] == steering_control)
        & (merged_df["patch_control"] == patch_control)
    ].copy()
    return {
        "steering_control": steering_control,
        "patch_control": patch_control,
        "n_items": int(len(rows)),
        "mean_steering_label_prob_effect": float(rows["steering_label_prob_effect"].mean()),
        "mean_steering_signed_margin_effect": float(rows["steering_signed_margin_effect"].mean()),
        "mean_patch_clean_prob_effect": float(rows["patch_clean_prob_effect"].mean()),
        "mean_patch_margin_effect": float(rows["patch_margin_effect"].mean()),
        "corr_label_prob_vs_clean_prob": safe_corr(
            rows["steering_label_prob_effect"], rows["patch_clean_prob_effect"]
        ),
        "corr_signed_margin_vs_patch_margin": safe_corr(
            rows["steering_signed_margin_effect"], rows["patch_margin_effect"]
        ),
        "corr_abs_label_prob_vs_abs_clean_prob": safe_corr(
            np.abs(rows["steering_label_prob_effect"]), np.abs(rows["patch_clean_prob_effect"])
        ),
        "corr_abs_margin_vs_abs_patch_margin": safe_corr(
            np.abs(rows["steering_signed_margin_effect"]), np.abs(rows["patch_margin_effect"])
        ),
        "sign_agreement_label_prob_vs_clean_prob": sign_agreement(
            rows["steering_label_prob_effect"], rows["patch_clean_prob_effect"]
        ),
        "sign_agreement_signed_margin_vs_patch_margin": sign_agreement(
            rows["steering_signed_margin_effect"], rows["patch_margin_effect"]
        ),
        "patch_contrast_valid_rate": float(rows["contrast_valid"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze consistency between steering and activation patching")
    parser.add_argument("--steering-detail-csv", type=str, default="logs/phase9/steering_dose_response_results_detail.csv")
    parser.add_argument("--patching-detail-csv", type=str, default="logs/phase9/activation_patching_consistency_results_detail.csv")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/steering_patching_consistency_summary.csv")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--steering-mode", type=str, default="add")
    parser.add_argument("--steering-alpha", type=float, default=12.5)
    parser.add_argument("--patch-alpha", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    merged_csv = infer_path(args.output_csv, "_merged")
    by_label_csv = infer_path(args.output_csv, "_by_label")

    steering_df = pd.read_csv(args.steering_detail_csv)
    steering_df = steering_df[
        (steering_df["layer"] == args.layer)
        & (steering_df["mode"] == args.steering_mode)
    ].copy()
    steering_df[["pair_id", "pair_label"]] = steering_df["item_name"].apply(parse_steering_item).apply(pd.Series)
    steering_df["label_pairwise_prob"] = np.where(
        steering_df["label"] == "math",
        steering_df["pairwise_math_prob"],
        steering_df["pairwise_creative_prob"],
    )

    steering_baseline = steering_df[steering_df["alpha"] == 0.0][
        ["layer", "control", "mode", "pair_id", "pair_label", "label_pairwise_prob", "signed_label_margin"]
    ].rename(
        columns={
            "control": "steering_control",
            "label_pairwise_prob": "baseline_label_pairwise_prob",
            "signed_label_margin": "baseline_signed_label_margin",
        }
    )
    steering_effect = steering_df[steering_df["alpha"] == args.steering_alpha][
        ["layer", "control", "mode", "alpha", "pair_id", "pair_label", "item_name", "label_pairwise_prob", "signed_label_margin"]
    ].rename(columns={"control": "steering_control", "alpha": "steering_alpha"})
    steering_effect = steering_effect.merge(
        steering_baseline,
        on=["layer", "steering_control", "mode", "pair_id", "pair_label"],
        how="left",
    )
    steering_effect["steering_label_prob_effect"] = (
        steering_effect["label_pairwise_prob"] - steering_effect["baseline_label_pairwise_prob"]
    )
    steering_effect["steering_signed_margin_effect"] = (
        steering_effect["signed_label_margin"] - steering_effect["baseline_signed_label_margin"]
    )

    patching_df = pd.read_csv(args.patching_detail_csv)
    patching_df = patching_df[
        (patching_df["layer"] == args.layer)
        & (patching_df["alpha"] == args.patch_alpha)
    ].copy()
    patching_df[["pair_id", "pair_label"]] = patching_df["item_name"].apply(parse_patching_item).apply(pd.Series)
    patching_df["patch_clean_prob_effect"] = patching_df["patched_pairwise_prob"] - patching_df["corrupt_pairwise_prob"]
    patching_df = patching_df.rename(columns={
        "control": "patch_control",
        "alpha": "patch_alpha",
        "patch_effect": "patch_margin_effect",
    })

    merged_df = steering_effect.merge(
        patching_df[
            [
                "layer",
                "patch_control",
                "patch_alpha",
                "pair_id",
                "pair_label",
                "item_name",
                "patch_clean_prob_effect",
                "patch_margin_effect",
                "contrast_valid",
                "restoration_fraction",
            ]
        ].rename(columns={"item_name": "patch_item_name"}),
        on=["layer", "pair_id", "pair_label"],
        how="inner",
    )

    summary_rows = []
    for steering_control in sorted(merged_df["steering_control"].unique()):
        for patch_control in sorted(merged_df["patch_control"].unique()):
            summary_rows.append(summarize_pair(merged_df, steering_control, patch_control))
    summary_df = pd.DataFrame(summary_rows)

    by_label_rows = []
    for (steering_control, patch_control, pair_label), rows in merged_df.groupby(
        ["steering_control", "patch_control", "pair_label"]
    ):
        by_label_rows.append({
            "steering_control": steering_control,
            "patch_control": patch_control,
            "pair_label": pair_label,
            "n_items": int(len(rows)),
            "mean_steering_label_prob_effect": float(rows["steering_label_prob_effect"].mean()),
            "mean_patch_clean_prob_effect": float(rows["patch_clean_prob_effect"].mean()),
            "corr_label_prob_vs_clean_prob": safe_corr(
                rows["steering_label_prob_effect"], rows["patch_clean_prob_effect"]
            ),
            "corr_abs_label_prob_vs_abs_clean_prob": safe_corr(
                np.abs(rows["steering_label_prob_effect"]), np.abs(rows["patch_clean_prob_effect"])
            ),
            "sign_agreement_label_prob_vs_clean_prob": sign_agreement(
                rows["steering_label_prob_effect"], rows["patch_clean_prob_effect"]
            ),
        })
    by_label_df = pd.DataFrame(by_label_rows)

    summary_df.to_csv(args.output_csv, index=False)
    by_label_df.to_csv(by_label_csv, index=False)
    merged_df.to_csv(merged_csv, index=False)

    print("\n--- STEERING VS PATCHING CONSISTENCY ---")
    print(summary_df.to_string(index=False))
    print("\n--- BY-LABEL CONSISTENCY ---")
    print(by_label_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"By-label summary saved to {by_label_csv}")
    print(f"Merged item-level data saved to {merged_csv}")


if __name__ == "__main__":
    main()