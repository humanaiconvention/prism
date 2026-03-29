"""Assemble a compact reviewer-facing Phase 9 summary from analysis artifacts."""

import argparse
import json
import os

import pandas as pd


def add_row(rows, section, metric, value, note=""):
    rows.append({"section": section, "metric": metric, "value": value, "note": note})


def main():
    parser = argparse.ArgumentParser(description="Build a compact reviewer-facing Phase 9 summary table")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--stability-csv", type=str, default="logs/phase9/direction_stability_summary.csv")
    parser.add_argument("--steering-curve-csv", type=str, default="logs/phase9/steering_dose_response_expanded_summary_curve_stats.csv")
    parser.add_argument("--patching-summary-csv", type=str, default="logs/phase9/activation_patching_expanded_results.csv")
    parser.add_argument("--consistency-csv", type=str, default="logs/phase9/steering_patching_consistency_expanded_summary.csv")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/phase9_reviewer_summary.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    items = eval_data["items"]

    stability_df = pd.read_csv(args.stability_csv)
    steering_curve_df = pd.read_csv(args.steering_curve_csv)
    patching_df = pd.read_csv(args.patching_summary_csv)
    consistency_df = pd.read_csv(args.consistency_csv)

    rows = []
    add_row(rows, "benchmark", "shared_eval_pairs", len(items), "Current shared Phase 9 held-out benchmark size.")
    add_row(rows, "benchmark", "shared_eval_items", len(items) * 2, "Math + creative prompt evaluations.")

    for _, row in stability_df[stability_df["layer"].isin([15, 29])].iterrows():
        prefix = f"L{int(row['layer'])}_{row['vector_key']}"
        add_row(rows, "stability", f"{prefix}_mean_pairwise_cosine", round(float(row["mean_pairwise_cosine"]), 6))
        add_row(rows, "stability", f"{prefix}_mean_cosine_to_full", round(float(row["mean_cosine_to_full"]), 6))
        add_row(rows, "stability", f"{prefix}_mean_retained_fraction", round(float(row["mean_retained_fraction"]), 6))

    for _, row in steering_curve_df.iterrows():
        prefix = f"L{int(row['layer'])}_{row['control']}_{row['mode']}"
        add_row(rows, "steering", f"{prefix}_endpoint_shift_pairwise_math_prob", round(float(row["endpoint_shift_pairwise_math_prob"]), 6))
        add_row(rows, "steering", f"{prefix}_endpoint_shift_signed_label_margin", round(float(row["endpoint_shift_signed_label_margin"]), 6))
        add_row(rows, "steering", f"{prefix}_slope_pairwise_math_prob", round(float(row["slope_pairwise_math_prob"]), 6))
        add_row(rows, "steering", f"{prefix}_slope_signed_label_margin", round(float(row["slope_signed_label_margin"]), 6))

    for _, row in patching_df[patching_df["alpha"] == 1.0].iterrows():
        prefix = f"L{int(row['layer'])}_{row['control']}_alpha1"
        add_row(rows, "patching", f"{prefix}_mean_patch_effect", round(float(row["mean_patch_effect"]), 6))
        add_row(rows, "patching", f"{prefix}_patched_clean_choice_rate", round(float(row["patched_clean_choice_rate"]), 6))
        add_row(rows, "patching", f"{prefix}_contrast_valid_rate", round(float(row["contrast_valid_rate"]), 6))

    for _, row in consistency_df.iterrows():
        prefix = f"{row['steering_control']}_vs_{row['patch_control']}"
        add_row(rows, "consistency", f"{prefix}_corr_label_prob_vs_clean_prob", round(float(row["corr_label_prob_vs_clean_prob"]), 6))
        add_row(rows, "consistency", f"{prefix}_corr_signed_margin_vs_patch_margin", round(float(row["corr_signed_margin_vs_patch_margin"]), 6))
        add_row(rows, "consistency", f"{prefix}_corr_abs_margin_vs_abs_patch_margin", round(float(row["corr_abs_margin_vs_abs_patch_margin"]), 6))
        add_row(rows, "consistency", f"{prefix}_sign_agreement_signed_margin_vs_patch_margin", round(float(row["sign_agreement_signed_margin_vs_patch_margin"]), 6))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_csv, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nReviewer summary saved to {args.output_csv}")


if __name__ == "__main__":
    main()