"""Analyze Phase 9 steering outputs as dose-response curves."""

import argparse
import os

import numpy as np
import pandas as pd


def infer_path(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext or '.csv'}"


def fit_slope(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2 or np.allclose(xs, xs[0]):
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


def endpoint_shift(df, value_col):
    ordered = df.sort_values("alpha")
    return float(ordered[value_col].iloc[-1] - ordered[value_col].iloc[0])


def baseline_delta(df, value_col):
    baseline = df[df["alpha"] == 0.0][["layer", "control", "mode", value_col]].rename(
        columns={value_col: f"baseline_{value_col}"}
    )
    merged = df.merge(baseline, on=["layer", "control", "mode"], how="left")
    merged[f"delta_from_baseline_{value_col}"] = merged[value_col] - merged[f"baseline_{value_col}"]
    return merged


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 9 steering dose-response outputs")
    parser.add_argument("--detail-csv", type=str, default="logs/phase9/steering_results_detail.csv")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/steering_dose_response_summary.csv")
    parser.add_argument("--mode", type=str, default="add")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    by_label_csv = infer_path(args.output_csv, "_by_label")
    curve_csv = infer_path(args.output_csv, "_curve_stats")
    label_curve_csv = infer_path(args.output_csv, "_label_curve_stats")

    detail_df = pd.read_csv(args.detail_csv)
    detail_df = detail_df[detail_df["mode"] == args.mode].copy()

    summary_df = (
        detail_df.groupby(["layer", "control", "mode", "alpha"], as_index=False)
        .agg(
            mean_math_bias_logprob=("math_minus_creative_logprob", "mean"),
            mean_pairwise_math_prob=("pairwise_math_prob", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            label_accuracy=("label_correct", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            mean_next_token_entropy=("next_token_entropy", "mean"),
            n_items=("item_name", "count"),
        )
    )
    for value_col in [
        "mean_math_bias_logprob",
        "mean_pairwise_math_prob",
        "mean_signed_label_margin",
        "label_accuracy",
    ]:
        summary_df = baseline_delta(summary_df, value_col)

    by_label_df = (
        detail_df.groupby(["layer", "control", "mode", "label", "alpha"], as_index=False)
        .agg(
            mean_math_bias_logprob=("math_minus_creative_logprob", "mean"),
            mean_pairwise_math_prob=("pairwise_math_prob", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            label_accuracy=("label_correct", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            mean_next_token_entropy=("next_token_entropy", "mean"),
            n_items=("item_name", "count"),
        )
    )

    curve_rows = []
    for (layer, control, mode), group in summary_df.groupby(["layer", "control", "mode"]):
        ordered = group.sort_values("alpha")
        curve_rows.append({
            "layer": layer,
            "control": control,
            "mode": mode,
            "alpha_min": float(ordered["alpha"].min()),
            "alpha_max": float(ordered["alpha"].max()),
            "endpoint_shift_math_bias_logprob": endpoint_shift(ordered, "mean_math_bias_logprob"),
            "endpoint_shift_pairwise_math_prob": endpoint_shift(ordered, "mean_pairwise_math_prob"),
            "endpoint_shift_signed_label_margin": endpoint_shift(ordered, "mean_signed_label_margin"),
            "endpoint_shift_label_accuracy": endpoint_shift(ordered, "label_accuracy"),
            "slope_math_bias_logprob": fit_slope(ordered["alpha"], ordered["mean_math_bias_logprob"]),
            "slope_pairwise_math_prob": fit_slope(ordered["alpha"], ordered["mean_pairwise_math_prob"]),
            "slope_signed_label_margin": fit_slope(ordered["alpha"], ordered["mean_signed_label_margin"]),
            "slope_label_accuracy": fit_slope(ordered["alpha"], ordered["label_accuracy"]),
            "n_alpha": int(len(ordered)),
        })

    label_curve_rows = []
    for (layer, control, mode, label), group in by_label_df.groupby(["layer", "control", "mode", "label"]):
        ordered = group.sort_values("alpha")
        label_curve_rows.append({
            "layer": layer,
            "control": control,
            "mode": mode,
            "label": label,
            "alpha_min": float(ordered["alpha"].min()),
            "alpha_max": float(ordered["alpha"].max()),
            "endpoint_shift_math_bias_logprob": endpoint_shift(ordered, "mean_math_bias_logprob"),
            "endpoint_shift_pairwise_math_prob": endpoint_shift(ordered, "mean_pairwise_math_prob"),
            "endpoint_shift_signed_label_margin": endpoint_shift(ordered, "mean_signed_label_margin"),
            "endpoint_shift_label_accuracy": endpoint_shift(ordered, "label_accuracy"),
            "slope_math_bias_logprob": fit_slope(ordered["alpha"], ordered["mean_math_bias_logprob"]),
            "slope_pairwise_math_prob": fit_slope(ordered["alpha"], ordered["mean_pairwise_math_prob"]),
            "slope_signed_label_margin": fit_slope(ordered["alpha"], ordered["mean_signed_label_margin"]),
            "slope_label_accuracy": fit_slope(ordered["alpha"], ordered["label_accuracy"]),
            "n_alpha": int(len(ordered)),
        })

    curve_df = pd.DataFrame(curve_rows)
    label_curve_df = pd.DataFrame(label_curve_rows)

    summary_df.to_csv(args.output_csv, index=False)
    by_label_df.to_csv(by_label_csv, index=False)
    curve_df.to_csv(curve_csv, index=False)
    label_curve_df.to_csv(label_curve_csv, index=False)

    print("\n--- STEERING DOSE-RESPONSE SUMMARY ---")
    print(summary_df.to_string(index=False))
    print("\n--- STEERING DOSE-RESPONSE CURVE STATS ---")
    print(curve_df.to_string(index=False))
    print(f"\nPer-alpha summary saved to {args.output_csv}")
    print(f"By-label summary saved to {by_label_csv}")
    print(f"Curve stats saved to {curve_csv}")
    print(f"Label curve stats saved to {label_curve_csv}")


if __name__ == "__main__":
    main()