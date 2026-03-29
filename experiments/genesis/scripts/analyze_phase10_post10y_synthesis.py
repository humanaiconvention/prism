import argparse
from pathlib import Path

import pandas as pd


SCALAR_COLS = [
    "baseline_signed_label_margin",
    "baseline_label_accuracy",
    "semantic_recipient_coeff",
    "abs_semantic_coeff_delta",
    "signed_counterfactual_gap",
]


def mean_or_nan(series):
    return float(series.mean()) if len(series) else float("nan")


def corr_or_nan(frame, col, target):
    if frame[col].nunique(dropna=True) < 2 or frame[target].nunique(dropna=True) < 2:
        return float("nan")
    return float(frame[col].corr(frame[target]))


def build_scalar_diagnostics(detail_df):
    rows = []
    for (dataset_name, target_layer), group in detail_df.groupby(["dataset_name", "target_layer"], dropna=False):
        effect = "toward_donor_shift_signed_label_margin"
        hurt_mask = group[effect] < 0.0
        help_group = group.loc[~hurt_mask]
        hurt_group = group.loc[hurt_mask]
        row = {
            "dataset_name": dataset_name,
            "target_layer": int(target_layer),
            "n_items": int(len(group)),
            "hurt_rate": float(hurt_mask.mean()),
            "mean_effect": mean_or_nan(group[effect]),
        }
        for col in SCALAR_COLS:
            row[f"corr_{col}"] = corr_or_nan(group, col, effect)
            row[f"help_minus_hurt_{col}"] = mean_or_nan(help_group[col]) - mean_or_nan(hurt_group[col])
        row["max_abs_corr"] = max(abs(row[f"corr_{col}"]) for col in SCALAR_COLS if pd.notna(row[f"corr_{col}"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset_name", "target_layer"])


def build_gate_summary(x_stats_df, y_stats_df):
    x_all = x_stats_df[
        (x_stats_df["comparison_type"] == "semantic_vs_random_toward_donor_shift")
        & (x_stats_df["slice_name"] == "all_items")
    ].copy()
    x_all["matched_control_positive"] = (
        (x_all["semantic_minus_random_mean_toward_donor_shift_signed_label_margin"] > 0.0)
        & (x_all["semantic_vs_random_pvalue"] < 0.05)
    )
    x_all["absolute_positive"] = (
        (x_all["semantic_mean_toward_donor_shift_signed_label_margin"] > 0.0)
        & (x_all["semantic_vs_zero_pvalue"] < 0.05)
    )
    x_all["random_antidonor"] = (
        (x_all["random_mean_toward_donor_shift_signed_label_margin"] < 0.0)
        & (x_all["random_vs_zero_pvalue"] < 0.05)
    )
    x_all = x_all[
        [
            "dataset_name",
            "target_layer",
            "semantic_mean_toward_donor_shift_signed_label_margin",
            "semantic_vs_zero_pvalue",
            "random_mean_toward_donor_shift_signed_label_margin",
            "random_vs_zero_pvalue",
            "semantic_minus_random_mean_toward_donor_shift_signed_label_margin",
            "semantic_vs_random_pvalue",
            "matched_control_positive",
            "absolute_positive",
            "random_antidonor",
        ]
    ]
    y_zero = y_stats_df[(y_stats_df["comparison_type"] == "full_donor_vs_zero") & (y_stats_df["slice_name"] == "all_items")].copy()
    y_orth = y_stats_df[(y_stats_df["comparison_type"] == "full_donor_vs_orth_semantic") & (y_stats_df["slice_name"] == "all_items")].copy()
    merged = x_all.merge(
        y_zero[["dataset_name", "target_layer", "condition_a_mean_toward_donor_shift_signed_label_margin", "condition_a_vs_zero_pvalue"]].rename(
            columns={
                "condition_a_mean_toward_donor_shift_signed_label_margin": "full_donor_mean",
                "condition_a_vs_zero_pvalue": "full_donor_vs_zero_pvalue",
            }
        ),
        on=["dataset_name", "target_layer"],
        how="left",
    ).merge(
        y_orth[["dataset_name", "target_layer", "condition_a_minus_b_mean_toward_donor_shift_signed_label_margin", "condition_a_vs_b_pvalue"]].rename(
            columns={
                "condition_a_minus_b_mean_toward_donor_shift_signed_label_margin": "full_minus_orth_mean",
                "condition_a_vs_b_pvalue": "full_vs_orth_pvalue",
            }
        ),
        on=["dataset_name", "target_layer"],
        how="left",
    )
    merged["full_donor_positive"] = (merged["full_donor_mean"] > 0.0) & (merged["full_donor_vs_zero_pvalue"] < 0.05)
    merged["full_beats_orth"] = (merged["full_minus_orth_mean"] > 0.0) & (merged["full_vs_orth_pvalue"] < 0.05)
    return merged.sort_values(["dataset_name", "target_layer"])


def build_summary_text(scalar_df, gate_df):
    scalar_structured = bool((scalar_df["max_abs_corr"] >= 0.25).any())
    matched_pos = int(gate_df["matched_control_positive"].sum())
    absolute_pos = int(gate_df["absolute_positive"].sum())
    antidonor_random = int(gate_df["random_antidonor"].sum())
    bundle_pos = int(gate_df["full_donor_positive"].sum())
    bundle_beats_orth = int(gate_df["full_beats_orth"].sum())
    next_step = "Phase 11 orthogonal-remainder component decomposition" if (not scalar_structured and bundle_beats_orth == 0) else "analysis-gated follow-up before any new benchmark"
    return "\n".join([
        "# Phase 10 post-10Y synthesis (artifact-only)",
        "",
        f"- Scalar-interference all-items diagnostic: {'no moderate all-items structure detected' if not scalar_structured else 'some moderate structure detected'}.",
        f"- 10X all-items matched-control-positive cells: {matched_pos}.",
        f"- 10X all-items absolute-positive cells: {absolute_pos}.",
        f"- 10X all-items random anti-donor cells: {antidonor_random}.",
        f"- 10Y all-items full-donor-positive cells: {bundle_pos}.",
        f"- 10Y all-items full-donor-beats-orth cells: {bundle_beats_orth}.",
        "",
        "Recommendation:",
        f"- Prefer {next_step} over a new scalar-blend 10Z portability variant.",
        "- Keep all-items as primary; treat donor-gap slices as secondary/supporting only.",
        "- Require both semantic-vs-random positivity and semantic-vs-zero positivity before promoting any new mechanism claim.",
    ])


def main():
    parser = argparse.ArgumentParser(description="Summarize the executed 10W-10Y artifacts into a post-10Y planning gate.")
    parser.add_argument("--output-prefix", default="logs/phase10/post10y_synthesis", help="Prefix for output artifacts.")
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    w_detail = pd.read_csv("logs/phase10/natural_scalar_interchange_summary_detail.csv")
    x_stats = pd.read_csv("logs/phase10/natural_orthogonal_interchange_summary_stats.csv")
    y_stats = pd.read_csv("logs/phase10/natural_bundle_interchange_summary_stats.csv")

    scalar_df = build_scalar_diagnostics(w_detail)
    gate_df = build_gate_summary(x_stats, y_stats)
    summary_text = build_summary_text(scalar_df, gate_df)

    scalar_df.to_csv(prefix.with_name(prefix.name + "_scalar_diagnostics.csv"), index=False)
    gate_df.to_csv(prefix.with_name(prefix.name + "_gate_summary.csv"), index=False)
    prefix.with_name(prefix.name + "_summary.md").write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()