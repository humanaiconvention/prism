"""Summarize the strongest non-promoted near-misses from executed Phase 11 artifacts."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.phase10_experiment_utils import infer_companion_csv


POOLED_REFERENCE_FOLD_ID = "POOLED_ALL_FOLDS"
SEMANTIC_MEAN = "semantic_mean_toward_donor_shift_signed_label_margin"
DELTA_MEAN = "semantic_minus_random_mean_toward_donor_shift_signed_label_margin"


def add_near_miss_features(df):
    out = df.copy()
    if out.empty:
        return out
    out["semantic_sign_positive"] = out[SEMANTIC_MEAN] > 0.0
    out["matched_control_sign_positive"] = out[DELTA_MEAN] > 0.0
    out["both_sign_positive"] = out["semantic_sign_positive"] & out["matched_control_sign_positive"]
    out["sign_support_count"] = out["semantic_sign_positive"].astype(int) + out["matched_control_sign_positive"].astype(int)
    out["best_gate_pvalue"] = out[["semantic_vs_zero_pvalue", "semantic_vs_random_pvalue"]].min(axis=1)
    out["worst_gate_pvalue"] = out[["semantic_vs_zero_pvalue", "semantic_vs_random_pvalue"]].max(axis=1)
    out["near_miss_tier"] = out["sign_support_count"].map(
        {2: "both_signs_positive", 1: "one_sign_positive", 0: "no_positive_gate_signs"}
    )
    return out


def build_pooled_table(pooled_gate_df):
    pooled = pooled_gate_df[(pooled_gate_df["slice_name"] == "all_items") & (~pooled_gate_df["promotion_eligible"])].copy()
    pooled = add_near_miss_features(pooled)
    pooled = pooled.sort_values(
        [
            "is_primary_selection_cell",
            "sign_support_count",
            "best_gate_pvalue",
            DELTA_MEAN,
            SEMANTIC_MEAN,
            "requested_subspace_rank",
        ],
        ascending=[False, False, True, False, False, True],
    ).reset_index(drop=True)
    pooled.insert(0, "near_miss_rank", range(1, len(pooled) + 1))
    return pooled[
        [
            "near_miss_rank", "dataset_name", "target_layer", "site", "component_label",
            "requested_subspace_rank", "effective_subspace_rank", "n_items", "pooled_reference_fold_count",
            "is_primary_selection_cell", "sign_support_count", "near_miss_tier",
            SEMANTIC_MEAN, "semantic_vs_zero_pvalue", DELTA_MEAN, "semantic_vs_random_pvalue",
            "matched_control_positive", "absolute_positive", "dual_gate_positive", "promotion_eligible",
        ]
    ]


def build_fold_support_table(gate_df):
    fold_df = gate_df[
        (gate_df["slice_name"] == "all_items")
        & (gate_df["reference_fold_id"] != POOLED_REFERENCE_FOLD_ID)
        & (~gate_df["promotion_eligible"])
    ].copy()
    fold_df = add_near_miss_features(fold_df)
    group_cols = [
        "dataset_name", "target_layer", "site", "reference_dataset", "reference_scheme",
        "component_kind", "component_label", "requested_subspace_rank", "effective_subspace_rank",
        "component_pc_index", "is_primary_selection_cell",
    ]
    summary = (
        fold_df.groupby(group_cols, dropna=False)
        .agg(
            n_folds=("reference_fold_id", "nunique"),
            semantic_positive_folds=("semantic_sign_positive", "sum"),
            matched_control_delta_positive_folds=("matched_control_sign_positive", "sum"),
            both_sign_positive_folds=("both_sign_positive", "sum"),
            mean_semantic_effect=(SEMANTIC_MEAN, "mean"),
            mean_delta_effect=(DELTA_MEAN, "mean"),
            best_semantic_pvalue=("semantic_vs_zero_pvalue", "min"),
            best_delta_pvalue=("semantic_vs_random_pvalue", "min"),
        )
        .reset_index()
    )
    summary["both_sign_positive_fraction"] = summary["both_sign_positive_folds"] / summary["n_folds"]
    summary = summary.sort_values(
        [
            "is_primary_selection_cell",
            "both_sign_positive_folds",
            "mean_delta_effect",
            "mean_semantic_effect",
            "requested_subspace_rank",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    summary.insert(0, "near_miss_rank", range(1, len(summary) + 1))
    return summary


def format_pooled_row(row):
    return (
        f"- L{int(row['target_layer'])} `{row['component_label']}` ({'primary' if row['is_primary_selection_cell'] else 'shadow'}): "
        f"sign_support={int(row['sign_support_count'])}/2, semantic_mean={row[SEMANTIC_MEAN]:+.4f}, "
        f"delta_mean={row[DELTA_MEAN]:+.4f}, p_zero={row['semantic_vs_zero_pvalue']:.4f}, "
        f"p_vs_random={row['semantic_vs_random_pvalue']:.4f}."
    )


def format_fold_row(row):
    return (
        f"- L{int(row['target_layer'])} `{row['component_label']}` ({'primary' if row['is_primary_selection_cell'] else 'shadow'}): "
        f"both-sign folds={int(row['both_sign_positive_folds'])}/{int(row['n_folds'])}, "
        f"mean_semantic={row['mean_semantic_effect']:+.4f}, mean_delta={row['mean_delta_effect']:+.4f}."
    )


def build_summary_text(summary_csv, pooled_table, fold_support_table, top_k):
    top_pooled = pooled_table.head(top_k)
    top_folds = fold_support_table.head(top_k)
    lines = [
        "# Phase 11 near-miss summary (artifact-only)",
        "",
        f"- Source bundle: `{summary_csv}`",
        "- Scope: executed full Phase 11 artifacts only; no new benchmark was launched.",
        "- Ranking rule: prioritize primary-site rows first, then stronger sign support on the two all-items gates, then smaller gate p-values, then larger positive effects.",
        f"- Pooled non-promoted rows with both gate signs positive: {int((pooled_table['sign_support_count'] == 2).sum())}/{len(pooled_table)}.",
        f"- Fold/component groups with any both-sign fold support: {int((fold_support_table['both_sign_positive_folds'] > 0).sum())}/{len(fold_support_table)}.",
        "",
        "## Best pooled non-promoted near-misses",
        *[format_pooled_row(row) for _, row in top_pooled.iterrows()],
        "",
        "## Best fold-support near-misses",
        *[format_fold_row(row) for _, row in top_folds.iterrows()],
        "",
        "Interpretation:",
        "- These rows are still non-promoted because they fail the strict all-items dual gate.",
        "- Pooled rows remain the primary decision surface; fold-level support is secondary diagnostic context only.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize the strongest non-promoted near-misses from a Phase 11 artifact bundle.")
    parser.add_argument("--summary-csv", default="logs/phase11/phase11_full_heldout_shared_summary.csv", help="Phase 11 summary CSV used to infer companion gate artifacts.")
    parser.add_argument("--output-prefix", default=None, help="Prefix for report outputs. Defaults to <summary_csv stem>_near_misses.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of rows to highlight in the Markdown summary.")
    args = parser.parse_args()

    summary_csv = str(args.summary_csv)
    gate_csv = infer_companion_csv(summary_csv, "gate_summary")
    pooled_gate_csv = infer_companion_csv(summary_csv, "pooled_gate_summary")
    output_prefix = args.output_prefix or (str(Path(summary_csv).with_suffix("")) + "_near_misses")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    gate_df = pd.read_csv(gate_csv)
    pooled_gate_df = pd.read_csv(pooled_gate_csv)
    pooled_table = build_pooled_table(pooled_gate_df)
    fold_support_table = build_fold_support_table(gate_df)
    summary_text = build_summary_text(summary_csv, pooled_table, fold_support_table, top_k=args.top_k)

    pooled_table.to_csv(prefix.with_name(prefix.name + "_pooled_rows.csv"), index=False)
    fold_support_table.to_csv(prefix.with_name(prefix.name + "_fold_support.csv"), index=False)
    prefix.with_name(prefix.name + "_summary.md").write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    print(f"\nSaved pooled near-miss rows to {prefix.with_name(prefix.name + '_pooled_rows.csv')}")
    print(f"Saved fold-support summary to {prefix.with_name(prefix.name + '_fold_support.csv')}")


if __name__ == "__main__":
    main()