"""Audit pair-local susceptibility structure across executed Phase 12 pair artifacts."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


REPRESENTATIVE_EFFECT_COLUMNS = [
    "phase12b_tminus1_w1_effect_signed_label_margin",
    "phase12d_forget_gate_effect_signed_label_margin",
    "phase12e_wo_full_effect_signed_label_margin",
    "phase12f_l11_top3_steer_effect_signed_label_margin",
]


def load_prompt_metadata(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = pd.DataFrame(payload["items"])
    return items[["name", "math_option"]].rename(columns={"name": "pair_name"})


def validate_pair_coverage(df, expected_pairs, label):
    pair_names = set(df["pair_name"])
    missing = sorted(expected_pairs - pair_names)
    extra = sorted(pair_names - expected_pairs)
    if missing or extra:
        raise ValueError(f"{label} pair coverage mismatch: missing={missing}, extra={extra}")
    if df["pair_name"].duplicated().any():
        dupes = sorted(df.loc[df["pair_name"].duplicated(), "pair_name"].unique())
        raise ValueError(f"{label} contains duplicate pair rows: {dupes}")


def select_phase12b_effects(csv_path, expected_pairs):
    df = pd.read_csv(csv_path)
    out = df[(df["answer_offset"] == 1) & (df["window_size"] == 1)][
        ["pair_name", "semantic_minus_random_delta_from_baseline_signed_label_margin"]
    ].rename(columns={
        "semantic_minus_random_delta_from_baseline_signed_label_margin": "phase12b_tminus1_w1_effect_signed_label_margin"
    })
    validate_pair_coverage(out, expected_pairs, "Phase 12B t_minus_1 window=1")
    return out


def select_phase12d_effects(csv_path, expected_pairs):
    out = pd.read_csv(csv_path)[["pair_name", "semantic_minus_random_delta_from_baseline_signed_label_margin"]].rename(columns={
        "semantic_minus_random_delta_from_baseline_signed_label_margin": "phase12d_forget_gate_effect_signed_label_margin"
    })
    validate_pair_coverage(out, expected_pairs, "Phase 12D pair effect")
    return out


def select_phase12e_effects(csv_path, expected_pairs):
    df = pd.read_csv(csv_path)
    out = df[df["subspace"] == "full"][
        ["pair_name", "semantic_minus_random_delta_from_baseline_signed_label_margin"]
    ].rename(columns={
        "semantic_minus_random_delta_from_baseline_signed_label_margin": "phase12e_wo_full_effect_signed_label_margin"
    })
    validate_pair_coverage(out, expected_pairs, "Phase 12E full-subspace pair effect")
    return out


def select_phase12f_effects_and_baseline(csv_path, expected_pairs):
    df = pd.read_csv(csv_path)
    out = df[df["bundle_name"] == "l11_top3"][
        [
            "pair_name",
            "baseline_signed_label_margin",
            "baseline_label_target_pairwise_prob",
            "semantic_steer_effect_signed_label_margin",
            "random_steer_effect_signed_label_margin",
        ]
    ].copy()
    out["phase12f_l11_top3_steer_effect_signed_label_margin"] = (
        out["semantic_steer_effect_signed_label_margin"] - out["random_steer_effect_signed_label_margin"]
    )
    out = out[
        [
            "pair_name",
            "baseline_signed_label_margin",
            "baseline_label_target_pairwise_prob",
            "phase12f_l11_top3_steer_effect_signed_label_margin",
        ]
    ]
    validate_pair_coverage(out, expected_pairs, "Phase 12F l11_top3 steer/baseline")
    return out


def recurrent_polarity(row):
    if row["positive_family_count"] >= 3:
        return "positive_recurrent"
    if row["negative_family_count"] >= 3:
        return "negative_recurrent"
    return "mixed"


def build_pair_table(prompt_json, phase12b_csv, phase12d_csv, phase12e_csv, phase12f_csv):
    prompt_df = load_prompt_metadata(prompt_json)
    expected_pairs = set(prompt_df["pair_name"])
    pair_table = prompt_df.copy()
    for artifact_df in [
        select_phase12b_effects(phase12b_csv, expected_pairs),
        select_phase12d_effects(phase12d_csv, expected_pairs),
        select_phase12e_effects(phase12e_csv, expected_pairs),
        select_phase12f_effects_and_baseline(phase12f_csv, expected_pairs),
    ]:
        pair_table = pair_table.merge(artifact_df, on="pair_name", how="left")
    if pair_table[REPRESENTATIVE_EFFECT_COLUMNS + ["baseline_signed_label_margin", "baseline_label_target_pairwise_prob"]].isnull().any().any():
        raise ValueError("Merged pair table contains missing representative-family or baseline values.")

    pair_table["baseline_abs_signed_label_margin"] = pair_table["baseline_signed_label_margin"].abs()
    pair_table["baseline_abs_pairwise_prob_gap"] = (pair_table["baseline_label_target_pairwise_prob"] - 0.5).abs()
    pair_table["mean_family_effect_signed_label_margin"] = pair_table[REPRESENTATIVE_EFFECT_COLUMNS].mean(axis=1)
    pair_table["median_family_effect_signed_label_margin"] = pair_table[REPRESENTATIVE_EFFECT_COLUMNS].median(axis=1)
    pair_table["positive_family_count"] = (pair_table[REPRESENTATIVE_EFFECT_COLUMNS] > 0.0).sum(axis=1)
    pair_table["negative_family_count"] = (pair_table[REPRESENTATIVE_EFFECT_COLUMNS] < 0.0).sum(axis=1)
    pair_table["recurrent_polarity"] = pair_table.apply(recurrent_polarity, axis=1)
    pair_table = pair_table.sort_values(
        ["mean_family_effect_signed_label_margin", "positive_family_count", "baseline_abs_signed_label_margin"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    pair_table.insert(0, "susceptibility_rank", range(1, len(pair_table) + 1))
    return pair_table[
        [
            "susceptibility_rank",
            "pair_name",
            "math_option",
            "baseline_signed_label_margin",
            "baseline_label_target_pairwise_prob",
            "baseline_abs_signed_label_margin",
            "baseline_abs_pairwise_prob_gap",
            *REPRESENTATIVE_EFFECT_COLUMNS,
            "mean_family_effect_signed_label_margin",
            "median_family_effect_signed_label_margin",
            "positive_family_count",
            "negative_family_count",
            "recurrent_polarity",
        ]
    ]


def variance_explained_from_r(r_value):
    return float(r_value ** 2) if pd.notna(r_value) else float("nan")


def eta_squared(values, groups):
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    overall_mean = frame["value"].mean()
    total_ss = ((frame["value"] - overall_mean) ** 2).sum()
    if total_ss == 0.0:
        return 0.0
    between_ss = frame.groupby("group")["value"].apply(lambda x: len(x) * ((x.mean() - overall_mean) ** 2)).sum()
    return float(between_ss / total_ss)


def build_predictor_summary(pair_table):
    target = pair_table["mean_family_effect_signed_label_margin"]
    rows = []
    for predictor_name in ["baseline_abs_signed_label_margin", "baseline_abs_pairwise_prob_gap"]:
        predictor = pair_table[predictor_name]
        pearson_r = predictor.corr(target)
        spearman_r = predictor.rank(method="average").corr(target.rank(method="average"))
        rows.append({
            "predictor_name": predictor_name,
            "predictor_type": "continuous",
            "primary_score_name": "variance_explained_r2",
            "primary_score_value": variance_explained_from_r(pearson_r),
            "secondary_score_name": "pearson_r",
            "secondary_score_value": float(pearson_r),
            "tertiary_score_name": "spearman_rho",
            "tertiary_score_value": float(spearman_r),
            "group_a_mean": float("nan"),
            "group_b_mean": float("nan"),
            "group_a_count": 0,
            "group_b_count": 0,
        })

    mean_a = float(target[pair_table["math_option"] == "A"].mean())
    mean_b = float(target[pair_table["math_option"] == "B"].mean())
    rows.append({
        "predictor_name": "math_option_A_vs_B",
        "predictor_type": "categorical",
        "primary_score_name": "eta_squared",
        "primary_score_value": eta_squared(target, pair_table["math_option"]),
        "secondary_score_name": "mean_difference_A_minus_B",
        "secondary_score_value": mean_a - mean_b,
        "tertiary_score_name": "group_mean_A",
        "tertiary_score_value": mean_a,
        "group_a_mean": mean_a,
        "group_b_mean": mean_b,
        "group_a_count": int((pair_table["math_option"] == "A").sum()),
        "group_b_count": int((pair_table["math_option"] == "B").sum()),
    })
    return pd.DataFrame(rows).sort_values("primary_score_value", ascending=False).reset_index(drop=True)


def mean_off_diagonal_family_correlation(pair_table):
    corr = pair_table[REPRESENTATIVE_EFFECT_COLUMNS].corr()
    stacked = corr.where(~pd.DataFrame([[i == j for j in range(len(corr.columns))] for i in range(len(corr.columns))], index=corr.index, columns=corr.columns)).stack()
    return float(stacked.mean())


def format_pair_row(row):
    return (
        f"- `{row['pair_name']}`: mean={row['mean_family_effect_signed_label_margin']:+.4f}, "
        f"positive_families={int(row['positive_family_count'])}/4, "
        f"negative_families={int(row['negative_family_count'])}/4, "
        f"|baseline_margin|={row['baseline_abs_signed_label_margin']:.4f}."
    )


def format_predictor_row(row):
    if row["predictor_type"] == "continuous":
        return (
            f"- `{row['predictor_name']}`: {row['primary_score_name']}={row['primary_score_value']:.4f}, "
            f"{row['secondary_score_name']}={row['secondary_score_value']:+.4f}, "
            f"{row['tertiary_score_name']}={row['tertiary_score_value']:+.4f}."
        )
    return (
        f"- `{row['predictor_name']}`: {row['primary_score_name']}={row['primary_score_value']:.4f}, "
        f"mean_A={row['group_a_mean']:+.4f} (n={int(row['group_a_count'])}), "
        f"mean_B={row['group_b_mean']:+.4f} (n={int(row['group_b_count'])}), "
        f"diff_A_minus_B={row['secondary_score_value']:+.4f}."
    )


def build_summary_text(args, pair_table, predictor_summary):
    positive_rows = pair_table[pair_table["positive_family_count"] >= 3].head(args.top_k)
    negative_rows = pair_table[pair_table["negative_family_count"] >= 3].sort_values("mean_family_effect_signed_label_margin").head(args.top_k)
    best_continuous = predictor_summary[predictor_summary["predictor_type"] == "continuous"].iloc[0]
    math_option_row = predictor_summary[predictor_summary["predictor_name"] == "math_option_A_vs_B"].iloc[0]
    if best_continuous["primary_score_value"] > math_option_row["primary_score_value"]:
        predictor_conclusion = (
            "- Baseline-geometry magnitude beats the simple global `math_option` split on this audit, "
            "favoring pair-local structure over a single global option-orientation explanation."
        )
    else:
        predictor_conclusion = (
            "- The simple global `math_option` split is at least as competitive as the baseline-geometry predictors on this audit, "
            "so pair-local structure is not yet clearly stronger than the available global label proxy."
        )
    if best_continuous["secondary_score_value"] > 0.0:
        direction_conclusion = (
            "- The leading continuous predictor is positively correlated with susceptibility, so larger baseline geometry magnitude "
            "tracks larger effect; this is not a clean near-zero boundary-fragility pattern."
        )
    elif best_continuous["secondary_score_value"] < 0.0:
        direction_conclusion = (
            "- The leading continuous predictor is negatively correlated with susceptibility, which is consistent with a near-boundary fragility story."
        )
    else:
        direction_conclusion = "- The leading continuous predictor is effectively flat on this audit."
    lines = [
        "# Phase 13A pair-susceptibility audit (artifact-only)",
        "",
        f"- Phase 12B source: `{args.phase12b_pair_csv}`",
        f"- Phase 12D source: `{args.phase12d_pair_csv}`",
        f"- Phase 12E source: `{args.phase12e_pair_csv}`",
        f"- Phase 12F source: `{args.phase12f_pair_csv}`",
        f"- Held-out prompt metadata: `{args.heldout_json}`",
        "- Representative families: `12B t_minus_1 window=1`, `12D forget gate`, `12E W_o full`, `12F l11_top3 steer delta`.",
        f"- Recurrent positive pairs (>=3/4 families positive): {int((pair_table['positive_family_count'] >= 3).sum())}/{len(pair_table)}.",
        f"- Recurrent negative pairs (>=3/4 families negative): {int((pair_table['negative_family_count'] >= 3).sum())}/{len(pair_table)}.",
        f"- Mean off-diagonal representative-family correlation: {mean_off_diagonal_family_correlation(pair_table):.4f}.",
        "",
        "## Predictor comparison",
        *[format_predictor_row(row) for _, row in predictor_summary.iterrows()],
        "",
        "## Strongest positive recurrent pairs",
        *([format_pair_row(row) for _, row in positive_rows.iterrows()] or ["- None."]),
        "",
        "## Strongest negative recurrent pairs",
        *([format_pair_row(row) for _, row in negative_rows.iterrows()] or ["- None."]),
        "",
        "Interpretation:",
        "- This is an artifact-only cross-family audit over executed Phase 12 pair tables; it does not establish causal mechanism by itself.",
        predictor_conclusion,
        direction_conclusion,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit pair-local susceptibility structure from executed Phase 12 pair artifacts.")
    parser.add_argument("--phase12b-pair-csv", default="logs/phase12/phase12b_full_heldout_shared_summary_pair_effect.csv")
    parser.add_argument("--phase12d-pair-csv", default="logs/phase12/phase12d_full_heldout_shared_summary_pair_effect.csv")
    parser.add_argument("--phase12e-pair-csv", default="logs/phase12/phase12e_full_heldout_shared_summary_pair_effect.csv")
    parser.add_argument("--phase12f-pair-csv", default="logs/phase12/phase12f_full_heldout_shared_summary_pair_interaction.csv")
    parser.add_argument("--heldout-json", default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--output-prefix", default="logs/phase13/phase13a_pair_susceptibility")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pair_table = build_pair_table(
        prompt_json=args.heldout_json,
        phase12b_csv=args.phase12b_pair_csv,
        phase12d_csv=args.phase12d_pair_csv,
        phase12e_csv=args.phase12e_pair_csv,
        phase12f_csv=args.phase12f_pair_csv,
    )
    predictor_summary = build_predictor_summary(pair_table)
    summary_text = build_summary_text(args, pair_table, predictor_summary)

    pair_path = prefix.with_name(prefix.name + "_pair_table.csv")
    predictor_path = prefix.with_name(prefix.name + "_predictor_summary.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.md")
    pair_table.to_csv(pair_path, index=False)
    predictor_summary.to_csv(predictor_path, index=False)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    print(f"\nSaved pair table to {pair_path}")
    print(f"Saved predictor summary to {predictor_path}")


if __name__ == "__main__":
    main()