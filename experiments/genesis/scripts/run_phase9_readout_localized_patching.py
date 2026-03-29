"""Phase 9P: readout-localized semantic patching around L17.

Projects the L17 semantic patch onto an item-specific clean-vs-corrupt output-head
axis, then evaluates the localized patch with the sign-aware toward-clean metrics
introduced in Phase 9O.
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
from scripts.run_phase9_semantic_readout_refinement import subgroup_name, safe_signed_ratio, safe_weighted_fraction


EXPECTED_COMPONENTS = {"semantic", "readout_localized", "readout_orthogonal"}
EXPECTED_SUBGROUPS = {"all_items", "clean_margin_gt_corrupt", "clean_margin_le_corrupt", "high_alignment", "low_alignment"}

REQUIRED_RESULTS_COLUMNS = [
    "layer", "component", "alpha", "net_signed_margin_fraction", "mean_component_fraction_of_semantic",
    "mean_readout_alignment", "mean_readout_axis_norm", "n_items",
]
REQUIRED_DETAIL_COLUMNS = [
    "layer", "component", "alpha", "item_name", "clean_margin", "corrupt_margin", "patched_margin",
    "semantic_norm", "component_norm", "component_fraction_of_semantic", "readout_axis_norm", "readout_alignment",
    "readout_localized_fraction", "readout_orthogonal_fraction", "margin_group", "alignment_group",
]
REQUIRED_SUBGROUP_COLUMNS = [
    "component", "alpha", "subgroup", "mean_signed_margin_shift", "net_signed_margin_fraction", "n_items",
]
REQUIRED_SUMMARY_COLUMNS = [
    "layer", "best_component_by_net_signed_margin", "best_alpha_by_net_signed_margin", "max_net_signed_margin_fraction",
    "semantic_reference_net_signed_margin_fraction", "readout_localized_reference_net_signed_margin_fraction",
    "readout_orthogonal_reference_net_signed_margin_fraction", "readout_localized_verdict",
]


def require_columns(df, df_name, required_columns):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def require_finite(df, df_name, columns, row_mask=None):
    finite_df = df.loc[row_mask, columns] if row_mask is not None else df[columns]
    bad_counts = {}
    for column in columns:
        values = pd.to_numeric(finite_df[column], errors="coerce")
        bad_count = int((~np.isfinite(values)).sum())
        if bad_count:
            bad_counts[column] = bad_count
    if bad_counts:
        raise ValueError(f"{df_name} has non-finite values in required columns: {bad_counts}")


def validate_phase9p_artifacts(results_df, detail_df, subgroup_df, summary_df):
    require_columns(results_df, "results_df", REQUIRED_RESULTS_COLUMNS)
    require_columns(detail_df, "detail_df", REQUIRED_DETAIL_COLUMNS)
    require_columns(subgroup_df, "subgroup_df", REQUIRED_SUBGROUP_COLUMNS)
    require_columns(summary_df, "summary_df", REQUIRED_SUMMARY_COLUMNS)

    require_finite(detail_df, "detail_df", ["clean_margin", "corrupt_margin", "patched_margin", "semantic_norm", "component_norm", "readout_axis_norm"])
    semantic_nonzero = pd.to_numeric(detail_df["semantic_norm"], errors="coerce") > 1e-8
    require_finite(
        detail_df,
        "detail_df",
        ["component_fraction_of_semantic", "readout_alignment", "readout_localized_fraction", "readout_orthogonal_fraction"],
        row_mask=semantic_nonzero,
    )
    require_finite(
        results_df,
        "results_df",
        ["alpha", "net_signed_margin_fraction", "mean_component_fraction_of_semantic", "mean_readout_alignment", "mean_readout_axis_norm", "n_items"],
    )
    require_finite(subgroup_df, "subgroup_df", ["alpha", "mean_signed_margin_shift", "net_signed_margin_fraction", "n_items"])
    require_finite(
        summary_df,
        "summary_df",
        [
            "best_alpha_by_net_signed_margin", "max_net_signed_margin_fraction",
            "semantic_reference_net_signed_margin_fraction", "readout_localized_reference_net_signed_margin_fraction",
            "readout_orthogonal_reference_net_signed_margin_fraction",
        ],
    )

    unexpected_components = set(results_df["component"].dropna()) - EXPECTED_COMPONENTS
    if unexpected_components:
        raise ValueError(f"Unexpected components in results_df: {sorted(unexpected_components)}")

    unexpected_subgroups = set(subgroup_df["subgroup"].dropna()) - EXPECTED_SUBGROUPS
    if unexpected_subgroups:
        raise ValueError(f"Unexpected subgroups in subgroup_df: {sorted(unexpected_subgroups)}")

    if "all_items" not in set(subgroup_df["subgroup"].dropna()):
        raise ValueError("subgroup_df must contain an all_items row")

    if len(summary_df) != 1:
        raise ValueError(f"summary_df must contain exactly one row, found {len(summary_df)}")


def validate_phase9p_artifacts_from_csvs(output_csv):
    output_path = Path(output_csv)
    detail_csv = infer_detail_csv(str(output_path))
    summary_csv = infer_summary_csv(str(output_path))
    subgroup_csv = str(output_path.with_name("readout_localized_patching_subgroups.csv"))
    validate_phase9p_artifacts(
        pd.read_csv(output_path),
        pd.read_csv(detail_csv),
        pd.read_csv(subgroup_csv),
        pd.read_csv(summary_csv),
    )


def normalize_or_zero(vec, eps=1e-8):
    norm = float(torch.norm(vec).item())
    if not np.isfinite(norm) or norm <= eps:
        return torch.zeros_like(vec), norm
    return vec / norm, norm


def project_onto_axis(vec, axis_unit):
    return torch.dot(vec, axis_unit) * axis_unit


def compute_output_readout_axis(model, clean_token_id, corrupt_token_id):
    readout_delta = model.lm_head.weight[int(clean_token_id)].detach().float() - model.lm_head.weight[int(corrupt_token_id)].detach().float()
    return normalize_or_zero(readout_delta.to(model.lm_head.weight.device))


def main():
    parser = argparse.ArgumentParser(description="Phase 9P: readout-localized semantic patching")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--alpha-sweep", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--components", type=str, default="semantic,readout_localized,readout_orthogonal")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/semantic_directions.json")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--patch-token-position", type=int, default=-1)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/readout_localized_patching_results.csv")
    parser.add_argument("--validate-existing", action="store_true", help="Validate existing 9P CSV artifacts and exit")
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = infer_detail_csv(args.output_csv)
    summary_csv = infer_summary_csv(args.output_csv)
    subgroup_csv = str(Path(args.output_csv).with_name("readout_localized_patching_subgroups.csv"))
    if args.validate_existing:
        validate_phase9p_artifacts_from_csvs(args.output_csv)
        print(f"Phase 9P artifact validation passed for {args.output_csv}")
        print(f"Detail checked: {detail_csv}")
        print(f"Subgroups checked: {subgroup_csv}")
        print(f"Summary checked: {summary_csv}")
        return

    alphas = parse_float_list(args.alpha_sweep)
    components = [c.strip().lower() for c in args.components.split(",") if c.strip()]
    for component in components:
        if component not in EXPECTED_COMPONENTS:
            raise ValueError(f"Unsupported component: {component}")

    items = load_swap_items(args.eval_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_items = prepare_items(items, tokenizer, device)
    direction = torch.tensor(
        load_semantic_direction(args.semantic_directions, args.layer, vector_key=args.vector_key),
        device=device,
        dtype=torch.float32,
    )

    print("\n=== PHASE 9P: READOUT-LOCALIZED PATCHING ===")
    print(f"Layer: {args.layer}")
    print(f"Alphas: {alphas}")
    print(f"Components: {components}")
    print(f"Eval items: {len(prepared_items)}")

    baseline_cache = []
    for item in tqdm(prepared_items, desc=f"Caching L{args.layer} readout-localized state", leave=False):
        clean_metrics = score_binary_choice(model, item["clean_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
        corrupt_metrics = score_binary_choice(model, item["corrupt_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
        clean_vec = capture_residual(model, item["clean_prompt_ids"], args.layer, args.patch_token_position)
        corrupt_vec = capture_residual(model, item["corrupt_prompt_ids"], args.layer, args.patch_token_position)
        delta = decompose_delta(clean_vec, corrupt_vec, direction)
        readout_unit, readout_axis_norm = compute_output_readout_axis(model, item["clean_token_id"], item["corrupt_token_id"])
        readout_localized = project_onto_axis(delta["semantic"], readout_unit)
        readout_orthogonal = delta["semantic"] - readout_localized
        semantic_norm = float(torch.norm(delta["semantic"]).item())
        localized_norm = float(torch.norm(readout_localized).item())
        orthogonal_norm = float(torch.norm(readout_orthogonal).item())
        readout_alignment = np.nan if semantic_norm <= 1e-8 else float(torch.dot(delta["semantic"], readout_unit).item() / semantic_norm)
        target_margin_delta = clean_metrics["clean_minus_corrupt_logprob"] - corrupt_metrics["clean_minus_corrupt_logprob"]
        target_prob_delta = clean_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"]
        target_choice_delta = clean_metrics["predicts_clean_option"] - corrupt_metrics["predicts_clean_option"]
        baseline_cache.append({
            "item": item,
            "clean_metrics": clean_metrics,
            "corrupt_metrics": corrupt_metrics,
            "target_margin_delta": target_margin_delta,
            "target_prob_delta": target_prob_delta,
            "target_choice_delta": target_choice_delta,
            "semantic_fraction": delta["semantic_fraction"],
            "signed_alignment": delta["signed_alignment"],
            "abs_signed_alignment": abs(delta["signed_alignment"]),
            "semantic_norm": semantic_norm,
            "readout_axis_norm": readout_axis_norm,
            "readout_alignment": readout_alignment,
            "readout_localized_fraction": np.nan if semantic_norm <= 1e-8 else localized_norm / semantic_norm,
            "readout_orthogonal_fraction": np.nan if semantic_norm <= 1e-8 else orthogonal_norm / semantic_norm,
            "semantic": delta["semantic"],
            "readout_localized": readout_localized,
            "readout_orthogonal": readout_orthogonal,
        })

    alignment_threshold = float(np.nanmedian([row["abs_signed_alignment"] for row in baseline_cache]))
    detail_rows = []
    for component in components:
        for alpha in tqdm(alphas, desc=f"Readout-localized L{args.layer} | {component}", leave=False):
            for cached in baseline_cache:
                patch_hook = ResidualDeltaPatchHook(
                    delta_vector=cached[component],
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

                component_norm = float(torch.norm(cached[component]).item())
                detail_rows.append({
                    "layer": args.layer,
                    "component": component,
                    "alpha": alpha,
                    "item_name": cached["item"]["name"],
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
                    "component_norm": component_norm,
                    "component_fraction_of_semantic": np.nan if cached["semantic_norm"] <= 1e-8 else component_norm / cached["semantic_norm"],
                    "semantic_fraction": cached["semantic_fraction"],
                    "signed_semantic_alignment": cached["signed_alignment"],
                    "abs_signed_alignment": cached["abs_signed_alignment"],
                    "readout_axis_norm": cached["readout_axis_norm"],
                    "readout_alignment": cached["readout_alignment"],
                    "readout_localized_fraction": cached["readout_localized_fraction"],
                    "readout_orthogonal_fraction": cached["readout_orthogonal_fraction"],
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
        detail_df.groupby(["layer", "component", "alpha"], as_index=False)
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
            patched_clean_choice_rate=("patched_predicts_clean", "mean"),
            patched_matches_clean_prediction_rate=("patched_matches_clean_prediction", "mean"),
            mean_component_fraction_of_semantic=("component_fraction_of_semantic", "mean"),
            mean_readout_alignment=("readout_alignment", "mean"),
            mean_readout_axis_norm=("readout_axis_norm", "mean"),
            n_items=("item_name", "count"),
        )
    )
    results_df["net_signed_margin_fraction"] = results_df.apply(
        lambda row: safe_weighted_fraction(
            detail_df.loc[
                (detail_df["component"] == row["component"]) & np.isclose(detail_df["alpha"], row["alpha"]),
                "signed_margin_shift",
            ],
            detail_df.loc[
                (detail_df["component"] == row["component"]) & np.isclose(detail_df["alpha"], row["alpha"]),
                "abs_target_margin_delta",
            ],
        ),
        axis=1,
    )
    results_df["net_signed_prob_fraction"] = results_df.apply(
        lambda row: safe_weighted_fraction(
            detail_df.loc[
                (detail_df["component"] == row["component"]) & np.isclose(detail_df["alpha"], row["alpha"]),
                "signed_prob_shift",
            ],
            detail_df.loc[
                (detail_df["component"] == row["component"]) & np.isclose(detail_df["alpha"], row["alpha"]),
                "abs_target_prob_delta",
            ],
        ),
        axis=1,
    )

    subgroup_rows = []
    for (component, alpha), group_df in detail_df.groupby(["component", "alpha"]):
        subgroup_rows.append({
            "component": component,
            "alpha": alpha,
            "subgroup": "all_items",
            "mean_signed_margin_shift": group_df["signed_margin_shift"].mean(),
            "mean_signed_margin_restoration": group_df["signed_margin_restoration"].mean(),
            "net_signed_margin_fraction": safe_weighted_fraction(group_df["signed_margin_shift"], group_df["abs_target_margin_delta"]),
            "margin_toward_clean_rate": group_df["toward_clean_margin"].mean(),
            "patched_matches_clean_prediction_rate": group_df["patched_matches_clean_prediction"].mean(),
            "n_items": int(len(group_df)),
        })
        for column in ["margin_group", "alignment_group"]:
            for subgroup, subgroup_df in group_df.groupby(column):
                subgroup_rows.append({
                    "component": component,
                    "alpha": alpha,
                    "subgroup": subgroup,
                    "mean_signed_margin_shift": subgroup_df["signed_margin_shift"].mean(),
                    "mean_signed_margin_restoration": subgroup_df["signed_margin_restoration"].mean(),
                    "net_signed_margin_fraction": safe_weighted_fraction(subgroup_df["signed_margin_shift"], subgroup_df["abs_target_margin_delta"]),
                    "margin_toward_clean_rate": subgroup_df["toward_clean_margin"].mean(),
                    "patched_matches_clean_prediction_rate": subgroup_df["patched_matches_clean_prediction"].mean(),
                    "n_items": int(len(subgroup_df)),
                })
    subgroup_df = pd.DataFrame(subgroup_rows)

    best_idx = results_df["net_signed_margin_fraction"].idxmax()
    best_row = results_df.loc[best_idx]
    reference_alpha = 1.0 if any(abs(alpha - 1.0) < 1e-8 for alpha in alphas) else float(alphas[0])
    reference_df = results_df[np.isclose(results_df["alpha"], reference_alpha)].copy()
    reference_map = reference_df.set_index("component")["net_signed_margin_fraction"].to_dict()
    best_map = results_df.groupby("component")["net_signed_margin_fraction"].max().to_dict()
    localized_best = float(best_map.get("readout_localized", np.nan))
    semantic_best = float(best_map.get("semantic", np.nan))
    orthogonal_best = float(best_map.get("readout_orthogonal", np.nan))

    if localized_best > 0.0:
        verdict = "readout_localization_recovers_toward_clean_signal"
    elif np.isfinite(localized_best) and np.isfinite(semantic_best) and np.isfinite(orthogonal_best) and localized_best > max(semantic_best, orthogonal_best):
        verdict = "readout_localization_less_negative_than_semantic_baselines"
    else:
        verdict = "no_readout_localized_rescue_detected"

    summary_df = pd.DataFrame([
        {
            "layer": int(args.layer),
            "best_component_by_net_signed_margin": best_row["component"],
            "best_alpha_by_net_signed_margin": float(best_row["alpha"]),
            "max_net_signed_margin_fraction": float(best_row["net_signed_margin_fraction"]),
            "reference_alpha": reference_alpha,
            "semantic_reference_net_signed_margin_fraction": float(reference_map.get("semantic", np.nan)),
            "readout_localized_reference_net_signed_margin_fraction": float(reference_map.get("readout_localized", np.nan)),
            "readout_orthogonal_reference_net_signed_margin_fraction": float(reference_map.get("readout_orthogonal", np.nan)),
            "semantic_best_net_signed_margin_fraction": semantic_best,
            "readout_localized_best_net_signed_margin_fraction": localized_best,
            "readout_orthogonal_best_net_signed_margin_fraction": orthogonal_best,
            "readout_localized_verdict": verdict,
        }
    ])

    validate_phase9p_artifacts(results_df, detail_df, subgroup_df, summary_df)

    results_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    subgroup_df.to_csv(subgroup_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- READOUT-LOCALIZED PATCHING SUMMARY ---")
    print(results_df.to_string(index=False))
    print(subgroup_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")
    print(f"Subgroups saved to {subgroup_csv}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()