import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import (
    TensorSiteDirectionalCoeffInterchangeHook,
    TensorSiteOrthogonalResidualInterchangeHook,
    TensorSiteSwapHook,
    parse_site_list,
)
from scripts.phase9_semantic_utils import parse_int_list
from scripts.run_phase9_semantic_steering import load_anchor_direction, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, prepare_eval_item
from scripts.run_phase10_natural_scalar_interchange import opposite_label
from scripts.run_phase10_tail_conditioned_necessity import (
    add_sign_aware_fields,
    capture_site_state,
    cosine_similarity,
    expand_pair_items,
    fit_direction_from_records,
    load_pair_items,
    pair_name_from_item_name,
    parse_float_list,
    parse_str_list,
)


def evaluate_with_scalar_hook(model, prepared_item, anchor_layer, anchor_direction, layer, site, vector, donor_coeff, alpha, position_fraction):
    hook = TensorSiteDirectionalCoeffInterchangeHook(
        direction=vector,
        donor_coeff=float(donor_coeff),
        alpha=float(alpha),
        position_fraction=float(position_fraction),
    )
    hook.attach(model, int(layer), site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer, anchor_direction))
    finally:
        hook.remove()


def evaluate_with_orth_hook(model, prepared_item, anchor_layer, anchor_direction, layer, site, vector, donor_tensor, alpha, position_fraction, donor_norm_match):
    hook = TensorSiteOrthogonalResidualInterchangeHook(
        direction=vector,
        donor_tensor=donor_tensor,
        alpha=float(alpha),
        position_fraction=float(position_fraction),
        donor_norm_match=bool(donor_norm_match),
    )
    hook.attach(model, int(layer), site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer, anchor_direction))
    finally:
        hook.remove()


def evaluate_with_swap_hook(model, prepared_item, anchor_layer, anchor_direction, layer, site, donor_tensor, position_fraction, norm_match):
    hook = TensorSiteSwapHook(source_tensor=donor_tensor, position_fraction=float(position_fraction), norm_match=bool(norm_match))
    hook.attach(model, int(layer), site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer, anchor_direction))
    finally:
        hook.remove()


def build_slice_rows(detail_df, quantiles, min_slice_items):
    rows = []
    meta_cols = [
        "dataset_name",
        "target_layer",
        "site",
        "item_name",
        "pair_name",
        "signed_counterfactual_gap",
        "abs_semantic_coeff_delta",
    ]
    score_df = detail_df[detail_df["condition"] == "scalar_semantic"][meta_cols].drop_duplicates()
    grouped = score_df.groupby(["dataset_name", "target_layer", "site"], dropna=False)
    for (dataset_name, target_layer, site), group in grouped:
        item_scores = group.set_index("item_name")
        score_values = item_scores["signed_counterfactual_gap"].to_numpy(dtype=np.float64)
        thresholds = [("all_items", np.nan)]
        for quantile in quantiles:
            thresholds.append((f"top_q{int(round(100 * quantile)):02d}", float(np.quantile(score_values, quantile))))
        sub = detail_df[
            (detail_df["dataset_name"] == dataset_name)
            & (detail_df["target_layer"] == target_layer)
            & (detail_df["site"] == site)
        ].copy()
        for slice_name, threshold in thresholds:
            if slice_name == "all_items":
                selected_items = set(item_scores.index)
                tail_quantile = 0.0
            else:
                selected_items = set(item_scores.index[item_scores["signed_counterfactual_gap"] >= threshold])
                tail_quantile = float(slice_name.replace("top_q", "")) / 100.0
            if len(selected_items) < int(min_slice_items):
                continue
            slice_part = sub[sub["item_name"].isin(selected_items)].copy()
            slice_part["slice_name"] = slice_name
            slice_part["tail_quantile"] = float(tail_quantile)
            slice_part["counterfactual_gap_threshold"] = float(threshold) if np.isfinite(threshold) else np.nan
            rows.append(slice_part)
    if not rows:
        return pd.DataFrame(columns=list(detail_df.columns) + ["slice_name", "tail_quantile", "counterfactual_gap_threshold"])
    return pd.concat(rows, axis=0, ignore_index=True)


def append_vs_zero_row(rows, *, group_key, values, condition_name, comparison_type, seed):
    stats = signflip_test(values, seed=seed)
    rows.append(
        {
            "comparison_type": comparison_type,
            "dataset_name": group_key[0],
            "target_layer": int(group_key[1]),
            "site": group_key[2],
            "slice_name": group_key[3],
            "tail_quantile": float(group_key[4]),
            "counterfactual_gap_threshold": float(group_key[5]) if np.isfinite(group_key[5]) else np.nan,
            "n_items": int(values.shape[0]),
            "condition_a": condition_name,
            "condition_b": "",
            "condition_a_mean_toward_donor_shift_signed_label_margin": stats["mean"],
            "condition_a_ci95_low": stats["ci95_low"],
            "condition_a_ci95_high": stats["ci95_high"],
            "condition_a_vs_zero_pvalue": stats["pvalue"],
            "condition_b_mean_toward_donor_shift_signed_label_margin": np.nan,
            "condition_b_ci95_low": np.nan,
            "condition_b_ci95_high": np.nan,
            "condition_b_vs_zero_pvalue": np.nan,
            "condition_a_minus_b_mean_toward_donor_shift_signed_label_margin": np.nan,
            "condition_a_minus_b_ci95_low": np.nan,
            "condition_a_minus_b_ci95_high": np.nan,
            "condition_a_vs_b_pvalue": np.nan,
        }
    )


def append_paired_row(rows, *, group_key, pivot, condition_a, condition_b, comparison_type, seed_base):
    if condition_a not in pivot.columns or condition_b not in pivot.columns:
        return
    paired = pivot[[condition_a, condition_b]].dropna()
    if paired.empty:
        return
    values_a = paired[condition_a].to_numpy(dtype=np.float64)
    values_b = paired[condition_b].to_numpy(dtype=np.float64)
    stats_a = signflip_test(values_a, seed=seed_base + 1)
    stats_b = signflip_test(values_b, seed=seed_base + 2)
    stats_diff = paired_signflip_test(values_a, values_b, seed=seed_base + 3)
    rows.append(
        {
            "comparison_type": comparison_type,
            "dataset_name": group_key[0],
            "target_layer": int(group_key[1]),
            "site": group_key[2],
            "slice_name": group_key[3],
            "tail_quantile": float(group_key[4]),
            "counterfactual_gap_threshold": float(group_key[5]) if np.isfinite(group_key[5]) else np.nan,
            "n_items": int(paired.shape[0]),
            "condition_a": condition_a,
            "condition_b": condition_b,
            "condition_a_mean_toward_donor_shift_signed_label_margin": stats_a["mean"],
            "condition_a_ci95_low": stats_a["ci95_low"],
            "condition_a_ci95_high": stats_a["ci95_high"],
            "condition_a_vs_zero_pvalue": stats_a["pvalue"],
            "condition_b_mean_toward_donor_shift_signed_label_margin": stats_b["mean"],
            "condition_b_ci95_low": stats_b["ci95_low"],
            "condition_b_ci95_high": stats_b["ci95_high"],
            "condition_b_vs_zero_pvalue": stats_b["pvalue"],
            "condition_a_minus_b_mean_toward_donor_shift_signed_label_margin": stats_diff["mean"],
            "condition_a_minus_b_ci95_low": stats_diff["ci95_low"],
            "condition_a_minus_b_ci95_high": stats_diff["ci95_high"],
            "condition_a_vs_b_pvalue": stats_diff["pvalue"],
        }
    )


def build_stats_rows(slice_df):
    rows = []
    grouped = slice_df.groupby(["dataset_name", "target_layer", "site", "slice_name", "tail_quantile", "counterfactual_gap_threshold"], dropna=False)
    for group_key, group in grouped:
        pivot = group.pivot(index="item_name", columns="condition", values="toward_donor_shift_signed_label_margin")
        if "full_donor" in pivot.columns:
            values = pivot["full_donor"].dropna().to_numpy(dtype=np.float64)
            if values.size:
                append_vs_zero_row(rows, group_key=group_key, values=values, condition_name="full_donor", comparison_type="full_donor_vs_zero", seed=4000 + 19 * int(group_key[1]))
        append_paired_row(rows, group_key=group_key, pivot=pivot, condition_a="scalar_semantic", condition_b="scalar_random", comparison_type="scalar_semantic_vs_random", seed_base=1000 + 19 * int(group_key[1]))
        append_paired_row(rows, group_key=group_key, pivot=pivot, condition_a="orth_semantic", condition_b="orth_random", comparison_type="orth_semantic_vs_random", seed_base=2000 + 19 * int(group_key[1]))
        append_paired_row(rows, group_key=group_key, pivot=pivot, condition_a="full_donor", condition_b="scalar_semantic", comparison_type="full_donor_vs_scalar_semantic", seed_base=3000 + 19 * int(group_key[1]))
        append_paired_row(rows, group_key=group_key, pivot=pivot, condition_a="full_donor", condition_b="orth_semantic", comparison_type="full_donor_vs_orth_semantic", seed_base=5000 + 19 * int(group_key[1]))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 10Y: natural paired-donor bundle interchange at the within-band corridor")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument(
        "--eval-jsons",
        type=str,
        default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json",
    )
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--target-layers", type=str, default="7,11")
    parser.add_argument("--sites", type=str, default="attn_output")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--k-bulk", type=int, default=70)
    parser.add_argument("--min-retained-fraction", type=float, default=0.10)
    parser.add_argument("--rank-tol", type=float, default=1e-8)
    parser.add_argument("--tail-quantiles", type=str, default="0.67,0.80")
    parser.add_argument("--min-slice-items", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--orth-donor-norm-match", action="store_true")
    parser.add_argument("--full-donor-norm-match", action="store_true")
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/natural_bundle_interchange_summary.csv")
    args = parser.parse_args()

    eval_jsons = parse_str_list(args.eval_jsons)
    dataset_labels = parse_str_list(args.dataset_labels)
    if len(eval_jsons) != len(dataset_labels):
        raise ValueError("--eval-jsons and --dataset-labels must have the same number of entries.")
    target_layers = parse_int_list(args.target_layers)
    sites = parse_site_list(args.sites)
    tail_quantiles = parse_float_list(args.tail_quantiles)

    print("Loading Genesis model for Phase 10Y natural paired-donor bundle interchange...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    anchor_direction = load_anchor_direction(args.data_dir, args.anchor_layer, allow_invalid_metadata=args.allow_invalid_metadata)

    datasets = []
    for dataset_label, eval_json in zip(dataset_labels, eval_jsons):
        pair_items = load_pair_items(eval_json, max_eval_items=args.max_eval_items)
        eval_items = expand_pair_items(pair_items)
        prepared_items = [prepare_eval_item(tokenizer, item, device) for item in eval_items]
        baseline_metrics = {}
        print(f"Preparing dataset={dataset_label} with {len(pair_items)} pairs / {len(prepared_items)} eval items")
        for prepared in prepared_items:
            baseline_metrics[prepared["item"]["name"]] = add_sign_aware_fields(prepared, evaluate_prepared_item(model, prepared, args.anchor_layer, anchor_direction))
        state_cache = {}
        for target_layer in target_layers:
            for site in sites:
                state_cache[(int(target_layer), site)] = {}
                for prepared in prepared_items:
                    item_name = prepared["item"]["name"]
                    state_cache[(int(target_layer), site)][item_name] = capture_site_state(
                        model,
                        prepared["prompt_ids"],
                        target_layer,
                        site,
                        args.position_fraction,
                    )
        datasets.append(
            {
                "label": dataset_label,
                "pair_items": pair_items,
                "prepared_items": prepared_items,
                "baseline_metrics": baseline_metrics,
                "state_cache": state_cache,
            }
        )

    print("Fitting same-family leave-one-pair-out directions at target sites...")
    dataset_fits = {}
    vector_rows = []
    for dataset in datasets:
        dataset_label = dataset["label"]
        dataset_fits[dataset_label] = {}
        for target_layer in target_layers:
            for site in sites:
                records = []
                for prepared in dataset["prepared_items"]:
                    item_name = prepared["item"]["name"]
                    records.append(
                        {
                            "pair_name": pair_name_from_item_name(item_name),
                            "label": prepared["item"]["label"],
                            "state": dataset["state_cache"][(int(target_layer), site)][item_name].detach().cpu().numpy().astype(np.float64),
                        }
                    )
                full_fit = fit_direction_from_records(
                    records,
                    vector_key=args.vector_key,
                    k_bulk=args.k_bulk,
                    min_retained_fraction=args.min_retained_fraction,
                    rank_tol=args.rank_tol,
                )
                dataset_fits[dataset_label][(int(target_layer), site)] = {"full": full_fit, "crossfit": {}}
                vector_rows.append(
                    {
                        "dataset_name": dataset_label,
                        "target_layer": int(target_layer),
                        "site": site,
                        "fit_scope": "full",
                        "heldout_pair_name": "ALL",
                        "n_pairs_fit": full_fit["n_pairs_fit"],
                        "n_prompt_records_fit": full_fit["n_prompt_records_fit"],
                        "raw_norm": float(full_fit["raw_norm"]),
                        "perp_norm": float(full_fit["perp_norm"]),
                        "selected_vector_norm": float(full_fit["selected_vector_norm"]),
                        "retained_fraction": float(full_fit["retained_fraction"]),
                        "bulk_variance_explained": float(full_fit["bulk_variance_explained"]),
                        "k_bulk_effective": int(full_fit["k_bulk_effective"]),
                        "numerical_rank": int(full_fit["numerical_rank"]),
                        "sample_rank_cap": int(full_fit["sample_rank_cap"]),
                        "effective_rank": float(full_fit["effective_rank"]),
                        "cosine_to_full": 1.0,
                    }
                )
                for pair_name in [item["name"] for item in dataset["pair_items"]]:
                    remaining = [row for row in records if row["pair_name"] != pair_name]
                    crossfit = fit_direction_from_records(
                        remaining,
                        vector_key=args.vector_key,
                        k_bulk=args.k_bulk,
                        min_retained_fraction=args.min_retained_fraction,
                        rank_tol=args.rank_tol,
                    )
                    dataset_fits[dataset_label][(int(target_layer), site)]["crossfit"][pair_name] = crossfit
                    vector_rows.append(
                        {
                            "dataset_name": dataset_label,
                            "target_layer": int(target_layer),
                            "site": site,
                            "fit_scope": "leave_one_pair_out",
                            "heldout_pair_name": pair_name,
                            "n_pairs_fit": crossfit["n_pairs_fit"],
                            "n_prompt_records_fit": crossfit["n_prompt_records_fit"],
                            "raw_norm": float(crossfit["raw_norm"]),
                            "perp_norm": float(crossfit["perp_norm"]),
                            "selected_vector_norm": float(crossfit["selected_vector_norm"]),
                            "retained_fraction": float(crossfit["retained_fraction"]),
                            "bulk_variance_explained": float(crossfit["bulk_variance_explained"]),
                            "k_bulk_effective": int(crossfit["k_bulk_effective"]),
                            "numerical_rank": int(crossfit["numerical_rank"]),
                            "sample_rank_cap": int(crossfit["sample_rank_cap"]),
                            "effective_rank": float(crossfit["effective_rank"]),
                            "cosine_to_full": cosine_similarity(crossfit["unit_vector"], full_fit["unit_vector"]),
                        }
                    )

    print("Running paired opposite-label natural bundle interchange evaluation...")
    detail_rows = []
    for dataset_idx, dataset in enumerate(datasets):
        dataset_label = dataset["label"]
        for item_idx, prepared in enumerate(dataset["prepared_items"]):
            item_name = prepared["item"]["name"]
            pair_name = pair_name_from_item_name(item_name)
            donor_item_name = f"{pair_name}__{opposite_label(prepared['item']['label'])}"
            baseline = dataset["baseline_metrics"][item_name]
            label_sign = float(prepared["label_sign"])
            for target_layer in target_layers:
                for site_idx, site in enumerate(sites):
                    fit = dataset_fits[dataset_label][(int(target_layer), site)]["crossfit"][pair_name]
                    semantic_vec = torch.tensor(fit["unit_vector"], device=device, dtype=torch.float32)
                    random_vec = make_random_orthogonal_control(
                        semantic_vec,
                        seed=args.seed + 100000 * dataset_idx + 1000 * int(target_layer) + 100 * site_idx + item_idx,
                    )
                    recipient_state = dataset["state_cache"][(int(target_layer), site)][item_name]
                    donor_state = dataset["state_cache"][(int(target_layer), site)][donor_item_name]
                    semantic_recipient_coeff = float(torch.dot(recipient_state, semantic_vec).item())
                    semantic_donor_coeff = float(torch.dot(donor_state, semantic_vec).item())
                    random_recipient_coeff = float(torch.dot(recipient_state, random_vec).item())
                    random_donor_coeff = float(torch.dot(donor_state, random_vec).item())
                    semantic_recipient_orth = recipient_state - (semantic_recipient_coeff * semantic_vec)
                    semantic_donor_orth = donor_state - (semantic_donor_coeff * semantic_vec)
                    random_recipient_orth = recipient_state - (random_recipient_coeff * random_vec)
                    random_donor_orth = donor_state - (random_donor_coeff * random_vec)
                    signed_counterfactual_gap = float(label_sign * (semantic_recipient_coeff - semantic_donor_coeff))

                    condition_metrics = {
                        "scalar_semantic": evaluate_with_scalar_hook(
                            model, prepared, args.anchor_layer, anchor_direction, target_layer, site, semantic_vec, semantic_donor_coeff, args.alpha, args.position_fraction
                        ),
                        "scalar_random": evaluate_with_scalar_hook(
                            model, prepared, args.anchor_layer, anchor_direction, target_layer, site, random_vec, random_donor_coeff, args.alpha, args.position_fraction
                        ),
                        "orth_semantic": evaluate_with_orth_hook(
                            model, prepared, args.anchor_layer, anchor_direction, target_layer, site, semantic_vec, donor_state, args.alpha, args.position_fraction, args.orth_donor_norm_match
                        ),
                        "orth_random": evaluate_with_orth_hook(
                            model, prepared, args.anchor_layer, anchor_direction, target_layer, site, random_vec, donor_state, args.alpha, args.position_fraction, args.orth_donor_norm_match
                        ),
                        "full_donor": evaluate_with_swap_hook(
                            model, prepared, args.anchor_layer, anchor_direction, target_layer, site, donor_state, args.position_fraction, args.full_donor_norm_match
                        ),
                    }

                    for condition_name, metrics in condition_metrics.items():
                        detail_rows.append(
                            {
                                "dataset_name": dataset_label,
                                "target_layer": int(target_layer),
                                "site": site,
                                "condition": condition_name,
                                "fit_scope": "leave_one_pair_out",
                                "item_name": item_name,
                                "pair_name": pair_name,
                                "label": prepared["item"]["label"],
                                "donor_item_name": donor_item_name,
                                "donor_label": opposite_label(prepared["item"]["label"]),
                                "baseline_signed_label_margin": float(baseline["signed_label_margin"]),
                                "baseline_label_target_pairwise_prob": float(baseline["label_target_pairwise_prob"]),
                                "baseline_label_accuracy": float(baseline["label_accuracy"]),
                                "signed_label_margin": float(metrics["signed_label_margin"]),
                                "label_target_pairwise_prob": float(metrics["label_target_pairwise_prob"]),
                                "label_accuracy": float(metrics["label_accuracy"]),
                                "anchor_cosine": float(metrics["anchor_cosine"]),
                                "delta_from_baseline_signed_label_margin": float(metrics["signed_label_margin"] - baseline["signed_label_margin"]),
                                "delta_from_baseline_label_target_pairwise_prob": float(metrics["label_target_pairwise_prob"] - baseline["label_target_pairwise_prob"]),
                                "delta_from_baseline_label_accuracy": float(metrics["label_accuracy"] - baseline["label_accuracy"]),
                                "delta_from_baseline_anchor_cosine": float(metrics["anchor_cosine"] - baseline["anchor_cosine"]),
                                "toward_donor_shift_signed_label_margin": float(baseline["signed_label_margin"] - metrics["signed_label_margin"]),
                                "toward_donor_shift_label_target_pairwise_prob": float(baseline["label_target_pairwise_prob"] - metrics["label_target_pairwise_prob"]),
                                "toward_donor_shift_label_accuracy": float(baseline["label_accuracy"] - metrics["label_accuracy"]),
                                "semantic_recipient_coeff": float(semantic_recipient_coeff),
                                "semantic_donor_coeff": float(semantic_donor_coeff),
                                "semantic_coeff_delta": float(semantic_donor_coeff - semantic_recipient_coeff),
                                "abs_semantic_coeff_delta": float(abs(semantic_donor_coeff - semantic_recipient_coeff)),
                                "signed_counterfactual_gap": float(signed_counterfactual_gap),
                                "random_recipient_coeff": float(random_recipient_coeff),
                                "random_donor_coeff": float(random_donor_coeff),
                                "random_coeff_delta": float(random_donor_coeff - random_recipient_coeff),
                                "recipient_state_norm": float(torch.norm(recipient_state).item()),
                                "donor_state_norm": float(torch.norm(donor_state).item()),
                                "semantic_recipient_orth_norm": float(torch.norm(semantic_recipient_orth).item()),
                                "semantic_donor_orth_norm": float(torch.norm(semantic_donor_orth).item()),
                                "random_recipient_orth_norm": float(torch.norm(random_recipient_orth).item()),
                                "random_donor_orth_norm": float(torch.norm(random_donor_orth).item()),
                            }
                        )

    detail_df = pd.DataFrame(detail_rows)
    slice_df = build_slice_rows(detail_df, quantiles=tail_quantiles, min_slice_items=args.min_slice_items)
    summary_df = (
        slice_df.groupby(["dataset_name", "target_layer", "site", "slice_name", "tail_quantile", "counterfactual_gap_threshold", "condition"], dropna=False)
        .agg(
            n_items=("item_name", "count"),
            mean_signed_counterfactual_gap=("signed_counterfactual_gap", "mean"),
            mean_abs_semantic_coeff_delta=("abs_semantic_coeff_delta", "mean"),
            mean_recipient_state_norm=("recipient_state_norm", "mean"),
            mean_donor_state_norm=("donor_state_norm", "mean"),
            mean_semantic_recipient_orth_norm=("semantic_recipient_orth_norm", "mean"),
            mean_semantic_donor_orth_norm=("semantic_donor_orth_norm", "mean"),
            toward_donor_shift_mean_signed_label_margin=("toward_donor_shift_signed_label_margin", "mean"),
            toward_donor_shift_mean_label_target_pairwise_prob=("toward_donor_shift_label_target_pairwise_prob", "mean"),
            toward_donor_shift_mean_label_accuracy=("toward_donor_shift_label_accuracy", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            delta_from_baseline_mean_anchor_cosine=("delta_from_baseline_anchor_cosine", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            mean_label_accuracy=("label_accuracy", "mean"),
        )
        .reset_index()
    )
    stats_df = build_stats_rows(slice_df)
    vector_df = pd.DataFrame(vector_rows)
    vector_stats_df = (
        vector_df.groupby(["dataset_name", "target_layer", "site", "fit_scope"], dropna=False)
        .agg(
            n_rows=("heldout_pair_name", "count"),
            mean_retained_fraction=("retained_fraction", "mean"),
            min_retained_fraction=("retained_fraction", "min"),
            mean_bulk_variance_explained=("bulk_variance_explained", "mean"),
            mean_effective_rank=("effective_rank", "mean"),
            mean_cosine_to_full=("cosine_to_full", "mean"),
            min_cosine_to_full=("cosine_to_full", "min"),
        )
        .reset_index()
    )

    detail_path = infer_companion_csv(args.output_csv, "detail")
    stats_path = infer_companion_csv(args.output_csv, "stats")
    vector_path = infer_companion_csv(args.output_csv, "vector_fits")
    vector_stats_path = infer_companion_csv(args.output_csv, "vector_fit_stats")
    slice_detail_path = infer_companion_csv(args.output_csv, "slice_detail")
    for path in (args.output_csv, detail_path, slice_detail_path, stats_path, vector_path, vector_stats_path):
        ensure_parent_dir(path)
    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_path, index=False)
    slice_df.to_csv(slice_detail_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    vector_df.to_csv(vector_path, index=False)
    vector_stats_df.to_csv(vector_stats_path, index=False)

    print(f"Saved summary to {args.output_csv}")
    print(f"Saved detail to {detail_path}")
    print(f"Saved slice detail to {slice_detail_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Saved vector fits to {vector_path}")
    print(f"Saved vector fit stats to {vector_stats_path}")


if __name__ == "__main__":
    main()