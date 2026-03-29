import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import TensorSiteCaptureHook, TensorSiteInterventionHook, parse_site_list
from scripts.phase9_semantic_utils import parse_int_list
from scripts.run_phase9_extract import WelfordCovariance
from scripts.run_phase9_semantic_dirs import isolate_semantic_direction
from scripts.run_phase9_semantic_steering import load_anchor_direction, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, prepare_eval_item


def parse_str_list(raw):
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_float_list(raw):
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def pair_name_from_item_name(item_name):
    return item_name.rsplit("__", 1)[0]


def normalize_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError("Semantic direction norm is near zero.")
    return vec / norm, norm


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_pair_items(json_path, max_eval_items=None):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload["items"] if isinstance(payload, dict) else payload
    required = {"name", "math_prompt", "creative_prompt", "math_option"}
    for idx, item in enumerate(items):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Pair item {idx} in {json_path} missing keys: {sorted(missing)}")
    if max_eval_items is not None:
        max_pairs = max(1, int(math.ceil(int(max_eval_items) / 2.0)))
        items = items[:max_pairs]
    return items


def expand_pair_items(pair_items):
    expanded = []
    for item in pair_items:
        math_option = item["math_option"].strip().upper()
        expanded.extend(
            [
                {"name": f"{item['name']}__math", "label": "math", "math_option": math_option, "prompt": item["math_prompt"]},
                {"name": f"{item['name']}__creative", "label": "creative", "math_option": math_option, "prompt": item["creative_prompt"]},
            ]
        )
    return expanded


def add_sign_aware_fields(prepared_item, metrics):
    metrics = dict(metrics)
    label_target_prob = metrics["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else metrics["pairwise_creative_prob"]
    metrics["label_target_pairwise_prob"] = float(label_target_prob)
    metrics["label_accuracy"] = float(metrics["label_correct"])
    return metrics


def capture_site_state(model, prompt_ids, layer, site, position_fraction):
    hook = TensorSiteCaptureHook(position_fraction=position_fraction)
    hook.attach(model, int(layer), site)
    try:
        with torch.inference_mode():
            model(prompt_ids)
        if hook.captured is None:
            raise RuntimeError(f"No site state captured for layer={layer}, site={site}")
        return hook.captured.squeeze(0).detach().clone()
    finally:
        hook.remove()


def evaluate_with_hook(model, prepared_item, anchor_layer, anchor_direction, layer, site, vector, alpha, mode, position_fraction):
    hook = TensorSiteInterventionHook(vector=vector, alpha=float(alpha), mode=mode, position_fraction=float(position_fraction))
    hook.attach(model, int(layer), site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer, anchor_direction))
    finally:
        hook.remove()


def fit_direction_from_records(records, vector_key, k_bulk, min_retained_fraction, rank_tol):
    if not records:
        raise ValueError("Cannot fit semantic direction from an empty record set.")
    d_model = int(records[0]["state"].shape[-1])
    cov = WelfordCovariance(d_model)
    sums = {"math": np.zeros(d_model, dtype=np.float64), "creative": np.zeros(d_model, dtype=np.float64)}
    counts = {"math": 0, "creative": 0}
    for row in records:
        state = np.asarray(row["state"], dtype=np.float64)
        cov.update(state)
        sums[row["label"]] += state
        counts[row["label"]] += 1
    if counts["math"] == 0 or counts["creative"] == 0:
        raise ValueError(f"Both labels need coverage; got counts={counts}")
    fit = isolate_semantic_direction(
        cov=cov.get_covariance(),
        math_centroid=sums["math"] / counts["math"],
        creative_centroid=sums["creative"] / counts["creative"],
        k_bulk=int(k_bulk),
        min_retained_fraction=float(min_retained_fraction),
        rank_tol=float(rank_tol),
        n_samples=len(records),
    )
    unit_vec, selected_norm = normalize_vector(fit[vector_key])
    fit.update(
        {
            "unit_vector": unit_vec,
            "selected_vector_norm": float(selected_norm),
            "n_pairs_fit": int(min(counts["math"], counts["creative"])),
            "n_prompt_records_fit": int(len(records)),
        }
    )
    return fit


def build_slice_rows(detail_df, quantiles, min_slice_items):
    rows = []
    meta_cols = ["dataset_name", "target_layer", "site", "item_name", "pair_name", "signed_semantic_projection", "signed_semantic_cosine"]
    score_df = detail_df[detail_df["control"] == "semantic"][meta_cols].drop_duplicates()
    grouped = score_df.groupby(["dataset_name", "target_layer", "site"], dropna=False)
    for (dataset_name, target_layer, site), group in grouped:
        item_scores = group.set_index("item_name")
        score_values = item_scores["signed_semantic_cosine"].to_numpy(dtype=np.float64)
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
                selected_items = set(item_scores.index[item_scores["signed_semantic_cosine"] >= threshold])
                tail_quantile = float(slice_name.replace("top_q", "")) / 100.0
            if len(selected_items) < int(min_slice_items):
                continue
            slice_df = sub[sub["item_name"].isin(selected_items)].copy()
            slice_df["slice_name"] = slice_name
            slice_df["tail_quantile"] = float(tail_quantile)
            slice_df["occupancy_threshold_signed_cosine"] = float(threshold) if np.isfinite(threshold) else np.nan
            rows.append(slice_df)
    if not rows:
        return pd.DataFrame(columns=list(detail_df.columns) + ["slice_name", "tail_quantile", "occupancy_threshold_signed_cosine"])
    return pd.concat(rows, axis=0, ignore_index=True)


def build_stats_rows(slice_df):
    rows = []
    grouped = slice_df.groupby(["dataset_name", "target_layer", "site", "slice_name", "tail_quantile", "occupancy_threshold_signed_cosine"], dropna=False)
    for (dataset_name, target_layer, site, slice_name, tail_quantile, threshold), group in grouped:
        pivot = group.pivot(index="item_name", columns="control", values="delta_from_baseline_signed_label_margin").dropna()
        if pivot.empty or "semantic" not in pivot.columns or "random" not in pivot.columns:
            continue
        semantic = pivot["semantic"].to_numpy(dtype=np.float64)
        random = pivot["random"].to_numpy(dtype=np.float64)
        semantic_zero = signflip_test(semantic, seed=1000 + 13 * int(target_layer))
        random_zero = signflip_test(random, seed=2000 + 13 * int(target_layer))
        semantic_vs_random = paired_signflip_test(semantic, random, seed=3000 + 13 * int(target_layer))
        rows.append(
            {
                "comparison_type": "semantic_vs_random",
                "dataset_name": dataset_name,
                "target_layer": int(target_layer),
                "site": site,
                "slice_name": slice_name,
                "tail_quantile": float(tail_quantile),
                "occupancy_threshold_signed_cosine": float(threshold) if np.isfinite(threshold) else np.nan,
                "n_items": int(pivot.shape[0]),
                "semantic_mean_delta_signed_label_margin": semantic_zero["mean"],
                "semantic_ci95_low": semantic_zero["ci95_low"],
                "semantic_ci95_high": semantic_zero["ci95_high"],
                "semantic_vs_baseline_pvalue": semantic_zero["pvalue"],
                "random_mean_delta_signed_label_margin": random_zero["mean"],
                "random_ci95_low": random_zero["ci95_low"],
                "random_ci95_high": random_zero["ci95_high"],
                "random_vs_baseline_pvalue": random_zero["pvalue"],
                "semantic_minus_random_mean_delta_signed_label_margin": semantic_vs_random["mean"],
                "semantic_minus_random_ci95_low": semantic_vs_random["ci95_low"],
                "semantic_minus_random_ci95_high": semantic_vs_random["ci95_high"],
                "semantic_vs_random_pvalue": semantic_vs_random["pvalue"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 10U: tail-conditioned same-family necessity at the within-band corridor")
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
    parser.add_argument("--mode", type=str, choices=["ablate", "add"], default="ablate")
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
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/tail_conditioned_necessity_summary.csv")
    args = parser.parse_args()

    eval_jsons = parse_str_list(args.eval_jsons)
    dataset_labels = parse_str_list(args.dataset_labels)
    if len(eval_jsons) != len(dataset_labels):
        raise ValueError("--eval-jsons and --dataset-labels must have the same number of entries.")
    target_layers = parse_int_list(args.target_layers)
    sites = parse_site_list(args.sites)
    tail_quantiles = parse_float_list(args.tail_quantiles)

    print("Loading Genesis model for Phase 10U tail-conditioned necessity...")
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
            baseline_metrics[prepared["item"]["name"]] = add_sign_aware_fields(
                prepared,
                evaluate_prepared_item(model, prepared, args.anchor_layer, anchor_direction),
            )
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
    for dataset_idx, dataset in enumerate(datasets):
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
                pair_names = [item["name"] for item in dataset["pair_items"]]
                for pair_idx, pair_name in enumerate(pair_names):
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

    print("Running tail-conditioned necessity evaluation...")
    detail_rows = []
    for dataset_idx, dataset in enumerate(datasets):
        dataset_label = dataset["label"]
        for item_idx, prepared in enumerate(dataset["prepared_items"]):
            item_name = prepared["item"]["name"]
            pair_name = pair_name_from_item_name(item_name)
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
                    state = dataset["state_cache"][(int(target_layer), site)][item_name]
                    state_norm = float(torch.linalg.norm(state).item())
                    semantic_projection = float(torch.dot(state, semantic_vec).item())
                    semantic_cosine = float(semantic_projection / max(state_norm, 1e-8))
                    signed_projection = float(label_sign * semantic_projection)
                    signed_cosine = float(label_sign * semantic_cosine)
                    for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                        metrics = evaluate_with_hook(
                            model,
                            prepared,
                            args.anchor_layer,
                            anchor_direction,
                            target_layer,
                            site,
                            vector,
                            args.alpha,
                            args.mode,
                            args.position_fraction,
                        )
                        detail_rows.append(
                            {
                                "dataset_name": dataset_label,
                                "target_layer": int(target_layer),
                                "site": site,
                                "control": control_name,
                                "fit_scope": "leave_one_pair_out",
                                "item_name": item_name,
                                "pair_name": pair_name,
                                "label": prepared["item"]["label"],
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
                                "state_norm": float(state_norm),
                                "semantic_projection": float(semantic_projection),
                                "semantic_cosine": float(semantic_cosine),
                                "signed_semantic_projection": float(signed_projection),
                                "signed_semantic_cosine": float(signed_cosine),
                            }
                        )

    detail_df = pd.DataFrame(detail_rows)
    slice_df = build_slice_rows(detail_df, quantiles=tail_quantiles, min_slice_items=args.min_slice_items)
    summary_df = (
        slice_df.groupby(["dataset_name", "target_layer", "site", "slice_name", "tail_quantile", "occupancy_threshold_signed_cosine", "control"], dropna=False)
        .agg(
            n_items=("item_name", "count"),
            mean_signed_semantic_cosine=("signed_semantic_cosine", "mean"),
            min_signed_semantic_cosine=("signed_semantic_cosine", "min"),
            max_signed_semantic_cosine=("signed_semantic_cosine", "max"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
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