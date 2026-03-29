import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import format_chatml_prompt, load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import TensorSiteInterventionHook, parse_site_list
from scripts.phase9_semantic_utils import parse_int_list
from scripts.run_phase9_extract import MultiLayerCaptureHook, WelfordCovariance
from scripts.run_phase9_semantic_dirs import isolate_semantic_direction
from scripts.run_phase9_semantic_steering import load_anchor_direction, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, prepare_eval_item


def parse_str_list(raw):
    return [part.strip() for part in raw.split(",") if part.strip()]


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


def collect_pair_activations(model, tokenizer, pair_items, source_layer, device):
    hook = MultiLayerCaptureHook([int(source_layer)])
    hook.attach(model)
    records = []
    try:
        for item in pair_items:
            for label, prompt in (("math", item["math_prompt"]), ("creative", item["creative_prompt"])):
                prompt_ids = torch.tensor([tokenizer.encode(format_chatml_prompt(prompt))], device=device)
                hook.clear()
                with torch.inference_mode():
                    model(prompt_ids)
                captured = hook.captured_acts[int(source_layer)][-1]
                records.append(
                    {
                        "pair_name": item["name"],
                        "label": label,
                        "activation": captured[0].astype(np.float64),
                    }
                )
    finally:
        hook.remove()
    return records


def fit_direction_from_records(records, vector_key, k_bulk, min_retained_fraction, rank_tol):
    if not records:
        raise ValueError("Cannot fit semantic direction from an empty record set.")
    d_model = int(records[0]["activation"].shape[-1])
    cov = WelfordCovariance(d_model)
    sums = {"math": np.zeros(d_model, dtype=np.float64), "creative": np.zeros(d_model, dtype=np.float64)}
    counts = {"math": 0, "creative": 0}
    for row in records:
        cov.update(row["activation"])
        sums[row["label"]] += row["activation"]
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
            "math_count": int(counts["math"]),
            "creative_count": int(counts["creative"]),
        }
    )
    return fit


def add_sign_aware_fields(prepared_item, metrics):
    metrics = dict(metrics)
    label_target_prob = metrics["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else metrics["pairwise_creative_prob"]
    metrics["label_target_pairwise_prob"] = float(label_target_prob)
    metrics["label_accuracy"] = float(metrics["label_correct"])
    return metrics


def evaluate_with_hook(model, prepared_item, anchor_layer, anchor_direction, layer, site, vector, alpha, mode, position_fraction):
    hook = TensorSiteInterventionHook(
        vector=vector,
        alpha=float(alpha),
        mode=mode,
        position_fraction=float(position_fraction),
    )
    hook.attach(model, int(layer), site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer, anchor_direction))
    finally:
        hook.remove()


def build_stats_rows(detail_df):
    rows = []
    grouped = detail_df.groupby(["eval_dataset", "source_dataset", "target_layer", "site"])
    for (eval_dataset, source_dataset, target_layer, site), group in grouped:
        pivot = group.pivot(index="item_name", columns="control", values="delta_from_baseline_signed_label_margin").dropna()
        if pivot.empty or "semantic" not in pivot.columns or "random" not in pivot.columns:
            continue
        semantic = pivot["semantic"].to_numpy(dtype=np.float64)
        random = pivot["random"].to_numpy(dtype=np.float64)
        semantic_zero = signflip_test(semantic, seed=1000 + 17 * int(target_layer))
        random_zero = signflip_test(random, seed=2000 + 17 * int(target_layer))
        semantic_vs_random = paired_signflip_test(semantic, random, seed=3000 + 17 * int(target_layer))
        rows.append(
            {
                "comparison_type": "semantic_vs_random",
                "eval_dataset": eval_dataset,
                "source_dataset": source_dataset,
                "same_family": int(eval_dataset == source_dataset),
                "target_layer": int(target_layer),
                "site": site,
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

    semantic_only = detail_df[detail_df["control"] == "semantic"].copy()
    for (eval_dataset, target_layer, site), group in semantic_only.groupby(["eval_dataset", "target_layer", "site"]):
        pivot = group.pivot(index="item_name", columns="source_dataset", values="delta_from_baseline_signed_label_margin").dropna()
        if pivot.shape[1] < 2:
            continue
        cols = [str(col) for col in pivot.columns]
        if eval_dataset in cols:
            cross_cols = [col for col in cols if col != eval_dataset]
            if cross_cols:
                same_family = pivot[eval_dataset].to_numpy(dtype=np.float64)
                cross_family_mean = pivot[cross_cols].mean(axis=1).to_numpy(dtype=np.float64)
                same_vs_cross = paired_signflip_test(
                    same_family,
                    cross_family_mean,
                    seed=4000 + 29 * int(target_layer),
                )
                rows.append(
                    {
                        "comparison_type": "same_family_vs_cross_family_mean",
                        "eval_dataset": eval_dataset,
                        "target_layer": int(target_layer),
                        "site": site,
                        "cross_family_sources": ",".join(cross_cols),
                        "n_items": int(pivot.shape[0]),
                        "same_family_mean": same_vs_cross["mean_a"],
                        "cross_family_mean": same_vs_cross["mean_b"],
                        "same_minus_cross_mean": same_vs_cross["mean"],
                        "ci95_low": same_vs_cross["ci95_low"],
                        "ci95_high": same_vs_cross["ci95_high"],
                        "pvalue": same_vs_cross["pvalue"],
                    }
                )
        for idx, source_a in enumerate(cols):
            for source_b in cols[idx + 1 :]:
                contrast = paired_signflip_test(
                    pivot[source_a].to_numpy(dtype=np.float64),
                    pivot[source_b].to_numpy(dtype=np.float64),
                    seed=5000 + 31 * int(target_layer) + idx,
                )
                rows.append(
                    {
                        "comparison_type": "semantic_source_contrast",
                        "eval_dataset": eval_dataset,
                        "target_layer": int(target_layer),
                        "site": site,
                        "source_a": source_a,
                        "source_b": source_b,
                        "n_items": int(pivot.shape[0]),
                        "mean_a": contrast["mean_a"],
                        "mean_b": contrast["mean_b"],
                        "mean_a_minus_b": contrast["mean"],
                        "ci95_low": contrast["ci95_low"],
                        "ci95_high": contrast["ci95_high"],
                        "pvalue": contrast["pvalue"],
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 10T: family-specific semantic-direction portability matrix")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument(
        "--eval-jsons",
        type=str,
        default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json",
    )
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,11")
    parser.add_argument("--sites", type=str, default="attn_output")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, default="add")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--k-bulk", type=int, default=70)
    parser.add_argument("--min-retained-fraction", type=float, default=0.10)
    parser.add_argument("--rank-tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--disable-same-family-crossfit", action="store_true")
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/family_direction_transfer_summary.csv")
    args = parser.parse_args()

    eval_jsons = parse_str_list(args.eval_jsons)
    dataset_labels = parse_str_list(args.dataset_labels)
    if len(eval_jsons) != len(dataset_labels):
        raise ValueError("--eval-jsons and --dataset-labels must have the same number of entries.")
    target_layers = parse_int_list(args.target_layers)
    sites = parse_site_list(args.sites)

    print("Loading Genesis model for Phase 10T family-direction transfer...")
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
        activations = collect_pair_activations(model, tokenizer, pair_items, args.source_layer, device)
        datasets.append(
            {
                "label": dataset_label,
                "pair_items": pair_items,
                "prepared_items": prepared_items,
                "baseline_metrics": baseline_metrics,
                "activation_records": activations,
            }
        )

    print("Fitting family-specific source directions...")
    source_fits = {}
    vector_rows = []
    for dataset in datasets:
        source_label = dataset["label"]
        full_fit = fit_direction_from_records(
            dataset["activation_records"],
            vector_key=args.vector_key,
            k_bulk=args.k_bulk,
            min_retained_fraction=args.min_retained_fraction,
            rank_tol=args.rank_tol,
        )
        source_fits[source_label] = {"full": full_fit, "crossfit": {}}
        vector_rows.append(
            {
                "source_dataset": source_label,
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
        if args.disable_same_family_crossfit:
            continue
        pair_names = [item["name"] for item in dataset["pair_items"]]
        for pair_name in pair_names:
            remaining = [row for row in dataset["activation_records"] if row["pair_name"] != pair_name]
            crossfit = fit_direction_from_records(
                remaining,
                vector_key=args.vector_key,
                k_bulk=args.k_bulk,
                min_retained_fraction=args.min_retained_fraction,
                rank_tol=args.rank_tol,
            )
            source_fits[source_label]["crossfit"][pair_name] = crossfit
            vector_rows.append(
                {
                    "source_dataset": source_label,
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

    detail_rows = []
    print("Running transfer matrix evaluation...")
    for eval_idx, eval_dataset in enumerate(datasets):
        eval_label = eval_dataset["label"]
        for source_idx, source_dataset in enumerate(datasets):
            source_label = source_dataset["label"]
            full_source_vec = torch.tensor(source_fits[source_label]["full"]["unit_vector"], device=device, dtype=torch.float32)
            full_random_vec = make_random_orthogonal_control(full_source_vec, seed=args.seed + 1000 + source_idx)
            for prepared_idx, prepared in enumerate(eval_dataset["prepared_items"]):
                item_name = prepared["item"]["name"]
                pair_name = pair_name_from_item_name(item_name)
                baseline = eval_dataset["baseline_metrics"][item_name]
                if eval_label == source_label and not args.disable_same_family_crossfit:
                    semantic_vec = torch.tensor(source_fits[source_label]["crossfit"][pair_name]["unit_vector"], device=device, dtype=torch.float32)
                    random_vec = make_random_orthogonal_control(
                        semantic_vec,
                        seed=args.seed + 100000 + 1000 * source_idx + 100 * eval_idx + prepared_idx,
                    )
                    fit_scope = "leave_one_pair_out"
                else:
                    semantic_vec = full_source_vec
                    random_vec = full_random_vec
                    fit_scope = "full"
                for target_layer in target_layers:
                    for site in sites:
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
                                    "source_dataset": source_label,
                                    "eval_dataset": eval_label,
                                    "same_family": int(source_label == eval_label),
                                    "fit_scope": fit_scope,
                                    "source_layer": int(args.source_layer),
                                    "target_layer": int(target_layer),
                                    "site": site,
                                    "control": control_name,
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
                                }
                            )

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby(["source_dataset", "eval_dataset", "same_family", "fit_scope", "target_layer", "site", "control"], dropna=False)
        .agg(
            n_items=("item_name", "count"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            delta_from_baseline_mean_anchor_cosine=("delta_from_baseline_anchor_cosine", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            mean_label_accuracy=("label_accuracy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
        )
        .reset_index()
    )
    stats_df = build_stats_rows(detail_df)
    vector_df = pd.DataFrame(vector_rows)
    vector_stats_df = (
        vector_df.groupby(["source_dataset", "fit_scope"], dropna=False)
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
    for path in (args.output_csv, detail_path, stats_path, vector_path, vector_stats_path):
        ensure_parent_dir(path)
    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    vector_df.to_csv(vector_path, index=False)
    vector_stats_df.to_csv(vector_stats_path, index=False)

    print(f"Saved summary to {args.output_csv}")
    print(f"Saved detail to {detail_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Saved vector fits to {vector_path}")
    print(f"Saved vector fit stats to {vector_stats_path}")


if __name__ == "__main__":
    main()