import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, infer_companion_png
from scripts.phase9_semantic_utils import load_semantic_direction, parse_int_list
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, format_position_label, prepare_eval_item
from scripts.run_phase10_causal_subspace_intervention import (
    TensorSiteSubspaceInterventionHook,
    add_sign_aware_fields,
    build_stats_rows,
    capture_site_state,
    evaluate_with_hooks,
    fit_pca_subspace,
    make_random_subspace,
    project_state,
    save_plot,
)


def parse_str_list(raw):
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_pair_id(item_name):
    for suffix in ("__math", "__creative"):
        if item_name.endswith(suffix):
            return item_name[: -len(suffix)]
    return item_name


def fit_reference_bank(base_rows, target_layers, subspace_ranks, reference_dataset, reference_control, success_threshold, seed):
    ref_rows = [
        row
        for row in base_rows
        if row["dataset_name"] == reference_dataset
        and row["control"] == reference_control
        and row["delta_from_baseline_signed_label_margin"] > success_threshold
    ]
    if len(ref_rows) < 2:
        raise RuntimeError(f"Need at least two reference success rows in {reference_dataset}; found {len(ref_rows)}")

    heldout_pairs = sorted({row["pair_id"] for row in ref_rows})
    crossfit_refs = {}
    pooled_refs = {}

    for target_layer in target_layers:
        layer_rows = [row for row in ref_rows if row["target_layer"] == target_layer]
        if len(layer_rows) < 2:
            raise RuntimeError(f"Need at least two reference success rows for layer {target_layer}; found {len(layer_rows)}")

        pooled_states = [row["state"] for row in layer_rows]
        pooled_pair_count = len({row["pair_id"] for row in layer_rows})
        dim = int(np.asarray(pooled_states[0]).shape[0])
        for rank in subspace_ranks:
            pca_ref = fit_pca_subspace(pooled_states, requested_rank=rank)
            random_basis = make_random_subspace(dim, pca_ref["effective_rank"], seed=seed + (1000 * target_layer) + rank)
            pooled_common = {
                "reference_scheme": "heldout_pooled_transfer",
                "reference_fold_id": "pooled_all",
                "reference_excluded_pair_id": "none",
                "reference_n_items": int(len(pooled_states)),
                "reference_train_pair_count": int(pooled_pair_count),
                "requested_rank": int(rank),
            }
            pooled_refs[(target_layer, rank, "semantic_pca")] = {**pca_ref, **pooled_common}
            pooled_refs[(target_layer, rank, "random")] = {
                "mean": pca_ref["mean"],
                "basis": random_basis,
                "effective_rank": int(pca_ref["effective_rank"]),
                "explained_variance_ratio": np.nan,
                **pooled_common,
            }

        for pair_idx, pair_id in enumerate(heldout_pairs):
            train_rows = [row for row in layer_rows if row["pair_id"] != pair_id]
            if len(train_rows) < 2:
                raise RuntimeError(
                    f"Need at least two training success rows for layer {target_layer} excluding {pair_id}; found {len(train_rows)}"
                )
            train_states = [row["state"] for row in train_rows]
            train_pair_count = len({row["pair_id"] for row in train_rows})
            dim = int(np.asarray(train_states[0]).shape[0])
            for rank in subspace_ranks:
                pca_ref = fit_pca_subspace(train_states, requested_rank=rank)
                random_basis = make_random_subspace(
                    dim,
                    pca_ref["effective_rank"],
                    seed=seed + (100000 * target_layer) + (1000 * rank) + pair_idx,
                )
                common = {
                    "reference_scheme": "leave_one_pair_out",
                    "reference_fold_id": pair_id,
                    "reference_excluded_pair_id": pair_id,
                    "reference_n_items": int(len(train_states)),
                    "reference_train_pair_count": int(train_pair_count),
                    "requested_rank": int(rank),
                }
                crossfit_refs[(pair_id, target_layer, rank, "semantic_pca")] = {**pca_ref, **common}
                crossfit_refs[(pair_id, target_layer, rank, "random")] = {
                    "mean": pca_ref["mean"],
                    "basis": random_basis,
                    "effective_rank": int(pca_ref["effective_rank"]),
                    "explained_variance_ratio": np.nan,
                    **common,
                }

    return crossfit_refs, pooled_refs


def main():
    parser = argparse.ArgumentParser(description="Phase 10Q: cross-fitted corridor-input necessity test")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-jsons", type=str, default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json")
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/crossfit_subspace_necessity_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,11")
    parser.add_argument("--input-site", type=str, default="block_input")
    parser.add_argument("--intervention-site", type=str, default="attn_output")
    parser.add_argument("--steering-alpha", type=float, default=12.5)
    parser.add_argument("--steering-mode", type=str, choices=["add", "ablate"], default="add")
    parser.add_argument("--subspace-alpha", type=float, default=1.0)
    parser.add_argument("--subspace-mode", type=str, choices=["ablate", "keep_only"], default="ablate")
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--subspace-ranks", type=str, default="8,16")
    parser.add_argument("--plot-rank", type=int, default=16)
    parser.add_argument("--reference-dataset", type=str, default="heldout_shared")
    parser.add_argument("--reference-control", type=str, default="semantic")
    parser.add_argument("--success-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    eval_jsons = parse_str_list(args.eval_jsons)
    dataset_labels = parse_str_list(args.dataset_labels)
    if len(eval_jsons) != len(dataset_labels):
        raise ValueError("--eval-jsons and --dataset-labels must have the same length")
    if args.reference_dataset not in dataset_labels:
        raise ValueError(f"Reference dataset {args.reference_dataset!r} is not in --dataset-labels")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec = torch.tensor(load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key), device=device, dtype=torch.float32)
    random_vec = make_random_orthogonal_control(semantic_vec, seed=args.seed)
    anchor_direction = load_anchor_direction(args.data_dir, args.source_layer, allow_invalid_metadata=args.allow_invalid_metadata)
    target_layers = parse_int_list(args.target_layers)
    subspace_ranks = parse_int_list(args.subspace_ranks)

    dataset_payloads = []
    base_rows = []
    for dataset_name, eval_json in zip(dataset_labels, eval_jsons):
        items = load_eval_items(eval_json)
        if args.max_eval_items is not None:
            items = items[: args.max_eval_items]
        prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
        baseline_metrics = {
            item["item"]["name"]: add_sign_aware_fields(item, evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction))
            for item in prepared
        }
        state_cache = {}
        for target_layer in target_layers:
            state_cache[target_layer] = {
                item["item"]["name"]: capture_site_state(model, item["prompt_ids"], target_layer, args.input_site, args.position_fraction)
                for item in prepared
            }
            for item in prepared:
                item_name = item["item"]["name"]
                pair_id = parse_pair_id(item_name)
                baseline_row = baseline_metrics[item_name]
                state = state_cache[target_layer][item_name]
                for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                    steered = evaluate_with_hooks(
                        model,
                        item,
                        anchor_layer=args.source_layer,
                        anchor_direction=anchor_direction,
                        target_layer=target_layer,
                        intervention_site=args.intervention_site,
                        vector=vector,
                        alpha=args.steering_alpha,
                        mode=args.steering_mode,
                        position_fraction=args.position_fraction,
                    )
                    row = {
                        "dataset_name": dataset_name,
                        "item_name": item_name,
                        "pair_id": pair_id,
                        "target_layer": target_layer,
                        "input_site": args.input_site,
                        "intervention_site": args.intervention_site,
                        "control": control_name,
                        "subspace_type": "none",
                        "subspace_rank": 0,
                        "effective_subspace_rank": 0,
                        "precondition_mode": "none",
                        "projection_fraction": np.nan,
                        "projection_norm": np.nan,
                        "reference_scheme": "none",
                        "reference_fold_id": "none",
                        "reference_excluded_pair_id": "none",
                        "reference_n_items": 0,
                        "reference_train_pair_count": 0,
                        "reference_explained_variance_ratio": np.nan,
                        "steering_alpha": args.steering_alpha,
                        "steering_mode": args.steering_mode,
                        "subspace_alpha": 0.0,
                        "subspace_mode": "none",
                        "position_fraction": args.position_fraction,
                        "position_label": format_position_label(args.position_fraction),
                        "reference_dataset": args.reference_dataset,
                        "reference_control": args.reference_control,
                        "state": state,
                    }
                    row.update(steered)
                    for column in ["signed_label_margin", "label_target_pairwise_prob", "label_accuracy", "anchor_cosine", "math_minus_creative_logprob"]:
                        row[f"delta_from_baseline_{column}"] = float(steered[column] - baseline_row[column])
                    row["steering_gain_signed_label_margin"] = float(row["delta_from_baseline_signed_label_margin"] / max(args.steering_alpha, 1e-8))
                    base_rows.append(row)
        dataset_payloads.append({"dataset_name": dataset_name, "prepared": prepared, "baseline_metrics": baseline_metrics, "state_cache": state_cache})

    crossfit_refs, pooled_refs = fit_reference_bank(
        base_rows,
        target_layers=target_layers,
        subspace_ranks=subspace_ranks,
        reference_dataset=args.reference_dataset,
        reference_control=args.reference_control,
        success_threshold=args.success_threshold,
        seed=args.seed,
    )

    detail_rows = []
    for row in base_rows:
        base_row = dict(row)
        base_row.pop("state", None)
        detail_rows.append(base_row)

    for payload in dataset_payloads:
        dataset_name = payload["dataset_name"]
        prepared = payload["prepared"]
        baseline_metrics = payload["baseline_metrics"]
        state_cache = payload["state_cache"]
        for target_layer in target_layers:
            for item in prepared:
                item_name = item["item"]["name"]
                pair_id = parse_pair_id(item_name)
                baseline_row = baseline_metrics[item_name]
                state = state_cache[target_layer][item_name]
                for rank in subspace_ranks:
                    for subspace_type in ("semantic_pca", "random"):
                        if dataset_name == args.reference_dataset:
                            ref = crossfit_refs[(pair_id, target_layer, rank, subspace_type)]
                        else:
                            ref = pooled_refs[(target_layer, rank, subspace_type)]
                        projection_metrics = project_state(state, ref["mean"], ref["basis"])
                        pre_hook = TensorSiteSubspaceInterventionHook(
                            site=args.input_site,
                            mean=torch.tensor(ref["mean"], device=device, dtype=torch.float32),
                            basis=torch.tensor(ref["basis"], device=device, dtype=torch.float32),
                            alpha=args.subspace_alpha,
                            mode=args.subspace_mode,
                            position_fraction=args.position_fraction,
                        )
                        for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                            steered = evaluate_with_hooks(
                                model,
                                item,
                                anchor_layer=args.source_layer,
                                anchor_direction=anchor_direction,
                                target_layer=target_layer,
                                intervention_site=args.intervention_site,
                                vector=vector,
                                alpha=args.steering_alpha,
                                mode=args.steering_mode,
                                position_fraction=args.position_fraction,
                                pre_hook=pre_hook,
                            )
                            row = {
                                "dataset_name": dataset_name,
                                "item_name": item_name,
                                "pair_id": pair_id,
                                "target_layer": target_layer,
                                "input_site": args.input_site,
                                "intervention_site": args.intervention_site,
                                "control": control_name,
                                "subspace_type": subspace_type,
                                "subspace_rank": int(rank),
                                "effective_subspace_rank": int(ref["effective_rank"]),
                                "precondition_mode": args.subspace_mode,
                                "projection_fraction": projection_metrics["projection_fraction"],
                                "projection_norm": projection_metrics["projection_norm"],
                                "reference_scheme": ref["reference_scheme"],
                                "reference_fold_id": ref["reference_fold_id"],
                                "reference_excluded_pair_id": ref["reference_excluded_pair_id"],
                                "reference_n_items": int(ref["reference_n_items"]),
                                "reference_train_pair_count": int(ref["reference_train_pair_count"]),
                                "reference_explained_variance_ratio": ref["explained_variance_ratio"],
                                "steering_alpha": args.steering_alpha,
                                "steering_mode": args.steering_mode,
                                "subspace_alpha": args.subspace_alpha,
                                "subspace_mode": args.subspace_mode,
                                "position_fraction": args.position_fraction,
                                "position_label": format_position_label(args.position_fraction),
                                "reference_dataset": args.reference_dataset,
                                "reference_control": args.reference_control,
                            }
                            row.update(steered)
                            for column in ["signed_label_margin", "label_target_pairwise_prob", "label_accuracy", "anchor_cosine", "math_minus_creative_logprob"]:
                                row[f"delta_from_baseline_{column}"] = float(steered[column] - baseline_row[column])
                            row["steering_gain_signed_label_margin"] = float(row["delta_from_baseline_signed_label_margin"] / max(args.steering_alpha, 1e-8))
                            detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    summary = (
        detail_df.groupby(
            [
                "dataset_name",
                "target_layer",
                "input_site",
                "intervention_site",
                "control",
                "subspace_type",
                "subspace_rank",
                "effective_subspace_rank",
                "precondition_mode",
                "reference_scheme",
                "steering_mode",
                "steering_alpha",
                "subspace_mode",
                "subspace_alpha",
                "position_fraction",
                "position_label",
            ],
            as_index=False,
        )
        .agg(
            mean_projection_fraction=("projection_fraction", "mean"),
            mean_projection_norm=("projection_norm", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            mean_steering_gain_signed_label_margin=("steering_gain_signed_label_margin", "mean"),
            reference_n_items_mean=("reference_n_items", "mean"),
            reference_n_items_min=("reference_n_items", "min"),
            reference_n_items_max=("reference_n_items", "max"),
            reference_train_pair_count_mean=("reference_train_pair_count", "mean"),
            reference_train_pair_count_min=("reference_train_pair_count", "min"),
            reference_train_pair_count_max=("reference_train_pair_count", "max"),
            reference_explained_variance_ratio_mean=("reference_explained_variance_ratio", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["dataset_name", "target_layer", "subspace_rank", "subspace_type", "control"])
    )
    stats_df = build_stats_rows(detail_df)

    output_path = args.output_csv
    detail_path = args.detail_csv or infer_companion_csv(output_path, "detail")
    stats_path = args.stats_csv or infer_companion_csv(output_path, "stats")
    plot_path = args.plot_path or infer_companion_png(output_path, "summary")
    ensure_parent_dir(output_path)
    summary.to_csv(output_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    save_plot(stats_df, plot_path, plot_rank=args.plot_rank)
    print(summary.to_string(index=False))
    if not stats_df.empty:
        print("\n[stats]")
        print(stats_df.to_string(index=False))
    print(f"[saved] summary -> {output_path}")
    print(f"[saved] detail -> {detail_path}")
    print(f"[saved] stats -> {stats_path}")
    print(f"[saved] plot -> {plot_path}")


if __name__ == "__main__":
    main()