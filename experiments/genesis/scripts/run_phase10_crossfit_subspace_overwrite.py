import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import (
    configure_matplotlib,
    ensure_parent_dir,
    infer_companion_csv,
    infer_companion_png,
    paired_signflip_test,
    signflip_test,
)
from scripts.phase10_site_hooks import TensorSiteSubspaceOverwriteHook
from scripts.phase9_semantic_utils import load_semantic_direction, parse_int_list
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, format_position_label, prepare_eval_item
from scripts.run_phase10_causal_subspace_intervention import (
    TensorSiteSubspaceInterventionHook,
    add_sign_aware_fields,
    capture_site_state,
    evaluate_with_hooks,
    fit_pca_subspace,
    make_random_subspace,
    project_state,
)


def parse_str_list(raw):
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_pair_id(item_name):
    for suffix in ("__math", "__creative"):
        if item_name.endswith(suffix):
            return item_name[: -len(suffix)]
    return item_name


def normalize_label(raw_label):
    return str(raw_label or "unknown").strip().lower()


def choose_donor_row(rows, label):
    if not rows:
        return None
    label = normalize_label(label)
    same_label = [row for row in rows if normalize_label(row.get("label")) == label]
    candidates = same_label if same_label else list(rows)
    donor = max(
        candidates,
        key=lambda row: (float(row["delta_from_baseline_signed_label_margin"]), float(row["signed_label_margin"]), row["item_name"]),
    )
    return {
        "state": donor["state"],
        "donor_item_name": donor["item_name"],
        "donor_pair_id": donor["pair_id"],
        "donor_label": donor["label"],
        "donor_reference_gain_signed_label_margin": float(donor["delta_from_baseline_signed_label_margin"]),
        "donor_reference_signed_label_margin": float(donor["signed_label_margin"]),
        "donor_label_matched": int(normalize_label(donor["label"]) == label),
        "donor_selection_scheme": "same_label" if same_label else "fallback_any_label",
    }


def fit_reference_bank_and_donors(
    base_rows,
    target_layers,
    subspace_ranks,
    reference_dataset,
    reference_control,
    success_threshold,
    seed,
):
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
    labels = sorted({normalize_label(row["label"]) for row in ref_rows})
    crossfit_refs = {}
    pooled_refs = {}
    crossfit_donors = {}
    pooled_donors = {}

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

        for label in labels:
            donor = choose_donor_row(layer_rows, label)
            if donor is None:
                raise RuntimeError(f"No pooled donor found for layer {target_layer}, label={label}")
            pooled_donors[(target_layer, label)] = {
                **donor,
                "reference_scheme": "heldout_pooled_transfer",
                "reference_fold_id": "pooled_all",
                "reference_excluded_pair_id": "none",
                "reference_n_items": int(len(layer_rows)),
                "reference_train_pair_count": int(pooled_pair_count),
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
            for label in labels:
                donor = choose_donor_row(train_rows, label)
                if donor is None:
                    raise RuntimeError(f"No cross-fit donor found for layer {target_layer}, pair={pair_id}, label={label}")
                crossfit_donors[(pair_id, target_layer, label)] = {
                    **donor,
                    "reference_scheme": "leave_one_pair_out",
                    "reference_fold_id": pair_id,
                    "reference_excluded_pair_id": pair_id,
                    "reference_n_items": int(len(train_rows)),
                    "reference_train_pair_count": int(train_pair_count),
                }

    return crossfit_refs, pooled_refs, crossfit_donors, pooled_donors


def build_stats_rows(detail_df):
    rows = []
    effect_grouped = detail_df.groupby(
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
        ]
    )
    for keys, group in effect_grouped:
        dataset_name, target_layer, input_site, intervention_site, control, subspace_type, subspace_rank, effective_rank, precondition_mode = keys
        delta = group["delta_from_baseline_signed_label_margin"].to_numpy(dtype=np.float64)
        effect = signflip_test(delta, seed=1000 + int(target_layer) + int(subspace_rank))
        rows.append(
            {
                "comparison": "effect_vs_baseline",
                "dataset_name": dataset_name,
                "target_layer": int(target_layer),
                "input_site": input_site,
                "intervention_site": intervention_site,
                "control": control,
                "subspace_type": subspace_type,
                "subspace_rank": int(subspace_rank),
                "effective_subspace_rank": int(effective_rank),
                "precondition_mode": precondition_mode,
                "n_items": int(len(group)),
                "mean": effect["mean"],
                "ci95_low": effect["ci95_low"],
                "ci95_high": effect["ci95_high"],
                "pvalue": effect["pvalue"],
            }
        )

    pivot = (
        detail_df.pivot_table(
            index=[
                "dataset_name",
                "target_layer",
                "item_name",
                "input_site",
                "intervention_site",
                "subspace_type",
                "subspace_rank",
                "effective_subspace_rank",
                "precondition_mode",
            ],
            columns="control",
            values="delta_from_baseline_signed_label_margin",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    if not {"semantic", "random"}.issubset(pivot.columns):
        return pd.DataFrame(rows)

    pivot = pivot.dropna(subset=["semantic", "random"]).copy()
    pivot["semantic_minus_random"] = pivot["semantic"] - pivot["random"]

    grouped_gap = pivot.groupby(
        [
            "dataset_name",
            "target_layer",
            "input_site",
            "intervention_site",
            "subspace_type",
            "subspace_rank",
            "effective_subspace_rank",
            "precondition_mode",
        ]
    )
    for keys, group in grouped_gap:
        dataset_name, target_layer, input_site, intervention_site, subspace_type, subspace_rank, effective_rank, precondition_mode = keys
        gap = paired_signflip_test(
            group["semantic"].to_numpy(dtype=np.float64),
            group["random"].to_numpy(dtype=np.float64),
            seed=2000 + int(target_layer) + int(subspace_rank),
        )
        rows.append(
            {
                "comparison": "semantic_minus_random",
                "dataset_name": dataset_name,
                "target_layer": int(target_layer),
                "input_site": input_site,
                "intervention_site": intervention_site,
                "control": "semantic_minus_random",
                "subspace_type": subspace_type,
                "subspace_rank": int(subspace_rank),
                "effective_subspace_rank": int(effective_rank),
                "precondition_mode": precondition_mode,
                "n_items": int(len(group)),
                "mean": gap["mean"],
                "ci95_low": gap["ci95_low"],
                "ci95_high": gap["ci95_high"],
                "pvalue": gap["pvalue"],
                "mean_semantic": gap["mean_a"],
                "mean_random": gap["mean_b"],
            }
        )

    base = pivot[pivot["subspace_type"] == "none"][
        ["dataset_name", "target_layer", "item_name", "semantic_minus_random"]
    ].rename(columns={"semantic_minus_random": "base_gap"})
    comparisons = pivot[pivot["subspace_type"].isin(["semantic_pca", "random"])].merge(
        base,
        on=["dataset_name", "target_layer", "item_name"],
        how="inner",
    )
    if not comparisons.empty:
        comparisons["shift_vs_none"] = comparisons["semantic_minus_random"] - comparisons["base_gap"]
        for keys, pane in comparisons.groupby(
            [
                "dataset_name",
                "target_layer",
                "input_site",
                "intervention_site",
                "subspace_type",
                "subspace_rank",
                "effective_subspace_rank",
                "precondition_mode",
            ]
        ):
            dataset_name, target_layer, input_site, intervention_site, subspace_type, subspace_rank, effective_rank, precondition_mode = keys
            shift = signflip_test(
                pane["shift_vs_none"].to_numpy(dtype=np.float64),
                seed=3000 + int(target_layer) + int(subspace_rank),
            )
            rows.append(
                {
                    "comparison": "semantic_minus_random_shift_vs_none",
                    "dataset_name": dataset_name,
                    "target_layer": int(target_layer),
                    "input_site": input_site,
                    "intervention_site": intervention_site,
                    "control": "semantic_minus_random",
                    "subspace_type": subspace_type,
                    "subspace_rank": int(subspace_rank),
                    "effective_subspace_rank": int(effective_rank),
                    "precondition_mode": precondition_mode,
                    "n_items": int(len(pane)),
                    "mean": shift["mean"],
                    "ci95_low": shift["ci95_low"],
                    "ci95_high": shift["ci95_high"],
                    "pvalue": shift["pvalue"],
                    "mean_with_precondition": float(np.mean(pane["semantic_minus_random"])),
                    "mean_without_precondition": float(np.mean(pane["base_gap"])),
                }
            )

    matched = pivot[pivot["subspace_type"].isin(["semantic_pca", "random"])].copy()
    for (dataset_name, target_layer, subspace_rank, precondition_mode), pane in matched.groupby(
        ["dataset_name", "target_layer", "subspace_rank", "precondition_mode"]
    ):
        semantic = pane[pane["subspace_type"] == "semantic_pca"][
            ["item_name", "semantic_minus_random", "input_site", "intervention_site", "effective_subspace_rank"]
        ].rename(columns={"semantic_minus_random": "semantic_gap", "effective_subspace_rank": "semantic_effective_rank"})
        random = pane[pane["subspace_type"] == "random"][
            ["item_name", "semantic_minus_random", "effective_subspace_rank"]
        ].rename(columns={"semantic_minus_random": "random_gap", "effective_subspace_rank": "random_effective_rank"})
        merged = semantic.merge(random, on="item_name", how="inner")
        if merged.empty:
            continue
        diff = signflip_test(
            (merged["semantic_gap"] - merged["random_gap"]).to_numpy(dtype=np.float64),
            seed=4000 + int(target_layer) + int(subspace_rank),
        )
        rows.append(
            {
                "comparison": f"semantic_pca_minus_random_{precondition_mode}_gap",
                "dataset_name": dataset_name,
                "target_layer": int(target_layer),
                "input_site": merged["input_site"].iloc[0],
                "intervention_site": merged["intervention_site"].iloc[0],
                "control": "semantic_minus_random",
                "subspace_type": "semantic_pca_vs_random",
                "subspace_rank": int(subspace_rank),
                "effective_subspace_rank": int(
                    min(merged["semantic_effective_rank"].iloc[0], merged["random_effective_rank"].iloc[0])
                ),
                "precondition_mode": precondition_mode,
                "n_items": int(len(merged)),
                "mean": diff["mean"],
                "ci95_low": diff["ci95_low"],
                "ci95_high": diff["ci95_high"],
                "pvalue": diff["pvalue"],
                "mean_semantic_pca_gap": float(np.mean(merged["semantic_gap"])),
                "mean_random_gap": float(np.mean(merged["random_gap"])),
            }
        )

    rescue_source = pivot[pivot["subspace_type"].isin(["semantic_pca", "random"])].copy()
    rescue_rows = []
    for (dataset_name, target_layer, subspace_type, subspace_rank), pane in rescue_source.groupby(
        ["dataset_name", "target_layer", "subspace_type", "subspace_rank"]
    ):
        ablate = pane[pane["precondition_mode"] == "ablate"][
            ["item_name", "semantic_minus_random", "input_site", "intervention_site", "effective_subspace_rank"]
        ].rename(columns={"semantic_minus_random": "ablate_gap", "effective_subspace_rank": "ablate_effective_rank"})
        overwrite = pane[pane["precondition_mode"] == "overwrite"][
            ["item_name", "semantic_minus_random", "effective_subspace_rank"]
        ].rename(columns={"semantic_minus_random": "overwrite_gap", "effective_subspace_rank": "overwrite_effective_rank"})
        merged = ablate.merge(overwrite, on="item_name", how="inner")
        if merged.empty:
            continue
        merged["overwrite_rescue_vs_ablate"] = merged["overwrite_gap"] - merged["ablate_gap"]
        rescue = signflip_test(
            merged["overwrite_rescue_vs_ablate"].to_numpy(dtype=np.float64),
            seed=5000 + int(target_layer) + int(subspace_rank),
        )
        rows.append(
            {
                "comparison": "semantic_minus_random_overwrite_rescue_vs_ablate",
                "dataset_name": dataset_name,
                "target_layer": int(target_layer),
                "input_site": merged["input_site"].iloc[0],
                "intervention_site": merged["intervention_site"].iloc[0],
                "control": "semantic_minus_random",
                "subspace_type": subspace_type,
                "subspace_rank": int(subspace_rank),
                "effective_subspace_rank": int(
                    min(merged["ablate_effective_rank"].iloc[0], merged["overwrite_effective_rank"].iloc[0])
                ),
                "precondition_mode": "overwrite_vs_ablate",
                "n_items": int(len(merged)),
                "mean": rescue["mean"],
                "ci95_low": rescue["ci95_low"],
                "ci95_high": rescue["ci95_high"],
                "pvalue": rescue["pvalue"],
                "mean_overwrite_gap": float(np.mean(merged["overwrite_gap"])),
                "mean_ablate_gap": float(np.mean(merged["ablate_gap"])),
            }
        )
        rescue_rows.append(
            merged.assign(
                dataset_name=dataset_name,
                target_layer=int(target_layer),
                subspace_type=subspace_type,
                subspace_rank=int(subspace_rank),
            )
        )

    if rescue_rows:
        rescue_df = pd.concat(rescue_rows, ignore_index=True)
        for (dataset_name, target_layer, subspace_rank), pane in rescue_df.groupby(
            ["dataset_name", "target_layer", "subspace_rank"]
        ):
            semantic = pane[pane["subspace_type"] == "semantic_pca"][
                ["item_name", "overwrite_rescue_vs_ablate", "input_site", "intervention_site"]
            ].rename(columns={"overwrite_rescue_vs_ablate": "semantic_rescue"})
            random = pane[pane["subspace_type"] == "random"][
                ["item_name", "overwrite_rescue_vs_ablate"]
            ].rename(columns={"overwrite_rescue_vs_ablate": "random_rescue"})
            merged = semantic.merge(random, on="item_name", how="inner")
            if merged.empty:
                continue
            diff = signflip_test(
                (merged["semantic_rescue"] - merged["random_rescue"]).to_numpy(dtype=np.float64),
                seed=6000 + int(target_layer) + int(subspace_rank),
            )
            rows.append(
                {
                    "comparison": "semantic_pca_minus_random_overwrite_rescue_gap",
                    "dataset_name": dataset_name,
                    "target_layer": int(target_layer),
                    "input_site": merged["input_site"].iloc[0],
                    "intervention_site": merged["intervention_site"].iloc[0],
                    "control": "semantic_minus_random",
                    "subspace_type": "semantic_pca_vs_random",
                    "subspace_rank": int(subspace_rank),
                    "effective_subspace_rank": int(subspace_rank),
                    "precondition_mode": "overwrite_vs_ablate",
                    "n_items": int(len(merged)),
                    "mean": diff["mean"],
                    "ci95_low": diff["ci95_low"],
                    "ci95_high": diff["ci95_high"],
                    "pvalue": diff["pvalue"],
                    "mean_semantic_rescue": float(np.mean(merged["semantic_rescue"])),
                    "mean_random_rescue": float(np.mean(merged["random_rescue"])),
                }
            )

    return pd.DataFrame(rows)


def save_plot(stats_df, output_path, plot_rank):
    plt = configure_matplotlib()
    sub = stats_df[stats_df["comparison"] == "semantic_minus_random"].copy()
    if sub.empty:
        return
    datasets = list(sub["dataset_name"].drop_duplicates())
    layers = sorted(sub["target_layer"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(layers), figsize=(7.0 * len(layers), 3.8 * len(datasets)), squeeze=False)
    order = [
        ("none", 0, "none", "baseline"),
        ("semantic_pca", int(plot_rank), "ablate", "semantic ablate"),
        ("semantic_pca", int(plot_rank), "overwrite", "semantic overwrite"),
        ("random", int(plot_rank), "ablate", "random ablate"),
        ("random", int(plot_rank), "overwrite", "random overwrite"),
    ]
    colors = ["#4477AA", "#CC6677", "#EEAA33", "#228833", "#66CCAA"]
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, layer in enumerate(layers):
            ax = axes[row_idx, col_idx]
            pane = sub[(sub["dataset_name"] == dataset_name) & (sub["target_layer"] == layer)]
            if pane.empty:
                ax.axis("off")
                continue
            xs = np.arange(len(order))
            heights = []
            lowers = []
            uppers = []
            labels = []
            for subspace_type, subspace_rank, precondition_mode, label in order:
                row = pane[
                    (pane["subspace_type"] == subspace_type)
                    & (pane["subspace_rank"] == int(subspace_rank))
                    & (pane["precondition_mode"] == precondition_mode)
                ]
                if row.empty:
                    heights.append(np.nan)
                    lowers.append(0.0)
                    uppers.append(0.0)
                else:
                    rec = row.iloc[0]
                    heights.append(float(rec["mean"]))
                    lowers.append(float(rec["mean"] - rec["ci95_low"]))
                    uppers.append(float(rec["ci95_high"] - rec["mean"]))
                labels.append(label)
            ax.bar(xs, heights, color=colors, alpha=0.84)
            ax.errorbar(xs, heights, yerr=np.vstack([lowers, uppers]), fmt="none", ecolor="black", elinewidth=1.2, capsize=4)
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
            ax.set_xticks(xs, labels, rotation=18)
            ax.set_title(f"{dataset_name} | L{layer} | rank={plot_rank}")
            if col_idx == 0:
                ax.set_ylabel("Semantic − random Δ signed label margin")
            ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 10R: cross-fit corridor-input overwrite/rescue sufficiency test")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument(
        "--eval-jsons",
        type=str,
        default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json",
    )
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/crossfit_subspace_overwrite_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7")
    parser.add_argument("--input-site", type=str, default="block_input")
    parser.add_argument("--intervention-site", type=str, default="attn_output")
    parser.add_argument("--steering-alpha", type=float, default=12.5)
    parser.add_argument("--steering-mode", type=str, choices=["add", "ablate"], default="add")
    parser.add_argument("--subspace-alpha", type=float, default=1.0)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--subspace-ranks", type=str, default="8")
    parser.add_argument("--plot-rank", type=int, default=8)
    parser.add_argument("--reference-dataset", type=str, default="heldout_shared")
    parser.add_argument("--reference-control", type=str, default="semantic")
    parser.add_argument("--success-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    parser.add_argument("--donor-norm-match", action="store_true")
    args = parser.parse_args()

    eval_jsons = parse_str_list(args.eval_jsons)
    dataset_labels = parse_str_list(args.dataset_labels)
    if len(eval_jsons) != len(dataset_labels):
        raise ValueError("--eval-jsons and --dataset-labels must have the same length")
    if args.reference_dataset not in dataset_labels:
        raise ValueError(f"Reference dataset {args.reference_dataset!r} is not in --dataset-labels")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec = torch.tensor(
        load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key),
        device=device,
        dtype=torch.float32,
    )
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
            item["item"]["name"]: add_sign_aware_fields(
                item,
                evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction),
            )
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
                label = normalize_label(item["item"].get("label"))
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
                        "label": label,
                        "target_layer": target_layer,
                        "input_site": args.input_site,
                        "intervention_site": args.intervention_site,
                        "control": control_name,
                        "subspace_type": "none",
                        "subspace_rank": 0,
                        "effective_subspace_rank": 0,
                        "precondition_mode": "baseline",
                        "projection_fraction": np.nan,
                        "projection_norm": np.nan,
                        "donor_projection_fraction": np.nan,
                        "donor_projection_norm": np.nan,
                        "reference_scheme": "none",
                        "reference_fold_id": "none",
                        "reference_excluded_pair_id": "none",
                        "reference_n_items": 0,
                        "reference_train_pair_count": 0,
                        "reference_explained_variance_ratio": np.nan,
                        "donor_item_name": "none",
                        "donor_pair_id": "none",
                        "donor_label": "none",
                        "donor_reference_gain_signed_label_margin": np.nan,
                        "donor_reference_signed_label_margin": np.nan,
                        "donor_label_matched": np.nan,
                        "donor_selection_scheme": "none",
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
                    for column in [
                        "signed_label_margin",
                        "label_target_pairwise_prob",
                        "label_accuracy",
                        "anchor_cosine",
                        "math_minus_creative_logprob",
                    ]:
                        row[f"delta_from_baseline_{column}"] = float(steered[column] - baseline_row[column])
                    row["steering_gain_signed_label_margin"] = float(
                        row["delta_from_baseline_signed_label_margin"] / max(args.steering_alpha, 1e-8)
                    )
                    base_rows.append(row)
        dataset_payloads.append(
            {
                "dataset_name": dataset_name,
                "prepared": prepared,
                "baseline_metrics": baseline_metrics,
                "state_cache": state_cache,
            }
        )

    crossfit_refs, pooled_refs, crossfit_donors, pooled_donors = fit_reference_bank_and_donors(
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
                label = normalize_label(item["item"].get("label"))
                baseline_row = baseline_metrics[item_name]
                state = state_cache[target_layer][item_name]
                for rank in subspace_ranks:
                    for subspace_type in ("semantic_pca", "random"):
                        if dataset_name == args.reference_dataset:
                            ref = crossfit_refs[(pair_id, target_layer, rank, subspace_type)]
                            donor = crossfit_donors[(pair_id, target_layer, label)]
                        else:
                            ref = pooled_refs[(target_layer, rank, subspace_type)]
                            donor = pooled_donors[(target_layer, label)]
                        projection_metrics = project_state(state, ref["mean"], ref["basis"])
                        donor_projection_metrics = project_state(donor["state"], ref["mean"], ref["basis"])
                        mean_tensor = torch.tensor(ref["mean"], device=device, dtype=torch.float32)
                        basis_tensor = torch.tensor(ref["basis"], device=device, dtype=torch.float32)
                        donor_tensor = torch.tensor(donor["state"], device=device, dtype=torch.float32)
                        hooks = {
                            "ablate": TensorSiteSubspaceInterventionHook(
                                site=args.input_site,
                                mean=mean_tensor,
                                basis=basis_tensor,
                                alpha=args.subspace_alpha,
                                mode="ablate",
                                position_fraction=args.position_fraction,
                            ),
                            "overwrite": TensorSiteSubspaceOverwriteHook(
                                site=args.input_site,
                                mean=mean_tensor,
                                basis=basis_tensor,
                                donor_tensor=donor_tensor,
                                alpha=args.subspace_alpha,
                                position_fraction=args.position_fraction,
                                donor_norm_match=args.donor_norm_match,
                            ),
                        }
                        for precondition_mode, pre_hook in hooks.items():
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
                                    "label": label,
                                    "target_layer": target_layer,
                                    "input_site": args.input_site,
                                    "intervention_site": args.intervention_site,
                                    "control": control_name,
                                    "subspace_type": subspace_type,
                                    "subspace_rank": int(rank),
                                    "effective_subspace_rank": int(ref["effective_rank"]),
                                    "precondition_mode": precondition_mode,
                                    "projection_fraction": projection_metrics["projection_fraction"],
                                    "projection_norm": projection_metrics["projection_norm"],
                                    "donor_projection_fraction": donor_projection_metrics["projection_fraction"] if precondition_mode == "overwrite" else np.nan,
                                    "donor_projection_norm": donor_projection_metrics["projection_norm"] if precondition_mode == "overwrite" else np.nan,
                                    "reference_scheme": ref["reference_scheme"],
                                    "reference_fold_id": ref["reference_fold_id"],
                                    "reference_excluded_pair_id": ref["reference_excluded_pair_id"],
                                    "reference_n_items": int(ref["reference_n_items"]),
                                    "reference_train_pair_count": int(ref["reference_train_pair_count"]),
                                    "reference_explained_variance_ratio": ref["explained_variance_ratio"],
                                    "donor_item_name": donor["donor_item_name"] if precondition_mode == "overwrite" else "none",
                                    "donor_pair_id": donor["donor_pair_id"] if precondition_mode == "overwrite" else "none",
                                    "donor_label": donor["donor_label"] if precondition_mode == "overwrite" else "none",
                                    "donor_reference_gain_signed_label_margin": donor["donor_reference_gain_signed_label_margin"] if precondition_mode == "overwrite" else np.nan,
                                    "donor_reference_signed_label_margin": donor["donor_reference_signed_label_margin"] if precondition_mode == "overwrite" else np.nan,
                                    "donor_label_matched": donor["donor_label_matched"] if precondition_mode == "overwrite" else np.nan,
                                    "donor_selection_scheme": donor["donor_selection_scheme"] if precondition_mode == "overwrite" else "none",
                                    "steering_alpha": args.steering_alpha,
                                    "steering_mode": args.steering_mode,
                                    "subspace_alpha": args.subspace_alpha,
                                    "subspace_mode": precondition_mode,
                                    "position_fraction": args.position_fraction,
                                    "position_label": format_position_label(args.position_fraction),
                                    "reference_dataset": args.reference_dataset,
                                    "reference_control": args.reference_control,
                                }
                                row.update(steered)
                                for column in [
                                    "signed_label_margin",
                                    "label_target_pairwise_prob",
                                    "label_accuracy",
                                    "anchor_cosine",
                                    "math_minus_creative_logprob",
                                ]:
                                    row[f"delta_from_baseline_{column}"] = float(steered[column] - baseline_row[column])
                                row["steering_gain_signed_label_margin"] = float(
                                    row["delta_from_baseline_signed_label_margin"] / max(args.steering_alpha, 1e-8)
                                )
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
            mean_donor_projection_fraction=("donor_projection_fraction", "mean"),
            mean_donor_projection_norm=("donor_projection_norm", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            mean_steering_gain_signed_label_margin=("steering_gain_signed_label_margin", "mean"),
            donor_reference_gain_signed_label_margin_mean=("donor_reference_gain_signed_label_margin", "mean"),
            donor_label_match_rate=("donor_label_matched", "mean"),
            reference_n_items_mean=("reference_n_items", "mean"),
            reference_n_items_min=("reference_n_items", "min"),
            reference_n_items_max=("reference_n_items", "max"),
            reference_train_pair_count_mean=("reference_train_pair_count", "mean"),
            reference_train_pair_count_min=("reference_train_pair_count", "min"),
            reference_train_pair_count_max=("reference_train_pair_count", "max"),
            reference_explained_variance_ratio_mean=("reference_explained_variance_ratio", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["dataset_name", "target_layer", "subspace_rank", "subspace_type", "precondition_mode", "control"])
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