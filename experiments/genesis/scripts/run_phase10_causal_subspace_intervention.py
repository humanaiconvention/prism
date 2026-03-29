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
from scripts.phase10_site_hooks import (
    TensorSiteInterventionHook,
    TensorSiteCaptureHook,
    extract_output_tensor,
    replace_output_tensor,
    resolve_site_module,
)
from scripts.phase9_semantic_utils import load_semantic_direction, parse_int_list
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, format_position_label, prepare_eval_item, resolve_position_index


def parse_str_list(raw):
    return [part.strip() for part in raw.split(",") if part.strip()]


def add_sign_aware_fields(prepared_item, metrics):
    metrics = dict(metrics)
    label_target_prob = metrics["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else metrics["pairwise_creative_prob"]
    metrics["label_target_pairwise_prob"] = float(label_target_prob)
    metrics["label_accuracy"] = float(metrics["label_correct"])
    return metrics


def capture_site_state(model, prompt_ids, layer, site, position_fraction):
    hook = TensorSiteCaptureHook(position_fraction=position_fraction)
    hook.attach(model, layer, site)
    try:
        with torch.inference_mode():
            model(prompt_ids)
        if hook.captured is None:
            raise RuntimeError(f"No site state captured for layer={layer}, site={site}")
        return hook.captured.squeeze(0).detach().cpu().to(dtype=torch.float32).numpy()
    finally:
        hook.remove()


def evaluate_with_hooks(
    model,
    prepared_item,
    anchor_layer,
    anchor_direction,
    target_layer,
    intervention_site,
    vector,
    alpha,
    mode,
    position_fraction,
    pre_hook=None,
):
    steering_hook = TensorSiteInterventionHook(vector=vector, alpha=alpha, mode=mode, position_fraction=position_fraction)
    if pre_hook is not None:
        pre_hook.attach(model, target_layer)
    steering_hook.attach(model, target_layer, intervention_site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer=anchor_layer, anchor_direction=anchor_direction))
    finally:
        steering_hook.remove()
        if pre_hook is not None:
            pre_hook.remove()


def fit_pca_subspace(states, requested_rank):
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Need at least two reference states to fit a PCA subspace")
    mean = np.mean(x, axis=0)
    centered = x - mean[None, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    max_rank = max(1, min(int(requested_rank), int(vh.shape[0]), int(x.shape[0] - 1)))
    basis = vh[:max_rank].T
    energy = singular_values ** 2
    explained = float(np.sum(energy[:max_rank]) / max(np.sum(energy), 1e-12))
    return {"mean": mean, "basis": basis, "effective_rank": max_rank, "explained_variance_ratio": explained}


def make_random_subspace(dim, rank, seed):
    rng = np.random.default_rng(int(seed))
    mat = rng.normal(size=(int(dim), int(rank)))
    q, _ = np.linalg.qr(mat)
    return q[:, : int(rank)]


def project_state(state, mean, basis):
    x = np.asarray(state, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    total_energy = float(np.dot(x, x))
    coeff = np.asarray(basis, dtype=np.float64).T @ x
    proj_energy = float(np.dot(coeff, coeff))
    return {
        "projection_fraction": float(proj_energy / max(total_energy, 1e-12)) if total_energy > 1e-12 else 0.0,
        "projection_norm": float(np.sqrt(max(proj_energy, 0.0))),
    }


class TensorSiteSubspaceInterventionHook:
    def __init__(self, site, mean, basis, alpha=1.0, mode="ablate", position_fraction=1.0):
        self.site = site
        self.mean = mean
        self.basis = basis
        self.alpha = float(alpha)
        self.mode = mode
        self.position_fraction = float(position_fraction)
        self.enabled = True
        self.handle = None

    def _apply_intervention(self, x):
        pos = resolve_position_index(x.shape[1], self.position_fraction)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        token_slice = x[:, pos, :]
        centered = token_slice - mean.unsqueeze(0)
        coeff = centered @ basis
        proj = coeff @ basis.transpose(0, 1)
        if self.mode == "ablate":
            updated = token_slice - (self.alpha * proj)
        elif self.mode == "keep_only":
            updated = mean.unsqueeze(0) + (self.alpha * proj)
        else:
            raise ValueError(f"Unsupported subspace intervention mode: {self.mode}")
        x_mod = x.clone()
        x_mod[:, pos, :] = updated
        return x_mod

    def _pre_hook(self):
        def hook_fn(module, args):
            if not self.enabled or self.alpha == 0.0:
                return None
            return (self._apply_intervention(args[0]), *args[1:])

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            if not self.enabled or self.alpha == 0.0:
                return output
            x_mod = self._apply_intervention(extract_output_tensor(output))
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer):
        module, kind = resolve_site_module(model, layer, self.site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
            return
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


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
    if {"semantic", "random"}.issubset(pivot.columns):
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

        base = pivot[pivot["subspace_type"] == "none"].copy()
        for (_, dataset_name, target_layer), group in base.groupby(["subspace_type", "dataset_name", "target_layer"]):
            base_gap = group[["item_name", "semantic_minus_random"]].rename(columns={"semantic_minus_random": "base_gap"})
            comparisons = pivot[
                (pivot["dataset_name"] == dataset_name)
                & (pivot["target_layer"] == target_layer)
                & (pivot["subspace_type"].isin(["semantic_pca", "random"]))
            ].copy()
            if comparisons.empty:
                continue
            merged = comparisons.merge(base_gap, on="item_name", how="inner")
            if merged.empty:
                continue
            merged["attenuation_vs_none"] = merged["semantic_minus_random"] - merged["base_gap"]
            for keys, pane in merged.groupby(["subspace_type", "subspace_rank", "effective_subspace_rank", "precondition_mode"]):
                subspace_type, subspace_rank, effective_rank, precondition_mode = keys
                attn = signflip_test(
                    pane["attenuation_vs_none"].to_numpy(dtype=np.float64),
                    seed=3000 + int(target_layer) + int(subspace_rank),
                )
                rows.append(
                    {
                        "comparison": "semantic_minus_random_attenuation_vs_none",
                        "dataset_name": dataset_name,
                        "target_layer": int(target_layer),
                        "input_site": pane["input_site"].iloc[0],
                        "intervention_site": pane["intervention_site"].iloc[0],
                        "control": "semantic_minus_random",
                        "subspace_type": subspace_type,
                        "subspace_rank": int(subspace_rank),
                        "effective_subspace_rank": int(effective_rank),
                        "precondition_mode": precondition_mode,
                        "n_items": int(len(pane)),
                        "mean": attn["mean"],
                        "ci95_low": attn["ci95_low"],
                        "ci95_high": attn["ci95_high"],
                        "pvalue": attn["pvalue"],
                        "mean_with_precondition": float(np.mean(pane["semantic_minus_random"])),
                        "mean_without_precondition": float(np.mean(pane["base_gap"])),
                    }
                )

        for (dataset_name, target_layer, subspace_rank), pane in pivot[
            pivot["subspace_type"].isin(["semantic_pca", "random"])
        ].groupby(["dataset_name", "target_layer", "subspace_rank"]):
            semantic = pane[pane["subspace_type"] == "semantic_pca"][
                ["item_name", "semantic_minus_random", "input_site", "intervention_site", "effective_subspace_rank", "precondition_mode"]
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
                    "comparison": "semantic_pca_minus_random_ablation_gap",
                    "dataset_name": dataset_name,
                    "target_layer": int(target_layer),
                    "input_site": merged["input_site"].iloc[0],
                    "intervention_site": merged["intervention_site"].iloc[0],
                    "control": "semantic_minus_random",
                    "subspace_type": "semantic_pca_vs_random",
                    "subspace_rank": int(subspace_rank),
                    "effective_subspace_rank": int(min(merged["semantic_effective_rank"].iloc[0], merged["random_effective_rank"].iloc[0])),
                    "precondition_mode": merged["precondition_mode"].iloc[0],
                    "n_items": int(len(merged)),
                    "mean": diff["mean"],
                    "ci95_low": diff["ci95_low"],
                    "ci95_high": diff["ci95_high"],
                    "pvalue": diff["pvalue"],
                    "mean_semantic_pca_gap": float(np.mean(merged["semantic_gap"])),
                    "mean_random_gap": float(np.mean(merged["random_gap"])),
                }
            )
    return pd.DataFrame(rows)


def save_plot(stats_df, output_path, plot_rank):
    plt = configure_matplotlib()
    sub = stats_df[(stats_df["comparison"] == "semantic_minus_random")].copy()
    if sub.empty:
        return
    datasets = list(sub["dataset_name"].drop_duplicates())
    layers = sorted(sub["target_layer"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(layers), figsize=(5.3 * len(layers), 3.8 * len(datasets)), squeeze=False)
    order = [("none", 0, "none"), ("semantic_pca", int(plot_rank), "semantic PCA ablate"), ("random", int(plot_rank), "random ablate")]
    colors = ["#4477AA", "#CC6677", "#228833"]
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
            for subspace_type, subspace_rank, label in order:
                if subspace_type == "none":
                    row = pane[(pane["subspace_type"] == "none") & (pane["subspace_rank"] == 0)]
                else:
                    row = pane[(pane["subspace_type"] == subspace_type) & (pane["subspace_rank"] == int(plot_rank))]
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
            ax.bar(xs, heights, color=colors, alpha=0.82)
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
    parser = argparse.ArgumentParser(description="Phase 10P: causal corridor-subspace intervention")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-jsons", type=str, default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json")
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/causal_subspace_intervention_summary.csv")
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
    parser.add_argument("--subspace-ranks", type=str, default="4,8,16")
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
                        "reference_n_items": 0,
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

    reference_by_layer_rank = {}
    for target_layer in target_layers:
        ref_states = [
            row["state"]
            for row in base_rows
            if row["dataset_name"] == args.reference_dataset
            and row["target_layer"] == target_layer
            and row["control"] == args.reference_control
            and row["delta_from_baseline_signed_label_margin"] > args.success_threshold
        ]
        if len(ref_states) < 2:
            raise RuntimeError(f"Need at least two reference success states for layer {target_layer}; found {len(ref_states)}")
        dim = int(np.asarray(ref_states[0]).shape[0])
        for rank in subspace_ranks:
            pca_ref = fit_pca_subspace(ref_states, requested_rank=rank)
            random_basis = make_random_subspace(dim, pca_ref["effective_rank"], seed=args.seed + (100 * target_layer) + rank)
            reference_by_layer_rank[(target_layer, rank, "semantic_pca")] = {
                **pca_ref,
                "reference_n_items": int(len(ref_states)),
                "requested_rank": int(rank),
            }
            reference_by_layer_rank[(target_layer, rank, "random")] = {
                "mean": pca_ref["mean"],
                "basis": random_basis,
                "effective_rank": int(pca_ref["effective_rank"]),
                "explained_variance_ratio": np.nan,
                "reference_n_items": int(len(ref_states)),
                "requested_rank": int(rank),
            }

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
                baseline_row = baseline_metrics[item_name]
                state = state_cache[target_layer][item_name]
                for rank in subspace_ranks:
                    for subspace_type in ("semantic_pca", "random"):
                        ref = reference_by_layer_rank[(target_layer, rank, subspace_type)]
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
                                "reference_n_items": int(ref["reference_n_items"]),
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
            reference_n_items=("reference_n_items", "max"),
            reference_explained_variance_ratio=("reference_explained_variance_ratio", "max"),
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