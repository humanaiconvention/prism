import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import configure_matplotlib, ensure_parent_dir, infer_companion_csv, infer_companion_png, signflip_test
from scripts.phase10_site_hooks import TensorSiteCaptureHook, TensorSiteInterventionHook
from scripts.phase9_semantic_utils import load_semantic_direction, parse_int_list
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, format_position_label, prepare_eval_item


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


def evaluate_with_hook(model, prepared_item, anchor_layer, anchor_direction, layer, site, vector, alpha, mode, position_fraction):
    hook = TensorSiteInterventionHook(vector=vector, alpha=alpha, mode=mode, position_fraction=position_fraction)
    hook.attach(model, layer, site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer=anchor_layer, anchor_direction=anchor_direction))
    finally:
        hook.remove()


def safe_corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2 or np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def fit_slope(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2 or np.allclose(xs, xs[0]):
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


def bootstrap_ci(xs, ys, stat_fn, n_boot=4000, seed=0):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    stats = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, xs.size, size=xs.size)
        stats.append(float(stat_fn(xs[idx], ys[idx])))
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


def permutation_corr_pvalue(xs, ys, n_perm=20000, seed=0):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    observed = safe_corr(xs, ys)
    if xs.size < 2 or np.isnan(observed):
        return np.nan
    rng = np.random.default_rng(int(seed))
    extreme = 0
    for _ in range(int(n_perm)):
        shuffled = ys[rng.permutation(ys.size)]
        candidate = safe_corr(xs, shuffled)
        if np.isfinite(candidate) and abs(candidate) >= abs(observed):
            extreme += 1
    return float((1.0 + extreme) / (int(n_perm) + 1.0))


def median_split_gap(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 4:
        return np.nan, np.nan, np.nan
    cutoff = float(np.median(xs))
    high = ys[xs >= cutoff]
    low = ys[xs < cutoff]
    if high.size == 0 or low.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(high) - np.mean(low)), float(np.mean(high)), float(np.mean(low))


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
    orth_energy = max(total_energy - proj_energy, 0.0)
    return {
        "distance_to_reference_mean": float(np.sqrt(max(total_energy, 0.0))),
        "projection_norm": float(np.sqrt(max(proj_energy, 0.0))),
        "orthogonal_norm": float(np.sqrt(orth_energy)),
        "projection_fraction": float(proj_energy / max(total_energy, 1e-12)) if total_energy > 1e-12 else 0.0,
    }


def build_stats_rows(detail_df):
    rows = []
    grouped = detail_df.groupby(["dataset_name", "target_layer", "input_site", "intervention_site", "control", "subspace_type", "subspace_rank", "effective_subspace_rank"])
    for keys, group in grouped:
        dataset_name, target_layer, input_site, intervention_site, control, subspace_type, subspace_rank, effective_rank = keys
        proj = group["projection_fraction"].to_numpy(dtype=np.float64)
        delta = group["delta_from_baseline_signed_label_margin"].to_numpy(dtype=np.float64)
        gain = group["steering_gain_signed_label_margin"].to_numpy(dtype=np.float64)
        zero = signflip_test(delta, seed=1000 + int(target_layer) + int(subspace_rank))
        corr = safe_corr(proj, delta)
        gap, high, low = median_split_gap(proj, delta)
        corr_low, corr_high = bootstrap_ci(proj, delta, safe_corr, seed=2000 + int(target_layer) + int(subspace_rank))
        slope_low, slope_high = bootstrap_ci(proj, delta, fit_slope, seed=3000 + int(target_layer) + int(subspace_rank))
        rows.append(
            {
                "dataset_name": dataset_name,
                "target_layer": int(target_layer),
                "input_site": input_site,
                "intervention_site": intervention_site,
                "control": control,
                "subspace_type": subspace_type,
                "subspace_rank": int(subspace_rank),
                "effective_subspace_rank": int(effective_rank),
                "n_items": int(len(group)),
                "effect_mean": zero["mean"],
                "effect_ci95_low": zero["ci95_low"],
                "effect_ci95_high": zero["ci95_high"],
                "effect_pvalue": zero["pvalue"],
                "gain_mean": float(np.mean(gain)),
                "mean_projection_fraction": float(np.mean(proj)),
                "projection_fraction_corr": corr,
                "projection_fraction_corr_ci95_low": corr_low,
                "projection_fraction_corr_ci95_high": corr_high,
                "projection_fraction_corr_pvalue": permutation_corr_pvalue(proj, delta, seed=4000 + int(target_layer) + int(subspace_rank)),
                "projection_fraction_slope": fit_slope(proj, delta),
                "projection_fraction_slope_ci95_low": slope_low,
                "projection_fraction_slope_ci95_high": slope_high,
                "projection_high_minus_low_mean_delta": gap,
                "projection_high_half_mean_delta": high,
                "projection_low_half_mean_delta": low,
            }
        )
    return pd.DataFrame(rows)


def save_plot(detail_df, output_path, plot_rank):
    plt = configure_matplotlib()
    sub = detail_df[(detail_df["control"] == "semantic") & (detail_df["subspace_rank"] == int(plot_rank))].copy()
    if sub.empty:
        return
    datasets = list(sub["dataset_name"].drop_duplicates())
    layers = sorted(sub["target_layer"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(layers), figsize=(5.2 * len(layers), 3.8 * len(datasets)), squeeze=False)
    styles = {
        "semantic_pca": {"color": "#4477AA", "marker": "o", "label": "heldout-success PCA"},
        "random": {"color": "#CC6677", "marker": "x", "label": "matched random"},
    }
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, layer in enumerate(layers):
            ax = axes[row_idx, col_idx]
            pane = sub[(sub["dataset_name"] == dataset_name) & (sub["target_layer"] == layer)]
            if pane.empty:
                ax.axis("off")
                continue
            for subspace_type, style in styles.items():
                ctrl = pane[pane["subspace_type"] == subspace_type]
                if ctrl.empty:
                    continue
                xs = ctrl["projection_fraction"].to_numpy(dtype=np.float64)
                ys = ctrl["delta_from_baseline_signed_label_margin"].to_numpy(dtype=np.float64)
                ax.scatter(xs, ys, s=28, alpha=0.85, color=style["color"], marker=style["marker"], label=style["label"] if (row_idx, col_idx) == (0, 0) else None)
                slope = fit_slope(xs, ys)
                if np.isfinite(slope) and len(xs) >= 2 and not np.allclose(xs, xs[0]):
                    intercept = float(np.mean(ys) - slope * np.mean(xs))
                    xline = np.linspace(float(np.min(xs)), float(np.max(xs)), 50)
                    ax.plot(xline, slope * xline + intercept, color=style["color"], alpha=0.8, linewidth=1.4)
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
            ax.set_title(f"{dataset_name} | L{layer} | rank={plot_rank}")
            ax.set_xlabel("Projection fraction onto reference input subspace")
            if col_idx == 0:
                ax.set_ylabel("Δ signed label margin")
            ax.grid(True, alpha=0.2)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 10O: corridor-input residual subspace diagnostic")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-jsons", type=str, default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json")
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/corridor_input_subspace_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,11")
    parser.add_argument("--input-site", type=str, default="block_input")
    parser.add_argument("--intervention-site", type=str, default="attn_output")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, choices=["add", "ablate"], default="add")
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--subspace-ranks", type=str, default="4,8,16")
    parser.add_argument("--plot-rank", type=int, default=8)
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
    trial_rows = []

    for dataset_name, eval_json in zip(dataset_labels, eval_jsons):
        items = load_eval_items(eval_json)
        if args.max_eval_items is not None:
            items = items[: args.max_eval_items]
        prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
        baseline_metrics = {
            item["item"]["name"]: add_sign_aware_fields(item, evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction))
            for item in prepared
        }
        for target_layer in target_layers:
            state_cache = {
                item["item"]["name"]: capture_site_state(model, item["prompt_ids"], target_layer, args.input_site, args.position_fraction)
                for item in prepared
            }
            for item in prepared:
                item_name = item["item"]["name"]
                baseline_row = baseline_metrics[item_name]
                state = state_cache[item_name]
                for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                    steered = evaluate_with_hook(
                        model,
                        item,
                        anchor_layer=args.source_layer,
                        anchor_direction=anchor_direction,
                        layer=target_layer,
                        site=args.intervention_site,
                        vector=vector,
                        alpha=args.alpha,
                        mode=args.mode,
                        position_fraction=args.position_fraction,
                    )
                    row = {
                        "dataset_name": dataset_name,
                        "item_name": item_name,
                        "target_layer": target_layer,
                        "input_site": args.input_site,
                        "intervention_site": args.intervention_site,
                        "control": control_name,
                        "mode": args.mode,
                        "alpha": args.alpha,
                        "position_fraction": args.position_fraction,
                        "position_label": format_position_label(args.position_fraction),
                        "reference_dataset": args.reference_dataset,
                        "reference_control": args.reference_control,
                        "state": state,
                    }
                    row.update(steered)
                    for column in ["signed_label_margin", "label_target_pairwise_prob", "label_accuracy", "anchor_cosine", "math_minus_creative_logprob"]:
                        row[f"delta_from_baseline_{column}"] = float(steered[column] - baseline_row[column])
                    row["steering_gain_signed_label_margin"] = float(row["delta_from_baseline_signed_label_margin"] / max(args.alpha, 1e-8))
                    trial_rows.append(row)

    reference_by_layer_rank = {}
    for target_layer in target_layers:
        ref_states = [
            row["state"]
            for row in trial_rows
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
    for row in trial_rows:
        state = row["state"]
        base_row = {k: v for k, v in row.items() if k != "state"}
        for rank in subspace_ranks:
            for subspace_type in ("semantic_pca", "random"):
                ref = reference_by_layer_rank[(row["target_layer"], rank, subspace_type)]
                metrics = project_state(state, ref["mean"], ref["basis"])
                detail_rows.append(
                    {
                        **base_row,
                        **metrics,
                        "subspace_type": subspace_type,
                        "subspace_rank": int(rank),
                        "effective_subspace_rank": int(ref["effective_rank"]),
                        "reference_n_items": int(ref["reference_n_items"]),
                        "reference_explained_variance_ratio": ref["explained_variance_ratio"],
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    summary = (
        detail_df.groupby(
            ["dataset_name", "target_layer", "input_site", "intervention_site", "control", "subspace_type", "subspace_rank", "effective_subspace_rank", "mode", "alpha", "position_fraction", "position_label"],
            as_index=False,
        )
        .agg(
            mean_projection_fraction=("projection_fraction", "mean"),
            mean_projection_norm=("projection_norm", "mean"),
            mean_distance_to_reference_mean=("distance_to_reference_mean", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            mean_steering_gain_signed_label_margin=("steering_gain_signed_label_margin", "mean"),
            reference_n_items=("reference_n_items", "max"),
            reference_explained_variance_ratio=("reference_explained_variance_ratio", "max"),
            n_items=("item_name", "count"),
        )
        .sort_values(["dataset_name", "target_layer", "control", "subspace_rank", "subspace_type"])
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
    save_plot(detail_df, plot_path, plot_rank=args.plot_rank)
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