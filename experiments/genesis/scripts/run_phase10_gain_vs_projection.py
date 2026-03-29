import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import configure_matplotlib, ensure_parent_dir, infer_companion_csv, infer_companion_png, signflip_test
from scripts.phase10_site_hooks import TensorSiteCaptureHook, TensorSiteInterventionHook, parse_site_list
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
        return hook.captured.squeeze(0).detach().clone()
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


def build_stats_rows(detail_df):
    rows = []
    grouped = detail_df.groupby(["dataset_name", "target_layer", "site", "control"])
    for (dataset_name, target_layer, site, control), group in grouped:
        proj = group["baseline_projection"].to_numpy(dtype=np.float64)
        cos = group["baseline_cosine"].to_numpy(dtype=np.float64)
        delta = group["delta_from_baseline_signed_label_margin"].to_numpy(dtype=np.float64)
        gain = group["steering_gain_signed_label_margin"].to_numpy(dtype=np.float64)
        zero = signflip_test(delta, seed=1000 + int(target_layer))
        proj_corr = safe_corr(proj, delta)
        cos_corr = safe_corr(cos, delta)
        proj_gap, proj_high, proj_low = median_split_gap(proj, delta)
        cos_gap, cos_high, cos_low = median_split_gap(cos, delta)
        rows.append(
            {
                "dataset_name": dataset_name,
                "target_layer": int(target_layer),
                "site": site,
                "control": control,
                "n_items": int(len(group)),
                "effect_mean": zero["mean"],
                "effect_ci95_low": zero["ci95_low"],
                "effect_ci95_high": zero["ci95_high"],
                "effect_pvalue": zero["pvalue"],
                "gain_mean": float(np.mean(gain)),
                "projection_corr": proj_corr,
                "projection_corr_ci95_low": bootstrap_ci(proj, delta, safe_corr, seed=2000 + int(target_layer))[0],
                "projection_corr_ci95_high": bootstrap_ci(proj, delta, safe_corr, seed=2000 + int(target_layer))[1],
                "projection_corr_pvalue": permutation_corr_pvalue(proj, delta, seed=3000 + int(target_layer)),
                "projection_slope": fit_slope(proj, delta),
                "projection_slope_ci95_low": bootstrap_ci(proj, delta, fit_slope, seed=4000 + int(target_layer))[0],
                "projection_slope_ci95_high": bootstrap_ci(proj, delta, fit_slope, seed=4000 + int(target_layer))[1],
                "projection_high_minus_low_mean_delta": proj_gap,
                "projection_high_half_mean_delta": proj_high,
                "projection_low_half_mean_delta": proj_low,
                "cosine_corr": cos_corr,
                "cosine_corr_ci95_low": bootstrap_ci(cos, delta, safe_corr, seed=5000 + int(target_layer))[0],
                "cosine_corr_ci95_high": bootstrap_ci(cos, delta, safe_corr, seed=5000 + int(target_layer))[1],
                "cosine_corr_pvalue": permutation_corr_pvalue(cos, delta, seed=6000 + int(target_layer)),
                "cosine_slope": fit_slope(cos, delta),
                "cosine_slope_ci95_low": bootstrap_ci(cos, delta, fit_slope, seed=7000 + int(target_layer))[0],
                "cosine_slope_ci95_high": bootstrap_ci(cos, delta, fit_slope, seed=7000 + int(target_layer))[1],
                "cosine_high_minus_low_mean_delta": cos_gap,
                "cosine_high_half_mean_delta": cos_high,
                "cosine_low_half_mean_delta": cos_low,
            }
        )
    return pd.DataFrame(rows)


def save_plot(detail_df, output_path):
    plt = configure_matplotlib()
    datasets = list(detail_df["dataset_name"].drop_duplicates())
    layers = sorted(detail_df["target_layer"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), len(layers), figsize=(5.2 * len(layers), 3.8 * len(datasets)), squeeze=False)
    styles = {
        "semantic": {"color": "#4477AA", "marker": "o", "label": "semantic"},
        "random": {"color": "#CC6677", "marker": "x", "label": "random"},
    }
    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, layer in enumerate(layers):
            ax = axes[row_idx, col_idx]
            sub = detail_df[(detail_df["dataset_name"] == dataset_name) & (detail_df["target_layer"] == layer)].copy()
            if sub.empty:
                ax.axis("off")
                continue
            for control, style in styles.items():
                ctrl = sub[sub["control"] == control].copy()
                if ctrl.empty:
                    continue
                xs = ctrl["baseline_cosine"].to_numpy(dtype=np.float64)
                ys = ctrl["delta_from_baseline_signed_label_margin"].to_numpy(dtype=np.float64)
                ax.scatter(xs, ys, s=28, alpha=0.85, color=style["color"], marker=style["marker"], label=style["label"] if (row_idx, col_idx) == (0, 0) else None)
                slope = fit_slope(xs, ys)
                if np.isfinite(slope) and len(xs) >= 2 and not np.allclose(xs, xs[0]):
                    intercept = float(np.mean(ys) - slope * np.mean(xs))
                    xline = np.linspace(float(np.min(xs)), float(np.max(xs)), 50)
                    ax.plot(xline, slope * xline + intercept, color=style["color"], alpha=0.8, linewidth=1.4)
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
            ax.set_title(f"{dataset_name} | L{layer}")
            ax.set_xlabel("Baseline cosine to injected vector")
            if col_idx == 0:
                ax.set_ylabel("Δ signed label margin")
            ax.grid(True, alpha=0.2)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 10N: gain-vs-baseline-projection diagnostic at narrowed corridor sites")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument(
        "--eval-jsons",
        type=str,
        default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json",
    )
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/gain_vs_projection_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,11")
    parser.add_argument("--sites", type=str, default="attn_output")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, choices=["add", "ablate"], default="add")
    parser.add_argument("--position-fraction", type=float, default=1.0)
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
    sites = parse_site_list(args.sites)
    detail_rows = []

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
            for site in sites:
                state_cache = {
                    item["item"]["name"]: capture_site_state(model, item["prompt_ids"], target_layer, site, args.position_fraction)
                    for item in prepared
                }
                for item in prepared:
                    item_name = item["item"]["name"]
                    baseline_row = baseline_metrics[item_name]
                    state = state_cache[item_name]
                    state_norm = float(torch.norm(state).item())
                    for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                        projection = float(torch.dot(state, vector).item())
                        cosine = float(projection / max(state_norm, 1e-8))
                        steered = evaluate_with_hook(
                            model,
                            item,
                            anchor_layer=args.source_layer,
                            anchor_direction=anchor_direction,
                            layer=target_layer,
                            site=site,
                            vector=vector,
                            alpha=args.alpha,
                            mode=args.mode,
                            position_fraction=args.position_fraction,
                        )
                        steered.update(
                            {
                                "dataset_name": dataset_name,
                                "target_layer": target_layer,
                                "site": site,
                                "control": control_name,
                                "mode": args.mode,
                                "alpha": args.alpha,
                                "position_fraction": args.position_fraction,
                                "position_label": format_position_label(args.position_fraction),
                                "baseline_projection": projection,
                                "baseline_cosine": cosine,
                                "baseline_state_norm": state_norm,
                            }
                        )
                        for column in ["signed_label_margin", "label_target_pairwise_prob", "label_accuracy", "anchor_cosine", "math_minus_creative_logprob"]:
                            steered[f"delta_from_baseline_{column}"] = float(steered[column] - baseline_row[column])
                        steered["steering_gain_signed_label_margin"] = float(steered["delta_from_baseline_signed_label_margin"] / max(args.alpha, 1e-8))
                        detail_rows.append(steered)

    detail_df = pd.DataFrame(detail_rows)
    summary = (
        detail_df.groupby(["dataset_name", "target_layer", "site", "control", "mode", "alpha", "position_fraction", "position_label"], as_index=False)
        .agg(
            mean_baseline_projection=("baseline_projection", "mean"),
            mean_baseline_cosine=("baseline_cosine", "mean"),
            mean_baseline_state_norm=("baseline_state_norm", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            mean_steering_gain_signed_label_margin=("steering_gain_signed_label_margin", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["dataset_name", "target_layer", "site", "control"])
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
    save_plot(detail_df, plot_path)
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