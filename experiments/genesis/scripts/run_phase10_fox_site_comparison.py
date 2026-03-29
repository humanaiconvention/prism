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
from scripts.phase10_site_hooks import TensorSiteInterventionHook, parse_site_list
from scripts.phase9_semantic_utils import load_semantic_direction, parse_int_list
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, format_position_label, prepare_eval_item


def add_sign_aware_fields(prepared_item, metrics):
    metrics = dict(metrics)
    label_target_prob = metrics["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else metrics["pairwise_creative_prob"]
    metrics["label_target_pairwise_prob"] = float(label_target_prob)
    metrics["label_accuracy"] = float(metrics["label_correct"])
    return metrics


def build_stats_rows(detail_df):
    rows = []
    for (target_layer, site), group in detail_df.groupby(["target_layer", "site"]):
        pivot = group.pivot(index="item_name", columns="control", values="delta_from_baseline_signed_label_margin").dropna()
        if pivot.empty or "semantic" not in pivot.columns or "random" not in pivot.columns:
            continue
        semantic = pivot["semantic"].to_numpy(dtype=np.float64)
        random = pivot["random"].to_numpy(dtype=np.float64)
        semantic_zero = signflip_test(semantic, seed=101 + int(target_layer))
        random_zero = signflip_test(random, seed=201 + int(target_layer))
        semantic_vs_random = paired_signflip_test(semantic, random, seed=301 + int(target_layer))
        rows.append(
            {
                "comparison_type": "semantic_vs_random",
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
    for site, group in semantic_only.groupby("site"):
        pivot = group.pivot(index="item_name", columns="target_layer", values="delta_from_baseline_signed_label_margin").dropna()
        for a, b in ((11, 7), (11, 15)):
            if pivot.empty or a not in pivot.columns or b not in pivot.columns:
                continue
            contrast = paired_signflip_test(pivot[a].to_numpy(dtype=np.float64), pivot[b].to_numpy(dtype=np.float64), seed=401 + a + b)
            rows.append(
                {
                    "comparison_type": "semantic_layer_contrast",
                    "site": site,
                    "layer_a": a,
                    "layer_b": b,
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


def save_plot(summary_df, stats_df, output_path):
    plt = configure_matplotlib()
    layers = sorted(summary_df["target_layer"].drop_duplicates())
    sites = list(summary_df["site"].drop_duplicates())
    fig, axes = plt.subplots(1, len(layers), figsize=(5.5 * len(layers), 4.5), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        sub = summary_df[summary_df["target_layer"] == layer].copy()
        stat_sub = stats_df[(stats_df["comparison_type"] == "semantic_vs_random") & (stats_df["target_layer"] == layer)].copy()
        x = np.arange(len(sites))
        width = 0.36
        semantic = sub[sub["control"] == "semantic"].set_index("site").reindex(sites)
        random = sub[sub["control"] == "random"].set_index("site").reindex(sites)
        ax.bar(x - width / 2, semantic["delta_from_baseline_mean_signed_label_margin"], width, label="semantic", color="#4477AA")
        ax.bar(x + width / 2, random["delta_from_baseline_mean_signed_label_margin"], width, label="random", color="#CC6677")
        for idx, site in enumerate(sites):
            row = stat_sub[stat_sub["site"] == site]
            if not row.empty:
                pvalue = float(row.iloc[0]["semantic_vs_random_pvalue"])
                y = max(semantic.iloc[idx]["delta_from_baseline_mean_signed_label_margin"], random.iloc[idx]["delta_from_baseline_mean_signed_label_margin"])
                ax.text(idx, y + 0.002, f"p={pvalue:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=25, ha="right")
        ax.set_ylabel("Δ signed label margin")
        ax.grid(True, axis="y", alpha=0.2)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 10J: cross-layer FoX site comparison")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/fox_site_comparison_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,11,15")
    parser.add_argument("--sites", type=str, default="block_input,attn_output,o_proj,block_output")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, choices=["add", "ablate"], default="add")
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec = torch.tensor(load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key), device=device, dtype=torch.float32)
    random_vec = make_random_orthogonal_control(semantic_vec, seed=args.seed)
    anchor_direction = load_anchor_direction(args.data_dir, args.source_layer, allow_invalid_metadata=args.allow_invalid_metadata)
    items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[: args.max_eval_items]
    prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
    target_layers = parse_int_list(args.target_layers)
    sites = parse_site_list(args.sites)
    detail_rows = []

    baseline_rows = []
    for item in prepared:
        metrics = add_sign_aware_fields(item, evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction))
        metrics.update({"control": "baseline"})
        baseline_rows.append(metrics)
        detail_rows.append(metrics.copy())
    baseline_df = pd.DataFrame(baseline_rows).set_index("item_name")

    for target_layer in target_layers:
        for site in sites:
            for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                hook = TensorSiteInterventionHook(vector=vector, alpha=args.alpha, mode=args.mode, position_fraction=args.position_fraction)
                hook.attach(model, target_layer, site)
                try:
                    for item in prepared:
                        metrics = add_sign_aware_fields(item, evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction))
                        metrics.update(
                            {
                                "control": control_name,
                                "site": site,
                                "target_layer": target_layer,
                                "mode": args.mode,
                                "alpha": args.alpha,
                                "position_fraction": args.position_fraction,
                                "position_label": format_position_label(args.position_fraction),
                            }
                        )
                        for column in ["signed_label_margin", "label_target_pairwise_prob", "label_accuracy", "anchor_cosine", "math_minus_creative_logprob"]:
                            metrics[f"delta_from_baseline_{column}"] = float(metrics[column] - baseline_df.loc[metrics["item_name"], column])
                        detail_rows.append(metrics)
                finally:
                    hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    conditioned = detail_df[detail_df["control"].isin(["semantic", "random"])].copy()
    summary = (
        conditioned.groupby(["target_layer", "site", "control", "mode", "alpha", "position_fraction", "position_label"], as_index=False)
        .agg(
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            label_accuracy=("label_accuracy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            mean_math_minus_creative_logprob=("math_minus_creative_logprob", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            delta_from_baseline_mean_anchor_cosine=("delta_from_baseline_anchor_cosine", "mean"),
            delta_from_baseline_mean_math_minus_creative_logprob=("delta_from_baseline_math_minus_creative_logprob", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["target_layer", "site", "control"])
    )
    stats_df = build_stats_rows(conditioned)

    output_path = args.output_csv
    detail_path = args.detail_csv or infer_companion_csv(output_path, "detail")
    stats_path = args.stats_csv or infer_companion_csv(output_path, "stats")
    plot_path = args.plot_path or infer_companion_png(output_path, "summary")
    ensure_parent_dir(output_path)
    summary.to_csv(output_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    save_plot(summary, stats_df, plot_path)
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