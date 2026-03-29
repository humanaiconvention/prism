import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import FOX_LAYER_INDICES, HiddenStateHook, load_genesis_model
from scripts.phase10_experiment_utils import (
    configure_matplotlib,
    ensure_parent_dir,
    infer_companion_csv,
    infer_companion_png,
    layer_type_name,
    paired_signflip_test,
    signflip_test,
)
from scripts.phase10_site_hooks import TensorSiteInterventionHook
from scripts.phase9_semantic_utils import load_semantic_direction, parse_int_list
from scripts.run_phase9_recurrent_state_patching import reset_model_decode_state
from scripts.run_phase9_semantic_steering import load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import format_position_label, prepare_eval_item


def unpack_logits(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output


def score_choice_logits(logits, prepared_item):
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    math_lp = float(log_probs[0, int(prepared_item["math_token_id"])].item())
    creative_lp = float(log_probs[0, int(prepared_item["creative_token_id"])].item())
    margin = math_lp - creative_lp
    label_sign = float(prepared_item["label_sign"])
    math_prob = float(torch.exp(log_probs[0, int(prepared_item["math_token_id"])]).item())
    creative_prob = float(torch.exp(log_probs[0, int(prepared_item["creative_token_id"])]).item())
    return {
        "signed_label_margin": label_sign * margin,
        "label_target_pairwise_prob": float(math_prob / max(math_prob + creative_prob, 1e-12)) if label_sign > 0 else float(creative_prob / max(math_prob + creative_prob, 1e-12)),
        "label_accuracy": float((margin >= 0.0) if label_sign > 0 else (margin <= 0.0)),
        "math_minus_creative_logprob": margin,
    }


def run_with_capture(model, prompt_ids, hidden_hook):
    reset_model_decode_state(model)
    hidden_hook.clear()
    with torch.inference_mode():
        logits = unpack_logits(model(prompt_ids))
    states = {layer: tensor.squeeze(0).detach().clone() for layer, tensor in enumerate(hidden_hook.get_hidden_states())}
    reset_model_decode_state(model)
    return logits.detach().clone(), states


def build_stats_rows(detail_df):
    rows = []
    for target_layer in sorted(detail_df["target_layer"].unique()):
        layer_group = detail_df[detail_df["target_layer"] == target_layer]
        pivot_margin = layer_group.pivot(index="item_name", columns="control", values="delta_from_baseline_signed_label_margin").dropna()
        pivot_late = layer_group.pivot(index="item_name", columns="control", values="late_delta_norm").dropna()
        pivot_growth = layer_group.pivot(index="item_name", columns="control", values="downstream_growth_ratio").dropna()
        if not pivot_margin.empty:
            semantic = pivot_margin["semantic"].to_numpy(dtype=np.float64)
            random = pivot_margin["random"].to_numpy(dtype=np.float64)
            margin_zero = signflip_test(semantic, seed=200 + target_layer)
            margin_pair = paired_signflip_test(semantic, random, seed=300 + target_layer)
            late_pair = paired_signflip_test(
                pivot_late["semantic"].to_numpy(dtype=np.float64),
                pivot_late["random"].to_numpy(dtype=np.float64),
                seed=400 + target_layer,
            )
            growth_pair = paired_signflip_test(
                pivot_growth["semantic"].to_numpy(dtype=np.float64),
                pivot_growth["random"].to_numpy(dtype=np.float64),
                seed=500 + target_layer,
            )
            rows.append(
                {
                    "test_family": "target_layer_effect",
                    "target_layer": int(target_layer),
                    "comparison": "semantic_vs_random",
                    "semantic_mean_delta_signed_label_margin": margin_zero["mean"],
                    "semantic_ci95_low": margin_zero["ci95_low"],
                    "semantic_ci95_high": margin_zero["ci95_high"],
                    "semantic_vs_baseline_pvalue": margin_zero["pvalue"],
                    "semantic_minus_random_mean_delta_signed_label_margin": margin_pair["mean"],
                    "semantic_minus_random_ci95_low": margin_pair["ci95_low"],
                    "semantic_minus_random_ci95_high": margin_pair["ci95_high"],
                    "semantic_vs_random_pvalue": margin_pair["pvalue"],
                    "late_delta_norm_semantic_minus_random": late_pair["mean"],
                    "late_delta_norm_ci95_low": late_pair["ci95_low"],
                    "late_delta_norm_ci95_high": late_pair["ci95_high"],
                    "late_delta_norm_pvalue": late_pair["pvalue"],
                    "growth_ratio_semantic_minus_random": growth_pair["mean"],
                    "growth_ratio_ci95_low": growth_pair["ci95_low"],
                    "growth_ratio_ci95_high": growth_pair["ci95_high"],
                    "growth_ratio_pvalue": growth_pair["pvalue"],
                    "n_items": int(pivot_margin.shape[0]),
                }
            )
    semantic_only = detail_df[detail_df["control"] == "semantic"]
    l11 = semantic_only[semantic_only["target_layer"] == 11].set_index("item_name")
    for other_layer in [7, 9, 13, 15]:
        other = semantic_only[semantic_only["target_layer"] == other_layer].set_index("item_name")
        common = l11.index.intersection(other.index)
        if len(common) == 0:
            continue
        margin_pair = paired_signflip_test(
            l11.loc[common, "delta_from_baseline_signed_label_margin"].to_numpy(dtype=np.float64),
            other.loc[common, "delta_from_baseline_signed_label_margin"].to_numpy(dtype=np.float64),
            seed=600 + other_layer,
        )
        growth_pair = paired_signflip_test(
            l11.loc[common, "downstream_growth_ratio"].to_numpy(dtype=np.float64),
            other.loc[common, "downstream_growth_ratio"].to_numpy(dtype=np.float64),
            seed=700 + other_layer,
        )
        rows.append(
            {
                "test_family": "l11_layer_contrast",
                "target_layer": 11,
                "comparison": f"semantic_L11_vs_L{other_layer}",
                "semantic_mean_delta_signed_label_margin": float(np.mean(l11.loc[common, "delta_from_baseline_signed_label_margin"])),
                "semantic_ci95_low": margin_pair["ci95_low"],
                "semantic_ci95_high": margin_pair["ci95_high"],
                "semantic_vs_baseline_pvalue": np.nan,
                "semantic_minus_random_mean_delta_signed_label_margin": margin_pair["mean"],
                "semantic_minus_random_ci95_low": margin_pair["ci95_low"],
                "semantic_minus_random_ci95_high": margin_pair["ci95_high"],
                "semantic_vs_random_pvalue": margin_pair["pvalue"],
                "late_delta_norm_semantic_minus_random": np.nan,
                "late_delta_norm_ci95_low": np.nan,
                "late_delta_norm_ci95_high": np.nan,
                "late_delta_norm_pvalue": np.nan,
                "growth_ratio_semantic_minus_random": growth_pair["mean"],
                "growth_ratio_ci95_low": growth_pair["ci95_low"],
                "growth_ratio_ci95_high": growth_pair["ci95_high"],
                "growth_ratio_pvalue": growth_pair["pvalue"],
                "n_items": int(len(common)),
            }
        )
    return pd.DataFrame(rows)


def save_summary_plot(summary_df, output_path):
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for control_name, color in (("semantic", "#4477AA"), ("random", "#CC6677")):
        sub = summary_df[summary_df["control"] == control_name].sort_values("target_layer")
        axes[0].plot(sub["target_layer"], sub["delta_from_baseline_mean_signed_label_margin"], marker="o", linewidth=2, color=color, label=control_name)
        axes[1].plot(sub["target_layer"], sub["mean_downstream_growth_ratio"], marker="o", linewidth=2, color=color, label=control_name)
    for ax, ylabel in zip(axes, ["Δ signed label margin", "Mean downstream growth ratio"]):
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Target layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Behavioral boundary sweep")
    axes[1].set_title("Downstream amplification")
    axes[0].legend(loc="best")
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_divergence_plot(divergence_df, output_path):
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, control_name in zip(axes, ["semantic", "random"]):
        sub = divergence_df[divergence_df["control"] == control_name]
        for target_layer in sorted(sub["target_layer"].unique()):
            curve = sub[sub["target_layer"] == target_layer].groupby("probe_layer", as_index=False)["delta_norm"].mean()
            ax.plot(curve["probe_layer"], curve["delta_norm"], linewidth=2, label=f"L{target_layer}")
        ax.set_title(f"{control_name}: downstream delta norms")
        ax.set_xlabel("Probe layer")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Mean ||Δ hidden||")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 10H: perturbation-boundary sweep around L11")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/perturbation_boundary_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--divergence-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--divergence-plot-path", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,9,11,13,15")
    parser.add_argument("--site", type=str, default="block_input")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-eval-items", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    hidden_hook = HiddenStateHook.attach_to_model(model)
    semantic_vec = torch.tensor(load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key), device=device, dtype=torch.float32)
    random_vec = make_random_orthogonal_control(semantic_vec, seed=args.seed)
    items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[: args.max_eval_items]
    prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
    target_layers = parse_int_list(args.target_layers)
    detail_rows = []
    divergence_rows = []

    try:
        for item in prepared:
            baseline_logits, baseline_states = run_with_capture(model, item["prompt_ids"], hidden_hook)
            baseline_scores = score_choice_logits(baseline_logits, item)
            for target_layer in target_layers:
                for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                    hook = TensorSiteInterventionHook(vector=vector, alpha=args.alpha, mode="add", position_fraction=args.position_fraction)
                    hook.attach(model, target_layer, args.site)
                    try:
                        logits, states = run_with_capture(model, item["prompt_ids"], hidden_hook)
                    finally:
                        hook.remove()
                    scores = score_choice_logits(logits, item)
                    delta_norms = {}
                    for probe_layer in sorted(states):
                        delta = states[probe_layer] - baseline_states[probe_layer]
                        delta_norm = float(torch.norm(delta).item())
                        delta_norms[probe_layer] = delta_norm
                        divergence_rows.append(
                            {
                                "item_name": item["item"]["name"],
                                "target_layer": target_layer,
                                "target_layer_type": layer_type_name(target_layer, FOX_LAYER_INDICES),
                                "control": control_name,
                                "site": args.site,
                                "probe_layer": probe_layer,
                                "probe_layer_type": layer_type_name(probe_layer, FOX_LAYER_INDICES),
                                "alpha": args.alpha,
                                "position_fraction": args.position_fraction,
                                "delta_norm": delta_norm,
                            }
                        )
                    post_layers = [layer for layer in sorted(delta_norms) if layer >= target_layer]
                    immediate = delta_norms[target_layer]
                    downstream = np.array([delta_norms[layer] for layer in post_layers], dtype=np.float64)
                    late = float(delta_norms[max(delta_norms)])
                    peak = float(np.max(downstream))
                    detail_rows.append(
                        {
                            "item_name": item["item"]["name"],
                            "target_layer": target_layer,
                            "target_layer_type": layer_type_name(target_layer, FOX_LAYER_INDICES),
                            "control": control_name,
                            "site": args.site,
                            "alpha": args.alpha,
                            "position_fraction": args.position_fraction,
                            "position_label": format_position_label(args.position_fraction),
                            "signed_label_margin": scores["signed_label_margin"],
                            "label_target_pairwise_prob": scores["label_target_pairwise_prob"],
                            "label_accuracy": scores["label_accuracy"],
                            "math_minus_creative_logprob": scores["math_minus_creative_logprob"],
                            "delta_from_baseline_signed_label_margin": float(scores["signed_label_margin"] - baseline_scores["signed_label_margin"]),
                            "delta_from_baseline_label_target_pairwise_prob": float(scores["label_target_pairwise_prob"] - baseline_scores["label_target_pairwise_prob"]),
                            "delta_from_baseline_label_accuracy": float(scores["label_accuracy"] - baseline_scores["label_accuracy"]),
                            "delta_from_baseline_math_minus_creative_logprob": float(scores["math_minus_creative_logprob"] - baseline_scores["math_minus_creative_logprob"]),
                            "immediate_delta_norm": float(immediate),
                            "late_delta_norm": late,
                            "peak_downstream_delta_norm": peak,
                            "mean_downstream_delta_norm": float(np.mean(downstream)),
                            "downstream_growth_ratio": float(late / max(immediate, 1e-8)),
                            "downstream_peak_ratio": float(peak / max(immediate, 1e-8)),
                        }
                    )
    finally:
        hidden_hook.remove_all()
        reset_model_decode_state(model)

    detail_df = pd.DataFrame(detail_rows)
    divergence_df = pd.DataFrame(divergence_rows)
    summary = (
        detail_df.groupby(["target_layer", "target_layer_type", "control", "site", "alpha", "position_fraction", "position_label"], as_index=False)
        .agg(
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            label_accuracy=("label_accuracy", "mean"),
            mean_math_minus_creative_logprob=("math_minus_creative_logprob", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            delta_from_baseline_mean_math_minus_creative_logprob=("delta_from_baseline_math_minus_creative_logprob", "mean"),
            mean_immediate_delta_norm=("immediate_delta_norm", "mean"),
            mean_late_delta_norm=("late_delta_norm", "mean"),
            mean_peak_downstream_delta_norm=("peak_downstream_delta_norm", "mean"),
            mean_mean_downstream_delta_norm=("mean_downstream_delta_norm", "mean"),
            mean_downstream_growth_ratio=("downstream_growth_ratio", "mean"),
            mean_downstream_peak_ratio=("downstream_peak_ratio", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["target_layer", "control"])
    )
    stats_df = build_stats_rows(detail_df)

    output_path = args.output_csv
    detail_path = args.detail_csv or infer_companion_csv(output_path, "detail")
    divergence_path = args.divergence_csv or infer_companion_csv(output_path, "divergence")
    stats_path = args.stats_csv or infer_companion_csv(output_path, "stats")
    plot_path = args.plot_path or infer_companion_png(output_path, "summary")
    divergence_plot_path = args.divergence_plot_path or infer_companion_png(output_path, "divergence")
    ensure_parent_dir(output_path)
    summary.to_csv(output_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    divergence_df.to_csv(divergence_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    save_summary_plot(summary, plot_path)
    save_divergence_plot(divergence_df, divergence_plot_path)
    print(summary.to_string(index=False))
    if not stats_df.empty:
        print("\n[stats]")
        print(stats_df.to_string(index=False))
    print(f"[saved] summary -> {output_path}")
    print(f"[saved] detail -> {detail_path}")
    print(f"[saved] divergence -> {divergence_path}")
    print(f"[saved] stats -> {stats_path}")
    print(f"[saved] plot -> {plot_path}")
    print(f"[saved] divergence plot -> {divergence_plot_path}")


if __name__ == "__main__":
    main()