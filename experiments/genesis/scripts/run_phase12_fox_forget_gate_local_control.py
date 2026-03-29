import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import FOX_LAYER_INDICES, load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import TensorSiteAnswerWindowCaptureHook, TensorSiteAnswerWindowInterventionHook, resolve_answer_window_positions
from scripts.phase9_semantic_utils import load_semantic_direction
from scripts.run_phase10_tail_conditioned_necessity import add_sign_aware_fields, expand_pair_items, load_pair_items, pair_name_from_item_name
from scripts.run_phase9_recurrent_state_patching import reset_model_decode_state
from scripts.run_phase9_semantic_steering import make_random_orthogonal_control, validate_upstream_metadata
from scripts.run_phase9_token_position_steering import prepare_eval_item


ALLOWED_CONTROLS = ("semantic", "random")
METRIC_SPECS = [
    "delta_from_baseline_signed_label_margin",
    "delta_from_baseline_label_target_pairwise_prob",
    "delta_from_baseline_fgate_logit_mean",
    "delta_from_baseline_fgate_prob_mean",
]
REQUIRED_PAIR_EFFECT_COLUMNS = [
    "dataset_name", "pair_name", "source_layer", "target_layer", "target_layer_type", "site", "capture_site", "vector_key",
    "mode", "alpha", "answer_offset", "answer_offset_label", "window_size", "n_items_per_pair",
    "semantic_delta_from_baseline_signed_label_margin", "random_delta_from_baseline_signed_label_margin",
    "semantic_delta_from_baseline_fgate_prob_mean", "random_delta_from_baseline_fgate_prob_mean",
]
REQUIRED_STATS_COLUMNS = [
    "dataset_name", "source_layer", "target_layer", "target_layer_type", "site", "capture_site", "vector_key", "mode",
    "alpha", "answer_offset", "answer_offset_label", "window_size", "metric_name", "comparison_type",
    "n_pairs", "mean_a", "mean_b", "mean_difference", "ci95_low", "ci95_high", "pvalue", "n_perm", "seed",
]


def phase12d_output_paths(output_csv):
    return {
        "summary": output_csv,
        "detail": infer_companion_csv(output_csv, "detail"),
        "pair_effect": infer_companion_csv(output_csv, "pair_effect"),
        "stats": infer_companion_csv(output_csv, "stats"),
    }


def layer_type_name(layer):
    return "fox" if int(layer) in FOX_LAYER_INDICES else "gla"


def parse_controls(raw_controls):
    ordered = []
    for token in raw_controls.split(","):
        control = token.strip().lower()
        if not control:
            continue
        if control not in ALLOWED_CONTROLS:
            raise ValueError(f"Unsupported control: {control}. Expected only {list(ALLOWED_CONTROLS)}")
        if control not in ordered:
            ordered.append(control)
    if set(ordered) != set(ALLOWED_CONTROLS):
        raise ValueError("Phase 12D requires exactly the semantic and random matched controls.")
    return ["semantic", "random"]


def answer_offset_label(answer_offset):
    return f"t_minus_{int(answer_offset)}"


def unpack_logits(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output


def run_with_capture(model, prompt_ids, capture_hook=None):
    reset_model_decode_state(model)
    if capture_hook is not None:
        capture_hook.clear()
    with torch.inference_mode():
        logits = unpack_logits(model(prompt_ids))
    captured = None
    if capture_hook is not None and capture_hook.get_captured() is not None:
        captured = capture_hook.get_captured().detach().clone()
    reset_model_decode_state(model)
    return logits.detach().clone(), captured


def score_choice_logits(logits, prepared_item):
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    math_lp = float(log_probs[0, int(prepared_item["math_token_id"])] .item())
    creative_lp = float(log_probs[0, int(prepared_item["creative_token_id"])] .item())
    math_minus_creative = math_lp - creative_lp
    pairwise_denom = np.exp(math_lp) + np.exp(creative_lp)
    metrics = {
        "math_logprob": math_lp,
        "creative_logprob": creative_lp,
        "math_minus_creative_logprob": float(math_minus_creative),
        "pairwise_math_prob": float(np.exp(math_lp) / max(pairwise_denom, 1e-12)),
        "pairwise_creative_prob": float(np.exp(creative_lp) / max(pairwise_denom, 1e-12)),
        "signed_label_margin": float(prepared_item["label_sign"] * math_minus_creative),
        "label_correct": int((math_minus_creative >= 0.0) == (prepared_item["label_sign"] > 0)),
    }
    return add_sign_aware_fields(prepared_item, metrics)


def summarize_gate_window(gate_window):
    logits_mean = float(gate_window.mean().item())
    prob_mean = float(torch.sigmoid(gate_window.float()).mean().item())
    return logits_mean, prob_mean


def build_summary_df(detail_df):
    return (
        detail_df.groupby(
            [
                "dataset_name", "source_layer", "target_layer", "target_layer_type", "site", "capture_site", "vector_key", "control",
                "alpha", "mode", "answer_offset", "answer_offset_label", "window_size",
            ],
            as_index=False,
        )
        .agg(
            n_items=("item_name", "count"),
            n_pairs=("pair_name", "nunique"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            mean_label_accuracy=("label_accuracy", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            mean_baseline_fgate_logit_mean=("baseline_fgate_logit_mean", "mean"),
            mean_fgate_logit_mean=("fgate_logit_mean", "mean"),
            delta_from_baseline_mean_fgate_logit_mean=("delta_from_baseline_fgate_logit_mean", "mean"),
            mean_baseline_fgate_prob_mean=("baseline_fgate_prob_mean", "mean"),
            mean_fgate_prob_mean=("fgate_prob_mean", "mean"),
            delta_from_baseline_mean_fgate_prob_mean=("delta_from_baseline_fgate_prob_mean", "mean"),
            mean_effective_window_size=("effective_window_size", "mean"),
        )
        .sort_values(["control"])
        .reset_index(drop=True)
    )


def build_pair_effect_df(detail_df):
    pair_control_df = (
        detail_df.groupby(
            [
                "dataset_name", "pair_name", "source_layer", "target_layer", "target_layer_type", "site", "capture_site", "vector_key",
                "mode", "alpha", "answer_offset", "answer_offset_label", "window_size", "control",
            ],
            as_index=False,
        )
        .agg(
            n_items_per_pair=("item_name", "count"),
            delta_from_baseline_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_fgate_logit_mean=("delta_from_baseline_fgate_logit_mean", "mean"),
            delta_from_baseline_fgate_prob_mean=("delta_from_baseline_fgate_prob_mean", "mean"),
        )
    )
    group_cols = [
        "dataset_name", "pair_name", "source_layer", "target_layer", "target_layer_type", "site", "capture_site", "vector_key",
        "mode", "alpha", "answer_offset", "answer_offset_label", "window_size",
    ]
    rows = []
    for group_key, group in pair_control_df.groupby(group_cols, dropna=False):
        control_table = group.set_index("control")
        if not set(ALLOWED_CONTROLS).issubset(set(control_table.index)):
            continue
        row = dict(zip(group_cols, group_key))
        row["n_items_per_pair"] = int(control_table.loc["semantic", "n_items_per_pair"])
        for metric_name in METRIC_SPECS:
            semantic_value = float(control_table.loc["semantic", metric_name])
            random_value = float(control_table.loc["random", metric_name])
            row[f"semantic_{metric_name}"] = semantic_value
            row[f"random_{metric_name}"] = random_value
            row[f"semantic_minus_random_{metric_name}"] = semantic_value - random_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pair_name"]).reset_index(drop=True)


def _stats_row(common, metric_name, comparison_type, result, *, n_perm, seed):
    return {
        **common,
        "metric_name": metric_name,
        "comparison_type": comparison_type,
        "n_pairs": int(result["n"]),
        "mean_a": float(result.get("mean_a", result["mean"])),
        "mean_b": float(result.get("mean_b", 0.0)),
        "mean_difference": float(result["mean"]),
        "ci95_low": float(result["ci95_low"]),
        "ci95_high": float(result["ci95_high"]),
        "pvalue": float(result["pvalue"]),
        "n_perm": int(n_perm),
        "seed": int(seed),
    }


def build_stats_rows(pair_effect_df, *, n_perm=100000, seed=1234):
    rows = []
    if pair_effect_df.empty:
        return rows
    group_cols = [
        "dataset_name", "source_layer", "target_layer", "target_layer_type", "site", "capture_site", "vector_key", "mode",
        "alpha", "answer_offset", "answer_offset_label", "window_size",
    ]
    for group_key, group in pair_effect_df.groupby(group_cols, dropna=False):
        common = dict(zip(group_cols, group_key))
        for metric_idx, metric_name in enumerate(METRIC_SPECS):
            semantic = group[f"semantic_{metric_name}"].to_numpy(dtype=np.float64)
            random = group[f"random_{metric_name}"].to_numpy(dtype=np.float64)
            seed_base = int(seed) + (metric_idx + 1) * 1000 + 97 * int(common["target_layer"]) + 13 * int(common["answer_offset"])
            semantic_zero = signflip_test(semantic, n_perm=n_perm, seed=seed_base + 1)
            random_zero = signflip_test(random, n_perm=n_perm, seed=seed_base + 2)
            semantic_vs_random = paired_signflip_test(semantic, random, n_perm=n_perm, seed=seed_base + 3)
            rows.append(_stats_row(common, metric_name, "semantic_vs_zero", semantic_zero, n_perm=n_perm, seed=seed_base + 1))
            rows.append(_stats_row(common, metric_name, "random_vs_zero", random_zero, n_perm=n_perm, seed=seed_base + 2))
            rows.append(_stats_row(common, metric_name, "semantic_vs_random", semantic_vs_random, n_perm=n_perm, seed=seed_base + 3))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 12D: FoX forget-gate-path local-control test at the answer boundary")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layer", type=int, default=11)
    parser.add_argument("--site", type=str, default="fgate_proj_input")
    parser.add_argument("--capture-site", type=str, default="fgate_proj")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, default="add", help="add or ablate")
    parser.add_argument("--controls", type=str, default="semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--answer-offset", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--dataset-name", type=str, default="heldout_shared")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase12/phase12d_fox_forget_gate_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--pair-effect-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    if args.mode not in {"add", "ablate"}:
        raise ValueError(f"Unsupported mode: {args.mode}")
    controls = parse_controls(args.controls)
    paths = phase12d_output_paths(args.output_csv)
    if args.detail_csv is not None:
        paths["detail"] = args.detail_csv
    if args.pair_effect_csv is not None:
        paths["pair_effect"] = args.pair_effect_csv
    if args.stats_csv is not None:
        paths["stats"] = args.stats_csv
    for path in paths.values():
        ensure_parent_dir(path)

    validate_upstream_metadata(args.data_dir, allow_invalid_metadata=args.allow_invalid_metadata)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec = torch.tensor(load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key), device=device, dtype=torch.float32)
    random_vec = make_random_orthogonal_control(semantic_vec, seed=args.seed + int(args.source_layer))
    direction_bank = {"semantic": semantic_vec, "random": random_vec}
    pair_items = load_pair_items(args.eval_json, max_eval_items=args.max_eval_items)
    prepared_items = [prepare_eval_item(tokenizer, item, device) for item in expand_pair_items(pair_items)]
    baseline_by_item = {}
    detail_rows = []

    baseline_capture_hook = TensorSiteAnswerWindowCaptureHook(answer_offset=args.answer_offset, window_size=args.window_size)
    baseline_capture_hook.attach(model, args.target_layer, args.capture_site)
    try:
        for prepared_item in tqdm(prepared_items, desc="Baseline", leave=False):
            logits, gate_window = run_with_capture(model, prepared_item["prompt_ids"], baseline_capture_hook)
            baseline_scores = score_choice_logits(logits, prepared_item)
            baseline_logit_mean, baseline_prob_mean = summarize_gate_window(gate_window)
            baseline_by_item[prepared_item["item"]["name"]] = {
                "scores": baseline_scores,
                "fgate_logit_mean": baseline_logit_mean,
                "fgate_prob_mean": baseline_prob_mean,
            }
    finally:
        baseline_capture_hook.remove()

    for control_name in controls:
        vector = direction_bank[control_name]
        hook = TensorSiteAnswerWindowInterventionHook(
            vector=vector,
            alpha=args.alpha,
            mode=args.mode,
            answer_offset=args.answer_offset,
            window_size=args.window_size,
        )
        hook.attach(model, args.target_layer, args.site)
        capture_hook = TensorSiteAnswerWindowCaptureHook(answer_offset=args.answer_offset, window_size=args.window_size)
        capture_hook.attach(model, args.target_layer, args.capture_site)
        try:
            for prepared_item in tqdm(prepared_items, desc=control_name, leave=False):
                item_name = prepared_item["item"]["name"]
                baseline = baseline_by_item[item_name]
                logits, gate_window = run_with_capture(model, prepared_item["prompt_ids"], capture_hook)
                scores = score_choice_logits(logits, prepared_item)
                fgate_logit_mean, fgate_prob_mean = summarize_gate_window(gate_window)
                positions = resolve_answer_window_positions(
                    prepared_item["prompt_token_count"],
                    answer_offset=args.answer_offset,
                    window_size=args.window_size,
                )
                detail_rows.append(
                    {
                        "dataset_name": args.dataset_name,
                        "item_name": item_name,
                        "pair_name": pair_name_from_item_name(item_name),
                        "source_layer": int(args.source_layer),
                        "target_layer": int(args.target_layer),
                        "target_layer_type": layer_type_name(args.target_layer),
                        "site": args.site,
                        "capture_site": args.capture_site,
                        "vector_key": args.vector_key,
                        "control": control_name,
                        "alpha": float(args.alpha),
                        "mode": args.mode,
                        "answer_offset": int(args.answer_offset),
                        "answer_offset_label": answer_offset_label(args.answer_offset),
                        "window_size": int(args.window_size),
                        "effective_window_size": int(len(positions)),
                        "window_start_index": int(positions[0]),
                        "window_end_index": int(positions[-1]),
                        "distance_to_answer_start": int(prepared_item["prompt_token_count"] - 1 - positions[0]),
                        "distance_to_answer_end": int(prepared_item["prompt_token_count"] - 1 - positions[-1]),
                        "math_minus_creative_logprob": float(scores["math_minus_creative_logprob"]),
                        "signed_label_margin": float(scores["signed_label_margin"]),
                        "label_target_pairwise_prob": float(scores["label_target_pairwise_prob"]),
                        "label_accuracy": float(scores["label_accuracy"]),
                        "baseline_signed_label_margin": float(baseline["scores"]["signed_label_margin"]),
                        "delta_from_baseline_signed_label_margin": float(scores["signed_label_margin"] - baseline["scores"]["signed_label_margin"]),
                        "baseline_label_target_pairwise_prob": float(baseline["scores"]["label_target_pairwise_prob"]),
                        "delta_from_baseline_label_target_pairwise_prob": float(
                            scores["label_target_pairwise_prob"] - baseline["scores"]["label_target_pairwise_prob"]
                        ),
                        "baseline_fgate_logit_mean": float(baseline["fgate_logit_mean"]),
                        "fgate_logit_mean": float(fgate_logit_mean),
                        "delta_from_baseline_fgate_logit_mean": float(fgate_logit_mean - baseline["fgate_logit_mean"]),
                        "baseline_fgate_prob_mean": float(baseline["fgate_prob_mean"]),
                        "fgate_prob_mean": float(fgate_prob_mean),
                        "delta_from_baseline_fgate_prob_mean": float(fgate_prob_mean - baseline["fgate_prob_mean"]),
                    }
                )
        finally:
            hook.remove()
            capture_hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = build_summary_df(detail_df)
    pair_effect_df = build_pair_effect_df(detail_df)
    stats_df = pd.DataFrame(build_stats_rows(pair_effect_df, n_perm=args.n_perm, seed=args.seed))

    summary_df.to_csv(paths["summary"], index=False)
    detail_df.to_csv(paths["detail"], index=False)
    pair_effect_df.to_csv(paths["pair_effect"], index=False)
    stats_df.to_csv(paths["stats"], index=False)

    primary_stats = stats_df[
        stats_df["metric_name"].isin(["delta_from_baseline_signed_label_margin", "delta_from_baseline_fgate_prob_mean"])
    ].copy()
    print("\n--- PHASE 12D SUMMARY ---")
    print(summary_df.to_string(index=False))
    print("\n--- PHASE 12D PRIMARY STATS ---")
    print(primary_stats.to_string(index=False))
    print(f"\nSummary saved to {paths['summary']}")
    print(f"Detail saved to {paths['detail']}")
    print(f"Pair effects saved to {paths['pair_effect']}")
    print(f"Stats saved to {paths['stats']}")


if __name__ == "__main__":
    main()