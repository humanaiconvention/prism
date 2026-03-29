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
from scripts.phase10_site_hooks import TensorSiteAnswerWindowInterventionHook, resolve_answer_window_positions
from scripts.phase9_semantic_utils import load_semantic_direction
from scripts.run_phase10_tail_conditioned_necessity import add_sign_aware_fields, expand_pair_items, load_pair_items, pair_name_from_item_name
from scripts.run_phase9_recurrent_state_patching import reset_model_decode_state
from scripts.run_phase9_semantic_steering import make_random_orthogonal_control, validate_upstream_metadata
from scripts.run_phase9_token_position_steering import prepare_eval_item


CONDITION_BASELINE = "baseline"
CONDITION_SEMANTIC_EARLY_ONLY = "semantic_early_only"
CONDITION_SEMANTIC_LATE_ONLY = "semantic_late_only"
CONDITION_SEMANTIC_JOINT = "semantic_joint"
CONDITION_RANDOM_EARLY_ONLY = "random_early_only"
CONDITION_RANDOM_LATE_ONLY = "random_late_only"
CONDITION_RANDOM_JOINT = "random_joint"
CONDITION_ORDER = [
    CONDITION_BASELINE,
    CONDITION_SEMANTIC_EARLY_ONLY,
    CONDITION_SEMANTIC_LATE_ONLY,
    CONDITION_SEMANTIC_JOINT,
    CONDITION_RANDOM_EARLY_ONLY,
    CONDITION_RANDOM_LATE_ONLY,
    CONDITION_RANDOM_JOINT,
]
ALLOWED_CONTROLS = ("semantic", "random")
METRIC_SPECS = [
    ("signed_label_margin", "signed_label_margin"),
    ("label_target_pairwise_prob", "label_target_pairwise_prob"),
    ("label_accuracy", "label_accuracy"),
]
REQUIRED_PAIR_INTERACTION_COLUMNS = [
    "dataset_name", "pair_name", "source_layer", "early_target_layer", "early_target_layer_type",
    "late_target_layer", "late_target_layer_type", "site", "vector_key", "mode", "alpha",
    "answer_offset", "answer_offset_label", "window_size", "n_items_per_pair",
    "semantic_joint_interaction_signed_label_margin", "random_joint_interaction_signed_label_margin",
    "semantic_joint_minus_best_single_signed_label_margin", "random_joint_minus_best_single_signed_label_margin",
]
REQUIRED_STATS_COLUMNS = [
    "dataset_name", "source_layer", "early_target_layer", "early_target_layer_type", "late_target_layer",
    "late_target_layer_type", "site", "vector_key", "mode", "alpha", "answer_offset",
    "answer_offset_label", "window_size", "metric_name", "comparison_type", "n_pairs", "mean_a",
    "mean_b", "mean_difference", "ci95_low", "ci95_high", "pvalue", "n_perm", "seed",
]


def phase12c_output_paths(output_csv):
    return {
        "summary": output_csv,
        "detail": infer_companion_csv(output_csv, "detail"),
        "pair_interaction": infer_companion_csv(output_csv, "pair_interaction"),
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
        raise ValueError("Phase 12C requires exactly the semantic and random matched controls.")
    return ["semantic", "random"]


def answer_offset_label(answer_offset):
    return f"t_minus_{int(answer_offset)}"


def unpack_logits(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output


def run_forward(model, prompt_ids):
    reset_model_decode_state(model)
    with torch.inference_mode():
        logits = unpack_logits(model(prompt_ids))
    reset_model_decode_state(model)
    return logits.detach().clone()


def score_choice_logits(logits, prepared_item):
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    math_lp = float(log_probs[0, int(prepared_item["math_token_id"])].item())
    creative_lp = float(log_probs[0, int(prepared_item["creative_token_id"])].item())
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


def add_condition_row(detail_rows, *, baseline, scores, prepared_item, args, condition, control_name, target_layers):
    positions = resolve_answer_window_positions(
        prepared_item["prompt_token_count"],
        answer_offset=args.answer_offset,
        window_size=args.window_size,
    )
    detail_rows.append(
        {
            "dataset_name": args.dataset_name,
            "item_name": prepared_item["item"]["name"],
            "pair_name": pair_name_from_item_name(prepared_item["item"]["name"]),
            "source_layer": int(args.source_layer),
            "early_target_layer": int(args.early_target_layer),
            "early_target_layer_type": layer_type_name(args.early_target_layer),
            "late_target_layer": int(args.late_target_layer),
            "late_target_layer_type": layer_type_name(args.late_target_layer),
            "site": args.site,
            "vector_key": args.vector_key,
            "condition": condition,
            "control": control_name,
            "alpha": float(args.alpha) if condition != CONDITION_BASELINE else 0.0,
            "mode": args.mode,
            "answer_offset": int(args.answer_offset),
            "answer_offset_label": answer_offset_label(args.answer_offset),
            "window_size": int(args.window_size),
            "effective_window_size": int(len(positions)),
            "window_start_index": int(positions[0]),
            "window_end_index": int(positions[-1]),
            "distance_to_answer_start": int(prepared_item["prompt_token_count"] - 1 - positions[0]),
            "distance_to_answer_end": int(prepared_item["prompt_token_count"] - 1 - positions[-1]),
            "hooked_layer_count": int(len(target_layers)),
            "hooked_layers_label": "+".join([f"L{int(layer)}" for layer in target_layers]) if target_layers else "none",
            "math_minus_creative_logprob": float(scores["math_minus_creative_logprob"]),
            "signed_label_margin": float(scores["signed_label_margin"]),
            "label_target_pairwise_prob": float(scores["label_target_pairwise_prob"]),
            "label_accuracy": float(scores["label_accuracy"]),
            "baseline_signed_label_margin": float(baseline["signed_label_margin"]),
            "delta_from_baseline_signed_label_margin": float(scores["signed_label_margin"] - baseline["signed_label_margin"]),
            "baseline_label_target_pairwise_prob": float(baseline["label_target_pairwise_prob"]),
            "delta_from_baseline_label_target_pairwise_prob": float(
                scores["label_target_pairwise_prob"] - baseline["label_target_pairwise_prob"]
            ),
            "baseline_label_accuracy": float(baseline["label_accuracy"]),
            "delta_from_baseline_label_accuracy": float(scores["label_accuracy"] - baseline["label_accuracy"]),
        }
    )


def build_summary_df(detail_df):
    summary_df = (
        detail_df.groupby(
            [
                "dataset_name", "source_layer", "early_target_layer", "early_target_layer_type",
                "late_target_layer", "late_target_layer_type", "site", "vector_key", "control",
                "condition", "alpha", "mode", "answer_offset", "answer_offset_label", "window_size",
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
            delta_from_baseline_mean_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
        )
    )
    order_map = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
    summary_df["_condition_order"] = summary_df["condition"].map(order_map).fillna(len(order_map)).astype(int)
    return summary_df.sort_values(["_condition_order"]).drop(columns=["_condition_order"]).reset_index(drop=True)


def build_pair_interaction_df(detail_df):
    pair_condition_df = (
        detail_df.groupby(
            [
                "dataset_name", "pair_name", "source_layer", "early_target_layer", "early_target_layer_type",
                "late_target_layer", "late_target_layer_type", "site", "vector_key", "mode", "alpha",
                "answer_offset", "answer_offset_label", "window_size", "condition",
            ],
            as_index=False,
        )
        .agg(
            n_items_per_pair=("item_name", "count"),
            signed_label_margin=("signed_label_margin", "mean"),
            label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            label_accuracy=("label_accuracy", "mean"),
        )
    )
    required_conditions = set(CONDITION_ORDER)
    group_cols = [
        "dataset_name", "pair_name", "source_layer", "early_target_layer", "early_target_layer_type",
        "late_target_layer", "late_target_layer_type", "site", "vector_key", "mode", "answer_offset",
        "answer_offset_label", "window_size",
    ]
    rows = []
    for group_key, group in pair_condition_df.groupby(group_cols, dropna=False):
        condition_table = group.set_index("condition")
        if not required_conditions.issubset(set(condition_table.index)):
            continue
        row = dict(zip(group_cols, group_key))
        row["alpha"] = float(group["alpha"].max())
        row["n_items_per_pair"] = int(condition_table.loc[CONDITION_BASELINE, "n_items_per_pair"])
        for metric_col, metric_name in METRIC_SPECS:
            baseline = float(condition_table.loc[CONDITION_BASELINE, metric_col])
            semantic_early = float(condition_table.loc[CONDITION_SEMANTIC_EARLY_ONLY, metric_col])
            semantic_late = float(condition_table.loc[CONDITION_SEMANTIC_LATE_ONLY, metric_col])
            semantic_joint = float(condition_table.loc[CONDITION_SEMANTIC_JOINT, metric_col])
            random_early = float(condition_table.loc[CONDITION_RANDOM_EARLY_ONLY, metric_col])
            random_late = float(condition_table.loc[CONDITION_RANDOM_LATE_ONLY, metric_col])
            random_joint = float(condition_table.loc[CONDITION_RANDOM_JOINT, metric_col])
            row[f"baseline_{metric_name}"] = baseline
            row[f"semantic_early_only_{metric_name}"] = semantic_early
            row[f"semantic_late_only_{metric_name}"] = semantic_late
            row[f"semantic_joint_{metric_name}"] = semantic_joint
            row[f"random_early_only_{metric_name}"] = random_early
            row[f"random_late_only_{metric_name}"] = random_late
            row[f"random_joint_{metric_name}"] = random_joint
            row[f"semantic_joint_effect_{metric_name}"] = semantic_joint - baseline
            row[f"random_joint_effect_{metric_name}"] = random_joint - baseline
            row[f"semantic_joint_interaction_{metric_name}"] = semantic_joint - semantic_early - semantic_late + baseline
            row[f"random_joint_interaction_{metric_name}"] = random_joint - random_early - random_late + baseline
            row[f"semantic_joint_minus_best_single_{metric_name}"] = semantic_joint - max(semantic_early, semantic_late)
            row[f"random_joint_minus_best_single_{metric_name}"] = random_joint - max(random_early, random_late)
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


def build_stats_rows(pair_interaction_df, *, n_perm=100000, seed=1234):
    rows = []
    if pair_interaction_df.empty:
        return rows
    group_cols = [
        "dataset_name", "source_layer", "early_target_layer", "early_target_layer_type", "late_target_layer",
        "late_target_layer_type", "site", "vector_key", "mode", "alpha", "answer_offset",
        "answer_offset_label", "window_size",
    ]
    comparison_specs = [
        ("joint_interaction", "joint_interaction"),
        ("joint_minus_best_single", "joint_minus_best_single"),
    ]
    for group_key, group in pair_interaction_df.groupby(group_cols, dropna=False):
        common = dict(zip(group_cols, group_key))
        for metric_idx, (_, metric_name) in enumerate(METRIC_SPECS):
            for comp_idx, (field_prefix, label_prefix) in enumerate(comparison_specs):
                semantic = group[f"semantic_{field_prefix}_{metric_name}"].to_numpy(dtype=np.float64)
                random = group[f"random_{field_prefix}_{metric_name}"].to_numpy(dtype=np.float64)
                seed_base = (
                    int(seed)
                    + (metric_idx + 1) * 1000
                    + (comp_idx + 1) * 100
                    + 17 * int(common["early_target_layer"])
                    + 31 * int(common["late_target_layer"])
                )
                semantic_zero = signflip_test(semantic, n_perm=n_perm, seed=seed_base + 1)
                random_zero = signflip_test(random, n_perm=n_perm, seed=seed_base + 2)
                semantic_vs_random = paired_signflip_test(semantic, random, n_perm=n_perm, seed=seed_base + 3)
                rows.append(_stats_row(common, metric_name, f"semantic_{label_prefix}_vs_zero", semantic_zero, n_perm=n_perm, seed=seed_base + 1))
                rows.append(_stats_row(common, metric_name, f"random_{label_prefix}_vs_zero", random_zero, n_perm=n_perm, seed=seed_base + 2))
                rows.append(
                    _stats_row(
                        common,
                        metric_name,
                        f"semantic_{label_prefix}_vs_random_{label_prefix}",
                        semantic_vs_random,
                        n_perm=n_perm,
                        seed=seed_base + 3,
                    )
                )
    return rows


def build_condition_specs(args, controls):
    specs = []
    for control_name in controls:
        prefix = "semantic" if control_name == "semantic" else "random"
        vector_layers = {
            f"{prefix}_early_only": [int(args.early_target_layer)],
            f"{prefix}_late_only": [int(args.late_target_layer)],
            f"{prefix}_joint": [int(args.early_target_layer), int(args.late_target_layer)],
        }
        for condition, target_layers in vector_layers.items():
            specs.append((condition, control_name, target_layers))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Phase 12C: joint L7 + L11 answer-adjacent synergy test")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--early-target-layer", type=int, default=7)
    parser.add_argument("--late-target-layer", type=int, default=11)
    parser.add_argument("--site", type=str, default="attn_output")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, default="add", help="add or ablate")
    parser.add_argument("--controls", type=str, default="semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--answer-offset", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--dataset-name", type=str, default="heldout_shared")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase12/phase12c_joint_l7_l11_synergy_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--pair-interaction-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    if args.mode not in {"add", "ablate"}:
        raise ValueError(f"Unsupported mode: {args.mode}")
    if int(args.early_target_layer) == int(args.late_target_layer):
        raise ValueError("Phase 12C requires distinct early and late target layers.")
    controls = parse_controls(args.controls)
    paths = phase12c_output_paths(args.output_csv)
    if args.detail_csv is not None:
        paths["detail"] = args.detail_csv
    if args.pair_interaction_csv is not None:
        paths["pair_interaction"] = args.pair_interaction_csv
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

    for prepared_item in tqdm(prepared_items, desc="Baseline", leave=False):
        scores = score_choice_logits(run_forward(model, prepared_item["prompt_ids"]), prepared_item)
        item_name = prepared_item["item"]["name"]
        baseline_by_item[item_name] = scores
        add_condition_row(
            detail_rows,
            baseline=scores,
            scores=scores,
            prepared_item=prepared_item,
            args=args,
            condition=CONDITION_BASELINE,
            control_name="none",
            target_layers=[],
        )

    for condition, control_name, target_layers in build_condition_specs(args, controls):
        vector = direction_bank[control_name]
        hooks = []
        for layer in target_layers:
            hook = TensorSiteAnswerWindowInterventionHook(
                vector=vector,
                alpha=args.alpha,
                mode=args.mode,
                answer_offset=args.answer_offset,
                window_size=args.window_size,
            )
            hook.attach(model, layer, args.site)
            hooks.append(hook)
        try:
            for prepared_item in tqdm(prepared_items, desc=condition, leave=False):
                item_name = prepared_item["item"]["name"]
                scores = score_choice_logits(run_forward(model, prepared_item["prompt_ids"]), prepared_item)
                add_condition_row(
                    detail_rows,
                    baseline=baseline_by_item[item_name],
                    scores=scores,
                    prepared_item=prepared_item,
                    args=args,
                    condition=condition,
                    control_name=control_name,
                    target_layers=target_layers,
                )
        finally:
            for hook in reversed(hooks):
                hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = build_summary_df(detail_df)
    pair_interaction_df = build_pair_interaction_df(detail_df)
    stats_df = pd.DataFrame(build_stats_rows(pair_interaction_df, n_perm=args.n_perm, seed=args.seed))

    summary_df.to_csv(paths["summary"], index=False)
    detail_df.to_csv(paths["detail"], index=False)
    pair_interaction_df.to_csv(paths["pair_interaction"], index=False)
    stats_df.to_csv(paths["stats"], index=False)

    primary_stats = stats_df[
        (stats_df["metric_name"] == "signed_label_margin")
        & stats_df["comparison_type"].isin([
            "semantic_joint_interaction_vs_zero",
            "semantic_joint_interaction_vs_random_joint_interaction",
            "semantic_joint_minus_best_single_vs_zero",
            "semantic_joint_minus_best_single_vs_random_joint_minus_best_single",
        ])
    ].copy()
    print("\n--- PHASE 12C SUMMARY ---")
    print(summary_df.to_string(index=False))
    print("\n--- PHASE 12C PRIMARY STATS ---")
    print(primary_stats.to_string(index=False))
    print(f"\nSummary saved to {paths['summary']}")
    print(f"Detail saved to {paths['detail']}")
    print(f"Pair interactions saved to {paths['pair_interaction']}")
    print(f"Stats saved to {paths['stats']}")


if __name__ == "__main__":
    main()