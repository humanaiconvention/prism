import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import resolve_answer_window_positions
from scripts.run_phase10_tail_conditioned_necessity import expand_pair_items, load_pair_items, pair_name_from_item_name
from scripts.run_phase9_recurrent_state_patching import (
    blend_states,
    is_gla_layer,
    reset_model_decode_state,
    sample_random_like,
    unpack_logits_and_cache,
)
from scripts.run_phase9_token_position_steering import prepare_eval_item


ALLOWED_CONTROLS = ("semantic", "random")
METRIC_SPECS = [
    "delta_from_baseline_signed_label_margin",
    "delta_from_baseline_label_target_pairwise_prob",
]
DONOR_STRATEGY = "same_label_leave_one_pair_out"
INTERVENTION_KIND = "bounded_recurrent_state_overwrite"
TARGET_LAYER_TYPE = "gla"


def phase13b_output_paths(output_csv):
    return {
        "summary": output_csv,
        "detail": infer_companion_csv(output_csv, "detail"),
        "pair_effect": infer_companion_csv(output_csv, "pair_effect"),
        "stats": infer_companion_csv(output_csv, "stats"),
        "selected_pairs": infer_companion_csv(output_csv, "selected_pairs"),
    }


def score_choice_logits(logits, prepared_item):
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    math_lp = float(log_probs[0, int(prepared_item["math_token_id"])].item())
    creative_lp = float(log_probs[0, int(prepared_item["creative_token_id"])].item())
    label_sign = float(prepared_item["label_sign"])
    margin = math_lp - creative_lp
    math_prob = float(torch.exp(log_probs[0, int(prepared_item["math_token_id"])]).item())
    creative_prob = float(torch.exp(log_probs[0, int(prepared_item["creative_token_id"])]).item())
    pairwise_denom = max(math_prob + creative_prob, 1e-12)
    return {
        "signed_label_margin": label_sign * margin,
        "label_target_pairwise_prob": float(math_prob / pairwise_denom) if label_sign > 0 else float(creative_prob / pairwise_denom),
        "label_accuracy": float((margin >= 0.0) if label_sign > 0 else (margin <= 0.0)),
        "math_minus_creative_logprob": margin,
    }


def validate_ranking_df(ranking_df):
    required = {
        "pair_name",
        "susceptibility_rank",
        "baseline_abs_signed_label_margin",
        "recurrent_polarity",
        "positive_family_count",
        "negative_family_count",
    }
    missing = required - set(ranking_df.columns)
    if missing:
        raise ValueError(f"Pair ranking CSV missing required columns: {sorted(missing)}")


def select_target_pairs(ranking_df, top_k, recurrent_polarity="positive_recurrent"):
    validate_ranking_df(ranking_df)
    filtered = ranking_df[ranking_df["recurrent_polarity"] == recurrent_polarity].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for recurrent_polarity={recurrent_polarity!r}")
    filtered = filtered.sort_values(
        ["susceptibility_rank", "baseline_abs_signed_label_margin"],
        ascending=[True, False],
    ).reset_index(drop=True)
    selected = filtered.head(int(top_k)).copy()
    if len(selected) < 2:
        raise ValueError("Phase 13B requires at least two selected pairs so leave-one-pair-out donors remain available.")
    selected["selected_subset_name"] = f"top{len(selected)}_{recurrent_polarity}"
    return selected


def resolve_patch_positions(prompt_token_count, answer_offset, window_size):
    raw_positions = resolve_answer_window_positions(prompt_token_count, answer_offset=answer_offset, window_size=window_size)
    patch_positions = [int(pos) for pos in raw_positions if int(pos) > 0]
    patch_answer_offsets = [int(prompt_token_count - pos) for pos in patch_positions]
    return patch_positions, patch_answer_offsets


def run_incremental_prompt_window(
    model,
    prompt_ids,
    *,
    capture_layer=None,
    patch_layer=None,
    patch_positions=None,
    source_window_states=None,
    patch_alpha=1.0,
    patch_source="semantic",
    capture_to_cpu=True,
    random_seed_base=None,
):
    patch_positions = [int(pos) for pos in (patch_positions or [])]
    source_window_states = list(source_window_states or [])
    patch_position_to_state = {
        int(position): source_window_states[idx]
        for idx, position in enumerate(patch_positions)
        if idx < len(source_window_states)
    }

    trajectory = []
    final_logits = None
    past_key_values = None

    reset_model_decode_state(model)

    with torch.inference_mode():
        for step_idx in range(prompt_ids.shape[1]):
            source_state = patch_position_to_state.get(int(step_idx)) if patch_layer is not None else None
            if step_idx > 0 and source_state is not None:
                current_states = model.get_segment_states()
                target_state = current_states.get(int(patch_layer))
                if target_state is not None:
                    if patch_source == "random":
                        if random_seed_base is not None:
                            local_seed = int(random_seed_base) + 997 * int(step_idx) + 31 * int(patch_layer)
                            torch.manual_seed(local_seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(local_seed)
                        patch_state = sample_random_like(source_state)
                    else:
                        patch_state = source_state
                    current_states[int(patch_layer)] = blend_states(target_state, patch_state, patch_alpha)
                    model.set_segment_states(current_states)

            model_output = model(
                prompt_ids[:, step_idx:step_idx + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            final_logits, past_key_values = unpack_logits_and_cache(model_output)
            if final_logits is None or past_key_values is None:
                raise RuntimeError("Cache-enabled forward did not return logits/past_key_values")

            if capture_layer is not None:
                states = model.get_segment_states()
                state = states.get(int(capture_layer))
                if state is None:
                    trajectory.append(None)
                else:
                    capture = state.detach().clone()
                    if capture_to_cpu:
                        capture = capture.cpu()
                    trajectory.append(capture)

    reset_model_decode_state(model)

    if final_logits is None:
        raise RuntimeError("Prompt produced no logits")
    return final_logits, trajectory


def build_source_window_for_item(baseline_records, target_item_name):
    target = baseline_records[target_item_name]
    donor_candidates = [
        record
        for item_name, record in baseline_records.items()
        if item_name != target_item_name
        and record["label"] == target["label"]
        and record["pair_name"] != target["pair_name"]
    ]
    if not donor_candidates:
        raise ValueError(f"No donor candidates available for {target_item_name}")

    window_states = []
    donor_pair_names = set()
    donor_item_names = set()
    donor_count_per_offset = []

    for answer_offset in target["patch_answer_offsets"]:
        donor_states = []
        for donor in donor_candidates:
            donor_position = int(donor["prompt_token_count"] - answer_offset)
            if donor_position < 0 or donor_position >= len(donor["trajectory"]):
                continue
            donor_state = donor["trajectory"][donor_position]
            if donor_state is None:
                continue
            donor_states.append(donor_state.float())
            donor_pair_names.add(donor["pair_name"])
            donor_item_names.add(donor["item_name"])
        if not donor_states:
            raise ValueError(f"No donor states available for {target_item_name} at answer_offset={answer_offset}")
        donor_count_per_offset.append(len(donor_states))
        window_states.append(torch.stack(donor_states, dim=0).mean(dim=0))

    return {
        "window_states": window_states,
        "donor_pair_count": len(donor_pair_names),
        "donor_item_count": len(donor_item_names),
        "mean_donor_count_per_offset": float(np.mean(donor_count_per_offset)),
        "source_state_mean_norm": float(np.mean([state.float().norm().item() for state in window_states])),
    }


def build_summary_df(detail_df):
    return (
        detail_df.groupby(
            [
                "dataset_name",
                "target_layer",
                "target_layer_type",
                "intervention_kind",
                "donor_strategy",
                "selected_subset_name",
                "control",
                "patch_alpha",
                "answer_offset",
                "window_size",
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
            mean_effective_window_size=("effective_window_size", "mean"),
            mean_donor_pair_count=("donor_pair_count", "mean"),
            mean_donor_item_count=("donor_item_count", "mean"),
            mean_source_state_mean_norm=("source_state_mean_norm", "mean"),
        )
        .sort_values(["answer_offset", "window_size", "control"])
        .reset_index(drop=True)
    )


def build_pair_effect_df(detail_df):
    pair_control_df = (
        detail_df.groupby(
            [
                "dataset_name",
                "pair_name",
                "susceptibility_rank",
                "recurrent_polarity",
                "target_layer",
                "target_layer_type",
                "intervention_kind",
                "donor_strategy",
                "selected_subset_name",
                "patch_alpha",
                "answer_offset",
                "window_size",
                "control",
            ],
            as_index=False,
        )
        .agg(
            n_items_per_pair=("item_name", "count"),
            baseline_abs_signed_label_margin=("baseline_abs_signed_label_margin", "first"),
            delta_from_baseline_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            donor_pair_count=("donor_pair_count", "mean"),
            donor_item_count=("donor_item_count", "mean"),
        )
    )
    group_cols = [
        "dataset_name",
        "pair_name",
        "susceptibility_rank",
        "recurrent_polarity",
        "target_layer",
        "target_layer_type",
        "intervention_kind",
        "donor_strategy",
        "selected_subset_name",
        "patch_alpha",
        "answer_offset",
        "window_size",
    ]
    rows = []
    for group_key, group in pair_control_df.groupby(group_cols, dropna=False):
        control_table = group.set_index("control")
        if not set(ALLOWED_CONTROLS).issubset(set(control_table.index)):
            continue
        row = dict(zip(group_cols, group_key))
        row["n_items_per_pair"] = int(control_table.loc["semantic", "n_items_per_pair"])
        row["baseline_abs_signed_label_margin"] = float(control_table.loc["semantic", "baseline_abs_signed_label_margin"])
        row["donor_pair_count"] = float(control_table.loc["semantic", "donor_pair_count"])
        row["donor_item_count"] = float(control_table.loc["semantic", "donor_item_count"])
        for metric_name in METRIC_SPECS:
            semantic_value = float(control_table.loc["semantic", metric_name])
            random_value = float(control_table.loc["random", metric_name])
            row[f"semantic_{metric_name}"] = semantic_value
            row[f"random_{metric_name}"] = random_value
            row[f"semantic_minus_random_{metric_name}"] = semantic_value - random_value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["susceptibility_rank", "pair_name"]).reset_index(drop=True)


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
        "dataset_name",
        "target_layer",
        "target_layer_type",
        "intervention_kind",
        "donor_strategy",
        "selected_subset_name",
        "patch_alpha",
        "answer_offset",
        "window_size",
    ]
    for group_key, group in pair_effect_df.groupby(group_cols, dropna=False):
        common = dict(zip(group_cols, group_key))
        for metric_idx, metric_name in enumerate(METRIC_SPECS):
            semantic = group[f"semantic_{metric_name}"].to_numpy(dtype=np.float64)
            random = group[f"random_{metric_name}"].to_numpy(dtype=np.float64)
            seed_base = int(seed) + (metric_idx + 1) * 1000 + 97 * int(common["answer_offset"]) + 13 * int(common["window_size"])
            semantic_zero = signflip_test(semantic, n_perm=n_perm, seed=seed_base + 1)
            random_zero = signflip_test(random, n_perm=n_perm, seed=seed_base + 2)
            semantic_vs_random = paired_signflip_test(semantic, random, n_perm=n_perm, seed=seed_base + 3)
            rows.append(_stats_row(common, metric_name, "semantic_vs_zero", semantic_zero, n_perm=n_perm, seed=seed_base + 1))
            rows.append(_stats_row(common, metric_name, "random_vs_zero", random_zero, n_perm=n_perm, seed=seed_base + 2))
            rows.append(_stats_row(common, metric_name, "semantic_vs_random", semantic_vs_random, n_perm=n_perm, seed=seed_base + 3))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Phase 13B: bounded upstream-state causal test on top 13A-susceptible held-out pairs"
    )
    parser.add_argument("--pair-ranking-csv", type=str, default="logs/phase13/phase13a_pair_susceptibility_pair_table.csv")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--dataset-name", type=str, default="heldout_shared")
    parser.add_argument("--target-layer", type=int, default=14)
    parser.add_argument("--answer-offset", type=int, default=2, help="Patch window ends this many tokens before the final prompt token")
    parser.add_argument("--window-size", type=int, default=4, help="Number of late-upstream prompt steps to overwrite")
    parser.add_argument("--patch-alpha", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recurrent-polarity", type=str, default="positive_recurrent")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-csv", type=str, default="logs/phase13/phase13b_bounded_state_summary.csv")
    args = parser.parse_args()

    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2 for leave-one-pair-out donor selection")
    if args.answer_offset < 1:
        raise ValueError("--answer-offset must be >= 1")
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")
    if not (0.0 <= float(args.patch_alpha) <= 1.0):
        raise ValueError("--patch-alpha must be in [0, 1]")

    output_paths = phase13b_output_paths(args.output_csv)
    for path in output_paths.values():
        ensure_parent_dir(path)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    ranking_df = pd.read_csv(args.pair_ranking_csv)
    selected_pairs_df = select_target_pairs(ranking_df, top_k=args.top_k, recurrent_polarity=args.recurrent_polarity)
    selected_pairs_df.to_csv(output_paths["selected_pairs"], index=False)
    selected_pair_names = set(selected_pairs_df["pair_name"].tolist())
    selected_subset_name = str(selected_pairs_df["selected_subset_name"].iloc[0])
    ranking_meta = selected_pairs_df.set_index("pair_name").to_dict(orient="index")

    pair_items = load_pair_items(args.eval_json, max_eval_items=args.max_eval_items)
    pair_items = [item for item in pair_items if item["name"] in selected_pair_names]
    if len(pair_items) != len(selected_pair_names):
        found = {item["name"] for item in pair_items}
        missing = sorted(selected_pair_names - found)
        raise ValueError(f"Selected pairs missing from eval JSON: {missing}")
    eval_items = expand_pair_items(pair_items)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    if not is_gla_layer(model, args.target_layer):
        raise ValueError(f"target_layer={args.target_layer} is not a GLA layer and cannot be used for recurrent-state overwrite")

    prepared_items = [prepare_eval_item(tokenizer, item, device) for item in eval_items]

    baseline_records = {}
    for prepared_item in tqdm(prepared_items, desc="Phase 13B baseline capture"):
        item = prepared_item["item"]
        item_name = item["name"]
        pair_name = pair_name_from_item_name(item_name)
        patch_positions, patch_answer_offsets = resolve_patch_positions(
            prepared_item["prompt_token_count"],
            args.answer_offset,
            args.window_size,
        )
        if not patch_positions:
            raise ValueError(
                f"No valid patch positions for {item_name}; prompt_token_count={prepared_item['prompt_token_count']}, "
                f"answer_offset={args.answer_offset}, window_size={args.window_size}"
            )

        baseline_logits, trajectory = run_incremental_prompt_window(
            model,
            prepared_item["prompt_ids"],
            capture_layer=args.target_layer,
        )
        baseline_metrics = score_choice_logits(baseline_logits, prepared_item)
        baseline_records[item_name] = {
            "item_name": item_name,
            "pair_name": pair_name,
            "label": item["label"],
            "prompt_token_count": int(prepared_item["prompt_token_count"]),
            "trajectory": trajectory,
            "patch_positions": patch_positions,
            "patch_answer_offsets": patch_answer_offsets,
            "baseline_metrics": baseline_metrics,
            "prepared_item": prepared_item,
        }

    detail_rows = []
    for item_idx, prepared_item in enumerate(tqdm(prepared_items, desc="Phase 13B bounded overwrite")):
        item = prepared_item["item"]
        item_name = item["name"]
        pair_name = pair_name_from_item_name(item_name)
        baseline_record = baseline_records[item_name]
        donor_bundle = build_source_window_for_item(baseline_records, item_name)
        ranking_row = ranking_meta[pair_name]

        for control_idx, control in enumerate(ALLOWED_CONTROLS):
            logits, _ = run_incremental_prompt_window(
                model,
                prepared_item["prompt_ids"],
                patch_layer=args.target_layer,
                patch_positions=baseline_record["patch_positions"],
                source_window_states=donor_bundle["window_states"],
                patch_alpha=args.patch_alpha,
                patch_source=control,
                random_seed_base=int(args.seed) + 100000 * (control_idx + 1) + 101 * item_idx,
            )
            metrics = score_choice_logits(logits, prepared_item)
            detail_rows.append(
                {
                    "dataset_name": args.dataset_name,
                    "pair_name": pair_name,
                    "item_name": item_name,
                    "label": item["label"],
                    "math_option": prepared_item["math_letter"],
                    "control": control,
                    "susceptibility_rank": int(ranking_row["susceptibility_rank"]),
                    "recurrent_polarity": str(ranking_row["recurrent_polarity"]),
                    "baseline_abs_signed_label_margin": float(ranking_row["baseline_abs_signed_label_margin"]),
                    "target_layer": int(args.target_layer),
                    "target_layer_type": TARGET_LAYER_TYPE,
                    "intervention_kind": INTERVENTION_KIND,
                    "donor_strategy": DONOR_STRATEGY,
                    "selected_subset_name": selected_subset_name,
                    "patch_alpha": float(args.patch_alpha),
                    "answer_offset": int(args.answer_offset),
                    "window_size": int(args.window_size),
                    "effective_window_size": int(len(baseline_record["patch_positions"])),
                    "patch_start_answer_offset": int(max(baseline_record["patch_answer_offsets"])),
                    "patch_end_answer_offset": int(min(baseline_record["patch_answer_offsets"])),
                    "donor_pair_count": int(donor_bundle["donor_pair_count"]),
                    "donor_item_count": int(donor_bundle["donor_item_count"]),
                    "mean_donor_count_per_offset": float(donor_bundle["mean_donor_count_per_offset"]),
                    "source_state_mean_norm": float(donor_bundle["source_state_mean_norm"]),
                    "prompt_token_count": int(prepared_item["prompt_token_count"]),
                    "baseline_signed_label_margin": float(baseline_record["baseline_metrics"]["signed_label_margin"]),
                    "baseline_label_target_pairwise_prob": float(baseline_record["baseline_metrics"]["label_target_pairwise_prob"]),
                    "baseline_label_accuracy": float(baseline_record["baseline_metrics"]["label_accuracy"]),
                    "signed_label_margin": float(metrics["signed_label_margin"]),
                    "label_target_pairwise_prob": float(metrics["label_target_pairwise_prob"]),
                    "label_accuracy": float(metrics["label_accuracy"]),
                    "math_minus_creative_logprob": float(metrics["math_minus_creative_logprob"]),
                    "delta_from_baseline_signed_label_margin": float(
                        metrics["signed_label_margin"] - baseline_record["baseline_metrics"]["signed_label_margin"]
                    ),
                    "delta_from_baseline_label_target_pairwise_prob": float(
                        metrics["label_target_pairwise_prob"] - baseline_record["baseline_metrics"]["label_target_pairwise_prob"]
                    ),
                    "delta_from_baseline_label_accuracy": float(
                        metrics["label_accuracy"] - baseline_record["baseline_metrics"]["label_accuracy"]
                    ),
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["susceptibility_rank", "item_name", "control"]).reset_index(drop=True)
    summary_df = build_summary_df(detail_df)
    pair_effect_df = build_pair_effect_df(detail_df)
    stats_df = pd.DataFrame(build_stats_rows(pair_effect_df, n_perm=args.n_perm, seed=args.seed))

    summary_df.to_csv(output_paths["summary"], index=False)
    detail_df.to_csv(output_paths["detail"], index=False)
    pair_effect_df.to_csv(output_paths["pair_effect"], index=False)
    stats_df.to_csv(output_paths["stats"], index=False)

    print(f"Saved selected pairs to {output_paths['selected_pairs']}")
    print(f"Saved Phase 13B summary to {output_paths['summary']}")
    print(f"Saved Phase 13B detail to {output_paths['detail']}")
    print(f"Saved Phase 13B pair effects to {output_paths['pair_effect']}")
    print(f"Saved Phase 13B stats to {output_paths['stats']}")

    primary_rows = stats_df[
        (stats_df["metric_name"] == "delta_from_baseline_signed_label_margin")
        & (stats_df["comparison_type"] == "semantic_vs_random")
    ]
    if not primary_rows.empty:
        row = primary_rows.iloc[0]
        print(
            "Primary semantic-vs-random result: "
            f"mean_difference={row['mean_difference']:.4f}, "
            f"ci95=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"p={row['pvalue']:.4g}, n_pairs={int(row['n_pairs'])}"
        )


if __name__ == "__main__":
    main()