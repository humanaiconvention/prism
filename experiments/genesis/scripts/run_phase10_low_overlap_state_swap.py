import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import TensorSiteInterventionHook, TensorSiteSwapHook
from scripts.phase9_semantic_utils import load_semantic_direction
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, prepare_eval_item
from scripts.run_phase10_state_conditional_swap import capture_site_state, cosine_or_zero


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def add_sign_aware_fields(prepared_item, metrics):
    metrics = dict(metrics)
    target_prob = metrics["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else metrics["pairwise_creative_prob"]
    metrics["label_target_pairwise_prob"] = float(target_prob)
    metrics["label_accuracy"] = float(metrics["label_correct"])
    return metrics


def split_family_id(item_name):
    return item_name.split("__", 1)[0]


def build_item_info(prepared_items):
    info = {}
    for item in prepared_items:
        name = item["item"]["name"]
        info[name] = {
            "label": item["item"]["label"].strip().lower(),
            "family_id": split_family_id(name),
        }
    return info


def candidate_names_for_swap(info, item_name, swap_type):
    meta = info[item_name]
    other_label = "creative" if meta["label"] == "math" else "math"
    candidates = []
    for other_name, other_meta in info.items():
        if other_name == item_name:
            continue
        if swap_type == "paired_opposite_label":
            if other_meta["family_id"] == meta["family_id"] and other_meta["label"] == other_label:
                candidates.append(other_name)
        elif swap_type == "cross_family_same_label":
            if other_meta["family_id"] != meta["family_id"] and other_meta["label"] == meta["label"]:
                candidates.append(other_name)
        elif swap_type == "cross_family_opposite_label":
            if other_meta["family_id"] != meta["family_id"] and other_meta["label"] == other_label:
                candidates.append(other_name)
        else:
            raise ValueError(f"Unsupported swap type: {swap_type}")
    return sorted(candidates)


def select_donor(current_name, candidates, state_cache, selector):
    if not candidates:
        return None, {
            "candidate_count": 0,
            "selected_rank": np.nan,
            "selected_cosine": np.nan,
            "candidate_min_cosine": np.nan,
            "candidate_q25_cosine": np.nan,
            "candidate_median_cosine": np.nan,
            "candidate_max_cosine": np.nan,
        }
    recipient = state_cache[current_name]
    scored = []
    for candidate in candidates:
        score = cosine_or_zero(recipient, state_cache[candidate])
        scored.append((float(score), candidate))
    scored.sort(key=lambda pair: (pair[0], pair[1]))
    cosines = np.array([pair[0] for pair in scored], dtype=np.float64)
    if selector == "min_cosine":
        selected_rank = 0
    else:
        raise ValueError(f"Unsupported donor selector: {selector}")
    selected_cosine, selected_name = scored[selected_rank]
    return selected_name, {
        "candidate_count": int(len(scored)),
        "selected_rank": int(selected_rank + 1),
        "selected_cosine": float(selected_cosine),
        "candidate_min_cosine": float(np.min(cosines)),
        "candidate_q25_cosine": float(np.quantile(cosines, 0.25)),
        "candidate_median_cosine": float(np.quantile(cosines, 0.5)),
        "candidate_max_cosine": float(np.max(cosines)),
    }


def build_state_aware_donor_map(info, state_cache, swap_types, selector):
    donor_map = {}
    donor_selection = {}
    for item_name in info:
        donor_map[item_name] = {"none": None}
        donor_selection[item_name] = {
            "none": {
                "candidate_count": 0,
                "selected_rank": np.nan,
                "selected_cosine": 1.0,
                "candidate_min_cosine": 1.0,
                "candidate_q25_cosine": 1.0,
                "candidate_median_cosine": 1.0,
                "candidate_max_cosine": 1.0,
            }
        }
        for swap_type in swap_types:
            if swap_type == "none":
                continue
            candidates = candidate_names_for_swap(info, item_name, swap_type)
            donor_name, selection_meta = select_donor(item_name, candidates, state_cache, selector)
            donor_map[item_name][swap_type] = donor_name
            donor_selection[item_name][swap_type] = selection_meta
    return donor_map, donor_selection


def evaluate_with_hooks(model, prepared_item, anchor_layer, anchor_direction, hooks):
    for hook, layer, site in hooks:
        hook.attach(model, layer, site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer=anchor_layer, anchor_direction=anchor_direction))
    finally:
        for hook, _, _ in reversed(hooks):
            hook.remove()


def assign_overlap_flags(detail_df, overlap_threshold):
    detail_df = detail_df.copy()
    detail_df["overlap_group_q50"] = np.nan
    detail_df["overlap_group_q25"] = np.nan
    detail_df["is_low_overlap_half"] = False
    detail_df["is_low_overlap_quartile"] = False
    detail_df["is_overlap_threshold"] = False
    baseline_rows = detail_df[(detail_df["control"] == "baseline") & (detail_df["swap_type"] != "none")].copy()
    flag_rows = []
    for (dataset_name, target_layer, swap_type), group in baseline_rows.groupby(["dataset_name", "target_layer", "swap_type"]):
        q50 = float(group["donor_recipient_cosine"].quantile(0.5))
        q25 = float(group["donor_recipient_cosine"].quantile(0.25))
        for _, row in group.iterrows():
            cosine = float(row["donor_recipient_cosine"])
            flag_rows.append(
                {
                    "dataset_name": dataset_name,
                    "target_layer": int(target_layer),
                    "swap_type": swap_type,
                    "item_name": row["item_name"],
                    "overlap_group_q50": q50,
                    "overlap_group_q25": q25,
                    "is_low_overlap_half": bool(cosine <= q50),
                    "is_low_overlap_quartile": bool(cosine <= q25),
                    "is_overlap_threshold": bool(cosine <= float(overlap_threshold)),
                }
            )
    if flag_rows:
        flags_df = pd.DataFrame(flag_rows)
        detail_df = detail_df.drop(columns=["overlap_group_q50", "overlap_group_q25", "is_low_overlap_half", "is_low_overlap_quartile", "is_overlap_threshold"])
        detail_df = detail_df.merge(flags_df, on=["dataset_name", "target_layer", "swap_type", "item_name"], how="left")
        for column in ["is_low_overlap_half", "is_low_overlap_quartile", "is_overlap_threshold"]:
            detail_df[column] = detail_df[column].astype("boolean").fillna(False).astype(bool)
    return detail_df


def expand_overlap_slices(detail_df, overlap_threshold):
    threshold_label = f"cos_le_{str(overlap_threshold).replace('.', 'p')}"
    frames = [detail_df.assign(overlap_slice="all")]
    for slice_name, flag_column in [
        ("low_overlap_half", "is_low_overlap_half"),
        ("low_overlap_quartile", "is_low_overlap_quartile"),
        (threshold_label, "is_overlap_threshold"),
    ]:
        subset = detail_df[detail_df[flag_column]].copy()
        if subset.empty:
            continue
        subset["overlap_slice"] = slice_name
        frames.append(subset)
    return pd.concat(frames, ignore_index=True)


def build_stats_rows(expanded_df):
    rows = []
    conditioned = expanded_df[expanded_df["control"].isin(["semantic", "rotated"])].copy()
    group_cols = ["dataset_name", "target_layer", "swap_type", "overlap_slice"]
    for group_key, group in conditioned.groupby(group_cols):
        dataset_name, target_layer, swap_type, overlap_slice = group_key
        pivot = group.pivot(index="item_name", columns="control", values="delta_from_condition_baseline_signed_label_margin").dropna()
        if pivot.empty or "semantic" not in pivot.columns or "rotated" not in pivot.columns:
            continue
        semantic = pivot["semantic"].to_numpy(dtype=np.float64)
        rotated = pivot["rotated"].to_numpy(dtype=np.float64)
        row = {
            "dataset_name": dataset_name,
            "target_layer": int(target_layer),
            "swap_type": swap_type,
            "overlap_slice": overlap_slice,
            "n_items": int(pivot.shape[0]),
        }
        row.update({f"semantic_{k}": v for k, v in signflip_test(semantic, seed=201).items() if k != "n"})
        row.update({f"rotated_{k}": v for k, v in signflip_test(rotated, seed=202).items() if k != "n"})
        row.update({f"semantic_vs_rotated_{k}": v for k, v in paired_signflip_test(semantic, rotated, seed=203).items() if k not in {"n", "mean_a", "mean_b"}})
        if swap_type != "none":
            none_group = conditioned[
                (conditioned["dataset_name"] == dataset_name)
                & (conditioned["target_layer"] == target_layer)
                & (conditioned["swap_type"] == "none")
                & (conditioned["control"] == "semantic")
                & (conditioned["overlap_slice"] == "all")
            ][["item_name", "delta_from_condition_baseline_signed_label_margin"]].rename(columns={"delta_from_condition_baseline_signed_label_margin": "none_semantic_delta"})
            merged = pivot[["semantic"]].reset_index().merge(none_group, on="item_name", how="inner")
            if not merged.empty:
                drop_stats = paired_signflip_test(
                    merged["semantic"].to_numpy(dtype=np.float64),
                    merged["none_semantic_delta"].to_numpy(dtype=np.float64),
                    seed=204,
                )
                row.update(
                    {
                        "semantic_effect_minus_noswap_mean": drop_stats["mean"],
                        "semantic_effect_minus_noswap_ci95_low": drop_stats["ci95_low"],
                        "semantic_effect_minus_noswap_ci95_high": drop_stats["ci95_high"],
                        "semantic_effect_minus_noswap_pvalue": drop_stats["pvalue"],
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 10M: low-overlap state-swap stress test at FoX attention outputs")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-jsons", type=str, default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json")
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/low_overlap_state_swap_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,11")
    parser.add_argument("--site", type=str, default="attn_output")
    parser.add_argument("--swap-types", type=str, default="none,paired_opposite_label,cross_family_same_label,cross_family_opposite_label")
    parser.add_argument("--donor-selector", type=str, default="min_cosine")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--overlap-threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    eval_jsons = parse_str_list(args.eval_jsons)
    dataset_labels = parse_str_list(args.dataset_labels)
    swap_types = parse_str_list(args.swap_types)
    target_layers = parse_int_list(args.target_layers)
    if len(eval_jsons) != len(dataset_labels):
        raise ValueError("--eval-jsons and --dataset-labels must have the same length")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec = torch.tensor(load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key), device=device, dtype=torch.float32)
    rotated_vec = make_random_orthogonal_control(semantic_vec, seed=args.seed) * torch.norm(semantic_vec)
    anchor_direction = load_anchor_direction(args.data_dir, args.source_layer, allow_invalid_metadata=args.allow_invalid_metadata)
    detail_rows = []

    for dataset_name, eval_json in zip(dataset_labels, eval_jsons):
        items = load_eval_items(eval_json)
        if args.max_eval_items is not None:
            items = items[: args.max_eval_items]
        prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
        item_info = build_item_info(prepared)
        baseline_metrics = {
            item["item"]["name"]: add_sign_aware_fields(item, evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction))
            for item in prepared
        }
        for target_layer in target_layers:
            state_cache = {
                item["item"]["name"]: capture_site_state(model, item["prompt_ids"], target_layer, args.site, args.position_fraction)
                for item in prepared
            }
            donor_map, donor_selection = build_state_aware_donor_map(item_info, state_cache, swap_types, args.donor_selector)
            for item in prepared:
                item_name = item["item"]["name"]
                recipient_state = state_cache[item_name]
                for swap_type in swap_types:
                    donor_name = donor_map[item_name].get(swap_type)
                    donor_state = state_cache.get(donor_name) if donor_name else None
                    donor_meta = item_info.get(donor_name, {}) if donor_name else {}
                    selection_meta = donor_selection[item_name].get(swap_type, {})
                    if swap_type != "none" and donor_state is None:
                        continue
                    if swap_type == "none":
                        condition_baseline = dict(baseline_metrics[item_name])
                    else:
                        condition_baseline = evaluate_with_hooks(
                            model,
                            item,
                            anchor_layer=args.source_layer,
                            anchor_direction=anchor_direction,
                            hooks=[(TensorSiteSwapHook(donor_state, position_fraction=args.position_fraction, norm_match=True), target_layer, args.site)],
                        )
                    baseline_row = dict(condition_baseline)
                    baseline_row.update(
                        {
                            "dataset_name": dataset_name,
                            "target_layer": target_layer,
                            "site": args.site,
                            "swap_type": swap_type,
                            "control": "baseline",
                            "donor_selector": args.donor_selector,
                            "donor_item_name": donor_name,
                            "donor_label": donor_meta.get("label"),
                            "donor_family_id": donor_meta.get("family_id"),
                            "recipient_family_id": item_info[item_name]["family_id"],
                            "family_match": float(donor_meta.get("family_id") == item_info[item_name]["family_id"]) if donor_name else 1.0,
                            "label_match": float(donor_meta.get("label") == item_info[item_name]["label"]) if donor_name else 1.0,
                            "donor_recipient_cosine": cosine_or_zero(donor_state, recipient_state) if donor_state is not None else 1.0,
                            "donor_norm": float(torch.norm(donor_state).item()) if donor_state is not None else float(torch.norm(recipient_state).item()),
                            "recipient_norm": float(torch.norm(recipient_state).item()),
                            "donor_to_recipient_norm_ratio": float(torch.norm(donor_state).item() / max(torch.norm(recipient_state).item(), 1e-8)) if donor_state is not None else 1.0,
                            "candidate_count": selection_meta.get("candidate_count", 0),
                            "selected_rank": selection_meta.get("selected_rank", np.nan),
                            "candidate_min_cosine": selection_meta.get("candidate_min_cosine", np.nan),
                            "candidate_q25_cosine": selection_meta.get("candidate_q25_cosine", np.nan),
                            "candidate_median_cosine": selection_meta.get("candidate_median_cosine", np.nan),
                            "candidate_max_cosine": selection_meta.get("candidate_max_cosine", np.nan),
                            "overlap_threshold": args.overlap_threshold,
                            "alpha": args.alpha,
                            "position_fraction": args.position_fraction,
                        }
                    )
                    detail_rows.append(baseline_row)
                    for control_name, vector in (("semantic", semantic_vec), ("rotated", rotated_vec)):
                        hooks = []
                        if donor_state is not None:
                            hooks.append((TensorSiteSwapHook(donor_state, position_fraction=args.position_fraction, norm_match=True), target_layer, args.site))
                        hooks.append((TensorSiteInterventionHook(vector=vector, alpha=args.alpha, mode="add", position_fraction=args.position_fraction), target_layer, args.site))
                        metrics = evaluate_with_hooks(
                            model,
                            item,
                            anchor_layer=args.source_layer,
                            anchor_direction=anchor_direction,
                            hooks=hooks,
                        )
                        metrics.update(
                            {
                                "dataset_name": dataset_name,
                                "target_layer": target_layer,
                                "site": args.site,
                                "swap_type": swap_type,
                                "control": control_name,
                                "donor_selector": args.donor_selector,
                                "donor_item_name": donor_name,
                                "donor_label": donor_meta.get("label"),
                                "donor_family_id": donor_meta.get("family_id"),
                                "recipient_family_id": item_info[item_name]["family_id"],
                                "family_match": float(donor_meta.get("family_id") == item_info[item_name]["family_id"]) if donor_name else 1.0,
                                "label_match": float(donor_meta.get("label") == item_info[item_name]["label"]) if donor_name else 1.0,
                                "donor_recipient_cosine": cosine_or_zero(donor_state, recipient_state) if donor_state is not None else 1.0,
                                "donor_norm": float(torch.norm(donor_state).item()) if donor_state is not None else float(torch.norm(recipient_state).item()),
                                "recipient_norm": float(torch.norm(recipient_state).item()),
                                "donor_to_recipient_norm_ratio": float(torch.norm(donor_state).item() / max(torch.norm(recipient_state).item(), 1e-8)) if donor_state is not None else 1.0,
                                "candidate_count": selection_meta.get("candidate_count", 0),
                                "selected_rank": selection_meta.get("selected_rank", np.nan),
                                "candidate_min_cosine": selection_meta.get("candidate_min_cosine", np.nan),
                                "candidate_q25_cosine": selection_meta.get("candidate_q25_cosine", np.nan),
                                "candidate_median_cosine": selection_meta.get("candidate_median_cosine", np.nan),
                                "candidate_max_cosine": selection_meta.get("candidate_max_cosine", np.nan),
                                "overlap_threshold": args.overlap_threshold,
                                "alpha": args.alpha,
                                "position_fraction": args.position_fraction,
                            }
                        )
                        for column in ["signed_label_margin", "label_target_pairwise_prob", "label_accuracy", "anchor_cosine", "math_minus_creative_logprob"]:
                            metrics[f"delta_from_condition_baseline_{column}"] = float(metrics[column] - condition_baseline[column])
                            metrics[f"delta_from_noswap_baseline_{column}"] = float(metrics[column] - baseline_metrics[item_name][column])
                        detail_rows.append(metrics)

    detail_df = assign_overlap_flags(pd.DataFrame(detail_rows), args.overlap_threshold)
    expanded_df = expand_overlap_slices(detail_df, args.overlap_threshold)
    conditioned = expanded_df[expanded_df["control"].isin(["semantic", "rotated"])].copy()
    summary = (
        conditioned.groupby(
            [
                "dataset_name",
                "target_layer",
                "site",
                "swap_type",
                "overlap_slice",
                "control",
                "donor_selector",
                "alpha",
                "position_fraction",
                "overlap_threshold",
            ],
            as_index=False,
        )
        .agg(
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            label_accuracy=("label_accuracy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            mean_donor_recipient_cosine=("donor_recipient_cosine", "mean"),
            mean_candidate_count=("candidate_count", "mean"),
            mean_donor_to_recipient_norm_ratio=("donor_to_recipient_norm_ratio", "mean"),
            delta_from_condition_baseline_mean_signed_label_margin=("delta_from_condition_baseline_signed_label_margin", "mean"),
            delta_from_condition_baseline_mean_label_target_pairwise_prob=("delta_from_condition_baseline_label_target_pairwise_prob", "mean"),
            delta_from_condition_baseline_label_accuracy=("delta_from_condition_baseline_label_accuracy", "mean"),
            delta_from_condition_baseline_mean_anchor_cosine=("delta_from_condition_baseline_anchor_cosine", "mean"),
            delta_from_noswap_baseline_mean_signed_label_margin=("delta_from_noswap_baseline_signed_label_margin", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["dataset_name", "target_layer", "swap_type", "overlap_slice", "control"])
    )
    stats_df = build_stats_rows(expanded_df)

    output_path = args.output_csv
    detail_path = args.detail_csv or infer_companion_csv(output_path, "detail")
    stats_path = args.stats_csv or infer_companion_csv(output_path, "stats")
    ensure_parent_dir(output_path)
    summary.to_csv(output_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    print(summary.to_string(index=False))
    if not stats_df.empty:
        print("\n[stats]")
        print(stats_df.to_string(index=False))
    print(f"[saved] summary -> {output_path}")
    print(f"[saved] detail -> {detail_path}")
    print(f"[saved] stats -> {stats_path}")


if __name__ == "__main__":
    main()