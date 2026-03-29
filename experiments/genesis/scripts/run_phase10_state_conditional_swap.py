import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import TensorSiteCaptureHook, TensorSiteInterventionHook, TensorSiteSwapHook
from scripts.phase9_semantic_utils import load_semantic_direction
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, prepare_eval_item


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


def cosine_or_zero(a, b):
    a_norm = float(torch.norm(a).item())
    b_norm = float(torch.norm(b).item())
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / (a_norm * b_norm))


def choose_next_donor(candidates, current_name):
    candidates = sorted([name for name in candidates if name != current_name])
    if not candidates:
        return None
    return candidates[0]


def build_swap_donor_map(prepared_items):
    info = {}
    for item in prepared_items:
        name = item["item"]["name"]
        info[name] = {
            "label": item["item"]["label"].strip().lower(),
            "family_id": split_family_id(name),
        }
    same_label = {}
    by_family = {}
    for name, meta in info.items():
        same_label.setdefault(meta["label"], []).append(name)
        by_family.setdefault(meta["family_id"], []).append(name)
    donor_map = {}
    for name, meta in info.items():
        other_label = "creative" if meta["label"] == "math" else "math"
        family_group = [candidate for candidate in by_family[meta["family_id"]] if candidate != name and info[candidate]["label"] == other_label]
        cross_same = [candidate for candidate in same_label[meta["label"]] if info[candidate]["family_id"] != meta["family_id"]]
        cross_opp = [candidate for candidate in same_label.get(other_label, []) if info[candidate]["family_id"] != meta["family_id"]]
        donor_map[name] = {
            "none": None,
            "paired_opposite_label": choose_next_donor(family_group, name),
            "cross_family_same_label": choose_next_donor(cross_same, name),
            "cross_family_opposite_label": choose_next_donor(cross_opp, name),
        }
    return donor_map, info


def evaluate_with_hooks(model, prepared_item, anchor_layer, anchor_direction, hooks):
    for hook, layer, site in hooks:
        hook.attach(model, layer, site)
    try:
        return add_sign_aware_fields(prepared_item, evaluate_prepared_item(model, prepared_item, anchor_layer=anchor_layer, anchor_direction=anchor_direction))
    finally:
        for hook, _, _ in reversed(hooks):
            hook.remove()


def build_stats_rows(detail_df):
    rows = []
    conditioned = detail_df[detail_df["control"].isin(["semantic", "rotated"])].copy()
    for (dataset_name, target_layer, swap_type), group in conditioned.groupby(["dataset_name", "target_layer", "swap_type"]):
        pivot = group.pivot(index="item_name", columns="control", values="delta_from_condition_baseline_signed_label_margin").dropna()
        if pivot.empty or "semantic" not in pivot.columns or "rotated" not in pivot.columns:
            continue
        semantic = pivot["semantic"].to_numpy(dtype=np.float64)
        rotated = pivot["rotated"].to_numpy(dtype=np.float64)
        row = {
            "dataset_name": dataset_name,
            "target_layer": int(target_layer),
            "swap_type": swap_type,
            "n_items": int(pivot.shape[0]),
        }
        row.update({f"semantic_{k}": v for k, v in signflip_test(semantic, seed=101).items() if k != "n"})
        row.update({f"rotated_{k}": v for k, v in signflip_test(rotated, seed=102).items() if k != "n"})
        row.update({f"semantic_vs_rotated_{k}": v for k, v in paired_signflip_test(semantic, rotated, seed=103).items() if k not in {"n", "mean_a", "mean_b"}})
        if swap_type != "none":
            none_group = conditioned[
                (conditioned["dataset_name"] == dataset_name)
                & (conditioned["target_layer"] == target_layer)
                & (conditioned["swap_type"] == "none")
                & (conditioned["control"] == "semantic")
            ][["item_name", "delta_from_condition_baseline_signed_label_margin"]].rename(columns={"delta_from_condition_baseline_signed_label_margin": "none_semantic_delta"})
            merged = pivot[["semantic"]].reset_index().merge(none_group, on="item_name", how="inner")
            if not merged.empty:
                drop_stats = paired_signflip_test(
                    merged["semantic"].to_numpy(dtype=np.float64),
                    merged["none_semantic_delta"].to_numpy(dtype=np.float64),
                    seed=104,
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
    parser = argparse.ArgumentParser(description="Phase 10L: state-conditional corridor swap at FoX attention outputs")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-jsons", type=str, default="prompts/phase9_shared_eval_heldout.json,prompts/phase10_ood_semantic_eval.json,prompts/phase10_ood_semantic_eval_family2.json")
    parser.add_argument("--dataset-labels", type=str, default="heldout_shared,ood_family1,ood_family2")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/state_conditional_swap_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="7,11")
    parser.add_argument("--site", type=str, default="attn_output")
    parser.add_argument("--swap-types", type=str, default="none,paired_opposite_label,cross_family_same_label")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--position-fraction", type=float, default=1.0)
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
        donor_map, item_info = build_swap_donor_map(prepared)
        baseline_metrics = {
            item["item"]["name"]: add_sign_aware_fields(item, evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction))
            for item in prepared
        }
        for target_layer in target_layers:
            state_cache = {
                item["item"]["name"]: capture_site_state(model, item["prompt_ids"], target_layer, args.site, args.position_fraction)
                for item in prepared
            }
            for item in prepared:
                item_name = item["item"]["name"]
                recipient_state = state_cache[item_name]
                for swap_type in swap_types:
                    donor_name = donor_map[item_name].get(swap_type)
                    donor_state = state_cache.get(donor_name) if donor_name else None
                    donor_meta = item_info.get(donor_name, {}) if donor_name else {}
                    if swap_type != "none" and donor_state is None:
                        continue
                    if swap_type == "none":
                        condition_baseline = dict(baseline_metrics[item_name])
                    else:
                        swap_hook = TensorSiteSwapHook(donor_state, position_fraction=args.position_fraction, norm_match=True)
                        condition_baseline = evaluate_with_hooks(
                            model,
                            item,
                            anchor_layer=args.source_layer,
                            anchor_direction=anchor_direction,
                            hooks=[(swap_hook, target_layer, args.site)],
                        )
                    baseline_row = dict(condition_baseline)
                    baseline_row.update(
                        {
                            "dataset_name": dataset_name,
                            "target_layer": target_layer,
                            "site": args.site,
                            "swap_type": swap_type,
                            "control": "baseline",
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
                                "alpha": args.alpha,
                                "position_fraction": args.position_fraction,
                            }
                        )
                        for column in ["signed_label_margin", "label_target_pairwise_prob", "label_accuracy", "anchor_cosine", "math_minus_creative_logprob"]:
                            metrics[f"delta_from_condition_baseline_{column}"] = float(metrics[column] - condition_baseline[column])
                            metrics[f"delta_from_noswap_baseline_{column}"] = float(metrics[column] - baseline_metrics[item_name][column])
                        detail_rows.append(metrics)

    detail_df = pd.DataFrame(detail_rows)
    conditioned = detail_df[detail_df["control"].isin(["semantic", "rotated"])].copy()
    summary = (
        conditioned.groupby(["dataset_name", "target_layer", "site", "swap_type", "control", "alpha", "position_fraction"], as_index=False)
        .agg(
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            label_accuracy=("label_accuracy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            mean_donor_recipient_cosine=("donor_recipient_cosine", "mean"),
            mean_donor_to_recipient_norm_ratio=("donor_to_recipient_norm_ratio", "mean"),
            delta_from_condition_baseline_mean_signed_label_margin=("delta_from_condition_baseline_signed_label_margin", "mean"),
            delta_from_condition_baseline_mean_label_target_pairwise_prob=("delta_from_condition_baseline_label_target_pairwise_prob", "mean"),
            delta_from_condition_baseline_label_accuracy=("delta_from_condition_baseline_label_accuracy", "mean"),
            delta_from_condition_baseline_mean_anchor_cosine=("delta_from_condition_baseline_anchor_cosine", "mean"),
            delta_from_noswap_baseline_mean_signed_label_margin=("delta_from_noswap_baseline_signed_label_margin", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["dataset_name", "target_layer", "swap_type", "control"])
    )
    stats_df = build_stats_rows(detail_df)

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