import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_site_hooks import DEFAULT_L11_SITES, TensorSiteInterventionHook, parse_site_list
from scripts.phase9_semantic_utils import infer_detail_csv, load_semantic_direction, parse_int_list
from scripts.run_phase9_semantic_steering import load_anchor_direction, load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import (
    evaluate_prepared_item,
    format_position_label,
    prepare_eval_item,
)


def add_sign_aware_fields(prepared_item, metrics):
    metrics = dict(metrics)
    label_target_prob = metrics["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else metrics["pairwise_creative_prob"]
    metrics["label_target_pairwise_prob"] = float(label_target_prob)
    metrics["label_accuracy"] = float(metrics["label_correct"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 10E: L11 subcomponent steering sweep")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/l11_subcomponent_steering_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="11")
    parser.add_argument("--sites", type=str, default=",".join(DEFAULT_L11_SITES))
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, choices=["add", "ablate"], default="add")
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec_np = load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key)
    semantic_vec = torch.tensor(semantic_vec_np, device=device, dtype=torch.float32)
    random_vec = make_random_orthogonal_control(semantic_vec, seed=args.seed)
    anchor_direction = load_anchor_direction(args.data_dir, args.source_layer, allow_invalid_metadata=args.allow_invalid_metadata)
    items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[: args.max_eval_items]
    prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
    detail_rows = []
    targets = parse_int_list(args.target_layers)
    sites = parse_site_list(args.sites)

    baseline_rows = []
    for item in prepared:
        metrics = add_sign_aware_fields(
            item,
            evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction),
        )
        metrics.update({"control": "baseline"})
        baseline_rows.append(metrics)
        detail_rows.append(metrics.copy())
    baseline_df = pd.DataFrame(baseline_rows).set_index("item_name")

    for target_layer in targets:
        for site in sites:
            for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
                hook = TensorSiteInterventionHook(vector=vector, alpha=args.alpha, mode=args.mode, position_fraction=args.position_fraction)
                hook.attach(model, target_layer, site)
                try:
                    for item in prepared:
                        metrics = add_sign_aware_fields(
                            item,
                            evaluate_prepared_item(model, item, anchor_layer=args.source_layer, anchor_direction=anchor_direction),
                        )
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
                        detail_rows.append(metrics)
                finally:
                    hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    conditioned = detail_df[detail_df["control"] != "baseline"].copy()
    for column in [
        "signed_label_margin",
        "label_target_pairwise_prob",
        "label_accuracy",
        "anchor_cosine",
        "math_minus_creative_logprob",
    ]:
        conditioned[f"delta_from_baseline_{column}"] = conditioned.apply(
            lambda row: row[column] - baseline_df.loc[row["item_name"], column], axis=1
        )
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
        .sort_values(["target_layer", "delta_from_baseline_mean_signed_label_margin"], ascending=[True, False])
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    detail_path = Path(args.detail_csv) if args.detail_csv else infer_detail_csv(output_path)
    detail_df.to_csv(detail_path, index=False)
    print(summary.to_string(index=False))
    print(f"[saved] summary -> {output_path}")
    print(f"[saved] detail -> {detail_path}")


if __name__ == "__main__":
    main()