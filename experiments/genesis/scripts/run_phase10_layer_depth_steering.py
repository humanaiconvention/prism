import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import FOX_LAYER_INDICES, load_genesis_model
from scripts.phase9_semantic_utils import infer_detail_csv, load_semantic_direction, parse_int_list
from scripts.run_phase9_semantic_steering import (
    load_anchor_direction,
    load_eval_items,
    make_random_orthogonal_control,
    validate_upstream_metadata,
)
from scripts.run_phase9_token_position_steering import (
    ResidualPositionInterventionHook,
    evaluate_prepared_item,
    format_position_label,
    prepare_eval_item,
    resolve_position_index,
)


def layer_type_name(layer):
    return "fox" if layer in FOX_LAYER_INDICES else "gla"


def main():
    parser = argparse.ArgumentParser(description="Phase 10A: layer-depth steering sweep")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--source-layer", type=int, default=15, help="Layer from which to load the steering vector")
    parser.add_argument(
        "--target-layers",
        type=str,
        default="1,7,11,15,19,23,27,29",
        help="Comma-separated target layers where the same source-layer vector is injected.",
    )
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, default="add", help="add or ablate")
    parser.add_argument("--controls", type=str, default="semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29)
    parser.add_argument("--position-fraction", type=float, default=1.0, help="Prompt position fraction; 1.0 = final prompt token")
    parser.add_argument(
        "--eval-json",
        type=str,
        default="prompts/phase9_shared_eval_heldout.json",
        help="Shared held-out benchmark by default.",
    )
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase10/layer_depth_steering_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    if args.mode not in {"add", "ablate"}:
        raise ValueError(f"Unsupported mode: {args.mode}")

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    target_layers = parse_int_list(args.target_layers)
    controls = [c.strip().lower() for c in args.controls.split(",") if c.strip()]

    validate_upstream_metadata(args.data_dir, allow_invalid_metadata=args.allow_invalid_metadata)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    anchor_direction = load_anchor_direction(
        args.data_dir,
        args.anchor_layer,
        allow_invalid_metadata=args.allow_invalid_metadata,
    )

    eval_items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        eval_items = eval_items[:args.max_eval_items]
    prepared_items = [prepare_eval_item(tokenizer, item, device) for item in eval_items]

    semantic_vec_np = load_semantic_direction(
        args.semantic_directions,
        args.source_layer,
        vector_key=args.vector_key,
    )
    semantic_vec = torch.tensor(semantic_vec_np, device=device, dtype=torch.float32)
    direction_bank = {"semantic": semantic_vec}
    if "random" in controls:
        direction_bank["random"] = make_random_orthogonal_control(semantic_vec, args.seed + args.source_layer)

    baseline_by_item = {}
    for prepared_item in tqdm(prepared_items, desc="Baseline", leave=False):
        baseline_by_item[prepared_item["item"]["name"]] = evaluate_prepared_item(
            model,
            prepared_item,
            args.anchor_layer,
            anchor_direction,
        )

    position_label = format_position_label(args.position_fraction)
    print("\n=== PHASE 10A: LAYER-DEPTH STEERING SWEEP ===")
    print(f"Source layer: {args.source_layer}")
    print(f"Target layers: {target_layers}")
    print(f"Mode: {args.mode}")
    print(f"Controls: {controls}")
    print(f"Alpha: {args.alpha}")
    print(f"Position: {position_label} ({args.position_fraction:.2f})")
    print(f"Eval items: {len(prepared_items)}")

    detail_rows = []
    for control_name in controls:
        if control_name not in direction_bank:
            raise ValueError(f"Unsupported control type: {control_name}")
        for target_layer in tqdm(target_layers, desc=f"{control_name} depth sweep", leave=False):
            hook = ResidualPositionInterventionHook(
                direction_bank[control_name],
                alpha=args.alpha,
                mode=args.mode,
                position_fraction=args.position_fraction,
            )
            hook.attach(model, target_layer)
            try:
                for prepared_item in prepared_items:
                    row = evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction)
                    baseline = baseline_by_item[prepared_item["item"]["name"]]
                    pos_idx = resolve_position_index(prepared_item["prompt_token_count"], args.position_fraction)
                    label_target_prob = (
                        row["pairwise_math_prob"]
                        if prepared_item["label_sign"] > 0
                        else row["pairwise_creative_prob"]
                    )
                    baseline_target_prob = (
                        baseline["pairwise_math_prob"]
                        if prepared_item["label_sign"] > 0
                        else baseline["pairwise_creative_prob"]
                    )
                    row.update(
                        {
                            "source_layer": args.source_layer,
                            "target_layer": target_layer,
                            "target_layer_type": layer_type_name(target_layer),
                            "control": control_name,
                            "mode": args.mode,
                            "alpha": args.alpha,
                            "position_fraction": float(args.position_fraction),
                            "position_label": position_label,
                            "selected_position_index": pos_idx,
                            "normalized_position": float(pos_idx / max(prepared_item["prompt_token_count"] - 1, 1)),
                            "distance_to_answer": int(prepared_item["prompt_token_count"] - 1 - pos_idx),
                            "label_target_pairwise_prob": label_target_prob,
                            "baseline_math_minus_creative_logprob": baseline["math_minus_creative_logprob"],
                            "delta_from_baseline_math_minus_creative_logprob": (
                                row["math_minus_creative_logprob"] - baseline["math_minus_creative_logprob"]
                            ),
                            "baseline_signed_label_margin": baseline["signed_label_margin"],
                            "delta_from_baseline_signed_label_margin": (
                                row["signed_label_margin"] - baseline["signed_label_margin"]
                            ),
                            "baseline_label_accuracy": baseline["label_correct"],
                            "delta_from_baseline_label_accuracy": row["label_correct"] - baseline["label_correct"],
                            "baseline_label_target_pairwise_prob": baseline_target_prob,
                            "delta_from_baseline_label_target_pairwise_prob": label_target_prob - baseline_target_prob,
                            "baseline_anchor_cosine": baseline["anchor_cosine"],
                            "delta_from_baseline_anchor_cosine": row["anchor_cosine"] - baseline["anchor_cosine"],
                        }
                    )
                    detail_rows.append(row)
            finally:
                hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby(
            [
                "source_layer",
                "target_layer",
                "target_layer_type",
                "control",
                "mode",
                "alpha",
                "position_label",
                "position_fraction",
            ],
            as_index=False,
        )
        .agg(
            mean_math_bias_logprob=("math_minus_creative_logprob", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            label_accuracy=("label_correct", "mean"),
            mean_next_token_entropy=("next_token_entropy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            baseline_mean_math_bias_logprob=("baseline_math_minus_creative_logprob", "mean"),
            delta_from_baseline_mean_math_bias_logprob=("delta_from_baseline_math_minus_creative_logprob", "mean"),
            baseline_mean_label_target_pairwise_prob=("baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            baseline_mean_signed_label_margin=("baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            baseline_label_accuracy=("baseline_label_accuracy", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            baseline_mean_anchor_cosine=("baseline_anchor_cosine", "mean"),
            delta_from_baseline_mean_anchor_cosine=("delta_from_baseline_anchor_cosine", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["control", "target_layer"])
    )

    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    print("\n--- LAYER-DEPTH STEERING SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")


if __name__ == "__main__":
    main()