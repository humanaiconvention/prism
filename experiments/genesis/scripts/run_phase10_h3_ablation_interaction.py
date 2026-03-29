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


def infer_interaction_csv(output_csv):
    root, ext = os.path.splitext(output_csv)
    return f"{root}_interaction{ext or '.csv'}"


def layer_type_name(layer):
    return "fox" if layer in FOX_LAYER_INDICES else "gla"


class OProjHeadColumnAblation:
    def __init__(self, model, layer, head, head_dim):
        self.model = model
        self.layer = int(layer)
        self.head = int(head)
        self.head_dim = int(head_dim)
        self.saved_slice = None

    def __enter__(self):
        attn = getattr(self.model.blocks[self.layer], "attn", None)
        if attn is None or not hasattr(attn, "o_proj") or not hasattr(attn.o_proj, "weight"):
            raise ValueError(f"Layer {self.layer} has no attn.o_proj weight to ablate")
        start = self.head * self.head_dim
        end = start + self.head_dim
        weight = attn.o_proj.weight
        if end > weight.shape[1]:
            raise ValueError(
                f"Head slice [{start}:{end}] exceeds o_proj input width {weight.shape[1]} at layer {self.layer}"
            )
        with torch.no_grad():
            self.saved_slice = weight[:, start:end].detach().clone()
            weight[:, start:end].zero_()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.saved_slice is None:
            return
        attn = self.model.blocks[self.layer].attn
        start = self.head * self.head_dim
        end = start + self.head_dim
        with torch.no_grad():
            attn.o_proj.weight[:, start:end].copy_(self.saved_slice)
        self.saved_slice = None


def add_condition_row(detail_rows, baseline, row, prepared_item, args, target_layer, position_label, condition, control_name):
    pos_idx = resolve_position_index(prepared_item["prompt_token_count"], args.position_fraction)
    label_target_prob = row["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else row["pairwise_creative_prob"]
    baseline_target_prob = (
        baseline["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else baseline["pairwise_creative_prob"]
    )
    enriched = dict(row)
    enriched.update(
        {
            "source_layer": args.source_layer,
            "target_layer": int(target_layer),
            "target_layer_type": layer_type_name(target_layer),
            "ablation_layer": args.ablation_layer,
            "ablation_head": args.ablation_head,
            "condition": condition,
            "control": control_name,
            "steering_enabled": int("steer" in condition),
            "head_ablation_enabled": int("head_ablate" in condition),
            "alpha": args.alpha if "steer" in condition else 0.0,
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
            "delta_from_baseline_signed_label_margin": row["signed_label_margin"] - baseline["signed_label_margin"],
            "baseline_label_accuracy": baseline["label_correct"],
            "delta_from_baseline_label_accuracy": row["label_correct"] - baseline["label_correct"],
            "baseline_label_target_pairwise_prob": baseline_target_prob,
            "delta_from_baseline_label_target_pairwise_prob": label_target_prob - baseline_target_prob,
            "baseline_anchor_cosine": baseline["anchor_cosine"],
            "delta_from_baseline_anchor_cosine": row["anchor_cosine"] - baseline["anchor_cosine"],
        }
    )
    detail_rows.append(enriched)


def main():
    parser = argparse.ArgumentParser(description="Phase 10B: L15-H3 ablation × steering interaction")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument(
        "--target-layers",
        type=str,
        default="11,15",
        help="Steering target layers. Defaults compare the strongest 10A corridor layer with the original local site.",
    )
    parser.add_argument("--ablation-layer", type=int, default=15)
    parser.add_argument("--ablation-head", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, default="add", help="add or ablate")
    parser.add_argument("--controls", type=str, default="semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase10/h3_ablation_interaction_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--interaction-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    if args.mode not in {"add", "ablate"}:
        raise ValueError(f"Unsupported mode: {args.mode}")

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    interaction_csv = args.interaction_csv or infer_interaction_csv(args.output_csv)
    target_layers = parse_int_list(args.target_layers)
    controls = [c.strip().lower() for c in args.controls.split(",") if c.strip()]

    validate_upstream_metadata(args.data_dir, allow_invalid_metadata=args.allow_invalid_metadata)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    anchor_direction = load_anchor_direction(
        args.data_dir,
        args.anchor_layer,
        allow_invalid_metadata=args.allow_invalid_metadata,
    )

    eval_items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        eval_items = eval_items[:args.max_eval_items]
    prepared_items = [prepare_eval_item(tokenizer, item, device) for item in eval_items]

    semantic_vec = torch.tensor(
        load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key),
        device=device,
        dtype=torch.float32,
    )
    direction_bank = {"semantic": semantic_vec}
    if "random" in controls:
        direction_bank["random"] = make_random_orthogonal_control(semantic_vec, args.seed + args.source_layer)

    baseline_by_item = {}
    for prepared_item in tqdm(prepared_items, desc="Baseline", leave=False):
        baseline_by_item[prepared_item["item"]["name"]] = evaluate_prepared_item(
            model, prepared_item, args.anchor_layer, anchor_direction
        )

    ablated_by_item = {}
    with OProjHeadColumnAblation(model, args.ablation_layer, args.ablation_head, config.head_dim):
        for prepared_item in tqdm(prepared_items, desc="Head ablation only", leave=False):
            ablated_by_item[prepared_item["item"]["name"]] = evaluate_prepared_item(
                model, prepared_item, args.anchor_layer, anchor_direction
            )

    position_label = format_position_label(args.position_fraction)
    print("\n=== PHASE 10B: H3 ABLATION × STEERING INTERACTION ===")
    print(f"Source layer: {args.source_layer}")
    print(f"Target layers: {target_layers}")
    print(f"Ablation head: L{args.ablation_layer}-H{args.ablation_head}")
    print(f"Mode: {args.mode}")
    print(f"Controls: {controls}")
    print(f"Alpha: {args.alpha}")
    print(f"Position: {position_label} ({args.position_fraction:.2f})")
    print(f"Eval items: {len(prepared_items)}")

    detail_rows = []
    for target_layer in target_layers:
        for prepared_item in prepared_items:
            item_name = prepared_item["item"]["name"]
            add_condition_row(
                detail_rows,
                baseline_by_item[item_name],
                baseline_by_item[item_name],
                prepared_item,
                args,
                target_layer,
                position_label,
                "baseline",
                "none",
            )
            add_condition_row(
                detail_rows,
                baseline_by_item[item_name],
                ablated_by_item[item_name],
                prepared_item,
                args,
                target_layer,
                position_label,
                "head_ablate_only",
                "none",
            )

        for control_name in tqdm(controls, desc=f"L{target_layer} controls", leave=False):
            if control_name not in direction_bank:
                raise ValueError(f"Unsupported control type: {control_name}")
            hook = ResidualPositionInterventionHook(
                direction_bank[control_name],
                alpha=args.alpha,
                mode=args.mode,
                position_fraction=args.position_fraction,
            )

            hook.attach(model, target_layer)
            try:
                for prepared_item in prepared_items:
                    steered = evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction)
                    add_condition_row(
                        detail_rows,
                        baseline_by_item[prepared_item["item"]["name"]],
                        steered,
                        prepared_item,
                        args,
                        target_layer,
                        position_label,
                        f"{control_name}_steer",
                        control_name,
                    )
            finally:
                hook.remove()

            hook.attach(model, target_layer)
            try:
                with OProjHeadColumnAblation(model, args.ablation_layer, args.ablation_head, config.head_dim):
                    for prepared_item in prepared_items:
                        both = evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction)
                        add_condition_row(
                            detail_rows,
                            baseline_by_item[prepared_item["item"]["name"]],
                            both,
                            prepared_item,
                            args,
                            target_layer,
                            position_label,
                            f"{control_name}_steer_plus_head_ablate",
                            control_name,
                        )
            finally:
                hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby(
            [
                "source_layer",
                "target_layer",
                "target_layer_type",
                "ablation_layer",
                "ablation_head",
                "condition",
                "control",
                "steering_enabled",
                "head_ablation_enabled",
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
        .sort_values(["target_layer", "condition"])
    )

    interaction_rows = []
    metric_specs = [
        ("math_minus_creative_logprob", "math_bias_logprob"),
        ("label_target_pairwise_prob", "label_target_pairwise_prob"),
        ("signed_label_margin", "signed_label_margin"),
        ("label_correct", "label_accuracy"),
        ("anchor_cosine", "anchor_cosine"),
    ]
    for target_layer in target_layers:
        baseline_subset = detail_df[
            (detail_df["target_layer"] == target_layer) & (detail_df["condition"] == "baseline")
        ]
        ablate_subset = detail_df[
            (detail_df["target_layer"] == target_layer) & (detail_df["condition"] == "head_ablate_only")
        ]
        for control_name in controls:
            steer_subset = detail_df[
                (detail_df["target_layer"] == target_layer) & (detail_df["condition"] == f"{control_name}_steer")
            ]
            both_subset = detail_df[
                (detail_df["target_layer"] == target_layer)
                & (detail_df["condition"] == f"{control_name}_steer_plus_head_ablate")
            ]
            merged = baseline_subset[["item_name"]].merge(
                baseline_subset[["item_name", *[m[0] for m in metric_specs]]].rename(
                    columns={m[0]: f"baseline_{m[1]}" for m in metric_specs}
                ),
                on="item_name",
            ).merge(
                ablate_subset[["item_name", *[m[0] for m in metric_specs]]].rename(
                    columns={m[0]: f"ablate_{m[1]}" for m in metric_specs}
                ),
                on="item_name",
            ).merge(
                steer_subset[["item_name", *[m[0] for m in metric_specs]]].rename(
                    columns={m[0]: f"steer_{m[1]}" for m in metric_specs}
                ),
                on="item_name",
            ).merge(
                both_subset[["item_name", *[m[0] for m in metric_specs]]].rename(
                    columns={m[0]: f"both_{m[1]}" for m in metric_specs}
                ),
                on="item_name",
            )
            row = {
                "source_layer": args.source_layer,
                "target_layer": int(target_layer),
                "target_layer_type": layer_type_name(target_layer),
                "ablation_layer": args.ablation_layer,
                "ablation_head": args.ablation_head,
                "control": control_name,
                "alpha": args.alpha,
                "position_label": position_label,
                "position_fraction": float(args.position_fraction),
                "n_items": int(len(merged)),
            }
            for _, metric_name in metric_specs:
                steer_effect = merged[f"steer_{metric_name}"] - merged[f"baseline_{metric_name}"]
                ablated_steer_effect = merged[f"both_{metric_name}"] - merged[f"ablate_{metric_name}"]
                interaction = ablated_steer_effect - steer_effect
                row[f"mean_steering_effect_{metric_name}"] = float(steer_effect.mean())
                row[f"mean_ablated_steering_effect_{metric_name}"] = float(ablated_steer_effect.mean())
                row[f"mean_interaction_{metric_name}"] = float(interaction.mean())
            interaction_rows.append(row)

    interaction_df = pd.DataFrame(interaction_rows).sort_values(["target_layer", "control"])
    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    interaction_df.to_csv(interaction_csv, index=False)

    print("\n--- H3 ABLATION × STEERING SUMMARY ---")
    print(summary_df.to_string(index=False))
    print("\n--- INTERACTION SUMMARY ---")
    print(interaction_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")
    print(f"Interaction saved to {interaction_csv}")


if __name__ == "__main__":
    main()