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
from scripts.run_phase9_activation_patching import capture_residual
from scripts.run_phase9_semantic_steering import (
    load_anchor_direction,
    load_eval_items,
    make_random_orthogonal_control,
    validate_upstream_metadata,
)
from scripts.run_phase9_token_position_steering import (
    evaluate_prepared_item,
    format_position_label,
    prepare_eval_item,
    resolve_position_index,
)


def layer_type_name(layer):
    return "fox" if layer in FOX_LAYER_INDICES else "gla"


class ResidualDecompositionHook:
    """Residual decomposition intervention at one prompt position."""

    def __init__(self, vector, condition, add_alpha=12.5, ablate_scale=1.0, position_fraction=1.0):
        self.vector = vector
        self.condition = condition
        self.add_alpha = float(add_alpha)
        self.ablate_scale = float(ablate_scale)
        self.position_fraction = float(position_fraction)
        self.handle = None

    def _make_hook(self):
        def hook_fn(module, args):
            x = args[0]
            pos = resolve_position_index(x.shape[1], self.position_fraction)
            x_mod = x.clone()
            token_slice = x_mod[:, pos, :]
            vec = self.vector.unsqueeze(0).to(device=token_slice.device, dtype=token_slice.dtype)
            coeff = torch.sum(token_slice * vec, dim=-1, keepdim=True)
            proj = coeff * vec

            if self.condition == "add":
                token_new = token_slice + (self.add_alpha * vec)
            elif self.condition == "axis_ablate":
                token_new = token_slice - (self.ablate_scale * proj)
            elif self.condition == "axis_ablate_plus_add":
                token_new = token_slice - (self.ablate_scale * proj) + (self.add_alpha * vec)
            elif self.condition == "parallel_only":
                token_new = proj
            else:
                raise ValueError(f"Unsupported condition: {self.condition}")

            x_mod[:, pos, :] = token_new
            return (x_mod, *args[1:])

        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def build_summary(detail_df):
    return (
        detail_df.groupby(
            [
                "source_layer",
                "target_layer",
                "target_layer_type",
                "control",
                "condition",
                "add_alpha",
                "ablate_scale",
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
            mean_baseline_axis_coeff=("baseline_axis_coeff", "mean"),
            mean_baseline_axis_abs_coeff=("baseline_axis_abs_coeff", "mean"),
            mean_baseline_axis_energy_fraction=("baseline_axis_energy_fraction", "mean"),
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
        .sort_values(["target_layer", "control", "condition"])
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 10D: orthogonal residual decomposition test")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layers", type=str, default="11,15")
    parser.add_argument("--add-alpha", type=float, default=12.5)
    parser.add_argument("--ablate-scale", type=float, default=1.0)
    parser.add_argument("--controls", type=str, default="semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--include-parallel-only", action="store_true")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase10/residual_decomposition_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    target_layers = parse_int_list(args.target_layers)
    controls = [c.strip().lower() for c in args.controls.split(",") if c.strip()]
    conditions = ["add", "axis_ablate", "axis_ablate_plus_add"]
    if args.include_parallel_only:
        conditions.append("parallel_only")

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

    residual_by_item_layer = {}
    for target_layer in tqdm(target_layers, desc="Capture baseline residuals", leave=False):
        for prepared_item in prepared_items:
            pos_idx = resolve_position_index(prepared_item["prompt_token_count"], args.position_fraction)
            residual_by_item_layer[(prepared_item["item"]["name"], target_layer)] = capture_residual(
                model,
                prepared_item["prompt_ids"],
                target_layer,
                pos_idx,
            )

    position_label = format_position_label(args.position_fraction)
    print("\n=== PHASE 10D: ORTHOGONAL RESIDUAL DECOMPOSITION ===")
    print(f"Source layer: {args.source_layer}")
    print(f"Target layers: {target_layers}")
    print(f"Controls: {controls}")
    print(f"Conditions: {conditions}")
    print(f"Add alpha: {args.add_alpha}")
    print(f"Ablate scale: {args.ablate_scale}")
    print(f"Position: {position_label} ({args.position_fraction:.2f})")
    print(f"Eval items: {len(prepared_items)}")

    detail_rows = []
    for control_name in controls:
        if control_name not in direction_bank:
            raise ValueError(f"Unsupported control type: {control_name}")
        vector = direction_bank[control_name]
        for target_layer in tqdm(target_layers, desc=f"{control_name} decomposition", leave=False):
            for condition in conditions:
                hook = ResidualDecompositionHook(
                    vector=vector,
                    condition=condition,
                    add_alpha=args.add_alpha,
                    ablate_scale=args.ablate_scale,
                    position_fraction=args.position_fraction,
                )
                hook.attach(model, target_layer)
                try:
                    for prepared_item in prepared_items:
                        item_name = prepared_item["item"]["name"]
                        baseline = baseline_by_item[item_name]
                        row = evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction)
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

                        baseline_residual = residual_by_item_layer[(item_name, target_layer)].to(vector.device)
                        axis_coeff = float(torch.dot(baseline_residual, vector).item())
                        residual_norm = float(torch.norm(baseline_residual).item())
                        axis_energy_fraction = float((axis_coeff ** 2) / max(residual_norm ** 2, 1e-10))

                        row.update(
                            {
                                "source_layer": args.source_layer,
                                "target_layer": target_layer,
                                "target_layer_type": layer_type_name(target_layer),
                                "control": control_name,
                                "condition": condition,
                                "add_alpha": args.add_alpha,
                                "ablate_scale": args.ablate_scale,
                                "position_fraction": float(args.position_fraction),
                                "position_label": position_label,
                                "selected_position_index": pos_idx,
                                "normalized_position": float(pos_idx / max(prepared_item["prompt_token_count"] - 1, 1)),
                                "distance_to_answer": int(prepared_item["prompt_token_count"] - 1 - pos_idx),
                                "label_target_pairwise_prob": label_target_prob,
                                "baseline_axis_coeff": axis_coeff,
                                "baseline_axis_abs_coeff": abs(axis_coeff),
                                "baseline_axis_energy_fraction": axis_energy_fraction,
                                "baseline_residual_norm": residual_norm,
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
    summary_df = build_summary(detail_df)

    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    print("\n--- RESIDUAL DECOMPOSITION SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")


if __name__ == "__main__":
    main()