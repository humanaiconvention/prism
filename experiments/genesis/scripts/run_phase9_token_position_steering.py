import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import format_chatml_prompt, load_genesis_model
from scripts.phase9_semantic_utils import infer_detail_csv, load_semantic_direction, parse_float_list
from scripts.run_phase9_semantic_steering import (
    anchor_cosine,
    load_anchor_direction,
    load_eval_items,
    make_random_orthogonal_control,
    next_token_entropy,
    next_token_logprob,
    validate_upstream_metadata,
)


def format_position_label(position_fraction):
    if abs(position_fraction) < 1e-8:
        return "first"
    if abs(position_fraction - 1.0) < 1e-8:
        return "last"
    return f"frac_{position_fraction:.2f}"


def resolve_position_index(seq_len, position_fraction):
    if seq_len <= 1:
        return 0
    frac = float(np.clip(position_fraction, 0.0, 1.0))
    return int(round(frac * (seq_len - 1)))


class ResidualPositionInterventionHook:
    """Intervene on one selected prompt position at the input of a Genesis block."""

    def __init__(self, vector=None, alpha=0.0, mode="add", position_fraction=1.0):
        self.vector = vector
        self.alpha = float(alpha)
        self.mode = mode
        self.position_fraction = float(position_fraction)
        self.handle = None

    def _make_hook(self):
        def hook_fn(module, args):
            if self.vector is None or self.alpha == 0.0:
                return None
            x = args[0]
            pos = resolve_position_index(x.shape[1], self.position_fraction)
            x_mod = x.clone()
            if self.mode == "add":
                x_mod[:, pos, :] = x_mod[:, pos, :] + (self.alpha * self.vector)
                return (x_mod, *args[1:])
            if self.mode == "ablate":
                token_slice = x_mod[:, pos, :]
                coeff = torch.sum(token_slice * self.vector.unsqueeze(0), dim=-1, keepdim=True)
                proj = coeff * self.vector.unsqueeze(0)
                x_mod[:, pos, :] = token_slice - (self.alpha * proj)
                return (x_mod, *args[1:])
            return None

        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def prepare_eval_item(tokenizer, item, device):
    prompt_ids = torch.tensor(
        [tokenizer.encode(format_chatml_prompt(item["prompt"]))],
        device=device,
    )
    math_letter = item["math_option"].strip().upper()
    creative_letter = "B" if math_letter == "A" else "A"
    math_token_ids = tokenizer.encode(f" {math_letter}")
    creative_token_ids = tokenizer.encode(f" {creative_letter}")
    if len(math_token_ids) != 1 or len(creative_token_ids) != 1:
        raise ValueError(
            f"Choice tokens must be single-token encodings; got {math_letter}={math_token_ids}, "
            f"{creative_letter}={creative_token_ids}"
        )
    return {
        "item": item,
        "prompt_ids": prompt_ids,
        "prompt_token_count": int(prompt_ids.shape[1]),
        "math_letter": math_letter,
        "creative_letter": creative_letter,
        "math_token_id": int(math_token_ids[0]),
        "creative_token_id": int(creative_token_ids[0]),
        "label_sign": 1.0 if item["label"].strip().lower() == "math" else -1.0,
    }


def evaluate_prepared_item(model, prepared_item, anchor_layer, anchor_direction):
    item = prepared_item["item"]
    prompt_ids = prepared_item["prompt_ids"]

    math_lp = next_token_logprob(model, prompt_ids, prepared_item["math_token_id"])
    creative_lp = next_token_logprob(model, prompt_ids, prepared_item["creative_token_id"])
    entropy = next_token_entropy(model, prompt_ids)
    anchor = anchor_cosine(model, prompt_ids, anchor_layer, anchor_direction)

    pairwise_denom = np.exp(math_lp) + np.exp(creative_lp)
    pairwise_math_prob = float(np.exp(math_lp) / max(pairwise_denom, 1e-12))
    pairwise_creative_prob = float(np.exp(creative_lp) / max(pairwise_denom, 1e-12))
    math_minus_creative = float(math_lp - creative_lp)
    signed_margin = prepared_item["label_sign"] * math_minus_creative

    return {
        "item_name": item["name"],
        "label": item["label"],
        "math_option": prepared_item["math_letter"],
        "creative_option": prepared_item["creative_letter"],
        "prompt_token_count": prepared_item["prompt_token_count"],
        "math_logprob": math_lp,
        "creative_logprob": creative_lp,
        "math_minus_creative_logprob": math_minus_creative,
        "pairwise_math_prob": pairwise_math_prob,
        "pairwise_creative_prob": pairwise_creative_prob,
        "signed_label_margin": signed_margin,
        "label_correct": int((math_minus_creative >= 0) == (prepared_item["label_sign"] > 0)),
        "next_token_entropy": entropy,
        "anchor_cosine": anchor,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 9U: token-position / lead-time steering sweep")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, default="add", help="add or ablate")
    parser.add_argument("--controls", type=str, default="semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29)
    parser.add_argument(
        "--position-fractions",
        type=str,
        default="0.10,0.30,0.50,0.70,1.00",
        help="Comma-separated normalized prompt positions in [0,1].",
    )
    parser.add_argument(
        "--eval-json",
        type=str,
        default="prompts/phase9_shared_eval_heldout.json",
        help="Shared held-out benchmark by default.",
    )
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/token_position_steering_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    if args.mode not in {"add", "ablate"}:
        raise ValueError(f"Unsupported mode: {args.mode}")

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    position_fractions = parse_float_list(args.position_fractions)
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

    semantic_vec_np = load_semantic_direction(args.semantic_directions, args.layer, vector_key=args.vector_key)
    semantic_vec = torch.tensor(semantic_vec_np, device=device, dtype=torch.float32)
    direction_bank = {"semantic": semantic_vec}
    if "random" in controls:
        direction_bank["random"] = make_random_orthogonal_control(semantic_vec, args.seed + args.layer)

    baseline_by_item = {}
    for prepared_item in tqdm(prepared_items, desc="Baseline", leave=False):
        baseline_by_item[prepared_item["item"]["name"]] = evaluate_prepared_item(
            model,
            prepared_item,
            args.anchor_layer,
            anchor_direction,
        )

    print("\n=== PHASE 9U: TOKEN-POSITION / LEAD-TIME STEERING SWEEP ===")
    print(f"Layer: {args.layer}")
    print(f"Mode: {args.mode}")
    print(f"Controls: {controls}")
    print(f"Alpha: {args.alpha}")
    print(f"Position fractions: {position_fractions}")
    print(f"Eval items: {len(prepared_items)}")

    detail_rows = []
    for control_name in controls:
        if control_name not in direction_bank:
            raise ValueError(f"Unsupported control type: {control_name}")
        for position_fraction in tqdm(position_fractions, desc=f"{control_name} sweep", leave=False):
            hook = ResidualPositionInterventionHook(
                direction_bank[control_name],
                alpha=args.alpha,
                mode=args.mode,
                position_fraction=position_fraction,
            )
            hook.attach(model, args.layer)
            try:
                for prepared_item in prepared_items:
                    row = evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction)
                    baseline = baseline_by_item[prepared_item["item"]["name"]]
                    pos_idx = resolve_position_index(prepared_item["prompt_token_count"], position_fraction)
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
                            "layer": args.layer,
                            "control": control_name,
                            "mode": args.mode,
                            "alpha": args.alpha,
                            "position_fraction": float(position_fraction),
                            "position_label": format_position_label(position_fraction),
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
            ["layer", "control", "mode", "alpha", "position_label", "position_fraction"],
            as_index=False,
        )
        .agg(
            mean_position_index=("selected_position_index", "mean"),
            mean_normalized_position=("normalized_position", "mean"),
            mean_distance_to_answer=("distance_to_answer", "mean"),
            mean_math_bias_logprob=("math_minus_creative_logprob", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            label_accuracy=("label_correct", "mean"),
            mean_next_token_entropy=("next_token_entropy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            baseline_mean_math_bias_logprob=("baseline_math_minus_creative_logprob", "mean"),
            delta_from_baseline_mean_math_bias_logprob=("delta_from_baseline_math_minus_creative_logprob", "mean"),
            baseline_mean_label_target_pairwise_prob=("baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=(
                "delta_from_baseline_label_target_pairwise_prob",
                "mean",
            ),
            baseline_mean_signed_label_margin=("baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            baseline_label_accuracy=("baseline_label_accuracy", "mean"),
            delta_from_baseline_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            baseline_mean_anchor_cosine=("baseline_anchor_cosine", "mean"),
            delta_from_baseline_mean_anchor_cosine=("delta_from_baseline_anchor_cosine", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["control", "position_fraction"])
    )

    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    print("\n--- TOKEN-POSITION STEERING SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")


if __name__ == "__main__":
    main()