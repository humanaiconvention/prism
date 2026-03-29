"""Phase 9I: Causal triangle diagnostic.

Combines three complementary causal probes on the shared 24-pair benchmark:
- additive steering along the candidate semantic direction
- clean-to-corrupt activation patching
- semantic-direction ablation

The summary emphasizes whether a layer shows a consistent positive signal on
all three legs of the triangle rather than over-reading any single metric.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model, format_chatml_prompt
from scripts.phase9_semantic_utils import (
    infer_detail_csv,
    infer_summary_csv,
    load_semantic_direction,
    parse_int_list,
)
from scripts.run_phase9_activation_patching import (
    ResidualPatchHook,
    capture_residual,
    encode_choice_token,
    encode_prompt,
    load_swap_items,
    prepare_items,
    score_binary_choice,
)
from scripts.run_phase9_semantic_steering import ResidualInterventionHook, load_eval_items


def score_style_item(model, tokenizer, item):
    device = next(model.parameters()).device
    prompt_ids = encode_prompt(tokenizer, item["prompt"], device)
    math_letter = item["math_option"].strip().upper()
    creative_letter = "B" if math_letter == "A" else "A"
    math_token_id = encode_choice_token(tokenizer, math_letter)
    creative_token_id = encode_choice_token(tokenizer, creative_letter)

    with torch.inference_mode():
        logits = model(prompt_ids)
    logits = logits[0] if isinstance(logits, tuple) else logits
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

    math_lp = float(log_probs[0, math_token_id].item())
    creative_lp = float(log_probs[0, creative_token_id].item())
    pairwise_denom = np.exp(math_lp) + np.exp(creative_lp)
    pairwise_math_prob = float(np.exp(math_lp) / max(pairwise_denom, 1e-12))
    label_is_math = item["label"].strip().lower() == "math"
    label_prob = pairwise_math_prob if label_is_math else (1.0 - pairwise_math_prob)
    signed_margin = (math_lp - creative_lp) * (1.0 if label_is_math else -1.0)

    return {
        "math_minus_creative_logprob": float(math_lp - creative_lp),
        "pairwise_math_prob": pairwise_math_prob,
        "label_probability": label_prob,
        "signed_label_margin": signed_margin,
        "label_correct": int(signed_margin >= 0.0),
    }


def evaluate_style_condition(model, tokenizer, item, layer, hook=None):
    if hook is not None:
        hook.attach(model, layer)
    try:
        return score_style_item(model, tokenizer, item)
    finally:
        if hook is not None:
            hook.remove()


def build_summary(results_df):
    summary_df = results_df.sort_values(["causal_consistent", "triangle_score"], ascending=[False, False]).reset_index(drop=True)
    summary_df["rank_by_triangle_score"] = np.arange(1, len(summary_df) + 1)
    summary_df["best_layer_by_triangle_score"] = (summary_df["rank_by_triangle_score"] == 1).astype(int)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Phase 9I: Causal triangle diagnostic")
    parser.add_argument("--layers", type=str, default="13,14,15,16,17")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/semantic_directions.json")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--steering-scale", type=float, default=12.5)
    parser.add_argument("--ablation-alpha", type=float, default=1.0)
    parser.add_argument("--patch-alpha", type=float, default=1.0)
    parser.add_argument("--patch-token-position", type=int, default=-1)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/causal_triangle_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    summary_csv = infer_summary_csv(args.output_csv)
    layers = parse_int_list(args.layers)

    style_items = load_eval_items(args.eval_json)
    swap_items = load_swap_items(args.eval_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_swaps = prepare_items(swap_items, tokenizer, device)

    print("\n=== PHASE 9I: CAUSAL TRIANGLE DIAGNOSTIC ===")
    print(f"Layers: {layers}")
    print(f"Steering scale: {args.steering_scale}")
    print(f"Ablation alpha: {args.ablation_alpha}")
    print(f"Patch alpha: {args.patch_alpha}")
    print(f"Eval items: {len(style_items)} steering / {len(prepared_swaps)} patching")

    detail_rows = []
    result_rows = []

    for layer in layers:
        semantic_vec = torch.tensor(
            load_semantic_direction(args.semantic_directions, layer, vector_key=args.vector_key),
            device=device,
            dtype=torch.float32,
        )

        steering_margin_effects = []
        steering_prob_effects = []
        ablation_margin_effects = []
        ablation_prob_effects = []
        steering_acc_effects = []
        ablation_acc_effects = []

        for item in tqdm(style_items, desc=f"Triangle style L{layer}", leave=False):
            baseline = evaluate_style_condition(model, tokenizer, item, layer)
            steered = evaluate_style_condition(
                model,
                tokenizer,
                item,
                layer,
                hook=ResidualInterventionHook(vector=semantic_vec, alpha=args.steering_scale, mode="add"),
            )
            ablated = evaluate_style_condition(
                model,
                tokenizer,
                item,
                layer,
                hook=ResidualInterventionHook(vector=semantic_vec, alpha=args.ablation_alpha, mode="ablate"),
            )

            steering_margin_effect = steered["signed_label_margin"] - baseline["signed_label_margin"]
            steering_prob_effect = steered["label_probability"] - baseline["label_probability"]
            ablation_margin_effect = baseline["signed_label_margin"] - ablated["signed_label_margin"]
            ablation_prob_effect = baseline["label_probability"] - ablated["label_probability"]

            steering_margin_effects.append(steering_margin_effect)
            steering_prob_effects.append(steering_prob_effect)
            ablation_margin_effects.append(ablation_margin_effect)
            ablation_prob_effects.append(ablation_prob_effect)
            steering_acc_effects.append(steered["label_correct"] - baseline["label_correct"])
            ablation_acc_effects.append(baseline["label_correct"] - ablated["label_correct"])

            detail_rows.append({
                "layer": layer,
                "family": "style",
                "item_name": item["name"],
                "baseline_signed_label_margin": baseline["signed_label_margin"],
                "baseline_label_probability": baseline["label_probability"],
                "baseline_label_correct": baseline["label_correct"],
                "steered_signed_label_margin": steered["signed_label_margin"],
                "steered_label_probability": steered["label_probability"],
                "steered_label_correct": steered["label_correct"],
                "ablation_signed_label_margin": ablated["signed_label_margin"],
                "ablation_label_probability": ablated["label_probability"],
                "ablation_label_correct": ablated["label_correct"],
                "steering_margin_effect": steering_margin_effect,
                "steering_prob_effect": steering_prob_effect,
                "ablation_margin_effect": ablation_margin_effect,
                "ablation_prob_effect": ablation_prob_effect,
            })

        patch_margin_effects = []
        patch_prob_effects = []
        patch_acc_effects = []
        restoration_fractions = []
        contrast_valids = []

        for item_idx, cached in enumerate(prepared_swaps):
            clean_metrics = score_binary_choice(model, cached["clean_prompt_ids"], cached["clean_token_id"], cached["corrupt_token_id"])
            corrupt_metrics = score_binary_choice(model, cached["corrupt_prompt_ids"], cached["clean_token_id"], cached["corrupt_token_id"])
            source_vector = capture_residual(model, cached["clean_prompt_ids"], layer, args.patch_token_position)

            patch_hook = ResidualPatchHook(
                source_vector=source_vector,
                alpha=args.patch_alpha,
                position=args.patch_token_position,
                control="clean",
                seed=args.seed + (1000 * layer) + item_idx,
            )
            patch_hook.attach(model, layer)
            try:
                patched_metrics = score_binary_choice(
                    model,
                    cached["corrupt_prompt_ids"],
                    cached["clean_token_id"],
                    cached["corrupt_token_id"],
                )
            finally:
                patch_hook.remove()

            clean_margin = clean_metrics["clean_minus_corrupt_logprob"]
            corrupt_margin = corrupt_metrics["clean_minus_corrupt_logprob"]
            patched_margin = patched_metrics["clean_minus_corrupt_logprob"]
            denom = clean_margin - corrupt_margin
            contrast_valid = int(denom > 1e-8)
            restoration_fraction = np.nan if not contrast_valid else (patched_margin - corrupt_margin) / denom
            patch_margin_effect = patched_margin - corrupt_margin
            patch_prob_effect = patched_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"]

            patch_margin_effects.append(patch_margin_effect)
            patch_prob_effects.append(patch_prob_effect)
            patch_acc_effects.append(patched_metrics["predicts_clean_option"] - corrupt_metrics["predicts_clean_option"])
            restoration_fractions.append(restoration_fraction)
            contrast_valids.append(contrast_valid)

            detail_rows.append({
                "layer": layer,
                "family": "patch",
                "item_name": cached["name"],
                "baseline_signed_label_margin": np.nan,
                "baseline_label_probability": np.nan,
                "baseline_label_correct": np.nan,
                "steered_signed_label_margin": np.nan,
                "steered_label_probability": np.nan,
                "steered_label_correct": np.nan,
                "ablation_signed_label_margin": np.nan,
                "ablation_label_probability": np.nan,
                "ablation_label_correct": np.nan,
                "steering_margin_effect": np.nan,
                "steering_prob_effect": np.nan,
                "ablation_margin_effect": np.nan,
                "ablation_prob_effect": np.nan,
                "clean_margin": clean_margin,
                "corrupt_margin": corrupt_margin,
                "patched_margin": patched_margin,
                "patch_margin_effect": patch_margin_effect,
                "patch_prob_effect": patch_prob_effect,
                "restoration_fraction": restoration_fraction,
                "contrast_valid": contrast_valid,
                "corrupt_predicts_clean": corrupt_metrics["predicts_clean_option"],
                "patched_predicts_clean": patched_metrics["predicts_clean_option"],
            })

        mean_steering_margin_effect = float(np.mean(steering_margin_effects))
        mean_patch_margin_effect = float(np.mean(patch_margin_effects))
        mean_ablation_margin_effect = float(np.mean(ablation_margin_effects))
        mean_steering_prob_effect = float(np.mean(steering_prob_effects))
        mean_patch_prob_effect = float(np.mean(patch_prob_effects))
        mean_ablation_prob_effect = float(np.mean(ablation_prob_effects))

        result_rows.append({
            "layer": layer,
            "mean_steering_margin_effect": mean_steering_margin_effect,
            "mean_patch_margin_effect": mean_patch_margin_effect,
            "mean_ablation_margin_effect": mean_ablation_margin_effect,
            "mean_steering_prob_effect": mean_steering_prob_effect,
            "mean_patch_prob_effect": mean_patch_prob_effect,
            "mean_ablation_prob_effect": mean_ablation_prob_effect,
            "mean_steering_accuracy_effect": float(np.mean(steering_acc_effects)),
            "mean_patch_accuracy_effect": float(np.mean(patch_acc_effects)),
            "mean_ablation_accuracy_effect": float(np.mean(ablation_acc_effects)),
            "mean_restoration_fraction": float(np.nanmean(restoration_fractions)),
            "contrast_valid_rate": float(np.mean(contrast_valids)),
            "triangle_score": mean_steering_margin_effect * mean_patch_margin_effect * mean_ablation_margin_effect,
            "triangle_prob_score": mean_steering_prob_effect * mean_patch_prob_effect * mean_ablation_prob_effect,
            "causal_consistent": int(
                (mean_steering_margin_effect > 0.0)
                and (mean_patch_margin_effect > 0.0)
                and (mean_ablation_margin_effect > 0.0)
            ),
        })

    detail_df = pd.DataFrame(detail_rows)
    results_df = pd.DataFrame(result_rows)
    summary_df = build_summary(results_df)

    results_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- CAUSAL TRIANGLE RESULTS ---")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()