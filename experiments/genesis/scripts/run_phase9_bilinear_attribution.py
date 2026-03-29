import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase9_semantic_utils import infer_detail_csv, parse_float_list
from scripts.run_phase9_activation_patching import load_swap_items, prepare_items
from scripts.run_phase9_fox_query_patching import blend_query, get_q_proj_module, is_fox_layer
from scripts.run_phase9_recurrent_state_patching import (
    blend_states,
    is_gla_layer,
    reset_model_decode_state,
    sample_random_like,
    score_binary_choice_from_logits,
    unpack_logits_and_cache,
)


class QueryCaptureHook:
    def __init__(self):
        self.handle = None
        self.trajectory = []

    def attach(self, model, layer):
        def hook_fn(module, args, output):
            self.trajectory.append(output.detach().clone())
            return None

        self.handle = get_q_proj_module(model, layer).register_forward_hook(hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class QueryPatchHook:
    def __init__(self, clean_trajectory, alpha=1.0, patch_source="clean"):
        self.clean_trajectory = list(clean_trajectory or [])
        self.alpha = float(alpha)
        self.patch_source = patch_source
        self.step_idx = 0
        self.handle = None

    def attach(self, model, layer):
        def hook_fn(module, args, output):
            if not self.clean_trajectory:
                self.step_idx += 1
                return output
            source = self.clean_trajectory[min(self.step_idx, len(self.clean_trajectory) - 1)]
            if self.patch_source == "random":
                source = sample_random_like(source)
            self.step_idx += 1
            return blend_query(output, source, self.alpha)

        self.handle = get_q_proj_module(model, layer).register_forward_hook(hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_incremental_prompt(model, prompt_ids, capture_state_layer=None, capture_query_layer=None, patch_state_layer=None, clean_state_trajectory=None, state_alpha=1.0, state_source="clean", patch_query_layer=None, clean_query_trajectory=None, query_alpha=1.0, query_source="clean"):
    state_trajectory, past_key_values, final_logits = [], None, None
    capture_hook = QueryCaptureHook() if capture_query_layer is not None else None
    patch_hook = QueryPatchHook(clean_query_trajectory, alpha=query_alpha, patch_source=query_source) if patch_query_layer is not None else None
    reset_model_decode_state(model)
    try:
        if capture_hook is not None:
            capture_hook.attach(model, capture_query_layer)
        if patch_hook is not None:
            patch_hook.attach(model, patch_query_layer)
        with torch.inference_mode():
            for step_idx in range(prompt_ids.shape[1]):
                if patch_state_layer is not None and step_idx > 0 and clean_state_trajectory:
                    source = clean_state_trajectory[min(step_idx - 1, len(clean_state_trajectory) - 1)]
                    current_states = model.get_segment_states()
                    target = current_states.get(patch_state_layer)
                    if source is not None and target is not None:
                        patch = source if state_source == "clean" else sample_random_like(source)
                        current_states[patch_state_layer] = blend_states(target, patch, state_alpha)
                        model.set_segment_states(current_states)
                model_output = model(prompt_ids[:, step_idx:step_idx + 1], past_key_values=past_key_values, use_cache=True)
                final_logits, past_key_values = unpack_logits_and_cache(model_output)
                if final_logits is None or past_key_values is None:
                    raise RuntimeError("Cache-enabled forward did not return logits/past_key_values")
                if capture_state_layer is not None:
                    states = model.get_segment_states()
                    source = states.get(capture_state_layer)
                    state_trajectory.append(None if source is None else source.detach().clone())
    finally:
        if patch_hook is not None:
            patch_hook.remove()
        if capture_hook is not None:
            capture_hook.remove()
    if final_logits is None:
        raise RuntimeError("Prompt produced no logits")
    return final_logits, state_trajectory, ([] if capture_hook is None else capture_hook.trajectory)


def main():
    parser = argparse.ArgumentParser(description="Phase 9T: bilinear state-query interaction diagnostic")
    parser.add_argument("--state-layer", type=int, default=14)
    parser.add_argument("--query-layer", type=int, default=15)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/bilinear_attribution_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--alphas", type=str, default="1.0", help="Comma-separated coupled alpha values for both state and query patches")
    parser.add_argument("--state-source", type=str, default="clean", choices=["clean", "random"])
    parser.add_argument("--query-source", type=str, default="clean", choices=["clean", "random"])
    parser.add_argument("--require-exact-token-alignment", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    alphas = parse_float_list(args.alphas)
    if not alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("All alphas must be in [0, 1] and at least one alpha is required")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    items = load_swap_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[:args.max_eval_items]
    model, tokenizer, _ = load_genesis_model(device="cuda" if torch.cuda.is_available() else "cpu")
    prepared_items = prepare_items(items, tokenizer, next(model.parameters()).device)
    alignment_matches = [item for item in prepared_items if item["clean_prompt_ids"].shape[1] == item["corrupt_prompt_ids"].shape[1]]
    if args.require_exact_token_alignment:
        prepared_items = alignment_matches
    if not is_gla_layer(model, args.state_layer):
        raise ValueError(f"State layer {args.state_layer} is not a valid GLA layer")
    if not is_fox_layer(model, args.query_layer):
        raise ValueError(f"Query layer {args.query_layer} is not a valid FoX layer")
    if not prepared_items:
        raise ValueError("No evaluation items remain after filtering")

    print("\n=== PHASE 9T: BILINEAR STATE-QUERY ATTRIBUTION ===")
    print(f"State layer: {args.state_layer} | Query layer: {args.query_layer}")
    print(f"Eval items: {len(prepared_items)}")
    print(f"Exact-token alignments available: {len(alignment_matches)}/{len(items)}")
    print(f"Eval benchmark: {args.eval_json}")
    print(f"Alphas: {alphas}")
    print(f"State source: {args.state_source} | Query source: {args.query_source}")
    print(f"Exact alignment only: {args.require_exact_token_alignment}")

    detail_rows = []
    for item in tqdm(prepared_items, desc="Bilinear attribution items"):
        clean_len = int(item["clean_prompt_ids"].shape[1]); corrupt_len = int(item["corrupt_prompt_ids"].shape[1])
        clean_logits, clean_state_traj, clean_query_traj = run_incremental_prompt(model, item["clean_prompt_ids"], capture_state_layer=args.state_layer, capture_query_layer=args.query_layer)
        corrupt_logits, _, _ = run_incremental_prompt(model, item["corrupt_prompt_ids"])
        clean_metrics = score_binary_choice_from_logits(clean_logits, item["clean_token_id"], item["corrupt_token_id"])
        corrupt_metrics = score_binary_choice_from_logits(corrupt_logits, item["clean_token_id"], item["corrupt_token_id"])
        for alpha in alphas:
            state_logits, _, _ = run_incremental_prompt(model, item["corrupt_prompt_ids"], patch_state_layer=args.state_layer, clean_state_trajectory=clean_state_traj, state_alpha=alpha, state_source=args.state_source)
            query_logits, _, _ = run_incremental_prompt(model, item["corrupt_prompt_ids"], patch_query_layer=args.query_layer, clean_query_trajectory=clean_query_traj, query_alpha=alpha, query_source=args.query_source)
            joint_logits, _, _ = run_incremental_prompt(model, item["corrupt_prompt_ids"], patch_state_layer=args.state_layer, clean_state_trajectory=clean_state_traj, state_alpha=alpha, state_source=args.state_source, patch_query_layer=args.query_layer, clean_query_trajectory=clean_query_traj, query_alpha=alpha, query_source=args.query_source)
            state_metrics = score_binary_choice_from_logits(state_logits, item["clean_token_id"], item["corrupt_token_id"])
            query_metrics = score_binary_choice_from_logits(query_logits, item["clean_token_id"], item["corrupt_token_id"])
            joint_metrics = score_binary_choice_from_logits(joint_logits, item["clean_token_id"], item["corrupt_token_id"])
            detail_rows.append({
                "state_layer": args.state_layer, "query_layer": args.query_layer, "alpha": alpha, "state_source": args.state_source, "query_source": args.query_source,
                "item_name": item["name"], "clean_option": item["clean_option"], "corrupt_option": item["corrupt_option"], "clean_prompt_tokens": clean_len, "corrupt_prompt_tokens": corrupt_len,
                "exact_token_alignment": int(clean_len == corrupt_len), "clean_margin": clean_metrics["clean_minus_corrupt_logprob"], "corrupt_margin": corrupt_metrics["clean_minus_corrupt_logprob"],
                "state_margin": state_metrics["clean_minus_corrupt_logprob"], "query_margin": query_metrics["clean_minus_corrupt_logprob"], "joint_margin": joint_metrics["clean_minus_corrupt_logprob"],
                "state_effect": state_metrics["clean_minus_corrupt_logprob"] - corrupt_metrics["clean_minus_corrupt_logprob"], "query_effect": query_metrics["clean_minus_corrupt_logprob"] - corrupt_metrics["clean_minus_corrupt_logprob"],
                "joint_effect": joint_metrics["clean_minus_corrupt_logprob"] - corrupt_metrics["clean_minus_corrupt_logprob"], "interaction_effect": joint_metrics["clean_minus_corrupt_logprob"] - state_metrics["clean_minus_corrupt_logprob"] - query_metrics["clean_minus_corrupt_logprob"] + corrupt_metrics["clean_minus_corrupt_logprob"],
                "state_prob_effect": state_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"], "query_prob_effect": query_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"],
                "joint_prob_effect": joint_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"], "interaction_prob_effect": joint_metrics["pairwise_clean_prob"] - state_metrics["pairwise_clean_prob"] - query_metrics["pairwise_clean_prob"] + corrupt_metrics["pairwise_clean_prob"],
                "state_predicts_clean": state_metrics["predicts_clean_option"], "query_predicts_clean": query_metrics["predicts_clean_option"], "joint_predicts_clean": joint_metrics["predicts_clean_option"],
            })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = detail_df.groupby(["state_layer", "query_layer", "alpha", "state_source", "query_source"], as_index=False).agg(mean_state_effect=("state_effect", "mean"), mean_query_effect=("query_effect", "mean"), mean_joint_effect=("joint_effect", "mean"), mean_interaction_effect=("interaction_effect", "mean"), mean_state_prob_effect=("state_prob_effect", "mean"), mean_query_prob_effect=("query_prob_effect", "mean"), mean_joint_prob_effect=("joint_prob_effect", "mean"), mean_interaction_prob_effect=("interaction_prob_effect", "mean"), state_accuracy=("state_predicts_clean", "mean"), query_accuracy=("query_predicts_clean", "mean"), joint_accuracy=("joint_predicts_clean", "mean")).sort_values(["state_layer", "query_layer", "alpha", "state_source", "query_source"])
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(args.output_csv, index=False)
    print("\n--- BILINEAR ATTRIBUTION SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")


if __name__ == "__main__":
    main()