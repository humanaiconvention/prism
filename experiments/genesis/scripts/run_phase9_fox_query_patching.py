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
from scripts.phase9_semantic_utils import infer_detail_csv, parse_float_list, parse_int_list
from scripts.run_phase9_activation_patching import load_swap_items, prepare_items
from scripts.run_phase9_recurrent_state_patching import (
    reset_model_decode_state,
    sample_random_like,
    score_binary_choice_from_logits,
    unpack_logits_and_cache,
)


def is_fox_layer(model, layer):
    if layer < 0 or layer >= len(model.blocks):
        return False
    block = model.blocks[layer]
    attn = getattr(block, "attn", None)
    return bool(getattr(block, "use_full_attention", False)) and attn is not None and hasattr(attn, "q_proj")


def get_q_proj_module(model, layer):
    if not is_fox_layer(model, layer):
        raise ValueError(f"Layer {layer} is not a FoX layer with q_proj")
    return model.blocks[layer].attn.q_proj


def blend_query(current_query, source_query, alpha):
    source_query = source_query.to(device=current_query.device, dtype=current_query.dtype)
    if alpha <= 0.0:
        return current_query.detach().clone()
    if alpha >= 1.0:
        return source_query.detach().clone()
    return ((1.0 - alpha) * current_query + alpha * source_query).detach().clone()


class QueryCaptureHook:
    def __init__(self):
        self.handle = None
        self.trajectory = []

    def _make_hook(self):
        def hook_fn(module, args, output):
            self.trajectory.append(output.detach().clone())
            return None

        return hook_fn

    def attach(self, model, layer):
        self.handle = get_q_proj_module(model, layer).register_forward_hook(self._make_hook())

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

    def _make_hook(self):
        def hook_fn(module, args, output):
            if not self.clean_trajectory:
                self.step_idx += 1
                return output

            source_idx = min(self.step_idx, len(self.clean_trajectory) - 1)
            source_query = self.clean_trajectory[source_idx]
            if self.patch_source == "random":
                source_query = sample_random_like(source_query)
            patched = blend_query(output, source_query, self.alpha)
            self.step_idx += 1
            return patched

        return hook_fn

    def attach(self, model, layer):
        self.handle = get_q_proj_module(model, layer).register_forward_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_incremental_prompt(
    model,
    prompt_ids,
    capture_layers=None,
    patch_layer=None,
    clean_query_trajectory=None,
    patch_alpha=1.0,
    patch_source="clean",
):
    capture_layers = tuple(capture_layers or ())
    capture_hooks = {}
    patch_hook = None
    final_logits = None
    past_key_values = None

    reset_model_decode_state(model)

    try:
        for layer in capture_layers:
            hook = QueryCaptureHook()
            hook.attach(model, layer)
            capture_hooks[layer] = hook

        if patch_layer is not None:
            patch_hook = QueryPatchHook(clean_query_trajectory, alpha=patch_alpha, patch_source=patch_source)
            patch_hook.attach(model, patch_layer)

        with torch.inference_mode():
            for step_idx in range(prompt_ids.shape[1]):
                model_output = model(
                    prompt_ids[:, step_idx:step_idx + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                final_logits, past_key_values = unpack_logits_and_cache(model_output)
                if final_logits is None or past_key_values is None:
                    raise RuntimeError("Cache-enabled forward did not return logits/past_key_values")
    finally:
        if patch_hook is not None:
            patch_hook.remove()
        for hook in capture_hooks.values():
            hook.remove()

    if final_logits is None:
        raise RuntimeError("Prompt produced no logits")
    return final_logits, {layer: hook.trajectory for layer, hook in capture_hooks.items()}


def main():
    parser = argparse.ArgumentParser(description="Phase 9S: FoX query-vector patching diagnostic")
    parser.add_argument("--layers", type=str, default="15", help="Comma-separated FoX layer indices to test")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/fox_query_patching_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument(
        "--alphas",
        type=str,
        default="1.0",
        help="Comma-separated interpolation coefficients from corrupt query (0.0) to patch source (1.0)",
    )
    parser.add_argument(
        "--patch-source",
        type=str,
        default="clean",
        choices=["clean", "random"],
        help="Patch source to inject at the target FoX q_proj output",
    )
    parser.add_argument(
        "--require-exact-token-alignment",
        action="store_true",
        help="Only score items whose clean/corrupt prompts have identical token lengths",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    requested_layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    if not alphas:
        raise ValueError("At least one interpolation alpha is required")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("All interpolation alphas must be in [0, 1]")

    items = load_swap_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[:args.max_eval_items]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_items = prepare_items(items, tokenizer, device)
    alignment_matches = [item for item in prepared_items if item["clean_prompt_ids"].shape[1] == item["corrupt_prompt_ids"].shape[1]]
    if args.require_exact_token_alignment:
        prepared_items = alignment_matches

    valid_layers = [layer for layer in requested_layers if is_fox_layer(model, layer)]
    skipped_layers = [layer for layer in requested_layers if layer not in valid_layers]
    if skipped_layers:
        print(f"[Phase 9S] Skipping non-FoX layers with no q_proj path: {skipped_layers}")
    if not valid_layers:
        raise ValueError("No valid FoX layers were requested")
    if not prepared_items:
        raise ValueError("No evaluation items remain after filtering")

    print("\n=== PHASE 9S: FOX QUERY PATCHING ===")
    print(f"Requested layers: {requested_layers}")
    print(f"Active FoX layers: {valid_layers}")
    print(f"Eval items: {len(prepared_items)}")
    print(f"Exact-token alignments available: {len(alignment_matches)}/{len(items)}")
    print(f"Eval benchmark: {args.eval_json}")
    print(f"Patch source: {args.patch_source}")
    print(f"Interpolation alphas: {alphas}")
    print(f"Exact alignment only: {args.require_exact_token_alignment}")
    print("Note: patches FoX q_proj outputs during cache-enabled incremental replay.")

    detail_rows = []
    for item in tqdm(prepared_items, desc="FoX query patch items"):
        clean_len = int(item["clean_prompt_ids"].shape[1])
        corrupt_len = int(item["corrupt_prompt_ids"].shape[1])
        exact_token_alignment = int(clean_len == corrupt_len)

        clean_logits, clean_trajectories = run_incremental_prompt(model, item["clean_prompt_ids"], capture_layers=valid_layers)
        corrupt_logits, _ = run_incremental_prompt(model, item["corrupt_prompt_ids"])
        clean_metrics = score_binary_choice_from_logits(clean_logits, item["clean_token_id"], item["corrupt_token_id"])
        corrupt_metrics = score_binary_choice_from_logits(corrupt_logits, item["clean_token_id"], item["corrupt_token_id"])

        for layer in valid_layers:
            layer_trajectory = clean_trajectories.get(layer, [])
            for alpha in alphas:
                patched_logits, _ = run_incremental_prompt(
                    model,
                    item["corrupt_prompt_ids"],
                    patch_layer=layer,
                    clean_query_trajectory=layer_trajectory,
                    patch_alpha=alpha,
                    patch_source=args.patch_source,
                )
                patched_metrics = score_binary_choice_from_logits(
                    patched_logits,
                    item["clean_token_id"],
                    item["corrupt_token_id"],
                )
                detail_rows.append({
                    "layer": layer,
                    "alpha": alpha,
                    "patch_source": args.patch_source,
                    "item_name": item["name"],
                    "clean_option": item["clean_option"],
                    "corrupt_option": item["corrupt_option"],
                    "clean_prompt_tokens": clean_len,
                    "corrupt_prompt_tokens": corrupt_len,
                    "exact_token_alignment": exact_token_alignment,
                    "clean_margin": clean_metrics["clean_minus_corrupt_logprob"],
                    "corrupt_margin": corrupt_metrics["clean_minus_corrupt_logprob"],
                    "patched_margin": patched_metrics["clean_minus_corrupt_logprob"],
                    "patch_effect": patched_metrics["clean_minus_corrupt_logprob"] - corrupt_metrics["clean_minus_corrupt_logprob"],
                    "clean_pairwise_prob": clean_metrics["pairwise_clean_prob"],
                    "corrupt_pairwise_prob": corrupt_metrics["pairwise_clean_prob"],
                    "patched_pairwise_prob": patched_metrics["pairwise_clean_prob"],
                    "patch_prob_effect": patched_metrics["pairwise_clean_prob"] - corrupt_metrics["pairwise_clean_prob"],
                    "patched_entropy": patched_metrics["next_token_entropy"],
                    "patched_predicts_clean": patched_metrics["predicts_clean_option"],
                })

    detail_df = pd.DataFrame(detail_rows)
    summary_group_cols = ["layer"]
    if len(alphas) > 1 or args.patch_source != "clean":
        summary_group_cols.extend(["alpha", "patch_source"])
    summary_df = (
        detail_df.groupby(summary_group_cols, as_index=False)
        .agg(
            mean_patch_effect=("patch_effect", "mean"),
            mean_patch_prob_effect=("patch_prob_effect", "mean"),
            accuracy=("patched_predicts_clean", "mean"),
        )
        .sort_values(summary_group_cols)
    )

    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(args.output_csv, index=False)

    print("\n--- FOX QUERY PATCHING SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")


if __name__ == "__main__":
    main()