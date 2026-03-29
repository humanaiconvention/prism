import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase9_semantic_utils import infer_detail_csv, parse_float_list, parse_int_list
from scripts.run_phase9_activation_patching import load_swap_items, prepare_items


CACHE_ATTRS = (
    "_conv_cache_q",
    "_conv_cache_k",
    "_conv_cache_v",
    "_k_conv_cache",
    "_v_conv_cache",
    "_causal_cache",
)


def unpack_logits_and_cache(model_output):
    if isinstance(model_output, tuple):
        logits = model_output[0]
        past_key_values = model_output[3] if len(model_output) > 3 else None
        return logits, past_key_values
    return model_output, None


def score_binary_choice_from_logits(logits, clean_token_id, corrupt_token_id):
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    clean_lp = float(log_probs[0, int(clean_token_id)].item())
    corrupt_lp = float(log_probs[0, int(corrupt_token_id)].item())
    clean_prob = float(torch.exp(log_probs[0, int(clean_token_id)]).item())
    corrupt_prob = float(torch.exp(log_probs[0, int(corrupt_token_id)]).item())
    pairwise_clean_prob = clean_prob / max(clean_prob + corrupt_prob, 1e-12)
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    entropy = float((-(probs * torch.log(probs + 1e-10)).sum(dim=-1)).item())
    margin = clean_lp - corrupt_lp
    return {
        "clean_logprob": clean_lp,
        "corrupt_logprob": corrupt_lp,
        "clean_minus_corrupt_logprob": margin,
        "pairwise_clean_prob": float(pairwise_clean_prob),
        "next_token_entropy": entropy,
        "predicts_clean_option": int(margin >= 0.0),
    }


def is_gla_layer(model, layer):
    if layer < 0 or layer >= len(model.blocks):
        return False
    return not bool(getattr(model.blocks[layer], "use_full_attention", False))


def reset_model_decode_state(model):
    with torch.inference_mode():
        if hasattr(model, "reset_segment_states"):
            model.reset_segment_states()
        for block in getattr(model, "blocks", []):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            for attr in CACHE_ATTRS:
                if hasattr(attn, attr):
                    delattr(attn, attr)


def blend_states(current_state, source_state, alpha):
    source_state = source_state.to(device=current_state.device, dtype=current_state.dtype)
    if alpha <= 0.0:
        return current_state.detach().clone()
    if alpha >= 1.0:
        return source_state.detach().clone()
    return ((1.0 - alpha) * current_state + alpha * source_state).detach().clone()


def sample_random_like(reference_state):
    ref = reference_state.detach().float()
    std = ref.std(unbiased=False).clamp_min(1e-6)
    sample = torch.randn_like(ref) * std + ref.mean()
    return sample.to(device=reference_state.device, dtype=reference_state.dtype)


def run_incremental_prompt(
    model,
    prompt_ids,
    capture_layers=None,
    patch_layer=None,
    clean_trajectory=None,
    patch_alpha=1.0,
    patch_source="clean",
):
    capture_layers = tuple(capture_layers or ())
    trajectory = []
    final_logits = None
    past_key_values = None

    reset_model_decode_state(model)

    with torch.inference_mode():
        for step_idx in range(prompt_ids.shape[1]):
            if patch_layer is not None and step_idx > 0 and clean_trajectory:
                source_idx = min(step_idx - 1, len(clean_trajectory) - 1)
                source_state = clean_trajectory[source_idx].get(patch_layer)
                if source_state is not None:
                    current_states = model.get_segment_states()
                    target_state = current_states.get(patch_layer)
                    if target_state is not None:
                        patch_state = source_state if patch_source == "clean" else sample_random_like(source_state)
                        current_states[patch_layer] = blend_states(target_state, patch_state, patch_alpha)
                        model.set_segment_states(current_states)

            model_output = model(
                prompt_ids[:, step_idx:step_idx + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            final_logits, past_key_values = unpack_logits_and_cache(model_output)
            if final_logits is None or past_key_values is None:
                raise RuntimeError("Cache-enabled forward did not return logits/past_key_values")

            if capture_layers:
                states = model.get_segment_states()
                trajectory.append({layer: states[layer].detach().clone() for layer in capture_layers if layer in states})

    if final_logits is None:
        raise RuntimeError("Prompt produced no logits")
    return final_logits, trajectory


def main():
    parser = argparse.ArgumentParser(description="Phase 9Q/9R: Recurrent state patching on GLA segment states")
    parser.add_argument("--layers", type=str, default="13,14,15,16,17", help="Comma-separated layer indices to test")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/state_patching_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument(
        "--alphas",
        type=str,
        default="1.0",
        help="Comma-separated interpolation coefficients from corrupt state (0.0) to patch source (1.0)",
    )
    parser.add_argument(
        "--patch-source",
        type=str,
        default="clean",
        choices=["clean", "random"],
        help="Patch source to inject at the target GLA layer",
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

    valid_layers = [layer for layer in requested_layers if is_gla_layer(model, layer)]
    skipped_layers = [layer for layer in requested_layers if layer not in valid_layers]
    if skipped_layers:
        print(f"[Phase 9Q] Skipping non-GLA layers with no recurrent state: {skipped_layers}")
    if not valid_layers:
        raise ValueError("No valid GLA layers were requested")
    if not prepared_items:
        raise ValueError("No evaluation items remain after filtering")

    print("\n=== PHASE 9Q: RECURRENT STATE PATCHING ===")
    print(f"Requested layers: {requested_layers}")
    print(f"Active GLA layers: {valid_layers}")
    print(f"Eval items: {len(prepared_items)}")
    print(f"Exact-token alignments available: {len(alignment_matches)}/{len(items)}")
    print(f"Eval benchmark: {args.eval_json}")
    print(f"Patch source: {args.patch_source}")
    print(f"Interpolation alphas: {alphas}")
    print(f"Exact alignment only: {args.require_exact_token_alignment}")
    print("Note: uses cache-enabled incremental forward to access GLA segment states.")

    detail_rows = []
    for item in tqdm(prepared_items, desc="State patch items"):
        clean_len = int(item["clean_prompt_ids"].shape[1])
        corrupt_len = int(item["corrupt_prompt_ids"].shape[1])
        exact_token_alignment = int(clean_len == corrupt_len)
        clean_logits, clean_trajectory = run_incremental_prompt(model, item["clean_prompt_ids"], capture_layers=valid_layers)
        corrupt_logits, _ = run_incremental_prompt(model, item["corrupt_prompt_ids"])
        clean_metrics = score_binary_choice_from_logits(clean_logits, item["clean_token_id"], item["corrupt_token_id"])
        corrupt_metrics = score_binary_choice_from_logits(corrupt_logits, item["clean_token_id"], item["corrupt_token_id"])

        for layer in valid_layers:
            for alpha in alphas:
                patched_logits, _ = run_incremental_prompt(
                    model,
                    item["corrupt_prompt_ids"],
                    patch_layer=layer,
                    clean_trajectory=clean_trajectory,
                    patch_alpha=alpha,
                    patch_source=args.patch_source,
                )
                patched_metrics = score_binary_choice_from_logits(patched_logits, item["clean_token_id"], item["corrupt_token_id"])
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

    print("\n--- RECURRENT STATE PATCHING SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")


if __name__ == "__main__":
    main()