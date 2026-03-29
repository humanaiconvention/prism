"""Phase 9K: Counterfactual trajectory patching.

Injects a clean residual state into a corrupt prompt at a start layer, then
captures the resulting downstream residual trajectory to see whether it stays
closer to the clean path or relaxes back toward the corrupt path.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase9_semantic_utils import infer_detail_csv, infer_summary_csv, load_semantic_direction, parse_int_list
from scripts.run_phase9_activation_patching import (
    ResidualPatchHook,
    capture_residual,
    load_swap_items,
    prepare_items,
    score_binary_choice,
)


class LayerCaptureHook:
    def __init__(self):
        self.handle = None
        self.captured = None

    def _make_hook(self):
        def hook_fn(module, args):
            self.captured = args[0][:, -1, :].detach().clone()
            return None
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def capture_layers(model, prompt_ids, layers, patch_hook=None):
    hooks = []
    try:
        if patch_hook is not None:
            patch_hook.attach(model, min(layers))
        for layer in layers:
            hook = LayerCaptureHook()
            hook.attach(model, layer)
            hooks.append((layer, hook))
        with torch.inference_mode():
            model(prompt_ids)
        captures = {}
        for layer, hook in hooks:
            if hook.captured is None:
                raise RuntimeError(f"No residual captured for layer {layer}")
            captures[layer] = hook.captured.squeeze(0).detach().clone()
        return captures
    finally:
        if patch_hook is not None:
            patch_hook.remove()
        for _, hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description="Phase 9K: counterfactual trajectory patching")
    parser.add_argument("--layers", type=str, default="13,14,15,16,17")
    parser.add_argument("--start-layer", type=int, default=15)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/semantic_directions.json")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--patch-alpha", type=float, default=1.0)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/counterfactual_trajectory_results.csv")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = infer_detail_csv(args.output_csv)
    summary_csv = infer_summary_csv(args.output_csv)
    layers = parse_int_list(args.layers)
    capture_layers_list = [layer for layer in layers if layer >= args.start_layer]
    if args.start_layer not in capture_layers_list:
        raise ValueError("--start-layer must be included in --layers")

    items = load_swap_items(args.eval_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_items = prepare_items(items, tokenizer, device)

    layer_directions = {
        layer: torch.tensor(
            load_semantic_direction(args.semantic_directions, layer, vector_key=args.vector_key),
            device=device,
            dtype=torch.float32,
        )
        for layer in capture_layers_list
    }

    print("\n=== PHASE 9K: COUNTERFACTUAL TRAJECTORY PATCHING ===")
    print(f"Layers: {layers}")
    print(f"Start layer: {args.start_layer}")
    print(f"Capture layers: {capture_layers_list}")
    print(f"Eval items: {len(prepared_items)}")

    detail_rows = []
    for item_idx, item in enumerate(tqdm(prepared_items, desc="Trajectory items")):
        source_vector = capture_residual(model, item["clean_prompt_ids"], args.start_layer, -1)
        clean_captures = capture_layers(model, item["clean_prompt_ids"], capture_layers_list)
        corrupt_captures = capture_layers(model, item["corrupt_prompt_ids"], capture_layers_list)

        patch_hook = ResidualPatchHook(
            source_vector=source_vector,
            alpha=args.patch_alpha,
            position=-1,
            control="clean",
            seed=args.seed + item_idx,
        )
        patched_captures = capture_layers(model, item["corrupt_prompt_ids"], capture_layers_list, patch_hook=patch_hook)

        corrupt_metrics = score_binary_choice(model, item["corrupt_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
        patched_behavior_hook = ResidualPatchHook(
            source_vector=source_vector,
            alpha=args.patch_alpha,
            position=-1,
            control="clean",
            seed=args.seed + item_idx,
        )
        patched_behavior_hook.attach(model, args.start_layer)
        try:
            patched_metrics = score_binary_choice(model, item["corrupt_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
        finally:
            patched_behavior_hook.remove()

        for layer in capture_layers_list:
            clean_vec = clean_captures[layer]
            corrupt_vec = corrupt_captures[layer]
            patched_vec = patched_captures[layer]
            direction = layer_directions[layer]
            clean_distance = float(torch.norm(patched_vec - clean_vec).item())
            corrupt_distance = float(torch.norm(patched_vec - corrupt_vec).item())
            cosine_to_clean = float(torch.nn.functional.cosine_similarity(patched_vec.unsqueeze(0), clean_vec.unsqueeze(0)).item())
            cosine_to_corrupt = float(torch.nn.functional.cosine_similarity(patched_vec.unsqueeze(0), corrupt_vec.unsqueeze(0)).item())
            clean_activation = float(torch.dot(clean_vec, direction).item())
            corrupt_activation = float(torch.dot(corrupt_vec, direction).item())
            patched_activation = float(torch.dot(patched_vec, direction).item())

            detail_rows.append({
                "item_name": item["name"],
                "start_layer": args.start_layer,
                "layer": layer,
                "clean_distance": clean_distance,
                "corrupt_distance": corrupt_distance,
                "relative_clean_bias": corrupt_distance - clean_distance,
                "closer_to_clean": int(clean_distance < corrupt_distance),
                "cosine_to_clean": cosine_to_clean,
                "cosine_to_corrupt": cosine_to_corrupt,
                "clean_activation": clean_activation,
                "corrupt_activation": corrupt_activation,
                "patched_activation": patched_activation,
                "activation_shift_toward_clean": abs(corrupt_activation - clean_activation) - abs(patched_activation - clean_activation),
                "corrupt_margin": corrupt_metrics["clean_minus_corrupt_logprob"],
                "patched_margin": patched_metrics["clean_minus_corrupt_logprob"],
                "behavior_patch_effect": patched_metrics["clean_minus_corrupt_logprob"] - corrupt_metrics["clean_minus_corrupt_logprob"],
                "patched_predicts_clean": patched_metrics["predicts_clean_option"],
            })

    detail_df = pd.DataFrame(detail_rows)
    results_df = (
        detail_df.groupby(["start_layer", "layer"], as_index=False)
        .agg(
            mean_relative_clean_bias=("relative_clean_bias", "mean"),
            clean_convergence_rate=("closer_to_clean", "mean"),
            mean_cosine_to_clean=("cosine_to_clean", "mean"),
            mean_cosine_to_corrupt=("cosine_to_corrupt", "mean"),
            mean_activation_shift_toward_clean=("activation_shift_toward_clean", "mean"),
            mean_behavior_patch_effect=("behavior_patch_effect", "mean"),
            patched_clean_rate=("patched_predicts_clean", "mean"),
            n_items=("item_name", "count"),
        )
    )
    results_df["trajectory_state"] = np.where(
        results_df["mean_relative_clean_bias"] > 0.0,
        "toward_clean",
        "toward_corrupt",
    )

    terminal_layer = max(capture_layers_list)
    terminal = results_df[results_df["layer"] == terminal_layer].iloc[0]
    summary_df = pd.DataFrame([
        {
            "start_layer": args.start_layer,
            "terminal_layer": terminal_layer,
            "terminal_mean_relative_clean_bias": float(terminal["mean_relative_clean_bias"]),
            "terminal_clean_convergence_rate": float(terminal["clean_convergence_rate"]),
            "terminal_activation_shift_toward_clean": float(terminal["mean_activation_shift_toward_clean"]),
            "terminal_behavior_patch_effect": float(terminal["mean_behavior_patch_effect"]),
            "trajectory_verdict": "converges_toward_clean" if float(terminal["mean_relative_clean_bias"]) > 0.0 else "collapses_back_toward_corrupt",
        }
    ])

    results_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- COUNTERFACTUAL TRAJECTORY SUMMARY ---")
    print(results_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()