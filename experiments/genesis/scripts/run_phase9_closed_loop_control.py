"""Phase 9L: Closed-loop behavioral control along the semantic direction.

This controller measures the current semantic activation and applies a feedback
correction toward a target level before the next-token decision is scored.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase9_semantic_utils import infer_summary_csv, load_semantic_direction, parse_float_list
from scripts.run_phase9_activation_patching import encode_choice_token, encode_prompt, resolve_position
from scripts.run_phase9_semantic_steering import load_eval_items


class ClosedLoopControlHook:
    def __init__(self, semantic_vec, target_level=0.0, control_strength=0.5, position=-1):
        self.semantic_vec = semantic_vec
        self.target_level = float(target_level)
        self.control_strength = float(control_strength)
        self.position = int(position)
        self.handle = None
        self.pre_activations = []
        self.post_activations = []

    def _make_hook(self):
        def hook_fn(module, args):
            x = args[0]
            idx = resolve_position(x.shape[1], self.position)
            current = x[:, idx, :]
            activation = torch.sum(current * self.semantic_vec.unsqueeze(0), dim=-1, keepdim=True)
            error = self.target_level - activation
            correction = self.control_strength * error * self.semantic_vec.unsqueeze(0)
            x_mod = x.clone()
            x_mod[:, idx, :] = current + correction
            post_activation = torch.sum(x_mod[:, idx, :] * self.semantic_vec.unsqueeze(0), dim=-1, keepdim=True)
            self.pre_activations.extend(activation.squeeze(-1).detach().cpu().tolist())
            self.post_activations.extend(post_activation.squeeze(-1).detach().cpu().tolist())
            return (x_mod, *args[1:])
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


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
    signed_label_margin = (math_lp - creative_lp) * (1.0 if label_is_math else -1.0)

    return {
        "math_minus_creative_logprob": float(math_lp - creative_lp),
        "pairwise_math_prob": pairwise_math_prob,
        "label_correct": int(signed_label_margin >= 0.0),
    }


def summarize_control(results_df):
    sorted_df = results_df.sort_values("target_level")
    behavioral_sensitivity = float(np.polyfit(sorted_df["target_level"], sorted_df["mean_logit_margin"], 1)[0])
    activation_tracking_corr = float(np.corrcoef(sorted_df["target_level"], sorted_df["mean_semantic_activation"])[0, 1])
    tracking_mae = float(np.mean(np.abs(sorted_df["mean_semantic_activation"] - sorted_df["target_level"])))
    return pd.DataFrame([
        {
            "layer": int(sorted_df["layer"].iloc[0]),
            "control_strength": float(sorted_df["control_strength"].iloc[0]),
            "control_stability": float(sorted_df["semantic_variance"].mean()),
            "behavioral_sensitivity": behavioral_sensitivity,
            "control_effectiveness": activation_tracking_corr,
            "tracking_mae": tracking_mae,
        }
    ])


def main():
    parser = argparse.ArgumentParser(description="Phase 9L: Closed-loop behavioral control")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/semantic_directions.json")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--target-levels", type=str, default="-3,-1,0,1,3")
    parser.add_argument("--control-strength", type=float, default=0.5)
    parser.add_argument("--patch-token-position", type=int, default=-1)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/closed_loop_control_results.csv")
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    summary_csv = infer_summary_csv(args.output_csv)
    target_levels = parse_float_list(args.target_levels)
    items = load_eval_items(args.eval_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec = torch.tensor(
        load_semantic_direction(args.semantic_directions, args.layer, vector_key=args.vector_key),
        device=device,
        dtype=torch.float32,
    )

    print("\n=== PHASE 9L: CLOSED-LOOP CONTROL ===")
    print(f"Layer: {args.layer}")
    print(f"Target levels: {target_levels}")
    print(f"Control strength: {args.control_strength}")
    print(f"Eval items: {len(items)}")

    rows = []
    for target_level in target_levels:
        activations = []
        margins = []
        probs = []
        accs = []
        for item in items:
            hook = ClosedLoopControlHook(
                semantic_vec=semantic_vec,
                target_level=target_level,
                control_strength=args.control_strength,
                position=args.patch_token_position,
            )
            hook.attach(model, args.layer)
            try:
                metrics = score_style_item(model, tokenizer, item)
            finally:
                hook.remove()

            realized_activation = float(np.mean(hook.post_activations)) if hook.post_activations else np.nan
            activations.append(realized_activation)
            margins.append(metrics["math_minus_creative_logprob"])
            probs.append(metrics["pairwise_math_prob"])
            accs.append(metrics["label_correct"])

        rows.append({
            "layer": args.layer,
            "target_level": float(target_level),
            "control_strength": float(args.control_strength),
            "mean_semantic_activation": float(np.mean(activations)),
            "semantic_variance": float(np.var(activations)),
            "mean_logit_margin": float(np.mean(margins)),
            "target_probability": float(np.mean(probs)),
            "accuracy": float(np.mean(accs)),
        })

    results_df = pd.DataFrame(rows)
    summary_df = summarize_control(results_df)

    results_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- CLOSED-LOOP CONTROL SUMMARY ---")
    print(results_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {args.output_csv}")
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()