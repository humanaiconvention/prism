"""Phase 9F: Causal residual-stream intervention evaluation.

Upgrades the earlier steering demo into a more reviewer-facing intervention stage:
- additive residual intervention r' = r + alpha * d
- feature ablation / projection removal along d
- layer sweeps
- random orthogonal control directions
- token-level causal metrics based on contrastive answer log-probabilities
- optional qualitative greedy generations as side artifacts
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure spectral_microscope is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import HiddenStateHook, load_genesis_model, format_chatml_prompt


BUILTIN_EVAL_ITEMS = [
    {
        "name": "math_style_1",
        "label": "math",
        "math_option": "A",
        "prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: give a precise mathematical explanation.\n"
            "A) derive the result step by step using formal reasoning\n"
            "B) describe it with vivid imagery and emotion\n"
            "Answer:"
        ),
    },
    {
        "name": "math_style_2",
        "label": "math",
        "math_option": "B",
        "prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: analyze the claim like a careful proof.\n"
            "A) tell it as a dreamlike scene in a forest\n"
            "B) justify it with explicit logical steps\n"
            "Answer:"
        ),
    },
    {
        "name": "math_style_3",
        "label": "math",
        "math_option": "A",
        "prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: answer analytically and with technical precision.\n"
            "A) define the variables and compute the conclusion\n"
            "B) evoke mood, color, and metaphor\n"
            "Answer:"
        ),
    },
    {
        "name": "creative_style_1",
        "label": "creative",
        "math_option": "A",
        "prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: continue as imaginative creative writing.\n"
            "A) derive the answer with equations and definitions\n"
            "B) paint the idea through scene, rhythm, and feeling\n"
            "Answer:"
        ),
    },
    {
        "name": "creative_style_2",
        "label": "creative",
        "math_option": "B",
        "prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: write like a poet, not a textbook.\n"
            "A) use metaphor, cadence, and sensory detail\n"
            "B) present a strict derivation with numbered steps\n"
            "Answer:"
        ),
    },
    {
        "name": "creative_style_3",
        "label": "creative",
        "math_option": "A",
        "prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: continue with imaginative storytelling.\n"
            "A) let the response unfold through character and atmosphere\n"
            "B) formalize the claim and prove it rigorously\n"
            "Answer:"
        ),
    },
]


BUILTIN_GENERATION_PROMPTS = [
    "Write a highly creative poem about the ocean.",
    "A magical journey into the dark forest reveals",
    "Compose a romantic verse about the moonlit sky:",
]


class ResidualInterventionHook:
    """Intervene on the residual stream at the input of a Genesis block."""

    def __init__(self, vector=None, alpha=0.0, mode="add"):
        self.vector = vector
        self.alpha = float(alpha)
        self.mode = mode
        self.handle = None

    def _make_hook(self):
        def hook_fn(module, args):
            if self.vector is None:
                return None
            x = args[0]
            if self.mode == "add":
                if self.alpha == 0.0:
                    return None
                x_mod = x.clone()
                x_mod[:, -1, :] = x_mod[:, -1, :] + (self.alpha * self.vector)
                return (x_mod, *args[1:])
            if self.mode == "ablate":
                if self.alpha == 0.0:
                    return None
                x_mod = x.clone()
                coeff = torch.sum(x_mod[:, -1, :] * self.vector.unsqueeze(0), dim=-1, keepdim=True)
                proj = coeff * self.vector.unsqueeze(0)
                x_mod[:, -1, :] = x_mod[:, -1, :] - (self.alpha * proj)
                return (x_mod, *args[1:])
            return None
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def infer_detail_csv(output_csv):
    root, ext = os.path.splitext(output_csv)
    return f"{root}_detail{ext or '.csv'}"


def infer_examples_json(output_csv):
    root, _ = os.path.splitext(output_csv)
    return f"{root}_examples.json"


def is_pair_eval_item(item):
    required = {"name", "math_prompt", "creative_prompt", "math_option"}
    return required.issubset(item.keys())


def expand_pair_eval_items(items):
    expanded = []
    for item in items:
        math_option = item["math_option"].strip().upper()
        expanded.extend([
            {
                "name": f"{item['name']}__math",
                "label": "math",
                "math_option": math_option,
                "prompt": item["math_prompt"],
            },
            {
                "name": f"{item['name']}__creative",
                "label": "creative",
                "math_option": math_option,
                "prompt": item["creative_prompt"],
            },
        ])
    return expanded


def load_eval_items(eval_json=None):
    if eval_json is None:
        return BUILTIN_EVAL_ITEMS
    with open(eval_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload["items"] if isinstance(payload, dict) else payload
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("Evaluation JSON must contain a non-empty list of items.")
    if all(is_pair_eval_item(item) for item in items):
        return expand_pair_eval_items(items)
    required = {"name", "label", "math_option", "prompt"}
    for idx, item in enumerate(items):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Eval item {idx} missing required keys: {sorted(missing)}")
    return items


def validate_upstream_metadata(data_dir, allow_invalid_metadata=False):
    metadata_path = Path(data_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"[WARN] No metadata found at {metadata_path}; skipping category-count validation.")
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    math_count = int(metadata.get("math_count", 0))
    creative_count = int(metadata.get("creative_count", 0))
    if (math_count <= 0 or creative_count <= 0) and not allow_invalid_metadata:
        raise ValueError(
            f"Invalid Phase 9 metadata in {metadata_path}: math_count={math_count}, "
            f"creative_count={creative_count}. Re-run extraction with valid balanced categories, "
            f"or pass --allow-invalid-metadata for smoke tests only."
        )
    if math_count <= 0 or creative_count <= 0:
        print(
            f"[WARN] Upstream metadata is unbalanced (math_count={math_count}, "
            f"creative_count={creative_count}). Proceeding because --allow-invalid-metadata was set."
        )
    return metadata


def load_layer_vector(vector_dir, layer, vector_key="delta_perp"):
    vector_path = Path(vector_dir) / f"layer_{layer}_vector.npz"
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    payload = np.load(vector_path)
    if vector_key not in payload:
        raise KeyError(f"Vector key '{vector_key}' not found in {vector_path}")
    vec = payload[vector_key].astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError(f"Vector {vector_key} in {vector_path} has near-zero norm.")
    return vec / norm


def load_anchor_direction(data_dir, anchor_layer, allow_invalid_metadata=False):
    stats_path = Path(data_dir) / f"layer_{anchor_layer}_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Anchor stats file not found: {stats_path}")
    payload = np.load(stats_path)
    math_centroid = payload["math_centroid"].astype(np.float32)
    creative_centroid = payload["creative_centroid"].astype(np.float32)
    delta = math_centroid - creative_centroid
    norm = np.linalg.norm(delta)
    if norm < 1e-8:
        if not allow_invalid_metadata:
            raise ValueError(
                f"Anchor direction at layer {anchor_layer} is near zero. "
                "Re-run extraction with valid category coverage or use --allow-invalid-metadata for smoke tests."
            )
        print("[WARN] Anchor delta is near zero; falling back to normalized math centroid.")
        delta = math_centroid
        norm = np.linalg.norm(delta)
    if norm < 1e-8:
        raise ValueError(f"Anchor representation at layer {anchor_layer} is near zero.")
    return delta / norm


def make_random_orthogonal_control(semantic_vec, seed):
    g = torch.Generator(device=semantic_vec.device)
    g.manual_seed(int(seed))
    ctrl = torch.randn(semantic_vec.shape, generator=g, device=semantic_vec.device, dtype=semantic_vec.dtype)
    ctrl = ctrl - torch.dot(ctrl, semantic_vec) * semantic_vec
    norm = torch.norm(ctrl)
    if norm < 1e-8:
        raise ValueError("Random control vector collapsed after orthogonalization.")
    return ctrl / norm


def unpack_logits(model_output):
    if not isinstance(model_output, tuple):
        return model_output
    return model_output[0]


def next_token_logprob(model, prefix_ids, token_id):
    with torch.inference_mode():
        logits = unpack_logits(model(prefix_ids))
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    return float(log_probs[0, int(token_id)].item())


def next_token_entropy(model, prefix_ids):
    with torch.inference_mode():
        logits = unpack_logits(model(prefix_ids))
    probs = F.softmax(logits[:, -1, :], dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return float(entropy.item())


def anchor_cosine(model, prefix_ids, anchor_layer, anchor_direction):
    hs_hook = HiddenStateHook.attach_to_model(model)
    try:
        with torch.inference_mode():
            model(prefix_ids)
        hidden = hs_hook.get_hidden_states()[anchor_layer].squeeze(0).numpy()
    finally:
        hs_hook.remove_all()
    return float(np.dot(hidden, anchor_direction) / (np.linalg.norm(hidden) + 1e-10))


def greedy_generate(model, tokenizer, prompt_text, max_new_tokens):
    device = next(model.parameters()).device
    prompt_ids = torch.tensor([tokenizer.encode(format_chatml_prompt(prompt_text))], device=device)
    curr_ids = prompt_ids
    past_key_values = None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            idx_cond = curr_ids[:, -1:] if past_key_values is not None else curr_ids
            output = model(idx_cond, past_key_values=past_key_values, use_cache=True)
            logits = output[0]
            past_key_values = output[3] if len(output) > 3 else None
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
    return tokenizer.decode(curr_ids[0][prompt_ids.shape[1]:].tolist())


def evaluate_item(model, tokenizer, item, anchor_layer, anchor_direction):
    prompt_ids = torch.tensor(
        [tokenizer.encode(format_chatml_prompt(item["prompt"]))],
        device=next(model.parameters()).device,
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

    math_lp = next_token_logprob(model, prompt_ids, math_token_ids[0])
    creative_lp = next_token_logprob(model, prompt_ids, creative_token_ids[0])
    entropy = next_token_entropy(model, prompt_ids)
    anchor = anchor_cosine(model, prompt_ids, anchor_layer, anchor_direction)

    pairwise_denom = np.exp(math_lp) + np.exp(creative_lp)
    pairwise_math_prob = float(np.exp(math_lp) / max(pairwise_denom, 1e-12))
    pairwise_creative_prob = float(np.exp(creative_lp) / max(pairwise_denom, 1e-12))
    math_minus_creative = float(math_lp - creative_lp)
    label_sign = 1.0 if item["label"].strip().lower() == "math" else -1.0
    signed_margin = label_sign * math_minus_creative

    return {
        "item_name": item["name"],
        "label": item["label"],
        "math_option": math_letter,
        "creative_option": creative_letter,
        "math_logprob": math_lp,
        "creative_logprob": creative_lp,
        "math_minus_creative_logprob": math_minus_creative,
        "pairwise_math_prob": pairwise_math_prob,
        "pairwise_creative_prob": pairwise_creative_prob,
        "signed_label_margin": signed_margin,
        "label_correct": int((math_minus_creative >= 0) == (label_sign > 0)),
        "next_token_entropy": entropy,
        "anchor_cosine": anchor,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 9F: Causal residual intervention evaluation")
    parser.add_argument("--vector-dir", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices. Overrides --layer.")
    parser.add_argument("--lambda-sweep", type=str, default="0.0,5.0,12.5", help="Alias for additive alpha sweep.")
    parser.add_argument("--modes", type=str, default="add", help="Comma-separated: add,ablate")
    parser.add_argument("--controls", type=str, default="semantic", help="Comma-separated: semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29, help="Layer to use as semantic anchor")
    parser.add_argument("--eval-json", type=str, default=None, help="Optional JSON file with intervention evaluation items")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/steering_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--generate-examples", action="store_true")
    parser.add_argument("--examples-json", type=str, default=None)
    parser.add_argument("--example-max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    examples_json = args.examples_json or infer_examples_json(args.output_csv)

    validate_upstream_metadata(args.data_dir, allow_invalid_metadata=args.allow_invalid_metadata)
    layers = parse_int_list(args.layers) if args.layers else [args.layer]
    alphas = parse_float_list(args.lambda_sweep)
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    controls = [c.strip().lower() for c in args.controls.split(",") if c.strip()]
    eval_items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        eval_items = eval_items[:args.max_eval_items]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    anchor_direction = load_anchor_direction(
        args.data_dir,
        args.anchor_layer,
        allow_invalid_metadata=args.allow_invalid_metadata,
    )

    detail_rows = []
    example_rows = []

    print("\n=== PHASE 9F: CAUSAL RESIDUAL INTERVENTION ===")
    print(f"Layers: {layers}")
    print(f"Modes: {modes}")
    print(f"Controls: {controls}")
    print(f"Alphas: {alphas}")
    print(f"Eval items: {len(eval_items)}")

    for layer in layers:
        semantic_vec_np = load_layer_vector(args.vector_dir, layer, vector_key=args.vector_key)
        semantic_vec = torch.tensor(semantic_vec_np, device=device, dtype=torch.float32)
        direction_bank = {"semantic": semantic_vec}
        if "random" in controls:
            direction_bank["random"] = make_random_orthogonal_control(semantic_vec, args.seed + layer)

        for control_name in controls:
            if control_name not in direction_bank:
                raise ValueError(f"Unsupported control type: {control_name}")
            for mode in modes:
                if mode not in {"add", "ablate"}:
                    raise ValueError(f"Unsupported mode: {mode}")
                for alpha in tqdm(alphas, desc=f"Layer {layer} | {control_name} | {mode}", leave=False):
                    hook = ResidualInterventionHook(direction_bank[control_name], alpha=alpha, mode=mode)
                    hook.attach(model, layer)
                    try:
                        for item in eval_items:
                            row = evaluate_item(model, tokenizer, item, args.anchor_layer, anchor_direction)
                            row.update({
                                "layer": layer,
                                "control": control_name,
                                "mode": mode,
                                "alpha": alpha,
                            })
                            detail_rows.append(row)

                        if args.generate_examples:
                            for prompt in BUILTIN_GENERATION_PROMPTS:
                                example_rows.append({
                                    "layer": layer,
                                    "control": control_name,
                                    "mode": mode,
                                    "alpha": alpha,
                                    "prompt": prompt,
                                    "generation": greedy_generate(
                                        model,
                                        tokenizer,
                                        prompt,
                                        max_new_tokens=args.example_max_tokens,
                                    ),
                                })
                    finally:
                        hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby(["layer", "control", "mode", "alpha"], as_index=False)
        .agg(
            mean_math_bias_logprob=("math_minus_creative_logprob", "mean"),
            mean_pairwise_math_prob=("pairwise_math_prob", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            label_accuracy=("label_correct", "mean"),
            mean_next_token_entropy=("next_token_entropy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            n_items=("item_name", "count"),
        )
    )

    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    print("\n--- CAUSAL INTERVENTION SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")

    if args.generate_examples:
        with open(examples_json, "w", encoding="utf-8") as f:
            json.dump(example_rows, f, indent=2, ensure_ascii=False)
        print(f"Examples saved to {examples_json}")


if __name__ == "__main__":
    main()
