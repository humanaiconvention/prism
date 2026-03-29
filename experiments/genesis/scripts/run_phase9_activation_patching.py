"""Phase 9G: Activation patching / representation swap evaluation.

Reviewer-facing causal test:
- capture a clean residual state at a chosen block input and token position
- patch that state into a corrupted prompt run at the same layer/position
- measure restoration of next-token A/B choice probabilities
- compare against a matched-norm random-direction control
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import format_chatml_prompt, load_genesis_model


BUILTIN_SWAP_ITEMS = [
    {
        "name": "math_to_creative_1",
        "clean_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: give a precise mathematical explanation.\n"
            "A) derive the result step by step using formal reasoning\n"
            "B) describe it with vivid imagery and emotion\n"
            "Answer:"
        ),
        "corrupt_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: continue as imaginative creative writing.\n"
            "A) derive the result step by step using formal reasoning\n"
            "B) describe it with vivid imagery and emotion\n"
            "Answer:"
        ),
        "clean_option": "A",
        "corrupt_option": "B",
    },
    {
        "name": "math_to_creative_2",
        "clean_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: analyze the claim like a careful proof.\n"
            "A) tell it as a dreamlike scene in a forest\n"
            "B) justify it with explicit logical steps\n"
            "Answer:"
        ),
        "corrupt_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: write like a poet, not a textbook.\n"
            "A) tell it as a dreamlike scene in a forest\n"
            "B) justify it with explicit logical steps\n"
            "Answer:"
        ),
        "clean_option": "B",
        "corrupt_option": "A",
    },
    {
        "name": "math_to_creative_3",
        "clean_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: answer analytically and with technical precision.\n"
            "A) define the variables and compute the conclusion\n"
            "B) evoke mood, color, and metaphor\n"
            "Answer:"
        ),
        "corrupt_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: continue with imaginative storytelling.\n"
            "A) define the variables and compute the conclusion\n"
            "B) evoke mood, color, and metaphor\n"
            "Answer:"
        ),
        "clean_option": "A",
        "corrupt_option": "B",
    },
    {
        "name": "creative_to_math_1",
        "clean_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: continue as imaginative creative writing.\n"
            "A) derive the answer with equations and definitions\n"
            "B) paint the idea through scene, rhythm, and feeling\n"
            "Answer:"
        ),
        "corrupt_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: give a precise mathematical explanation.\n"
            "A) derive the answer with equations and definitions\n"
            "B) paint the idea through scene, rhythm, and feeling\n"
            "Answer:"
        ),
        "clean_option": "B",
        "corrupt_option": "A",
    },
    {
        "name": "creative_to_math_2",
        "clean_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: write like a poet, not a textbook.\n"
            "A) use metaphor, cadence, and sensory detail\n"
            "B) present a strict derivation with numbered steps\n"
            "Answer:"
        ),
        "corrupt_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: analyze the claim like a careful proof.\n"
            "A) use metaphor, cadence, and sensory detail\n"
            "B) present a strict derivation with numbered steps\n"
            "Answer:"
        ),
        "clean_option": "A",
        "corrupt_option": "B",
    },
    {
        "name": "creative_to_math_3",
        "clean_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: continue with imaginative storytelling.\n"
            "A) formalize the claim and prove it rigorously\n"
            "B) let the response unfold through character and atmosphere\n"
            "Answer:"
        ),
        "corrupt_prompt": (
            "Choose the continuation that best matches the requested style.\n"
            "Request: answer analytically and with technical precision.\n"
            "A) formalize the claim and prove it rigorously\n"
            "B) let the response unfold through character and atmosphere\n"
            "Answer:"
        ),
        "clean_option": "B",
        "corrupt_option": "A",
    },
]


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def infer_detail_csv(output_csv):
    root, ext = os.path.splitext(output_csv)
    return f"{root}_detail{ext or '.csv'}"


def is_pair_eval_item(item):
    required = {"name", "math_prompt", "creative_prompt", "math_option"}
    return required.issubset(item.keys())


def expand_pair_swap_items(items):
    expanded = []
    for item in items:
        math_option = item["math_option"].strip().upper()
        creative_option = "B" if math_option == "A" else "A"
        expanded.extend([
            {
                "name": f"{item['name']}__math_to_creative",
                "clean_prompt": item["math_prompt"],
                "corrupt_prompt": item["creative_prompt"],
                "clean_option": math_option,
                "corrupt_option": creative_option,
            },
            {
                "name": f"{item['name']}__creative_to_math",
                "clean_prompt": item["creative_prompt"],
                "corrupt_prompt": item["math_prompt"],
                "clean_option": creative_option,
                "corrupt_option": math_option,
            },
        ])
    return expanded


def load_swap_items(eval_json=None):
    if eval_json is None:
        return BUILTIN_SWAP_ITEMS
    with open(eval_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload["items"] if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("Activation patching eval JSON must contain a non-empty list of items.")
    if all(is_pair_eval_item(item) for item in items):
        return expand_pair_swap_items(items)
    required = {"name", "clean_prompt", "corrupt_prompt", "clean_option", "corrupt_option"}
    for idx, item in enumerate(items):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Eval item {idx} missing required keys: {sorted(missing)}")
    return items


def resolve_position(seq_len, position):
    idx = position if position >= 0 else seq_len + position
    if idx < 0 or idx >= seq_len:
        raise IndexError(f"Patch position {position} resolves to invalid index {idx} for seq_len={seq_len}")
    return idx


def resolve_window_positions(seq_len, window_size, end_position=-1):
    end_idx = resolve_position(seq_len, end_position)
    start_idx = max(0, end_idx - int(window_size) + 1)
    return list(range(start_idx, end_idx + 1))


class ResidualCaptureHook:
    def __init__(self, position=-1):
        self.position = int(position)
        self.handle = None
        self.captured = None

    def _make_hook(self):
        def hook_fn(module, args):
            x = args[0]
            idx = resolve_position(x.shape[1], self.position)
            self.captured = x[:, idx, :].detach().clone()
            return None
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ResidualPatchHook:
    def __init__(self, source_vector, alpha=1.0, position=-1, control="clean", seed=1234):
        self.source_vector = source_vector
        self.alpha = float(alpha)
        self.position = int(position)
        self.control = control
        self.seed = int(seed)
        self.handle = None

    def _random_delta(self, reference_delta):
        rand = torch.randn(reference_delta.shape[-1], generator=torch.Generator().manual_seed(self.seed))
        rand = rand.to(reference_delta.device, dtype=reference_delta.dtype)
        ref_norm = torch.norm(reference_delta)
        if ref_norm > 1e-8:
            ref_unit = reference_delta / ref_norm
            rand = rand - torch.dot(rand, ref_unit) * ref_unit
        rand_norm = torch.norm(rand)
        if rand_norm < 1e-8:
            rand = torch.roll(reference_delta, shifts=1, dims=0)
            rand_norm = torch.norm(rand)
        rand = rand / (rand_norm + 1e-10)
        return rand * ref_norm

    def _make_hook(self):
        def hook_fn(module, args):
            if self.alpha == 0.0:
                return None
            x = args[0]
            idx = resolve_position(x.shape[1], self.position)
            current = x[:, idx, :]
            source = self.source_vector.unsqueeze(0).expand_as(current)
            if self.control == "clean":
                delta = source - current
            elif self.control == "random":
                delta = torch.stack([self._random_delta(source[i] - current[i]) for i in range(current.shape[0])], dim=0)
            else:
                raise ValueError(f"Unsupported control type: {self.control}")
            x_mod = x.clone()
            x_mod[:, idx, :] = current + (self.alpha * delta)
            return (x_mod, *args[1:])
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ResidualWindowCaptureHook:
    def __init__(self, positions):
        self.positions = [int(p) for p in positions]
        self.handle = None
        self.captured = None

    def _make_hook(self):
        def hook_fn(module, args):
            x = args[0]
            self.captured = x[:, self.positions, :].detach().clone()
            return None
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ResidualWindowPatchHook:
    def __init__(self, source_window, alpha=1.0, positions=None, control="clean", seed=1234):
        self.source_window = source_window
        self.alpha = float(alpha)
        self.positions = [int(p) for p in (positions or [])]
        self.control = control
        self.seed = int(seed)
        self.handle = None

    def _random_delta(self, reference_delta):
        rand = torch.randn(reference_delta.shape[-1], generator=torch.Generator().manual_seed(self.seed))
        rand = rand.to(reference_delta.device, dtype=reference_delta.dtype)
        ref_norm = torch.norm(reference_delta)
        if ref_norm > 1e-8:
            ref_unit = reference_delta / ref_norm
            rand = rand - torch.dot(rand, ref_unit) * ref_unit
        rand_norm = torch.norm(rand)
        if rand_norm < 1e-8:
            rand = torch.roll(reference_delta, shifts=1, dims=0)
            rand_norm = torch.norm(rand)
        rand = rand / (rand_norm + 1e-10)
        return rand * ref_norm

    def _make_hook(self):
        def hook_fn(module, args):
            if self.alpha == 0.0:
                return None
            x = args[0]
            current = x[:, self.positions, :]
            source = self.source_window.unsqueeze(0).expand_as(current)
            if self.control == "clean":
                delta = source - current
            elif self.control == "random":
                delta_tokens = []
                for batch_idx in range(current.shape[0]):
                    token_deltas = []
                    for token_idx in range(current.shape[1]):
                        token_deltas.append(self._random_delta(source[batch_idx, token_idx] - current[batch_idx, token_idx]))
                    delta_tokens.append(torch.stack(token_deltas, dim=0))
                delta = torch.stack(delta_tokens, dim=0)
            else:
                raise ValueError(f"Unsupported control type: {self.control}")
            x_mod = x.clone()
            x_mod[:, self.positions, :] = current + (self.alpha * delta)
            return (x_mod, *args[1:])
        return hook_fn

    def attach(self, model, layer):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def unpack_logits(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output


def encode_prompt(tokenizer, text, device):
    return torch.tensor([tokenizer.encode(format_chatml_prompt(text))], device=device)


def encode_choice_token(tokenizer, choice):
    token_ids = tokenizer.encode(f" {choice.strip().upper()}")
    if len(token_ids) != 1:
        raise ValueError(f"Choice '{choice}' must encode to a single token, got {token_ids}")
    return token_ids[0]


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


def score_binary_choice(model, prompt_ids, clean_token_id, corrupt_token_id):
    clean_lp = next_token_logprob(model, prompt_ids, clean_token_id)
    corrupt_lp = next_token_logprob(model, prompt_ids, corrupt_token_id)
    denom = np.exp(clean_lp) + np.exp(corrupt_lp)
    pairwise_clean_prob = float(np.exp(clean_lp) / max(denom, 1e-12))
    margin = float(clean_lp - corrupt_lp)
    return {
        "clean_logprob": clean_lp,
        "corrupt_logprob": corrupt_lp,
        "clean_minus_corrupt_logprob": margin,
        "pairwise_clean_prob": pairwise_clean_prob,
        "next_token_entropy": next_token_entropy(model, prompt_ids),
        "predicts_clean_option": int(margin >= 0.0),
    }


def capture_residual(model, prompt_ids, layer, position):
    hook = ResidualCaptureHook(position=position)
    hook.attach(model, layer)
    try:
        with torch.inference_mode():
            model(prompt_ids)
        if hook.captured is None:
            raise RuntimeError(f"No residual captured for layer {layer}")
        return hook.captured.squeeze(0).detach().clone()
    finally:
        hook.remove()


def capture_residual_window(model, prompt_ids, layer, positions):
    hook = ResidualWindowCaptureHook(positions=positions)
    hook.attach(model, layer)
    try:
        with torch.inference_mode():
            model(prompt_ids)
        if hook.captured is None:
            raise RuntimeError(f"No residual window captured for layer {layer}")
        return hook.captured.squeeze(0).detach().clone()
    finally:
        hook.remove()


def prepare_items(items, tokenizer, device):
    prepared = []
    for item in items:
        prepared.append({
            "name": item["name"],
            "clean_prompt_ids": encode_prompt(tokenizer, item["clean_prompt"], device),
            "corrupt_prompt_ids": encode_prompt(tokenizer, item["corrupt_prompt"], device),
            "clean_token_id": encode_choice_token(tokenizer, item["clean_option"]),
            "corrupt_token_id": encode_choice_token(tokenizer, item["corrupt_option"]),
            "clean_option": item["clean_option"].strip().upper(),
            "corrupt_option": item["corrupt_option"].strip().upper(),
        })
    return prepared


def main():
    parser = argparse.ArgumentParser(description="Phase 9G: Activation patching / representation swap evaluation")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices. Overrides --layer.")
    parser.add_argument("--alpha-sweep", type=str, default="0.0,0.5,1.0", help="Interpolation sweep from corrupt to clean residual.")
    parser.add_argument("--controls", type=str, default="clean,random", help="Comma-separated: clean,random")
    parser.add_argument("--patch-token-position", type=int, default=-1, help="Token index to capture/patch. Default -1 = last token.")
    parser.add_argument("--eval-json", type=str, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/activation_patching_results.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    detail_csv = args.detail_csv or infer_detail_csv(args.output_csv)
    layers = parse_int_list(args.layers) if args.layers else [args.layer]
    alphas = parse_float_list(args.alpha_sweep)
    controls = [c.strip().lower() for c in args.controls.split(",") if c.strip()]
    items = load_swap_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[:args.max_eval_items]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    prepared_items = prepare_items(items, tokenizer, device)

    print("\n=== PHASE 9G: ACTIVATION PATCHING / REPRESENTATION SWAP ===")
    print(f"Layers: {layers}")
    print(f"Controls: {controls}")
    print(f"Alphas: {alphas}")
    print(f"Patch token position: {args.patch_token_position}")
    print(f"Eval items: {len(prepared_items)}")

    detail_rows = []
    baseline_cache = {}
    for layer in layers:
        layer_items = []
        for item in prepared_items:
            clean_metrics = score_binary_choice(model, item["clean_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
            corrupt_metrics = score_binary_choice(model, item["corrupt_prompt_ids"], item["clean_token_id"], item["corrupt_token_id"])
            source_vector = capture_residual(model, item["clean_prompt_ids"], layer, args.patch_token_position)
            layer_items.append({
                "item": item,
                "clean_metrics": clean_metrics,
                "corrupt_metrics": corrupt_metrics,
                "source_vector": source_vector,
            })
        baseline_cache[layer] = layer_items

    for layer in layers:
        for control in controls:
            if control not in {"clean", "random"}:
                raise ValueError(f"Unsupported control type: {control}")
            for alpha in tqdm(alphas, desc=f"Layer {layer} | {control}", leave=False):
                for item_idx, cached in enumerate(baseline_cache[layer]):
                    patch_hook = ResidualPatchHook(
                        source_vector=cached["source_vector"],
                        alpha=alpha,
                        position=args.patch_token_position,
                        control=control,
                        seed=args.seed + (1000 * layer) + item_idx,
                    )
                    patch_hook.attach(model, layer)
                    try:
                        patched_metrics = score_binary_choice(
                            model,
                            cached["item"]["corrupt_prompt_ids"],
                            cached["item"]["clean_token_id"],
                            cached["item"]["corrupt_token_id"],
                        )
                    finally:
                        patch_hook.remove()

                    clean_margin = cached["clean_metrics"]["clean_minus_corrupt_logprob"]
                    corrupt_margin = cached["corrupt_metrics"]["clean_minus_corrupt_logprob"]
                    patched_margin = patched_metrics["clean_minus_corrupt_logprob"]
                    denom = clean_margin - corrupt_margin
                    contrast_valid = int(denom > 1e-8)
                    restoration_fraction = np.nan if not contrast_valid else (patched_margin - corrupt_margin) / denom
                    detail_rows.append({
                        "layer": layer,
                        "control": control,
                        "alpha": alpha,
                        "item_name": cached["item"]["name"],
                        "clean_option": cached["item"]["clean_option"],
                        "corrupt_option": cached["item"]["corrupt_option"],
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patched_margin": patched_margin,
                        "patch_effect": patched_margin - corrupt_margin,
                        "contrast_valid": contrast_valid,
                        "restoration_fraction": restoration_fraction,
                        "clean_pairwise_prob": cached["clean_metrics"]["pairwise_clean_prob"],
                        "corrupt_pairwise_prob": cached["corrupt_metrics"]["pairwise_clean_prob"],
                        "patched_pairwise_prob": patched_metrics["pairwise_clean_prob"],
                        "clean_entropy": cached["clean_metrics"]["next_token_entropy"],
                        "corrupt_entropy": cached["corrupt_metrics"]["next_token_entropy"],
                        "patched_entropy": patched_metrics["next_token_entropy"],
                        "clean_predicts_clean": cached["clean_metrics"]["predicts_clean_option"],
                        "corrupt_predicts_clean": cached["corrupt_metrics"]["predicts_clean_option"],
                        "patched_predicts_clean": patched_metrics["predicts_clean_option"],
                    })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = (
        detail_df.groupby(["layer", "control", "alpha"], as_index=False)
        .agg(
            mean_clean_margin=("clean_margin", "mean"),
            mean_corrupt_margin=("corrupt_margin", "mean"),
            mean_patched_margin=("patched_margin", "mean"),
            mean_patch_effect=("patch_effect", "mean"),
            contrast_valid_rate=("contrast_valid", "mean"),
            mean_restoration_fraction=("restoration_fraction", "mean"),
            baseline_clean_choice_rate=("corrupt_predicts_clean", "mean"),
            patched_clean_choice_rate=("patched_predicts_clean", "mean"),
            mean_corrupt_entropy=("corrupt_entropy", "mean"),
            mean_patched_entropy=("patched_entropy", "mean"),
            n_items=("item_name", "count"),
        )
    )

    summary_df.to_csv(args.output_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    print("\n--- ACTIVATION PATCHING SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Detail saved to {detail_csv}")


if __name__ == "__main__":
    main()