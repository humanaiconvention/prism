import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_site_hooks import BlockOutputCaptureHook, TensorSiteInterventionHook
from scripts.phase9_semantic_utils import infer_detail_csv, load_semantic_direction, parse_int_list
from scripts.run_phase9_recurrent_state_patching import reset_model_decode_state, unpack_logits_and_cache
from scripts.run_phase9_semantic_steering import load_eval_items, make_random_orthogonal_control
from scripts.run_phase9_token_position_steering import prepare_eval_item


def score_choice_logits(logits, prepared_item):
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    math_lp = float(log_probs[0, int(prepared_item["math_token_id"])].item())
    creative_lp = float(log_probs[0, int(prepared_item["creative_token_id"])].item())
    label_sign = float(prepared_item["label_sign"])
    margin = math_lp - creative_lp
    math_prob = float(torch.exp(log_probs[0, int(prepared_item["math_token_id"])]).item())
    creative_prob = float(torch.exp(log_probs[0, int(prepared_item["creative_token_id"])]).item())
    return {
        "signed_label_margin": label_sign * margin,
        "label_target_pairwise_prob": float(math_prob / max(math_prob + creative_prob, 1e-12)) if label_sign > 0 else float(creative_prob / max(math_prob + creative_prob, 1e-12)),
        "label_accuracy": float((margin >= 0.0) if label_sign > 0 else (margin <= 0.0)),
        "math_minus_creative_logprob": margin,
    }


def cosine_or_zero(a, b):
    a_norm = float(torch.norm(a).item())
    b_norm = float(torch.norm(b).item())
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(torch.dot(a, b).item() / (a_norm * b_norm))


def run_incremental_probe(model, prompt_ids, forced_token_id, probe_layers, intervention_hook=None, intervene_step=None):
    captures = [BlockOutputCaptureHook(layer) for layer in probe_layers]
    for capture in captures:
        capture.attach(model)
    reset_model_decode_state(model)
    past_key_values = None
    prompt_states = None
    answer_states = None
    try:
        with torch.inference_mode():
            for step_idx in range(prompt_ids.shape[1]):
                if intervention_hook is not None:
                    intervention_hook.enabled = step_idx == intervene_step
                for capture in captures:
                    capture.clear()
                output = model(prompt_ids[:, step_idx : step_idx + 1], past_key_values=past_key_values, use_cache=True)
                first_logits, past_key_values = unpack_logits_and_cache(output)
                if step_idx == intervene_step:
                    prompt_states = {capture.layer: capture.captured.squeeze(0).detach().clone() for capture in captures}
            if intervention_hook is not None:
                intervention_hook.enabled = False
            for capture in captures:
                capture.clear()
            forced_ids = torch.tensor([[int(forced_token_id)]], device=prompt_ids.device)
            output = model(forced_ids, past_key_values=past_key_values, use_cache=True)
            second_logits, _ = unpack_logits_and_cache(output)
            answer_states = {capture.layer: capture.captured.squeeze(0).detach().clone() for capture in captures}
    finally:
        for capture in captures:
            capture.remove()
        if intervention_hook is not None:
            intervention_hook.enabled = False
        reset_model_decode_state(model)
    return first_logits.detach().clone(), second_logits.detach().clone(), prompt_states, answer_states


def main():
    parser = argparse.ArgumentParser(description="Phase 10F: L11 persistence and readout-locality follow-up")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/l11_persistence_readout_summary.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layer", type=int, default=11)
    parser.add_argument("--site", type=str, default="block_input")
    parser.add_argument("--probe-layers", type=str, default="11,15,29")
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-eval-items", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    semantic_vec_np = load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key)
    semantic_vec = torch.tensor(semantic_vec_np, device=device, dtype=torch.float32)
    random_vec = make_random_orthogonal_control(semantic_vec, seed=args.seed)
    items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[: args.max_eval_items]
    prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
    probe_layers = parse_int_list(args.probe_layers)
    detail_rows = []

    for item in prepared:
        forced_token_id = item["math_token_id"] if item["label_sign"] > 0 else item["creative_token_id"]
        baseline_first, baseline_second, baseline_prompt, baseline_answer = run_incremental_probe(
            model=model,
            prompt_ids=item["prompt_ids"],
            forced_token_id=forced_token_id,
            probe_layers=probe_layers,
            intervention_hook=None,
            intervene_step=item["prompt_ids"].shape[1] - 1,
        )
        baseline_first_scores = score_choice_logits(baseline_first, item)
        baseline_second_scores = score_choice_logits(baseline_second, item)
        for control_name, vector in (("semantic", semantic_vec), ("random", random_vec)):
            hook = TensorSiteInterventionHook(vector=vector, alpha=args.alpha, mode="add", position_fraction=args.position_fraction)
            hook.attach(model, args.target_layer, args.site)
            try:
                first_logits, second_logits, prompt_states, answer_states = run_incremental_probe(
                    model=model,
                    prompt_ids=item["prompt_ids"],
                    forced_token_id=forced_token_id,
                    probe_layers=probe_layers,
                    intervention_hook=hook,
                    intervene_step=item["prompt_ids"].shape[1] - 1,
                )
            finally:
                hook.remove()
            first_scores = score_choice_logits(first_logits, item)
            second_scores = score_choice_logits(second_logits, item)
            for layer in probe_layers:
                prompt_delta = prompt_states[layer] - baseline_prompt[layer]
                answer_delta = answer_states[layer] - baseline_answer[layer]
                detail_rows.append(
                    {
                        "item_name": item["item"]["name"],
                        "control": control_name,
                        "site": args.site,
                        "target_layer": args.target_layer,
                        "probe_layer": layer,
                        "alpha": args.alpha,
                        "position_fraction": args.position_fraction,
                        "first_answer_signed_label_margin": first_scores["signed_label_margin"],
                        "first_answer_delta_from_baseline_signed_label_margin": first_scores["signed_label_margin"] - baseline_first_scores["signed_label_margin"],
                        "first_answer_label_target_pairwise_prob": first_scores["label_target_pairwise_prob"],
                        "first_answer_delta_from_baseline_label_target_pairwise_prob": first_scores["label_target_pairwise_prob"] - baseline_first_scores["label_target_pairwise_prob"],
                        "second_step_signed_label_margin": second_scores["signed_label_margin"],
                        "second_step_delta_from_baseline_signed_label_margin": second_scores["signed_label_margin"] - baseline_second_scores["signed_label_margin"],
                        "second_step_label_target_pairwise_prob": second_scores["label_target_pairwise_prob"],
                        "second_step_delta_from_baseline_label_target_pairwise_prob": second_scores["label_target_pairwise_prob"] - baseline_second_scores["label_target_pairwise_prob"],
                        "prompt_delta_norm": float(torch.norm(prompt_delta).item()),
                        "answer_delta_norm": float(torch.norm(answer_delta).item()),
                        "answer_over_prompt_delta_ratio": float(torch.norm(answer_delta).item() / max(torch.norm(prompt_delta).item(), 1e-8)),
                        "delta_persistence_cosine": cosine_or_zero(prompt_delta, answer_delta),
                    }
                )

    detail_df = pd.DataFrame(detail_rows)
    summary = (
        detail_df.groupby(["target_layer", "site", "probe_layer", "control", "alpha", "position_fraction"], as_index=False)
        .agg(
            mean_first_answer_signed_label_margin=("first_answer_signed_label_margin", "mean"),
            delta_from_baseline_mean_first_answer_signed_label_margin=("first_answer_delta_from_baseline_signed_label_margin", "mean"),
            mean_first_answer_label_target_pairwise_prob=("first_answer_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_first_answer_label_target_pairwise_prob=("first_answer_delta_from_baseline_label_target_pairwise_prob", "mean"),
            mean_second_step_signed_label_margin=("second_step_signed_label_margin", "mean"),
            delta_from_baseline_mean_second_step_signed_label_margin=("second_step_delta_from_baseline_signed_label_margin", "mean"),
            mean_second_step_label_target_pairwise_prob=("second_step_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_second_step_label_target_pairwise_prob=("second_step_delta_from_baseline_label_target_pairwise_prob", "mean"),
            mean_prompt_delta_norm=("prompt_delta_norm", "mean"),
            mean_answer_delta_norm=("answer_delta_norm", "mean"),
            mean_answer_over_prompt_delta_ratio=("answer_over_prompt_delta_ratio", "mean"),
            mean_delta_persistence_cosine=("delta_persistence_cosine", "mean"),
            n_items=("item_name", "count"),
        )
        .sort_values(["probe_layer", "delta_from_baseline_mean_second_step_signed_label_margin"], ascending=[True, False])
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    detail_path = Path(args.detail_csv) if args.detail_csv else infer_detail_csv(output_path)
    detail_df.to_csv(detail_path, index=False)
    print(summary.to_string(index=False))
    print(f"[saved] summary -> {output_path}")
    print(f"[saved] detail -> {detail_path}")


if __name__ == "__main__":
    main()