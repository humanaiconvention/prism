"""Shannon and Rényi entropy profiles, and spectral-semantic coupling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

def compute_entropy_from_probs(probs: torch.Tensor) -> float:
    """Computes the Shannon entropy of a probability distribution tensor."""
    return float((-(probs * torch.log(probs + 1e-10)).sum()).item())

def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Computes the Kullback-Leibler divergence D_KL(P || Q) between two discrete probability distributions."""
    p = p + 1e-10
    q = q + 1e-10
    return float(torch.sum(p * torch.log(p / q)).item())

def unpack_logits_and_cache(model_output: Any) -> Tuple[torch.Tensor, Any]:
    """Mocks the behavior since transformer output formats vary."""
    if hasattr(model_output, "logits"):
        return model_output.logits, getattr(model_output, "past_key_values", None)
    return model_output[0], model_output[1] if len(model_output) > 1 else None

def compute_sequence_nll_tokenwise(model: nn.Module, prompt_ids: torch.Tensor, prompt_len: Optional[int]=None, reset_fn=None) -> float:
    """Mean NLL per token — mixed rollout version.
    Takes a mocked model and prompt_ids and calculates next-token NLL.
    """
    if reset_fn:
        reset_fn(model)
        
    seq_len = prompt_ids.shape[1]
    if seq_len < 2:
        return float("nan")
    
    if prompt_len is None:
        prompt_len = seq_len - 1 
    
    nlls = []
    past_kv = None
    with torch.inference_mode():
        # 1. Prefill prompt portion
        prefill_ids = prompt_ids[:, :prompt_len]
        model_output = model(prefill_ids, use_cache=True)
        logits, past_kv = unpack_logits_and_cache(model_output)
        
        # Score prefill predictions (tokens 1 to prompt_len)
        for i in range(prompt_len):
            if i + 1 < seq_len:
                logit_step = logits[0, i, :]
                if not torch.isfinite(logit_step).all():
                    return float("nan")
                log_probs = F.log_softmax(logit_step, dim=-1)
                target = int(prompt_ids[0, i + 1].item())
                nlls.append(-float(log_probs[target].item()))
        
        # 2. Incremental for remaining sequence (tokens prompt_len+1 to end)
        for i in range(prompt_len, seq_len - 1):
            cur_input = prompt_ids[:, i : i + 1]
            model_output = model(cur_input, past_key_values=past_kv, use_cache=True)
            logit, past_kv = unpack_logits_and_cache(model_output)
            logit_step = logit[0, -1, :]
            
            if not torch.isfinite(logit_step).all():
                return float("nan")
            log_probs = F.log_softmax(logit_step, dim=-1)
            target = int(prompt_ids[0, i + 1].item())
            nlls.append(-float(log_probs[target].item()))

    if reset_fn:
        reset_fn(model)
        
    return float(sum(nlls) / len(nlls)) if nlls else float("nan")

def compute_sequence_nll_tokenwise_with_traces(
    model: nn.Module, prompt_ids: torch.Tensor, prompt_len: Optional[int]=None, baseline_traces: Optional[List[Dict[str, Any]]]=None, top_k: int=10, save_full_logits: bool=False, reset_fn=None
) -> Tuple[float, List[float], List[Dict[str, Any]]]:
    """Extended tokenwise NLL with mixed rollout for artifact saving."""
    if reset_fn:
        reset_fn(model)
        
    seq_len = prompt_ids.shape[1]
    if seq_len < 2:
        return float("nan"), [], []

    if prompt_len is None:
        prompt_len = seq_len - 1

    nlls = []
    traces = []
    past_kv = None
    
    def process_logit(logit_step, pos, current_nlls, current_traces):
        if not torch.isfinite(logit_step).all():
            return False
        log_probs = F.log_softmax(logit_step, dim=-1)
        target = int(prompt_ids[0, pos].item())
        token_nll = -float(log_probs[target].item())
        current_nlls.append(token_nll)

        topk_logprobs, topk_ids = torch.topk(log_probs, min(top_k, log_probs.shape[-1]))
        trace = {
            "token_position": pos,
            "gold_token_id": target,
            "token_nll": token_nll,
            "topk_token_ids": topk_ids.cpu().tolist(),
            "topk_logprobs": topk_logprobs.cpu().tolist(),
        }
        if save_full_logits:
            trace["full_logprobs"] = log_probs.cpu().tolist()

        idx = pos - 1
        if baseline_traces is not None and idx < len(baseline_traces):
            bt = baseline_traces[idx]
            trace["baseline_token_nll"] = bt["token_nll"]
            trace["delta_token_nll"] = token_nll - bt["token_nll"]
            baseline_topk_ids = bt["topk_token_ids"]
            baseline_topk_logprobs = bt["topk_logprobs"]
            trace["baseline_topk_token_ids"] = baseline_topk_ids
            trace["baseline_topk_logprobs"] = baseline_topk_logprobs
            if baseline_topk_ids:
                baseline_ids_tensor = torch.tensor(baseline_topk_ids, device=log_probs.device)
                perturbed_at_baseline = log_probs[baseline_ids_tensor].detach().cpu().tolist()
                trace["perturbed_logprobs_at_baseline_topk"] = perturbed_at_baseline
                trace["delta_topk_logprobs"] = [p - bp for p, bp in zip(perturbed_at_baseline, baseline_topk_logprobs)]

        current_traces.append(trace)
        return True

    with torch.inference_mode():
        # 1. Prefill
        prefill_ids = prompt_ids[:, :prompt_len]
        model_output = model(prefill_ids, use_cache=True)
        logits, past_kv = unpack_logits_and_cache(model_output)
        for i in range(prompt_len):
            if i + 1 < seq_len:
                if not process_logit(logits[0, i, :], i + 1, nlls, traces):
                    break
        
        # 2. Incremental
        for i in range(prompt_len, seq_len - 1):
            cur_input = prompt_ids[:, i : i + 1]
            model_output = model(cur_input, past_key_values=past_kv, use_cache=True)
            logit, past_kv = unpack_logits_and_cache(model_output)
            if not process_logit(logit[0, -1, :], i + 1, nlls, traces):
                break

    if reset_fn:
        reset_fn(model)
        
    mean_nll = float(sum(nlls) / len(nlls)) if nlls else float("nan")
    return mean_nll, nlls, traces

def score_choice_logits(logits: torch.Tensor, prepared_item: Dict[str, Any]) -> Dict[str, float]:
    """Scores binary choice options from model logits."""
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    math_lp = float(log_probs[0, int(prepared_item["math_token_id"])].item())
    creative_lp = float(log_probs[0, int(prepared_item["creative_token_id"])].item())
    margin = math_lp - creative_lp
    label_sign = float(prepared_item["label_sign"])
    
    math_prob = float(torch.exp(log_probs[0, int(prepared_item["math_token_id"])]).item())
    creative_prob = float(torch.exp(log_probs[0, int(prepared_item["creative_token_id"])]).item())
    pairwise_denom = max(math_prob + creative_prob, 1e-12)
    
    return {
        "signed_label_margin": label_sign * margin,
        "label_target_pairwise_prob": float(math_prob / pairwise_denom) if label_sign > 0 else float(creative_prob / pairwise_denom),
        "label_accuracy": float((margin >= 0.0) if label_sign > 0 else (margin <= 0.0)),
    }

class EntropyDynamics:
    """Tracks entropy expansion/pruning and spectral-semantic coupling."""
    
    def __init__(self, model: nn.Module):
        self.model = model

    def compute_renyi_entropy(self, probabilities: torch.Tensor, alpha: float) -> float:
        """Computes Rényi entropy H_alpha for a given alpha."""
        if alpha == 1.0:
            return -torch.sum(probabilities * torch.log(probabilities + 1e-12)).item()
        if alpha == float('inf'):
            return -torch.log(torch.max(probabilities)).item()
        
        inner = torch.sum(probabilities**alpha)
        return (1.0 / (1.0 - alpha)) * torch.log(inner + 1e-12).item()

    def entropy_profile_tracking(self, layer_entropies: List[float]) -> List[str]:
        """Identifies expansion (dH > 0) vs pruning (H < 0) phases across depth."""
        states = []
        for i in range(1, len(layer_entropies)):
            diff = layer_entropies[i] - layer_entropies[i-1]
            states.append("expansion" if diff > 0 else "pruning")
        return states

    def spectral_semantic_coupling(self, spectral_entropies: List[float], semantic_entropies: List[float]) -> float:
        """Computes Pearson correlation between geometric and semantic collapse."""
        return np.corrcoef(spectral_entropies, semantic_entropies)[0, 1]

    def detect_entropy_phase_transitions(self, layer_entropies: List[float]) -> List[int]:
        """
        Detects layer-wise breakpoints where entropy undergoes discontinuous change.
        Identifies the 'inflection points' where the model commits to a prediction.
        """
        if len(layer_entropies) < 3:
            return []
            
        # Compute second derivative (rate of change of the rate of change)
        # Using simple finite differences
        ents = np.array(layer_entropies)
        second_deriv = np.diff(ents, n=2)
        
        # The transition is typically where the second derivative is maximized
        # (the sharpest 'knee' in the curve)
        breakpoint = np.argmax(np.abs(second_deriv)) + 1
        return [int(breakpoint)]

    def entropy_autocorrelation(self, entropy_trajectory: List[float]) -> float:
        """Measures how strongly token t's entropy predicts t+1."""
        x = np.array(entropy_trajectory)
        if len(x) < 2: return 0.0
        return np.corrcoef(x[:-1], x[1:])[0, 1]
