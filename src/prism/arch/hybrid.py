"""Cross-architecture and hybrid attention diagnostics."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from ..analysis import compute_spectral_metrics, compute_shannon_effective_rank
from ..architecture import TransformerArchitectureAdapter
import numpy as np

class HybridDiagnostics:
    """Tools for hybrid (linear+softmax) and recurrent models."""
    
    def __init__(self, model: nn.Module, adapter: Optional[TransformerArchitectureAdapter] = None):
        self.model = model
        self.adapter = adapter or TransformerArchitectureAdapter(model)

    def compare_attention_entropy(self, 
                                  softmax_attn_scores: torch.Tensor, 
                                  linear_recurrent_states: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compares the focusing behavior of softmax vs linear mechanisms.
        """
        results = {}
        with torch.no_grad():
            softmax_entropy = -torch.sum(softmax_attn_scores * torch.log(softmax_attn_scores + 1e-9), dim=-1).mean().item()
            results["softmax_entropy"] = softmax_entropy
            
        if linear_recurrent_states is not None:
            flat_state = linear_recurrent_states.view(-1, linear_recurrent_states.size(-1)).float()
            spectral_entropy, _ = compute_spectral_metrics(flat_state)
            results["linear_spectral_entropy"] = spectral_entropy
            
        return results

    def track_recurrent_attractors(self, state_trajectory: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Analyzes a sequence of recurrent states to find geometric attractors using Principal Angles.
        """
        if len(state_trajectory) < 2:
            return {"error": "Trajectory too short"}
            
        angles = []
        for i in range(len(state_trajectory) - 1):
            s1 = state_trajectory[i].float()
            s2 = state_trajectory[i+1].float()
            u1, _, _ = torch.linalg.svd(s1, full_matrices=False)
            u2, _, _ = torch.linalg.svd(s2, full_matrices=False)
            k = min(1, u1.size(1))
            m = u1[:, :k].t() @ u2[:, :k]
            _, s, _ = torch.linalg.svd(m)
            cos_theta = torch.clamp(s[0], 0, 1).item()
            angle_deg = np.degrees(np.arccos(cos_theta))
            angles.append(angle_deg)
            
        return {
            "mean_rotation_angle": sum(angles) / len(angles),
            "final_angle": angles[-1],
            "is_saturated": angles[-1] < 5.0
        }

    def measure_positional_sensitivity(self, 
                                       inputs: Dict[str, torch.Tensor], 
                                       layer_idx: int) -> Dict[str, float]:
        """
        Measures how much the representation changes when positional information is neutralized.
        This is a 'model-agnostic' heuristic that ablates the standard positional rotation 
        indices if the model uses RoPE.
        """
        self.model.eval()
        
        # 1. Baseline Run
        with torch.no_grad():
            out_base = self._forward_with_hidden_states(inputs)
            base_states = self._extract_hidden_states(out_base)
            base_idx = min(self.adapter.hidden_state_index_for_layer(layer_idx), len(base_states) - 1)
            state_base = base_states[base_idx].detach()

        # 2. Positional Ablation Run
        # To make this agnostic, we 'shift' the position_ids to all be 0.
        # This effectively tells RoPE to treat every token as the first token.
        ablated_inputs = inputs.copy()
        if "position_ids" in inputs:
            ablated_inputs["position_ids"] = torch.zeros_like(inputs["position_ids"])
        else:
            # Manually create zeroed position_ids for standard HF models
            batch_size, seq_len = inputs["input_ids"].shape
            ablated_inputs["position_ids"] = torch.zeros((batch_size, seq_len), dtype=torch.long, device=inputs["input_ids"].device)

        with torch.no_grad():
            out_ablated = self._forward_with_hidden_states(ablated_inputs)
            ablated_states = self._extract_hidden_states(out_ablated)
            ablated_idx = min(self.adapter.hidden_state_index_for_layer(layer_idx), len(ablated_states) - 1)
            state_ablated = ablated_states[ablated_idx].detach()

        # 3. Compute Sensitivity Metrics
        # Drift = 1 - CosineSimilarity(base, ablated)
        cos_sim = torch.nn.functional.cosine_similarity(state_base.view(-1), state_ablated.view(-1), dim=0).item()
        
        # Rank Collapse Ratio = Rank(ablated) / Rank(base)
        rank_base = compute_shannon_effective_rank(state_base[0])
        rank_ablated = compute_shannon_effective_rank(state_ablated[0])
        
        return {
            "positional_drift": 1.0 - cos_sim,
            "rank_collapse_ratio": rank_ablated / (rank_base + 1e-9),
            "is_spatially_rigid": (1.0 - cos_sim) > 0.5 # Heuristic
        }

    def _forward_with_hidden_states(self, inputs: Dict[str, torch.Tensor]) -> Any:
        try:
            return self.model(**inputs, output_hidden_states=True, return_dict=True)
        except TypeError:
            return self.model(**inputs, output_hidden_states=True)

    @staticmethod
    def _extract_hidden_states(outputs: Any) -> List[torch.Tensor]:
        for attr in ("decoder_hidden_states", "hidden_states"):
            states = getattr(outputs, attr, None)
            if states is not None:
                return list(states)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            return [last_hidden_state]
        raise ValueError(
            "Model outputs did not expose decoder_hidden_states, hidden_states, or last_hidden_state."
        )
