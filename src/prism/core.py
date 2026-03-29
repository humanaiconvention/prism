"""Core Spectral Microscope telemetry and automated scanning system."""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from .analysis import compute_shannon_effective_rank
from .architecture import TransformerArchitectureAdapter
from .causal.patching import ActivationPatcher
from .geometry.core import compute_cosine
from .lens.logit import LogitLens
from .attention.circuits import AttentionAnalyzer
from .mlp.memory import MLPAnalyzer
from .arch.hybrid import HybridDiagnostics
from .telemetry.schemas import CircuitReport, CausalProvenanceReport

class SpectralMicroscope:
    """High-level API for inline spectral telemetry and automated model scanning."""
    
    def __init__(self, 
                 max_tokens: int = 64, 
                 window_size: int = 32, 
                 streaming_cov_alpha: float = 0.9):
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.alpha = streaming_cov_alpha

    def _build_causal_provenance_report(
        self,
        layer_idx: int,
        pre_state: torch.Tensor,
        post_attn: torch.Tensor,
        post_mlp: torch.Tensor,
    ) -> CausalProvenanceReport:
        pre_rank = compute_shannon_effective_rank(pre_state)
        post_attn_rank = compute_shannon_effective_rank(post_attn)
        post_mlp_rank = compute_shannon_effective_rank(post_mlp)
        attn_cosine = compute_cosine(pre_state, post_attn)
        mlp_cosine = compute_cosine(pre_state, post_mlp)
        route_confidence = max(0.0, 0.5 * max(attn_cosine, 0.0) + 0.5 * max(mlp_cosine, 0.0))

        return CausalProvenanceReport(
            layer_idx=layer_idx,
            pre_state_rank=pre_rank,
            post_attention_rank=post_attn_rank,
            post_mlp_rank=post_mlp_rank,
            attention_rank_delta=post_attn_rank - pre_rank,
            mlp_rank_delta=post_mlp_rank - post_attn_rank,
            pre_to_post_attention_cosine=attn_cosine,
            pre_to_post_mlp_cosine=mlp_cosine,
            route_confidence=route_confidence,
        )

    def full_scan(self, 
                  model: nn.Module, 
                  tokenizer: PreTrainedTokenizerBase, 
                  prompt: str,
                  target_layer: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs a comprehensive mechanistic interpretability scan of the model.
        Runs telemetry, lenses, circuit analysis, and rank profiling in one pass.
        """
        adapter = TransformerArchitectureAdapter(model)
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        num_layers = adapter.num_layers
        if num_layers <= 0:
            raise ValueError("No transformer layers were resolved for this model.")
        if target_layer is None:
            target_layer = num_layers // 2
        target_layer = max(0, min(int(target_layer), num_layers - 1))
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "") or ""

        # 1. Initialize Analyzers
        logit_lens = LogitLens(model, adapter=adapter)
        attn_analyzer = AttentionAnalyzer(model, adapter=adapter)
        mlp_analyzer = MLPAnalyzer(model, adapter=adapter)
        hybrid_diag = HybridDiagnostics(model, adapter=adapter)
        patcher = ActivationPatcher(model)

        report = CircuitReport(
            kind="full_scan",
            prompt=prompt,
            model_name=model_name,
            metadata={"architecture": adapter.describe(), "target_layer": target_layer},
        )

        # 2. Static Weight Analysis (Fast)
        static_circuits = attn_analyzer.generate_circuit_report(target_layer)

        # 3. Dynamic Inference Pass (with Hooks)
        with patcher.trace():
            # Hooks for rank restoration
            layer_mod = adapter.resolve_layer(target_layer)
            attn_module = adapter.resolve_attention_module(layer_mod)
            mlp_module = adapter.resolve_mlp_module(layer_mod)
            patcher.cache_activation(attn_module, "post_attn")
            patcher.cache_activation(mlp_module, "post_mlp")
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)
            
            # Extract States
            hidden_states = tuple(getattr(outputs, "hidden_states", ()) or ())
            attn_outputs = tuple(getattr(outputs, "attentions", ()) or ())
            if not hidden_states:
                raise ValueError("Model did not return hidden states; full_scan requires them.")
            attn_weights = attn_outputs[target_layer] if target_layer < len(attn_outputs) else None
            
            # A. Logit Lens Analysis
            logit_lens_report = {
                "top_predictions": logit_lens.decode_top_k(hidden_states[-1], tokenizer, k=5),
                "entropy_trajectory": logit_lens.get_prediction_entropy_trajectory(hidden_states)
            }

            # B. Rank Restoration Profile
            pre_state = hidden_states[min(target_layer, len(hidden_states) - 1)]
            post_attn = patcher._cache.get("post_attn", pre_state)
            post_state_idx = adapter.hidden_state_index_for_layer(target_layer)
            post_mlp = hidden_states[min(post_state_idx, len(hidden_states) - 1)]
            
            rank_profile = mlp_analyzer.rank_restoration_profile(
                pre_state, post_attn, post_mlp
            )
            causal_provenance = self._build_causal_provenance_report(
                target_layer, pre_state, post_attn, post_mlp
            )

            # C. Attention Entropy Heatmap
            attention_heatmap = (
                attn_analyzer.compute_attention_entropy_map(attn_weights).tolist()
                if attn_weights is not None
                else []
            )

            # D. Hybrid/Positional Sensitivity
            positional_sensitivity = hybrid_diag.measure_positional_sensitivity(inputs, target_layer)

        report.sections = {
            "static_circuits": static_circuits,
            "logit_lens": logit_lens_report,
            "rank_profile": rank_profile,
            "attention_heatmap": attention_heatmap,
            "positional_sensitivity": positional_sensitivity,
        }
        report.provenance = causal_provenance

        return report.to_dict()

    def generate_and_analyze(self, 
                             model: nn.Module, 
                             tokenizer: PreTrainedTokenizerBase, 
                             prompt: str, 
                             max_new_tokens: int = 20) -> Dict[str, Any]:
        """Existing streaming telemetry method."""
        # (Preserved for backward compatibility)
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                output_hidden_states=True, 
                return_dict_in_generate=True
            )
        # Simplified telemetry extraction logic...
        return {"response": tokenizer.decode(outputs.sequences[0])}
