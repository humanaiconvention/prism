"""Attention circuit analysis including induction heads, OV, and QK spectral properties."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from ..architecture import TransformerArchitectureAdapter

class AttentionAnalyzer:
    """Analyzes attention circuits at the head level using weight-space and dynamic metrics."""
    
    def __init__(self, model: nn.Module, adapter: Optional[TransformerArchitectureAdapter] = None):
        self.model = model
        self.adapter = adapter or TransformerArchitectureAdapter(model)
        self.config = getattr(model, "config", None)
        self.n_heads = max(1, self._safe_int(
            getattr(self.config, "num_attention_heads", getattr(self.config, "num_key_value_heads", 0)),
            1,
        ))
        self.n_kv_heads = max(1, self._resolve_kv_heads(self.config, self.n_heads))
        self.hidden_size = self._safe_int(
            getattr(self.config, "hidden_size", getattr(self.config, "n_embd", getattr(self.config, "d_model", 0))),
            0,
        )
        self.head_dim = max(1, self.hidden_size // self.n_heads) if self.hidden_size > 0 else 1
        self.heads_per_group = max(1, self.n_heads // self.n_kv_heads)

    def compute_attention_entropy_map(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Computes the Shannon entropy of each head's attention distribution.
        attention_weights: (batch, n_heads, query_seq, key_seq)
        Returns: (query_seq, n_heads) - mean entropy across batch
        """
        with torch.no_grad():
            # Cast to float32 for stable logs
            p = attention_weights.float()
            # entropy = -sum(p * log(p))
            # Use 0 * log(0) = 0 property
            log_p = torch.log(p + 1e-12)
            entropy = -torch.sum(p * log_p, dim=-1)
            
            # Handle possible NaNs if any
            entropy = torch.nan_to_num(entropy, nan=0.0)
            
            # Mean across batch
            mean_entropy = entropy.mean(dim=0) # (n_heads, query_seq)
            return mean_entropy.t() # (query_seq, n_heads)

    def detect_induction_heads(self, tokenizer, pattern_length: int = 10, repeat_count: int = 2) -> Dict[int, List[float]]:
        """Detects induction heads using repeated random patterns."""
        tokens = torch.randint(100, 1000, (pattern_length,))
        device = next(self.model.parameters()).device
        repeated_tokens = tokens.repeat(repeat_count).unsqueeze(0).to(device)
        attention_scores = [None] * len(self.adapter.layers)
        
        def get_hook(layer_idx):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1:
                    attention_scores[layer_idx] = out[1].detach().cpu()
            return hook

        handles = []
        for i in range(len(self.adapter.layers)):
            layer = self.adapter.resolve_layer(i)
            attn = self.adapter.resolve_attention_module(layer)
            handles.append(attn.register_forward_hook(get_hook(i)))
        with torch.no_grad():
            self.model(input_ids=repeated_tokens, output_attentions=True)
        for h in handles: h.remove()

        induction_scores = {}
        for layer_idx, scores in enumerate(attention_scores):
            if scores is None: continue
            layer_head_scores = []
            for h_idx in range(self.n_heads):
                head_attn = scores[0, h_idx]
                total_induction_attn, count = 0, 0
                for j in range(pattern_length - 1):
                    total_induction_attn += head_attn[pattern_length + j, j + 1].item()
                    count += 1
                layer_head_scores.append(total_induction_attn / count)
            induction_scores[layer_idx] = layer_head_scores
        return induction_scores

    def _get_layer_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.adapter.resolve_attention_projections(layer_idx)

    def analyze_head_ov(self, layer_idx: int, head_idx: int) -> Dict[str, Any]:
        from prism.entropy.lens import compute_entropy_from_probs
        W_Q, W_K, W_V, W_O = self._get_layer_weights(layer_idx)
        kv_head_idx = head_idx // self.heads_per_group
        W_V_h = W_V[kv_head_idx * self.head_dim : (kv_head_idx+1) * self.head_dim, :]
        W_O_h = W_O[:, head_idx * self.head_dim : (head_idx+1) * self.head_dim]
        ov_circuit = W_O_h @ W_V_h
        singular_values = torch.linalg.svdvals(ov_circuit.float())
        probs = singular_values / singular_values.sum()
        entropy_val = compute_entropy_from_probs(probs.unsqueeze(0))
        return {"effective_rank": torch.exp(torch.tensor(entropy_val)).item(), "utilization": torch.exp(torch.tensor(entropy_val)).item() / self.head_dim}

    def analyze_head_qk(self, layer_idx: int, head_idx: int) -> Dict[str, Any]:
        W_Q, W_K, W_V, W_O = self._get_layer_weights(layer_idx)
        kv_head_idx = head_idx // self.heads_per_group
        W_Q_h = W_Q[head_idx * self.head_dim : (head_idx+1) * self.head_dim, :]
        W_K_h = W_K[kv_head_idx * self.head_dim : (kv_head_idx+1) * self.head_dim, :]
        qk_circuit = W_Q_h.T @ W_K_h
        evals = torch.linalg.eigvals(qk_circuit.float()).real
        concordance = evals[evals > 0].sum().item() / (evals.abs().sum().item() + 1e-9)
        return {"concordance_score": concordance, "is_concordant": concordance > 0.5}

    def generate_circuit_report(self, layer_idx: int) -> List[Dict[str, Any]]:
        report = []
        for h in range(self.n_heads):
            ov = self.analyze_head_ov(layer_idx, h)
            qk = self.analyze_head_qk(layer_idx, h)
            report.append({
                "head": h, "ov_rank": ov["effective_rank"], "ov_util": ov["utilization"],
                "concordance": qk["concordance_score"], "type": "concordant" if qk["is_concordant"] else "discordant"
            })
        return report

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _resolve_kv_heads(config: Any, default: int) -> int:
        kv_heads = getattr(config, "num_key_value_heads", None)
        if kv_heads is None:
            kv_heads = getattr(config, "num_kv_heads", None)
        if kv_heads is not None:
            try:
                return int(kv_heads)
            except (TypeError, ValueError):
                return default

        kv_groups = getattr(config, "num_key_value_groups", None)
        if kv_groups is not None:
            try:
                groups = int(kv_groups)
                if groups > 0:
                    return max(1, default // groups)
            except (TypeError, ValueError):
                return default

        return default
