"""FFN Key-Value memory analysis and rank restoration profiles."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from ..analysis import compute_shannon_effective_rank
from ..architecture import TransformerArchitectureAdapter

class MLPAnalyzer:
    """Mechanistic decomposition of feed-forward layers as Key-Value memories."""
    
    def __init__(self, model: nn.Module, adapter: Optional[TransformerArchitectureAdapter] = None):
        self.model = model
        self.adapter = adapter or TransformerArchitectureAdapter(model)
        
    def rank_restoration_profile(self, 
                                 pre_attn_state: torch.Tensor, 
                                 post_attn_state: torch.Tensor, 
                                 post_mlp_state: torch.Tensor) -> Dict[str, float]:
        """Computes effective rank at three stages of a block."""
        x_in = pre_attn_state.view(-1, pre_attn_state.size(-1)).float()
        x_mid = post_attn_state.view(-1, post_attn_state.size(-1)).float()
        x_out = post_mlp_state.view(-1, post_mlp_state.size(-1)).float()
        
        rank_in = compute_shannon_effective_rank(x_in)
        rank_mid = compute_shannon_effective_rank(x_mid)
        rank_out = compute_shannon_effective_rank(x_out)
        
        return {
            "input_rank": rank_in,
            "post_attn_rank": rank_mid,
            "post_mlp_rank": rank_out,
            "attn_impact": rank_mid - rank_in,
            "mlp_impact": rank_out - rank_mid,
            "net_impact": rank_out - rank_in
        }

    def map_neuron_to_vocabulary(self, 
                                 layer_idx: int, 
                                 neuron_idx: int, 
                                 lm_head: nn.Module, 
                                 final_norm: nn.Module, 
                                 tokenizer, 
                                 k: int = 10) -> List[Tuple[str, float]]:
        """
        Projects a specific MLP neuron's output direction (Value) to the vocabulary.
        In Qwen/Llama, the 'Value' is a row of the down_proj weight matrix.
        """
        layer = self.adapter.resolve_layer(layer_idx)
        down_proj = self.adapter.resolve_mlp_down_projection(layer)
        # down_proj shape: (hidden_size, intermediate_size)
        # We want the 'direction' this neuron writes to the residual stream
        value_direction = down_proj[:, neuron_idx].unsqueeze(0)
        
        head_dtype = next(lm_head.parameters()).dtype
        value_direction = value_direction.to(head_dtype)

        with torch.no_grad():
            normed_dir = final_norm(value_direction)
            logits = lm_head(normed_dir)
            probs = torch.softmax(logits, dim=-1)
            
        top_probs, top_indices = torch.topk(probs[0], k)
        results = []
        for i in range(k):
            results.append((tokenizer.decode([top_indices[i].item()]), top_probs[i].item()))
        return results

    def find_activating_tokens(self, 
                                layer_idx: int, 
                                neuron_idx: int, 
                                token_activations: torch.Tensor, 
                                tokenizer, 
                                tokens: torch.Tensor,
                                k: int = 10) -> List[Tuple[str, float]]:
        """
        Scans a set of pre-computed activations to find which tokens most strongly 
        activate a specific neuron (the 'Key' detector).
        token_activations: (N, intermediate_size)
        tokens: (N,) or (batch, seq) flattened
        """
        neuron_acts = token_activations[:, neuron_idx]
        top_vals, top_indices = torch.topk(neuron_acts, k)
        
        flat_tokens = tokens.view(-1)
        results = []
        for i in range(k):
            idx = top_indices[i].item()
            token_str = tokenizer.decode([flat_tokens[idx].item()])
            results.append((token_str, top_vals[i].item()))
        return results
