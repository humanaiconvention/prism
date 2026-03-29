"""Logit Lens for projecting intermediate states to vocabulary."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from ..architecture import TransformerArchitectureAdapter

class LogitLens:
    """Projects hidden states directly through the unembedding matrix."""
    
    def __init__(self, model: nn.Module, adapter: Optional[TransformerArchitectureAdapter] = None):
        self.model = model
        self.adapter = adapter or TransformerArchitectureAdapter(model)

        # 1. Find the unembedding head through the architecture adapter
        self.lm_head = self.adapter.resolve_lm_head()
        if self.lm_head is None:
            raise ValueError("Could not automatically locate the unembedding head (e.g., 'lm_head').")
            
        # 2. Find the final layer norm through the adapter
        self.final_norm = self.adapter.resolve_final_norm() or nn.Identity()

    def project_layer_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden state (batch, seq, dim) to vocab probabilities.
        """
        with torch.no_grad():
            # Standard path: hidden -> final_norm -> lm_head
            normed_states = self.final_norm(hidden_states)
            logits = self.lm_head(normed_states)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def decode_top_k(self, hidden_states: torch.Tensor, tokenizer, k: int = 5, position_idx: int = -1) -> List[Tuple[str, float]]:
        """
        Returns the top-k token strings and their probabilities for a specific sequence position.
        """
        probs = self.project_layer_states(hidden_states)
        
        # Extract the specific position (usually the last token generated)
        target_probs = probs[0, position_idx, :]
        top_probs, top_indices = torch.topk(target_probs, k)
        
        results = []
        for i in range(k):
            token_str = tokenizer.decode([top_indices[i].item()])
            prob_val = top_probs[i].item()
            results.append((token_str, prob_val))
            
        return results
        
    def get_prediction_entropy_trajectory(self, hidden_states_list: List[torch.Tensor], position_idx: int = -1) -> List[float]:
        """
        Tracks the Shannon entropy of the vocabulary distribution across depth.
        High entropy = uncertain. Low entropy = model has "decided" on a token.
        """
        entropies = []
        for h in hidden_states_list:
            with torch.no_grad():
                normed_states = self.final_norm(h)
                logits = self.lm_head(normed_states)
                
                from prism.entropy.lens import compute_entropy_from_probs
                
                # Get logits for the specific token position
                target_logits = logits[:, position_idx, :].float() # Cast to float32 for numerical stability
                
                # Compute stable entropy
                probs = torch.softmax(target_logits, dim=-1)
                entropy = compute_entropy_from_probs(probs)
                
            entropies.append(entropy)
            
        return entropies
