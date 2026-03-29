"""Analysis of SAE learned features."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class FeatureAnalyzer:
    """Analyzes monosemantic features extracted by SAE."""
    
    def __init__(self, sae_module: nn.Module):
        self.sae = sae_module
        # Decoder weights are the "feature directions" in residual space
        self.feature_directions = sae_module.decoder.weight.data.t() # (dict_size, hidden_size)
        
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes activations into the sparse SAE feature space."""
        self.sae.eval()
        with torch.no_grad():
            _, sparse_acts = self.sae(x)
        return sparse_acts

    def feature_activation_frequency(self, corpus_activations: torch.Tensor) -> torch.Tensor:
        """
        Tracks activation frequency of features across a corpus to flag dead/dense features.
        """
        sparse_acts = self.get_feature_activations(corpus_activations)
        # Frequency = (number of times feature > 0) / total samples
        freqs = (sparse_acts > 0).float().mean(dim=0)
        return freqs
        
    def feature_logit_attribution(self, feature_idx: int, lm_head: nn.Module, final_norm: nn.Module) -> List[Tuple[str, float]]:
        """
        Measures the direct effect of a single SAE feature on next-token predictions.
        Projects the feature's decoder direction through the model's unembedding head.
        """
        direction = self.feature_directions[feature_idx].unsqueeze(0) # (1, hidden_size)
        
        # Ensure dtype match (e.g., if model is float16 and SAE is float32)
        head_dtype = next(lm_head.parameters()).dtype
        direction = direction.to(head_dtype)
        
        with torch.no_grad():
            # Standard Unembedding path: direction -> norm -> lm_head
            normed_dir = final_norm(direction)
            logits = lm_head(normed_dir)
            probs = torch.softmax(logits, dim=-1)
            
        return probs

    def get_top_tokens_for_feature(self, feature_idx: int, lm_head: nn.Module, final_norm: nn.Module, tokenizer, k: int = 10):
        """Returns the human-readable tokens that this feature most strongly promotes."""
        probs = self.feature_logit_attribution(feature_idx, lm_head, final_norm)
        top_probs, top_indices = torch.topk(probs[0], k)
        
        results = []
        for i in range(k):
            token_str = tokenizer.decode([top_indices[i].item()])
            results.append((token_str, top_probs[i].item()))
        return results
