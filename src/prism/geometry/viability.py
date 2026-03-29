"""Intrinsic dimensionality, Fisher information, and noise sensitivity."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class GeometricViability:
    """Evaluates the structural health and robustness of the representation manifold."""
    
    def __init__(self, model: nn.Module):
        self.model = model

    def intrinsic_dimensionality_hunchback(self, hidden_states: List[torch.Tensor]) -> List[float]:
        """
        Estimates ID across layers using the Two-NN estimator (Levina & Bickel, 2004).
        Identifies the characteristic rise, peak, and collapse of representations.
        """
        ids = []
        for x in hidden_states:
            # Flatten to (tokens, dim)
            flat_x = x.view(-1, x.size(-1)).float().cpu()
            if flat_x.size(0) < 3:
                ids.append(0.0)
                continue
                
            # Compute pairwise distances
            dists = torch.cdist(flat_x, flat_x)
            # Find nearest and second nearest (skipping self at index 0)
            dists_sorted, _ = torch.sort(dists, dim=1)
            r1 = dists_sorted[:, 1]
            r2 = dists_sorted[:, 2]
            
            # Two-NN estimator: mu = r2/r1 -> ID = 1 / mean(log(mu))
            mu = r2 / (r1 + 1e-12)
            mu = mu[mu > 1.0] # Only valid ratios
            if len(mu) == 0:
                ids.append(0.0)
                continue
                
            layer_id = (len(mu) / torch.log(mu).sum()).item()
            ids.append(layer_id)
        return ids

    def compute_viability_score(self, effective_dimension: float, hidden_dim: int) -> float:
        """Computes normalized geometric health metric (ratio of utilization)."""
        return effective_dimension / hidden_dim

    def representational_noise_sensitivity(self, 
                                           layer_output: torch.Tensor, 
                                           noise_level: float = 0.1) -> float:
        """
        Measures the change in effective dimension when noise is injected.
        Low sensitivity = high slack/robustness.
        """
        from ..analysis import compute_spectral_metrics
        
        # 1. Base ID
        _, id_base = compute_spectral_metrics(layer_output)
        
        # 2. Additive Gaussian Noise
        noise = torch.randn_like(layer_output) * noise_level
        output_noisy = layer_output + noise
        
        # 3. Noisy ID
        _, id_noisy = compute_spectral_metrics(output_noisy)
        
        # Sensitivity = Delta ID normalized by base ID
        return (abs(id_noisy - id_base) / (id_base + 1e-12))

    def fisher_information_curvature(self, layer_output: torch.Tensor) -> float:
        """
        Approximates the trace of the Fisher Information Matrix (FIM) using 
        the variance of the activations. High curvature = tight representational manifold.
        """
        # FIM trace is approximated by the sum of activation variances
        # across the hidden dimension.
        variances = torch.var(layer_output.float(), dim=(0, 1))
        return variances.sum().item()

    def belief_state_fractal_dimension(self, hidden_states: torch.Tensor) -> float:
        """
        Measures fractal dimension of the point cloud via Box-Counting.
        Departure from expected signals manifold mismatch.
        """
        # Simplified: uses the slope of the cumulative energy vs rank
        # (Related to the spectral decay power law)
        from ..analysis import compute_top_eigenvalues
        evals = torch.tensor(compute_top_eigenvalues(hidden_states, k=32))
        log_rank = torch.log(torch.arange(1, len(evals) + 1).float())
        log_evals = torch.log(evals + 1e-12)
        
        # Slope of log-log plot is the fractal exponent
        slope = ((log_rank * log_evals).mean() - log_rank.mean() * log_evals.mean()) / ((log_rank**2).mean() - log_rank.mean()**2)
        return abs(slope.item())

    def grokking_monitor(self, activations: torch.Tensor) -> Dict[str, Any]:
        """
        Tracks the transition from 'glass phase' to 'hidden order'.
        Returns spectral tail decay exponent.
        """
        fractal_dim = self.belief_state_fractal_dimension(activations)
        return {
            "spectral_tail_decay": fractal_dim,
            "is_ordered": fractal_dim > 1.0 # Heuristic for hidden-order transition
        }
