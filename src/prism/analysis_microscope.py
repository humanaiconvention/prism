"""Legacy spectral scan compatibility wrapper.

This module exists for older experiment scripts that imported
``prism.analysis_microscope`` directly. New code should prefer
``prism.core.SpectralMicroscope``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .analysis import compute_eigenvalues_hardware_aware, compute_spectral_metrics
from .geometry.viability import GeometricViability


class SpectralMicroscope:
    """Compatibility wrapper around the modern Prism spectral helpers."""

    def __init__(self, model: Any = None):
        self.model = model

    def full_scan(
        self,
        activations: torch.Tensor,
        device_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan an activation matrix and return legacy-style spectral metrics."""
        if activations.ndim == 3:
            flat_acts = activations.view(-1, activations.shape[-1])
        else:
            flat_acts = activations

        gram_matrix = flat_acts @ flat_acts.T
        eigenvalues_tensor = compute_eigenvalues_hardware_aware(gram_matrix, device_override=device_override)
        eigenvalues = eigenvalues_tensor.cpu().numpy()

        spectral_entropy, effective_dimension = compute_spectral_metrics(flat_acts, device_override=device_override)
        positive_eigs = np.sort(eigenvalues[eigenvalues > 0])[::-1]
        if len(positive_eigs) > 0:
            total_variance = float(np.sum(positive_eigs))
            cumulative_variance = np.cumsum(positive_eigs)
            effective_dim_from_eigs = int(np.searchsorted(cumulative_variance, 0.90 * total_variance)) + 1
        else:
            effective_dim_from_eigs = 0

        hidden_dim = int(flat_acts.shape[-1]) if flat_acts.ndim >= 2 else 1
        viability = GeometricViability(self.model).compute_viability_score(float(effective_dimension), hidden_dim)

        return {
            "metrics": {
                "spectral_entropy": float(spectral_entropy),
                "effective_dimension": float(effective_dim_from_eigs),
                "viability_score": float(viability),
            },
            "effective_dimension": effective_dim_from_eigs,
            "eigenvalues": eigenvalues.tolist(),
        }
