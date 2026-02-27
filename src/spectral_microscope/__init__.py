"""Spectral Microscope: Telemetry and causal intervention for hybrid models.

This library provides utilities to track spectral entropy, effective dimension,
and layer-level attention metrics during inline text generation.
"""

from .analysis import (
    compute_spectral_metrics,
    compute_top_eigenvalues,
    compute_top_head_idx,
)
from .core import SpectralMicroscope

__all__ = [
    "compute_spectral_metrics",
    "compute_top_eigenvalues",
    "compute_top_head_idx",
    "SpectralMicroscope",
]
