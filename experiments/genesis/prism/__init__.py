"""Spectral Microscope: Telemetry and causal intervention for hybrid models.

This library provides utilities to track spectral entropy, effective dimension,
and layer-level attention metrics during inline text generation.
"""

__version__ = "0.1.0"

from .analysis import (
    compute_spectral_metrics,
    compute_shannon_effective_rank,
    compute_top_eigenvalues,
    compute_top_head_idx,
)
from .core import SpectralMicroscope

__all__ = [
    "__version__",
    "compute_spectral_metrics",
    "compute_shannon_effective_rank",
    "compute_top_eigenvalues",
    "compute_top_head_idx",
    "SpectralMicroscope",
]
