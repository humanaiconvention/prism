"""PRISM — Phase-based Research & Interpretability Spectral Microscope.

A model-agnostic mechanistic interpretability toolkit for transformer-family
language models.  Core capabilities:

* **Quantisation hostility profiling** — measure per-layer activation geometry
  that predicts KV-cache quantisation error (3 lines of code).
* **Full mechanistic scan** — logit lens, attention SVD, rank-restoration
  profile, causal patching, and 14 further analysis modules.
* **Verifiable telemetry** — structured snapshot schemas with delta proofs.

Quick start::

    from prism.geometry import scan_model_geometry

    results = scan_model_geometry("google/gemma-4-e2b-it")
    print(results["mean_quantization_hostility"])   # e.g. 0.914

See https://github.com/humanaiconvention/prism for documentation.
"""

__version__ = "1.0.0"

from .analysis import (
    compute_spectral_metrics,
    compute_shannon_effective_rank,
    compute_top_eigenvalues,
    compute_top_head_idx,
)
from .architecture import TransformerArchitectureAdapter
from .core import SpectralMicroscope

# Causal
from .causal.patching import ActivationPatcher, AttributionPatcher

# Lens
from .lens.logit import LogitLens
from .lens.tuned import TunedLens

# Attention
from .attention.circuits import AttentionAnalyzer

# Probing
from .probing.linear import ConceptProber, SteeringVectorExtractor

# SAE
from .sae.trainer import SAETrainer
from .sae.features import FeatureAnalyzer

# MLP
from .mlp.memory import MLPAnalyzer

# Arch
from .arch.hybrid import HybridDiagnostics

# Phase
from .phase.coherence import PhaseAnalyzer

# Entropy
from .entropy.lens import EntropyDynamics

# Geometry — core quantisation-hostility API (top-level for discoverability)
from .geometry.scanner import scan_model_geometry
from .geometry.core import outlier_geometry, outlier_geometry_numpy
from .geometry.viability import GeometricViability

# Telemetry
from .telemetry.snapshot import (
    take_snapshot,
    compute_entropy_delta,
    compute_geometric_health,
)
from .telemetry.schemas import (
    MetricSummary,
    SpectralMetricSummary,
    CausalProvenanceReport,
    CircuitReport,
    EntropySnapshot,
    EntropyDeltaProof,
    LayerEntropyProfile,
    GeometricHealthScore,
)

__all__ = [
    "__version__",
    # geometry / quantisation hostility (primary open-source surface)
    "scan_model_geometry",
    "outlier_geometry",
    "outlier_geometry_numpy",
    # spectral analysis
    "compute_spectral_metrics",
    "compute_shannon_effective_rank",
    "compute_top_eigenvalues",
    "compute_top_head_idx",
    "TransformerArchitectureAdapter",
    "SpectralMicroscope",
    "ActivationPatcher",
    "AttributionPatcher",
    "LogitLens",
    "TunedLens",
    "AttentionAnalyzer",
    "ConceptProber",
    "SteeringVectorExtractor",
    "SAETrainer",
    "FeatureAnalyzer",
    "MLPAnalyzer",
    "HybridDiagnostics",
    "PhaseAnalyzer",
    "EntropyDynamics",
    "GeometricViability",
    "take_snapshot",
    "compute_entropy_delta",
    "compute_geometric_health",
    "MetricSummary",
    "SpectralMetricSummary",
    "CausalProvenanceReport",
    "CircuitReport",
    "EntropySnapshot",
    "EntropyDeltaProof",
    "LayerEntropyProfile",
    "GeometricHealthScore",
]
