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

# Geometry
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
