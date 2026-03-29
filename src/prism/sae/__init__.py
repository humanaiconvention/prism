"""Sparse feature decomposition via Sparse Autoencoders (SAEs)."""

from .trainer import SAETrainer
from .features import FeatureAnalyzer

__all__ = ["SAETrainer", "FeatureAnalyzer"]
