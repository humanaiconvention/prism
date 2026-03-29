"""Entropy reduction dynamics and coupling analysis."""

from .lens import (
    EntropyDynamics, 
    compute_entropy_from_probs, 
    compute_kl_divergence,
    unpack_logits_and_cache,
    compute_sequence_nll_tokenwise,
    compute_sequence_nll_tokenwise_with_traces
)

__all__ = [
    "EntropyDynamics", 
    "compute_entropy_from_probs", 
    "compute_kl_divergence",
    "unpack_logits_and_cache",
    "compute_sequence_nll_tokenwise",
    "compute_sequence_nll_tokenwise_with_traces"
]
