"""Deterministic mock NLA for testing.

The mock implements the same :class:`NLAExplainer`-style surface used by
the real HTTP backend but does not call any external service. It generates
a deterministic text and FVE from a hash of the input vector so tests are
reproducible without GPUs or network access.

The mock is exposed as ``prism.nla.mock_explainer`` and is the primary
fixture used in :mod:`prism.tests.test_nla_inference` and
:mod:`prism.tests.test_nla_geometry`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .types import NLAExplanation


_TOPIC_WORDS = [
    "cause-and-effect reasoning",
    "spatial relationships",
    "numerical comparison",
    "named entity tracking",
    "syntactic agreement",
    "sentiment polarity",
    "temporal ordering",
    "negation handling",
    "discourse coherence",
    "domain-specific terminology",
]


def _hash_vector(vec: np.ndarray) -> int:
    """Stable hash of a 1-D array's bytes."""
    return int.from_bytes(
        hashlib.sha256(np.ascontiguousarray(vec, dtype=np.float32).tobytes()).digest()[:8],
        "big",
    )


class MockNLAExplainer:
    """Deterministic mock NLA.

    The mock conforms to the protocol expected by
    :func:`prism.geometry.scan_model_geometry`: ``explain`` for a single
    activation, ``explain_batch`` for many, and a ``d_model`` / ``layer_idx``
    pair describing what shape it expects.
    """

    def __init__(self, d_model: int, layer_idx: int, *, model_id: str = "mock") -> None:
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.model_id = model_id

    def _coerce(self, activation_vector: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(activation_vector, dtype=np.float32).ravel()
        if arr.shape[0] != self.d_model:
            raise ValueError(
                f"Mock NLA configured for d_model={self.d_model} but received "
                f"vector of length {arr.shape[0]}."
            )
        return arr

    def explain(self, activation_vector: Sequence[float] | np.ndarray) -> NLAExplanation:
        arr = self._coerce(activation_vector)
        h = _hash_vector(arr)
        topic = _TOPIC_WORDS[h % len(_TOPIC_WORDS)]
        # FVE deterministically in [0.4, 0.6] — matches the band described
        # in the brief and stays well clear of the "release-quality" 0.6-0.8
        # range so tests never confuse mock output with real output.
        fve = 0.4 + ((h >> 8) & 0xFFFF) / 0xFFFF * 0.2

        text = (
            f"[mock NLA explanation for layer {self.layer_idx}, model={self.model_id}] "
            f"activation appears to encode {topic}."
        )

        # Trivial reconstruction: scale the input by FVE so callers that
        # exercise reconstruction_vector can verify shape and dtype without
        # us pretending to compress anything.
        reconstruction = (arr * float(fve)).astype(np.float32)

        return NLAExplanation(
            text=text,
            reconstruction_fve=float(fve),
            reconstructed_vector=reconstruction,
            metadata={
                "model_id": self.model_id,
                "layer_idx": self.layer_idx,
                "d_model": self.d_model,
                "backend": "mock",
            },
        )

    def explain_batch(
        self, activation_vectors: Sequence[Sequence[float] | np.ndarray]
    ) -> List[NLAExplanation]:
        return [self.explain(v) for v in activation_vectors]


def mock_explainer(d_model: int, layer_idx: int, *, model_id: str = "mock") -> MockNLAExplainer:
    """Convenience constructor matching the user-facing API surface."""
    return MockNLAExplainer(d_model=d_model, layer_idx=layer_idx, model_id=model_id)
