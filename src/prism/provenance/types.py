"""Typed result objects for Model Provenance Kit (MPK) integration.

PRISM wraps Cisco's Model Provenance Kit (Apache-2.0, released
2026-05-04) to surface "is this model derived from X?" alongside the
existing geometry/NLA scans.  These dataclasses are the wire format
PRISM exposes; the actual MPK call lives in :mod:`prism.provenance.client`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


#: Threshold above which MPK considers two models to share provenance.
#: Sourced from the MPK README; do not change without rereading the
#: 111-pair benchmark calibration.
DEFAULT_PROVENANCE_THRESHOLD = 0.70


@dataclass
class ProvenanceSignals:
    """The five weight-level statistics MPK computes for every comparison.

    All values are in ``[0.0, 1.0]``.  Higher means more similar.  See
    ``docs/PROVENANCE.md`` for what each signal actually measures and the
    architectural assumptions baked into it.
    """

    eas: float  # Embedding Anchor Similarity
    end: float  # Embedding Norm Distribution
    nlf: float  # Norm Layer Fingerprint
    lep: float  # Layer Energy Profile
    wvc: float  # Weight-Value Cosine

    def as_dict(self) -> Dict[str, float]:
        return {"eas": self.eas, "end": self.end, "nlf": self.nlf, "lep": self.lep, "wvc": self.wvc}


@dataclass
class ProvenanceMatch:
    """A single (model, score) hit from a scan against MPK's database."""

    asset: str
    """The reference model the candidate matched against (HF model id)."""

    family: str
    """MPK family label (e.g. ``"gemma-3"``, ``"llama-3"``)."""

    composite_score: float
    """Aggregate identity score, ``[0.0, 1.0]``."""

    signals: ProvenanceSignals


@dataclass
class ProvenanceResult:
    """Result of a single ``scan_model_provenance`` or ``compare_models`` call.

    Attributes:
        model: The model that was scanned/compared (HF id or local path).
        method: Identifies the backend that produced this result.  ``"mpk"``
            for real Cisco MPK; ``"mock"`` for the deterministic mock.
        method_version: Version string of the backend.
        threshold: Composite score above which ``is_match`` is ``True``.
        matches: Sorted (descending by composite_score) list of hits.
            For pairwise compare, this is a single-element list.
        is_match: ``True`` iff the top match's composite score >= threshold.
        not_cryptographic: Always ``True`` — surfaced as a field so any
            downstream consumer that serialises the result automatically
            carries Cisco's "strong evidence, not absolute proof" caveat.
        metadata: Free-form dict (timestamps, signal weights, etc.).
    """

    model: str
    method: str
    method_version: str
    threshold: float
    matches: List[ProvenanceMatch]
    is_match: bool
    not_cryptographic: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_match(self) -> Optional[ProvenanceMatch]:
        return self.matches[0] if self.matches else None

    @property
    def composite_score(self) -> float:
        """Convenience accessor — top match's composite score, or 0.0."""
        return self.top_match.composite_score if self.top_match else 0.0

    def as_audit_dict(self) -> Dict[str, Any]:
        """Render as a flat JSON-friendly dict for receipt/audit logs.

        Includes the ``not_cryptographic`` flag verbatim so audit
        consumers cannot accidentally drop the caveat.
        """
        return {
            "model": self.model,
            "method": self.method,
            "method_version": self.method_version,
            "threshold": self.threshold,
            "is_match": self.is_match,
            "not_cryptographic": self.not_cryptographic,
            "composite_score": self.composite_score,
            "top_match_asset": self.top_match.asset if self.top_match else None,
            "top_match_family": self.top_match.family if self.top_match else None,
            "top_match_signals": (
                self.top_match.signals.as_dict() if self.top_match else None
            ),
            "n_matches": len(self.matches),
            "metadata": self.metadata,
        }
