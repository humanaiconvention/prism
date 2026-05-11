"""Deterministic mock provenance backend.

Used by tests and by any environment that doesn't want to pull MPK's
908 MB reference dataset.  The mock returns plausible-looking results
keyed on a hash of (candidate, parent) so calls are reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from .types import (
    DEFAULT_PROVENANCE_THRESHOLD,
    ProvenanceMatch,
    ProvenanceResult,
    ProvenanceSignals,
)


_MOCK_VERSION = "mock-0.1.0"


def _h(*parts: str) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(parts).encode("utf-8")).digest()[:8], "big"
    )


def _signals_from_seed(seed: int, base: float) -> ProvenanceSignals:
    """Build five signals clustered around *base* with deterministic jitter."""
    vals = []
    for shift in range(5):
        # 16-bit chunk per signal, mapped to ±0.04 jitter
        chunk = (seed >> (shift * 16)) & 0xFFFF
        jitter = (chunk / 0xFFFF - 0.5) * 0.08
        v = max(0.0, min(1.0, base + jitter))
        vals.append(v)
    return ProvenanceSignals(eas=vals[0], end=vals[1], nlf=vals[2], lep=vals[3], wvc=vals[4])


def mock_compare(
    model_a: str,
    model_b: str,
    *,
    score: Optional[float] = None,
    threshold: float = DEFAULT_PROVENANCE_THRESHOLD,
) -> ProvenanceResult:
    """Deterministic pairwise mock.

    Args:
        model_a: Candidate model id.
        model_b: Reference / claimed parent model id.
        score: Override the composite score directly.  When ``None``,
            a deterministic score is derived from the input ids — equal
            inputs return ~1.0, otherwise ~0.85 if the family substring
            of *model_b* appears in *model_a*, else ~0.30.
        threshold: ``is_match`` cutoff.
    """
    if score is None:
        if model_a == model_b:
            score = 1.0
        else:
            short_b = model_b.split("/")[-1].split("-")[0].lower()
            if short_b and short_b in model_a.lower():
                score = 0.85
            else:
                score = 0.30

    seed = _h(model_a, model_b)
    signals = _signals_from_seed(seed, score)
    match = ProvenanceMatch(
        asset=model_b,
        family=model_b.split("/")[-1].split("-")[0],
        composite_score=float(score),
        signals=signals,
    )
    return ProvenanceResult(
        model=model_a,
        method="mock",
        method_version=_MOCK_VERSION,
        threshold=threshold,
        matches=[match],
        is_match=float(score) >= threshold,
        metadata={"mode": "compare"},
    )


def mock_scan(
    model: str,
    *,
    candidate_assets: Optional[List[str]] = None,
    threshold: float = DEFAULT_PROVENANCE_THRESHOLD,
    top_k: int = 3,
) -> ProvenanceResult:
    """Deterministic scan mock.

    Builds *top_k* fake matches from *candidate_assets* (or a default
    set covering the families MPK's real database is known to cover).
    """
    if candidate_assets is None:
        candidate_assets = [
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
        ]

    matches: List[ProvenanceMatch] = []
    for asset in candidate_assets:
        cmp = mock_compare(model, asset, threshold=threshold)
        if cmp.top_match is not None:
            matches.append(cmp.top_match)

    matches.sort(key=lambda m: m.composite_score, reverse=True)
    matches = matches[:top_k]
    return ProvenanceResult(
        model=model,
        method="mock",
        method_version=_MOCK_VERSION,
        threshold=threshold,
        matches=matches,
        is_match=bool(matches) and matches[0].composite_score >= threshold,
        metadata={"mode": "scan", "n_candidates": len(candidate_assets)},
    )
