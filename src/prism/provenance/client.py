"""Thin wrapper around Cisco's Model Provenance Kit (MPK).

MPK is a separate package — `pip install provenancekit` — that PRISM
treats as an optional dependency.  Users who only need the mock can
operate without installing it at all.  When MPK *is* installed, this
client adapts its ``ModelProvenanceScanner`` API onto PRISM's
:class:`ProvenanceResult` schema.

Design notes
------------
* The MPK ``Settings`` and ``CacheService`` types are accessed lazily,
  so importing :mod:`prism.provenance` is free even on machines where
  MPK is not installed.  The first call to :class:`MPKBackend` raises
  a clear ImportError pointing at ``pip install provenancekit``.
* The MPK reference dataset is ~908 MB.  We surface ``cache_dir`` as
  an explicit knob so callers can pin where it lives and reuse it
  across runs.
* Cisco's documentation is explicit that MPK output is statistical
  evidence, not cryptographic proof.  Every :class:`ProvenanceResult`
  produced by this backend carries ``not_cryptographic=True``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Protocol

from .types import (
    DEFAULT_PROVENANCE_THRESHOLD,
    ProvenanceMatch,
    ProvenanceResult,
    ProvenanceSignals,
)


class _ScannerLike(Protocol):
    """Minimal duck-typed interface PRISM uses against any provenance backend."""

    def scan(self, model_id: str, *, top_k: int | None = None, threshold: float | None = None) -> Any: ...
    def compare(self, model_a: str, model_b: str) -> Any: ...


def _g(payload: Any, *names: str, default: Any = None) -> Any:
    """Return the first present attribute/key from *names*, or *default*.

    Used to robustly read MPK fields whose names may shift between
    minor versions or differ between CompareResult and ScanMatch.
    """
    if isinstance(payload, dict):
        for n in names:
            if n in payload:
                return payload[n]
        return default
    for n in names:
        v = getattr(payload, n, None)
        if v is not None:
            return v
    return default


def _coerce_signals(payload: Any) -> ProvenanceSignals:
    """Pull EAS/END/NLF/LEP/WVC out of MPK's signal-bearing object.

    Accepts:
      * a dict mapping signal keys to floats (legacy / test fakes);
      * a ``provenancekit.models.results.SignalScores`` (returned on
        ``CompareResult.signals``); or
      * a ``provenancekit.models.results.ScanMatchScores`` (returned on
        ``ScanMatch.scores`` — flatter shape with all signals inline
        plus ``pipeline_score`` etc).

    Missing values default to 0.0; the dataclass clips to ``[0, 1]``.
    """
    def _read(name: str) -> float:
        v = _g(payload, name, name.lower(), name.upper())
        return float(v) if v is not None else 0.0

    return ProvenanceSignals(
        eas=_read("eas"),
        end=_read("end"),
        nlf=_read("nlf"),
        lep=_read("lep"),
        wvc=_read("wvc"),
    )


def _coerce_score_and_signals(
    payload: Any,
) -> tuple[float, ProvenanceSignals]:
    """Find the pipeline/composite score AND signals, regardless of nesting.

    MPK 1.0.0 puts the composite score in different places depending on
    container type:

      CompareResult:  result.scores.pipeline_score   (signals: result.signals)
      ScanMatch:      match.scores.pipeline_score    (signals: match.scores)

    Test fakes from this codebase put it at the top level as
    ``composite_score`` with ``signals`` next to it.  Resolve all three.
    """
    # First try the test-fake shape: top-level composite_score / signals.
    top_signals = _g(payload, "signals", "signal_scores", default=None)
    top_score = _g(payload, "composite_score", "score", default=None)
    if top_score is not None and top_signals is not None:
        return float(top_score), _coerce_signals(top_signals)

    # MPK 1.0.0 shape: pipeline_score lives inside `.scores`; signals
    # may live on `.signals` (CompareResult) or be flattened into
    # `.scores` (ScanMatch with no separate `.signals` field).
    scores_obj = _g(payload, "scores", "scan_match_scores", "pipeline_score_obj", default=None)
    pipeline_score = _g(scores_obj, "pipeline_score", "score", default=None)
    # MPK CompareResult occasionally surfaces identity_score instead of
    # pipeline_score (when tokenizer_score is None) — fall back.
    if pipeline_score is None:
        pipeline_score = _g(scores_obj, "identity_score", "mfi_score", default=None)
    score = float(pipeline_score) if pipeline_score is not None else 0.0

    signals_source = top_signals
    if signals_source is None:
        # ScanMatchScores carries the five signals inline.
        signals_source = scores_obj
    signals = _coerce_signals(signals_source if signals_source is not None else payload)
    return score, signals


def _coerce_match(payload: Any, *, fallback_asset: str = "") -> ProvenanceMatch:
    """Build a :class:`ProvenanceMatch` from a single MPK record.

    Handles both ``CompareResult`` and ``ScanMatch`` shapes, as well as
    the simpler test-fake shape used in ``tests/test_provenance.py``.
    """
    # asset id: ScanMatch.asset_id / ScanMatch.model_id, CompareResult.model_b
    asset = str(_g(payload, "asset", "asset_id", "model_id", "model_b", "model", default=fallback_asset))
    # family: ScanMatch.family_id / family_name; CompareResult.family_b
    family = str(_g(
        payload,
        "family", "family_name", "family_id", "family_b",
        default="unknown",
    ))
    score, signals = _coerce_score_and_signals(payload)
    return ProvenanceMatch(asset=asset, family=family, composite_score=score, signals=signals)


class MPKBackend:
    """Default :class:`prism.provenance` backend backed by real MPK.

    Args:
        scanner: Pre-constructed ``ModelProvenanceScanner`` instance.
            Useful for dependency injection in tests; when ``None``
            (the default) a fresh scanner is built lazily on first use.
        cache_dir: Optional override for MPK's deep-signals cache
            location.  Passed to ``Settings`` when constructing a new
            scanner.
        method_version: Version label recorded in the result.  Defaults
            to MPK's reported package version when available.
    """

    def __init__(
        self,
        scanner: Optional[_ScannerLike] = None,
        *,
        cache_dir: Optional[str] = None,
        method_version: Optional[str] = None,
    ) -> None:
        self._scanner = scanner
        self._cache_dir = cache_dir
        self._method_version = method_version

    # ----------------------------------------------------------- lazy bootstrap

    def _build_scanner(self) -> _ScannerLike:
        try:
            from provenancekit import ModelProvenanceScanner, Settings  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "prism.provenance's real backend requires the 'provenancekit' "
                "package (pip install provenancekit).  For offline tests, use "
                "prism.provenance.mock_scan / mock_compare instead."
            ) from exc

        settings_kwargs: dict[str, Any] = {}
        if self._cache_dir is not None:
            settings_kwargs["cache_dir"] = self._cache_dir
        settings = Settings(**settings_kwargs) if settings_kwargs else Settings()
        return ModelProvenanceScanner(settings=settings)

    def _scanner_obj(self) -> _ScannerLike:
        if self._scanner is None:
            self._scanner = self._build_scanner()
        return self._scanner

    def _version(self) -> str:
        if self._method_version is not None:
            return self._method_version
        # MPK's PyPI distribution name is 'cisco-ai-provenance-kit'; the
        # import package is 'provenancekit'.  Try both.
        from importlib.metadata import PackageNotFoundError, version as _v
        for dist in ("cisco-ai-provenance-kit", "provenancekit"):
            try:
                return f"mpk-{_v(dist)}"
            except PackageNotFoundError:
                continue
        return "mpk-unknown"

    @staticmethod
    def _mpk_metadata(raw: Any) -> dict[str, Any]:
        """Extract MPK's own decision/interpretation fields into metadata.

        These don't fit cleanly into PRISM's typed schema but are valuable
        for audit logs — Cisco's "Confirmed Match" / "High-Confidence
        Match" labels reflect MPK's tier system more granularly than our
        single boolean ``is_match``.
        """
        out: dict[str, Any] = {}
        scores = _g(raw, "scores", default=None)
        if scores is not None:
            dec = _g(scores, "provenance_decision", default=None)
            if dec is not None:
                out["mpk_decision"] = str(dec)
            tier = _g(scores, "mfi_tier", default=None)
            if tier is not None:
                out["mpk_mfi_tier"] = int(tier)
        # ScanMatch carries decision/match_type at the top level.
        if "mpk_decision" not in out:
            dec = _g(raw, "provenance_decision", "match_type", default=None)
            if dec is not None:
                out["mpk_decision"] = str(dec)
        interpretation = _g(raw, "interpretation", default=None)
        if interpretation is not None:
            label = _g(interpretation, "label", default=None)
            if label is not None:
                out["mpk_interpretation"] = str(label)
        elapsed = _g(raw, "time_seconds", "elapsed_ms", default=None)
        if elapsed is not None:
            out["mpk_elapsed"] = float(elapsed)
        return out

    # ---------------------------------------------------------------- public

    def compare(
        self,
        model_a: str,
        model_b: str,
        *,
        threshold: float = DEFAULT_PROVENANCE_THRESHOLD,
    ) -> ProvenanceResult:
        raw = self._scanner_obj().compare(model_a, model_b)
        match = _coerce_match(raw, fallback_asset=model_b)
        metadata: dict[str, Any] = {"mode": "compare"}
        metadata.update(self._mpk_metadata(raw))
        return ProvenanceResult(
            model=model_a,
            method="mpk",
            method_version=self._version(),
            threshold=threshold,
            matches=[match],
            is_match=match.composite_score >= threshold,
            metadata=metadata,
        )

    def scan(
        self,
        model: str,
        *,
        threshold: float = DEFAULT_PROVENANCE_THRESHOLD,
        top_k: int = 5,
    ) -> ProvenanceResult:
        raw = self._scanner_obj().scan(model, top_k=top_k, threshold=threshold)
        # MPK's ScanResult typically exposes a `matches` iterable.
        if isinstance(raw, dict):
            raw_matches = raw.get("matches", [])
        else:
            raw_matches = getattr(raw, "matches", [])

        matches: List[ProvenanceMatch] = [_coerce_match(m) for m in raw_matches]
        matches.sort(key=lambda m: m.composite_score, reverse=True)
        matches = matches[:top_k]
        metadata: dict[str, Any] = {"mode": "scan", "top_k": top_k}
        # MPK ScanResult carries match_count, elapsed_ms, extract_seconds.
        for k in ("match_count", "elapsed_ms", "extract_seconds", "lookup_seconds"):
            v = _g(raw, k, default=None)
            if v is not None:
                metadata[f"mpk_{k}"] = v
        # Per-match metadata MPK provides that doesn't fit PRISM's typed
        # schema: match_type ('exact_arch' / 'weight_level' / etc.) and
        # provenance_decision ('Confirmed Match' / 'Probable Match' / ...).
        # Match_type especially matters: 'exact_arch' matches do NOT load
        # weights, so the per-match signals are not measured — they come
        # back as 0.0, which is NOT the same as "verified zero similarity."
        if isinstance(raw_matches, (list, tuple)) and raw_matches:
            mpk_match_types: list[str] = []
            mpk_decisions: list[str] = []
            for m in raw_matches[:top_k]:
                mt = _g(m, "match_type", default=None)
                pd = _g(m, "provenance_decision", default=None)
                if mt is not None:
                    mpk_match_types.append(str(mt))
                if pd is not None:
                    mpk_decisions.append(str(pd))
            if mpk_match_types:
                metadata["mpk_match_types"] = mpk_match_types
            if mpk_decisions:
                metadata["mpk_decisions"] = mpk_decisions
                metadata["mpk_top_decision"] = mpk_decisions[0]
        return ProvenanceResult(
            model=model,
            method="mpk",
            method_version=self._version(),
            threshold=threshold,
            matches=matches,
            is_match=bool(matches) and matches[0].composite_score >= threshold,
            metadata=metadata,
        )


# ─── module-level conveniences ───────────────────────────────────────────────


def scan_model_provenance(
    model: str,
    *,
    backend: Optional[Any] = None,
    threshold: float = DEFAULT_PROVENANCE_THRESHOLD,
    top_k: int = 5,
) -> ProvenanceResult:
    """Scan *model* against the configured backend's reference database.

    The default backend is :class:`MPKBackend`, which lazily requires
    the ``provenancekit`` package.  Pass ``backend=`` (any object with
    a ``scan(model, threshold=, top_k=)`` method returning a
    :class:`ProvenanceResult`) to override — including for tests.
    """
    if backend is None:
        backend = MPKBackend()
    return backend.scan(model, threshold=threshold, top_k=top_k)


def compare_models(
    candidate: str,
    parent: str,
    *,
    backend: Optional[Any] = None,
    threshold: float = DEFAULT_PROVENANCE_THRESHOLD,
) -> ProvenanceResult:
    """Pairwise compare two models.

    Returns a :class:`ProvenanceResult` whose ``is_match`` flag tells
    you whether MPK considers *candidate* derived from *parent*.
    """
    if backend is None:
        backend = MPKBackend()
    return backend.compare(candidate, parent, threshold=threshold)
