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


def _coerce_signals(payload: Any) -> ProvenanceSignals:
    """Pull EAS/END/NLF/LEP/WVC out of MPK's signal-bearing object.

    MPK's released types are still in flux; we accept either a dict or
    an object with attributes, and zero-default missing keys rather
    than crashing on a schema drift in a 1.0.0 dependency.
    """
    def _g(name: str) -> float:
        if isinstance(payload, dict):
            v = payload.get(name) or payload.get(name.lower()) or payload.get(name.upper())
        else:
            v = getattr(payload, name, None) or getattr(payload, name.lower(), None)
        if v is None:
            return 0.0
        return float(v)

    return ProvenanceSignals(
        eas=_g("eas"),
        end=_g("end"),
        nlf=_g("nlf"),
        lep=_g("lep"),
        wvc=_g("wvc"),
    )


def _coerce_match(payload: Any, *, fallback_asset: str = "") -> ProvenanceMatch:
    """Build a :class:`ProvenanceMatch` from a single MPK match record."""

    def _g(*names: str, default: Any = None) -> Any:
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

    asset = str(_g("asset", "asset_id", "model_id", "model", default=fallback_asset))
    family = str(_g("family", "family_id", default="unknown"))
    score = float(_g("composite_score", "score", default=0.0))
    signals_raw = _g("signals", "signal_scores", default={})
    signals = _coerce_signals(signals_raw)
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
        try:
            from importlib.metadata import version as _v
            return f"mpk-{_v('provenancekit')}"
        except Exception:
            return "mpk-unknown"

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
        return ProvenanceResult(
            model=model_a,
            method="mpk",
            method_version=self._version(),
            threshold=threshold,
            matches=[match],
            is_match=match.composite_score >= threshold,
            metadata={"mode": "compare"},
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
        return ProvenanceResult(
            model=model,
            method="mpk",
            method_version=self._version(),
            threshold=threshold,
            matches=matches,
            is_match=bool(matches) and matches[0].composite_score >= threshold,
            metadata={"mode": "scan", "top_k": top_k},
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
