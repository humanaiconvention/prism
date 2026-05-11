"""Unit tests for prism.provenance.

Covers the mock backend, the MPK adapter (with an injected fake
scanner — no real MPK install required), result schema invariants,
and audit-dict serialisation.  No network and no GPU.
"""

from __future__ import annotations

import pytest

from prism.provenance import (
    DEFAULT_PROVENANCE_THRESHOLD,
    MPKBackend,
    ProvenanceMatch,
    ProvenanceResult,
    ProvenanceSignals,
    compare_models,
    mock_compare,
    mock_scan,
    scan_model_provenance,
)


# ─── mock backend ───────────────────────────────────────────────────────────

class TestMockCompare:
    def test_identical_models_score_one(self):
        r = mock_compare("foo/bar", "foo/bar")
        assert r.composite_score == 1.0
        assert r.is_match

    def test_family_substring_match_above_threshold(self):
        r = mock_compare("haic-gemma4-v42", "google/gemma4-e2b-it")
        assert r.composite_score >= DEFAULT_PROVENANCE_THRESHOLD
        assert r.is_match

    def test_unrelated_models_below_threshold(self):
        r = mock_compare("totally/unrelated-thing-x", "anthropic/claude")
        assert r.composite_score < DEFAULT_PROVENANCE_THRESHOLD
        assert not r.is_match

    def test_score_override(self):
        r = mock_compare("a", "b", score=0.95)
        assert r.composite_score == pytest.approx(0.95)

    def test_signals_in_unit_interval(self):
        r = mock_compare("haic-gemma4-v42", "google/gemma4-e2b-it")
        s = r.top_match.signals
        for v in (s.eas, s.end, s.nlf, s.lep, s.wvc):
            assert 0.0 <= v <= 1.0

    def test_deterministic(self):
        r1 = mock_compare("x/y", "p/q")
        r2 = mock_compare("x/y", "p/q")
        assert r1.composite_score == r2.composite_score
        assert r1.top_match.signals.as_dict() == r2.top_match.signals.as_dict()


class TestMockScan:
    def test_returns_top_k(self):
        r = mock_scan("haic-gemma4-v42", top_k=2)
        assert len(r.matches) == 2

    def test_matches_sorted_descending(self):
        r = mock_scan("haic-gemma4-v42", top_k=4)
        scores = [m.composite_score for m in r.matches]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates(self):
        r = mock_scan("foo", candidate_assets=[])
        assert r.matches == []
        assert not r.is_match


# ─── result schema ──────────────────────────────────────────────────────────

class TestResultSchema:
    def test_not_cryptographic_flag_default_true(self):
        r = mock_compare("a", "b")
        assert r.not_cryptographic is True

    def test_audit_dict_includes_caveat(self):
        r = mock_compare("haic-gemma4-v42", "google/gemma4-e2b-it")
        d = r.as_audit_dict()
        # The brief's invariant: serialisers must not be able to drop
        # the not-cryptographic caveat without explicitly removing it.
        assert d["not_cryptographic"] is True
        assert "composite_score" in d
        assert "top_match_signals" in d
        assert d["top_match_asset"] == "google/gemma4-e2b-it"

    def test_top_match_none_when_no_matches(self):
        r = ProvenanceResult(
            model="x",
            method="mock",
            method_version="t",
            threshold=0.7,
            matches=[],
            is_match=False,
        )
        assert r.top_match is None
        assert r.composite_score == 0.0


# ─── MPK adapter w/ injected fake scanner ───────────────────────────────────

class _FakeMPKResult:
    def __init__(self, asset, family, score, signals):
        self.asset = asset
        self.family = family
        self.composite_score = score
        self.signals = signals


class _FakeMPKScanner:
    """Mimics the duck-typed surface of provenancekit.ModelProvenanceScanner."""

    def compare(self, model_a, model_b):
        return _FakeMPKResult(
            asset=model_b,
            family="gemma-4",
            score=0.91,
            signals={"eas": 0.9, "end": 0.85, "nlf": 0.95, "lep": 0.88, "wvc": 0.93},
        )

    def scan(self, model_id, *, top_k=None, threshold=None):
        class _R:
            matches = [
                _FakeMPKResult("google/gemma-4-e2b-it", "gemma-4", 0.92, {}),
                _FakeMPKResult("google/gemma-4-e4b-it", "gemma-4", 0.83, {}),
                _FakeMPKResult("unrelated/thing", "other", 0.41, {}),
            ]
        return _R()


class TestMPKBackend:
    def test_compare_adapts_fake_result(self):
        backend = MPKBackend(scanner=_FakeMPKScanner(), method_version="mpk-1.0.0-fake")
        r = backend.compare("haic-gemma4-v42", "google/gemma-4-e2b-it")
        assert r.method == "mpk"
        assert r.method_version == "mpk-1.0.0-fake"
        assert r.composite_score == pytest.approx(0.91)
        assert r.is_match
        s = r.top_match.signals
        assert s.nlf == pytest.approx(0.95)

    def test_scan_sorts_and_limits(self):
        backend = MPKBackend(scanner=_FakeMPKScanner(), method_version="mpk-1.0.0-fake")
        r = backend.scan("haic-gemma4-v42", top_k=2)
        assert len(r.matches) == 2
        assert r.matches[0].asset == "google/gemma-4-e2b-it"
        assert r.matches[0].composite_score >= r.matches[1].composite_score

    def test_module_helpers_accept_injected_backend(self):
        backend = MPKBackend(scanner=_FakeMPKScanner())
        r = compare_models("haic-gemma4-v42", "google/gemma-4-e2b-it", backend=backend)
        assert r.is_match
        r2 = scan_model_provenance("haic-gemma4-v42", backend=backend, top_k=1)
        assert len(r2.matches) == 1

    def test_real_backend_optional_dependency_error(self):
        """Without injection, MPKBackend should fail with a helpful error
        when provenancekit isn't installed."""
        backend = MPKBackend()  # no scanner injected
        try:
            import provenancekit  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="provenancekit"):
                backend.compare("a", "b")
        else:
            pytest.skip("provenancekit is installed; can't test the missing-dep path")
