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


# ─── real MPK 1.0.0 schema shape (regression guard) ────────────────────────

class _SignalScoresShape:
    """Mimics provenancekit.models.results.SignalScores."""

    def __init__(self, *, eas=None, nlf=None, lep=None, end=None, wvc=None):
        self.eas = eas
        self.nlf = nlf
        self.lep = lep
        self.end = end
        self.wvc = wvc


class _PipelineScoreShape:
    """Mimics provenancekit.models.results.PipelineScore."""

    def __init__(self, *, pipeline_score, mfi_score=1.0, mfi_tier=1,
                 provenance_decision="Confirmed Match", identity_score=None,
                 tokenizer_score=None):
        self.pipeline_score = pipeline_score
        self.mfi_score = mfi_score
        self.mfi_tier = mfi_tier
        self.provenance_decision = provenance_decision
        self.identity_score = identity_score
        self.tokenizer_score = tokenizer_score


class _CompareResultShape:
    """Mimics provenancekit.models.results.CompareResult — the real schema
    landed by MPK 1.0.0 (verified against the installed package on
    2026-05-11).  The original adapter built before this probe assumed
    a flat composite_score/signals shape and silently returned
    is_match=False on identical models; this test guards against that
    regression."""

    def __init__(self, *, model_a, model_b, family_a, family_b,
                 pipeline_score, signals_dict, interpretation_label="High-Confidence Match"):
        self.model_a = model_a
        self.model_b = model_b
        self.family_a = family_a
        self.family_b = family_b
        self.signals = _SignalScoresShape(**signals_dict)
        self.scores = _PipelineScoreShape(pipeline_score=pipeline_score)
        self.interpretation = type("I", (), {"label": interpretation_label, "colour": "#2ecc71"})()
        self.time_seconds = 3.5


class _ScanMatchScoresShape:
    """Mimics provenancekit.models.results.ScanMatchScores.

    ScanMatch tier-1 (exact_arch) matches do NOT load weights, so the
    five signals come back as None — not 0.0.  PRISM's coercion clips
    None to 0.0 but records ``match_type`` in metadata so audit
    consumers know to read pipeline_score, not signals.
    """

    def __init__(self, *, pipeline_score, eas=None, nlf=None, lep=None, end=None, wvc=None):
        self.pipeline_score = pipeline_score
        self.identity_score = None
        self.mfi_score = pipeline_score
        self.mfi_tier = 1
        self.mfi_match_type = "exact"
        self.tokenizer_score = None
        self.eas = eas
        self.nlf = nlf
        self.lep = lep
        self.end = end
        self.wvc = wvc
        self.tfv = None


class _ScanMatchShape:
    def __init__(self, *, asset_id, family_id, family_name, pipeline_score,
                 match_type="exact_arch", provenance_decision="Confirmed Match"):
        self.asset_id = asset_id
        self.model_id = asset_id
        self.family_id = family_id
        self.family_name = family_name
        self.param_bucket = "7B"
        self.match_type = match_type
        self.provenance_decision = provenance_decision
        self.scores = _ScanMatchScoresShape(pipeline_score=pipeline_score)
        self.elapsed_ms = 100.0


class _ScanResultShape:
    def __init__(self, matches):
        self.model_info = type("I", (), {"model_id": "test"})()
        self.matches = matches
        self.match_count = len(matches)
        self.elapsed_ms = 1234.5
        self.extract_seconds = 1.0
        self.lookup_seconds = 0.2


class _RealSchemaMPKScanner:
    """Returns objects that match MPK 1.0.0's real CompareResult / ScanResult."""

    def compare(self, model_a, model_b, *, on_phase=None):
        return _CompareResultShape(
            model_a=model_a,
            model_b=model_b,
            family_a="qwen",
            family_b="qwen",
            pipeline_score=1.0,
            signals_dict={"eas": 1.0, "nlf": 1.0, "lep": 1.0, "end": 1.0, "wvc": 1.0},
        )

    def scan(self, model_id, *, top_k=None, threshold=None, on_phase=None):
        return _ScanResultShape(matches=[
            _ScanMatchShape(asset_id="Qwen2.5-7B-Instruct__hf-safetensors",
                            family_id="qwen", family_name="Qwen",
                            pipeline_score=1.0),
            _ScanMatchShape(asset_id="Qwen2-7B-Instruct__hf-safetensors",
                            family_id="qwen", family_name="Qwen",
                            pipeline_score=0.92),
        ])


class TestMPKRealSchema:
    """Regression guards built from the real provenancekit==1.0.0 surface.

    These tests would have caught the bug found on 2026-05-11 where the
    initial duck-typed adapter returned is_match=False, composite_score=0.0
    on identical models because it looked for ``composite_score`` and
    ``signals`` at the top level instead of inside ``.scores`` /
    ``.signals``.
    """

    def test_compare_extracts_pipeline_score(self):
        backend = MPKBackend(scanner=_RealSchemaMPKScanner())
        r = backend.compare("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
        assert r.is_match
        assert r.composite_score == 1.0
        assert r.top_match.family == "qwen"
        assert r.top_match.signals.nlf == 1.0
        assert r.metadata["mpk_decision"] == "Confirmed Match"
        assert r.metadata["mpk_interpretation"] == "High-Confidence Match"
        assert r.metadata["mpk_mfi_tier"] == 1

    def test_scan_extracts_per_match_pipeline_scores(self):
        backend = MPKBackend(scanner=_RealSchemaMPKScanner())
        r = backend.scan("Qwen/Qwen2.5-7B-Instruct", top_k=2)
        assert r.is_match
        assert len(r.matches) == 2
        assert r.composite_score == 1.0
        # Tier-1 arch matches: weight-level signals are None in MPK; we
        # clip to 0.0 but record match_type in metadata.
        assert r.top_match.signals.eas == 0.0
        assert r.metadata["mpk_match_types"] == ["exact_arch", "exact_arch"]
        assert r.metadata["mpk_top_decision"] == "Confirmed Match"
        assert r.metadata["mpk_extract_seconds"] == 1.0

    def test_scan_match_with_none_signals_does_not_crash(self):
        """The exact_arch regression: None values must not crash the adapter."""
        backend = MPKBackend(scanner=_RealSchemaMPKScanner())
        r = backend.scan("Qwen/Qwen2.5-7B-Instruct", top_k=1)
        # ProvenanceSignals __post_init__ would have caught a non-float here
        for v in r.top_match.signals.as_dict().values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0
