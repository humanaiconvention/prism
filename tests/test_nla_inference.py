"""Unit tests for prism.nla — mock explainer and HTTP client."""

from __future__ import annotations

import numpy as np
import pytest

from prism.nla import (
    NLABatchResult,
    NLACheckpoint,
    NLAExplainer,
    NLAExplanation,
    find_for_target,
    get_checkpoint,
    is_supported,
    list_checkpoints,
    mock_explainer,
    summarize_layer,
)


# ─── mock determinism ───────────────────────────────────────────────────────

class TestMockExplainer:
    def test_returns_nla_explanation(self):
        exp = mock_explainer(d_model=8, layer_idx=2)
        r = exp.explain(np.zeros(8))
        assert isinstance(r, NLAExplanation)
        assert isinstance(r.text, str)
        assert isinstance(r.reconstruction_fve, float)
        assert isinstance(r.reconstructed_vector, np.ndarray)

    def test_text_format_marker(self):
        exp = mock_explainer(d_model=4, layer_idx=12)
        r = exp.explain([0.1, 0.2, 0.3, 0.4])
        # The brief requires this prefix so tests can detect the mock.
        assert r.text.startswith("[mock NLA explanation for ")

    def test_fve_in_documented_band(self):
        exp = mock_explainer(d_model=16, layer_idx=0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            r = exp.explain(rng.normal(size=16))
            assert 0.4 <= r.reconstruction_fve <= 0.6

    def test_deterministic_same_vector(self):
        exp = mock_explainer(d_model=16, layer_idx=5)
        v = np.linspace(-1, 1, 16, dtype=np.float32)
        r1 = exp.explain(v)
        r2 = exp.explain(v)
        assert r1.text == r2.text
        assert r1.reconstruction_fve == r2.reconstruction_fve

    def test_different_vectors_can_differ(self):
        exp = mock_explainer(d_model=16, layer_idx=5)
        v1 = np.zeros(16, dtype=np.float32)
        v2 = np.ones(16, dtype=np.float32)
        # Not all hash buckets must differ, but at least the FVE seeds do —
        # which means either text or fve will change.
        r1 = exp.explain(v1)
        r2 = exp.explain(v2)
        assert (r1.text, r1.reconstruction_fve) != (r2.text, r2.reconstruction_fve)

    def test_reconstruction_shape_matches_input(self):
        exp = mock_explainer(d_model=32, layer_idx=3)
        v = np.random.randn(32).astype(np.float32)
        r = exp.explain(v)
        assert r.reconstructed_vector.shape == v.shape

    def test_explain_batch_shape(self):
        exp = mock_explainer(d_model=8, layer_idx=1)
        rng = np.random.default_rng(42)
        batch = [rng.normal(size=8) for _ in range(5)]
        results = exp.explain_batch(batch)
        assert len(results) == 5
        assert all(isinstance(r, NLAExplanation) for r in results)

    def test_rejects_wrong_d_model(self):
        exp = mock_explainer(d_model=8, layer_idx=1)
        with pytest.raises(ValueError, match="d_model"):
            exp.explain(np.zeros(7))


# ─── NLAExplanation schema ──────────────────────────────────────────────────

class TestSchema:
    def test_fve_clipped_to_unit_interval(self):
        r = NLAExplanation(text="x", reconstruction_fve=1.7)
        assert r.reconstruction_fve == 1.0
        r2 = NLAExplanation(text="x", reconstruction_fve=-0.3)
        assert r2.reconstruction_fve == 0.0

    def test_metadata_independent_per_instance(self):
        a = NLAExplanation(text="a", reconstruction_fve=0.5)
        b = NLAExplanation(text="b", reconstruction_fve=0.5)
        a.metadata["k"] = "v"
        assert "k" not in b.metadata

    def test_text_must_be_str(self):
        with pytest.raises(TypeError):
            NLAExplanation(text=123, reconstruction_fve=0.5)  # type: ignore[arg-type]


# ─── registry ───────────────────────────────────────────────────────────────

class TestRegistry:
    def test_known_checkpoints_present(self):
        ids = {c.nla_id for c in list_checkpoints()}
        assert "kitft/nla-gemma-3-12b-it-layer32" in ids
        assert "kitft/nla-llama-3.3-70b-instruct-layer53" in ids

    def test_no_gemma4_nla(self):
        """The brief is explicit: there is no released Gemma-4 NLA."""
        assert not is_supported("google/gemma-4-e2b-it")
        assert find_for_target("google/gemma-4-e2b-it") == []

    def test_get_checkpoint_unknown_returns_none(self):
        assert get_checkpoint("does/not-exist") is None

    def test_checkpoints_carry_license(self):
        # The kitft release is Apache-2.0; PRISM is CC BY 4.0.  Both must
        # be discoverable from the metadata.
        for c in list_checkpoints():
            assert c.license == "Apache-2.0"

    def test_from_pretrained_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="No NLA checkpoint"):
            NLAExplainer.from_pretrained(
                "kitft/no-such-thing", server_url="http://localhost:8000"
            )


# ─── HTTP client w/ stub transport ──────────────────────────────────────────

class TestHTTPClient:
    def _ckpt(self):
        return NLACheckpoint(
            nla_id="test/dummy",
            target_model="test/model",
            target_layer=4,
            total_layers=8,
            d_model=16,
        )

    def test_explain_calls_transport_with_expected_payload(self):
        recorded = {}

        def transport(payload):
            recorded.update(payload)
            return {"text": "hello", "reconstruction_fve": 0.72}

        exp = NLAExplainer(self._ckpt(), server_url="http://x", transport=transport)
        r = exp.explain(np.arange(16, dtype=np.float32))
        assert recorded["nla_id"] == "test/dummy"
        assert recorded["layer_idx"] == 4
        assert len(recorded["activation_vector"]) == 16
        assert r.text == "hello"
        assert r.reconstruction_fve == 0.72
        assert r.metadata["backend"] == "http"

    def test_requires_server_url_or_transport(self):
        with pytest.raises(ValueError, match="server_url"):
            NLAExplainer(self._ckpt())

    def test_batch_payload_routes_through_transport(self):
        def transport(payload):
            n = len(payload["activation_vectors"])
            return {
                "results": [
                    {"text": f"sample {i}", "reconstruction_fve": 0.5}
                    for i in range(n)
                ]
            }

        exp = NLAExplainer(self._ckpt(), server_url="http://x", transport=transport)
        results = exp.explain_batch([np.zeros(16), np.ones(16), np.full(16, 2.0)])
        assert len(results) == 3
        assert results[0].text == "sample 0"

    def test_response_missing_text_raises(self):
        exp = NLAExplainer(self._ckpt(), server_url="http://x", transport=lambda p: {})
        with pytest.raises(ValueError, match="text"):
            exp.explain(np.zeros(16))


# ─── summary aggregation ────────────────────────────────────────────────────

class TestSummarizeLayer:
    def test_empty_batch(self):
        r = summarize_layer(layer_idx=3, explanations=[])
        assert isinstance(r, NLABatchResult)
        assert r.n_samples == 0
        assert r.summary == "no samples"

    def test_picks_common_themes(self):
        exps = [
            NLAExplanation(text="syntactic agreement and subject verb", reconstruction_fve=0.5),
            NLAExplanation(text="syntactic agreement features", reconstruction_fve=0.6),
            NLAExplanation(text="agreement between noun phrases", reconstruction_fve=0.4),
        ]
        r = summarize_layer(layer_idx=10, explanations=exps)
        assert r.n_samples == 3
        assert "agreement" in r.summary
        assert 0.4 <= r.mean_fve <= 0.6
