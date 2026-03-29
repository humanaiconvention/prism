"""Tests for the PRISM telemetry schemas and snapshot logic.

Coverage:
  TestPrismSchemas — shared data contracts for geometric telemetry
"""

import time
import pytest
import torch
import torch.nn as nn
from prism.telemetry.schemas import (
    EntropySnapshot, 
    EntropyDeltaProof, 
    GeometricHealthScore, 
    LayerEntropyProfile,
    MetricSummary,
    SpectralMetricSummary,
    CausalProvenanceReport,
    CircuitReport,
)
from prism.telemetry.snapshot import compute_entropy_delta, compute_geometric_health, take_snapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(gen_idx: int = 0, spectral_entropy: float = 3.0,
                   effective_dimension: float = 200.0, viability: float = 0.5,
                   coherence: float = 0.8) -> EntropySnapshot:
    return EntropySnapshot(
        timestamp=time.time(),
        model_name="test-model",
        generation_idx=gen_idx,
        regime="R1",
        mean_spectral_entropy=spectral_entropy,
        mean_effective_dimension=effective_dimension,
        mean_viability_score=viability,
        mean_fisher_curvature=1.5,
        layer_profiles=[
            LayerEntropyProfile(
                layer_idx=4,
                spectral_entropy=spectral_entropy,
                effective_dimension=effective_dimension,
                viability_score=viability,
                fisher_curvature=1.5,
                spectral_summary=SpectralMetricSummary(
                    spectral_entropy=MetricSummary.from_samples([spectral_entropy, spectral_entropy + 0.1]),
                    effective_dimension=MetricSummary.from_samples([effective_dimension, effective_dimension + 1.0]),
                ),
            )
        ],
        spectral_summary=SpectralMetricSummary(
            spectral_entropy=MetricSummary.from_samples([spectral_entropy, spectral_entropy + 0.2]),
            effective_dimension=MetricSummary.from_samples([effective_dimension, effective_dimension + 2.0]),
        ),
        mean_phase_coherence=coherence,
        noise_sensitivity=0.1,
        n_samples=5,
    )


def _make_delta(delta_se: float, epsilon: float = 0.01) -> EntropyDeltaProof:
    before = _make_snapshot(gen_idx=0, spectral_entropy=3.0)
    after = _make_snapshot(gen_idx=1, spectral_entropy=3.0 + delta_se)
    
    # Use actual compute_entropy_delta function to build the proof
    return compute_entropy_delta(before, after, epsilon=epsilon)


class _DummyBatch(dict):
    def to(self, device):
        return _DummyBatch({key: value.to(device) for key, value in self.items()})


class _DummyTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 512):
        base = max(2, min(5, len(prompt) % 5 + 2))
        tokens = (torch.arange(base).unsqueeze(0) + len(prompt)) % 31
        return _DummyBatch({"input_ids": tokens.long()})


class _DummySnapshotModel(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 2):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "_name_or_path": "dummy-snapshot-model",
            },
        )()
        self.embedding = nn.Embedding(32, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        x = self.embedding(input_ids).float()
        prompt_bias = input_ids.float().mean(dim=1, keepdim=True).unsqueeze(-1) * 0.05
        h0 = x + prompt_bias
        h1 = self.norm(h0 + 0.1)
        h2 = self.norm(h1 + 0.2)
        out = type("Output", (), {})()
        out.hidden_states = [h0, h1, h2]
        return out

# ---------------------------------------------------------------------------
# TestPrismSchemas
# ---------------------------------------------------------------------------

class TestPrismSchemas:
    """Tests for EntropySnapshot, EntropyDeltaProof, GeometricHealthScore."""

    def test_entropy_snapshot_roundtrip(self):
        """to_dict() / from_dict() round-trip preserves all scalar fields."""
        snap = _make_snapshot(gen_idx=2, spectral_entropy=4.1, effective_dimension=150.0)
        d = snap.to_dict()
        snap2 = EntropySnapshot.from_dict(d)

        assert snap2.generation_idx == snap.generation_idx
        assert snap2.regime == snap.regime
        assert abs(snap2.mean_spectral_entropy - snap.mean_spectral_entropy) < 1e-9
        assert abs(snap2.mean_effective_dimension - snap.mean_effective_dimension) < 1e-9
        assert abs(snap2.mean_viability_score - snap.mean_viability_score) < 1e-9
        assert abs(snap2.mean_phase_coherence - snap.mean_phase_coherence) < 1e-9
        assert abs(snap2.noise_sensitivity - snap.noise_sensitivity) < 1e-9
        assert snap2.n_samples == snap.n_samples

    def test_entropy_snapshot_layer_profiles_roundtrip(self):
        """Layer profiles are preserved through serialization."""
        snap = _make_snapshot()
        d = snap.to_dict()
        snap2 = EntropySnapshot.from_dict(d)
        assert len(snap2.layer_profiles) == 1
        lp = snap2.layer_profiles[0]
        assert lp.layer_idx == 4
        assert abs(lp.spectral_entropy - snap.mean_spectral_entropy) < 1e-9
        assert lp.spectral_summary is not None
        assert lp.spectral_summary.spectral_entropy is not None
        assert lp.spectral_summary.spectral_entropy.n_samples == 2

    def test_snapshot_spectral_summary_roundtrip(self):
        snap = _make_snapshot()
        d = snap.to_dict()
        snap2 = EntropySnapshot.from_dict(d)
        assert snap2.spectral_summary is not None
        assert snap2.spectral_summary.spectral_entropy is not None
        assert snap2.spectral_summary.effective_dimension is not None
        assert snap2.spectral_summary.spectral_entropy.ci_high >= snap2.spectral_summary.spectral_entropy.ci_low

    def test_circuit_report_roundtrip(self):
        provenance = CausalProvenanceReport(
            layer_idx=3,
            pre_state_rank=10.0,
            post_attention_rank=8.0,
            post_mlp_rank=9.0,
            attention_rank_delta=-2.0,
            mlp_rank_delta=1.0,
            pre_to_post_attention_cosine=0.4,
            pre_to_post_mlp_cosine=0.7,
            route_confidence=0.55,
        )
        report = CircuitReport(
            kind="full_scan",
            prompt="hello",
            model_name="test-model",
            metadata={"target_layer": 3},
            sections={"static_circuits": [{"head": 0, "ov_rank": 1.2}]},
            provenance=provenance,
        )
        restored = CircuitReport.from_dict(report.to_dict())
        assert restored.kind == "full_scan"
        assert restored.sections["static_circuits"][0]["head"] == 0
        assert restored.provenance is not None
        assert restored.provenance.layer_idx == 3

    def test_take_snapshot_includes_spectral_stability(self):
        model = _DummySnapshotModel()
        snapshot = take_snapshot(
            model=model,
            tokenizer=_DummyTokenizer(),
            eval_prompts=["short prompt", "a much longer prompt for variance"],
            layers_to_profile=[1],
            n_samples=2,
        )

        assert snapshot.spectral_summary is not None
        assert snapshot.spectral_summary.spectral_entropy is not None
        assert snapshot.spectral_summary.effective_dimension is not None
        assert snapshot.spectral_summary.spectral_entropy.n_samples == 2
        assert snapshot.spectral_summary.spectral_entropy.ci_high >= snapshot.spectral_summary.spectral_entropy.ci_low
        assert snapshot.layer_profiles
        assert snapshot.layer_profiles[0].spectral_summary is not None
        assert snapshot.layer_profiles[0].spectral_summary.spectral_entropy is not None
        assert snapshot.layer_profiles[0].spectral_summary.spectral_entropy.n_samples == 2
        assert snapshot.mean_spectral_entropy == pytest.approx(snapshot.spectral_summary.spectral_entropy.mean)

    def test_entropy_delta_verified_when_negative(self):
        """delta_spectral_entropy < −ε → reduction_verified=True."""
        proof = _make_delta(delta_se=-0.5, epsilon=0.01)
        assert proof.reduction_verified is True

    def test_entropy_delta_not_verified_when_positive(self):
        """delta_spectral_entropy > 0 → reduction_verified=False."""
        proof = _make_delta(delta_se=0.2, epsilon=0.01)
        assert proof.reduction_verified is False

    def test_entropy_delta_not_verified_below_epsilon(self):
        """delta_spectral_entropy = −0.005 (< ε=0.01) → reduction_verified=False."""
        proof = _make_delta(delta_se=-0.005, epsilon=0.01)
        assert proof.reduction_verified is False

    def test_entropy_delta_roundtrip(self):
        """to_dict() preserves delta and verification flag."""
        proof = _make_delta(delta_se=-0.3)
        d = proof.to_dict()
        assert abs(d["delta_spectral_entropy"] - (-0.3)) < 1e-9
        assert d["reduction_verified"] is True

    def test_geometric_health_score_computation(self):
        """Test compute_geometric_health and range of composite score."""
        from unittest.mock import MagicMock, patch
        
        with patch("prism.telemetry.snapshot.take_snapshot") as mock_snap:
            # Baseline test
            snap = _make_snapshot(viability=0.9, spectral_entropy=1.0)
            snap.noise_sensitivity = 0.05
            mock_snap.return_value = snap
            
            # Create dummy model with config
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            
            ghs = compute_geometric_health(mock_model, MagicMock(), ["prompt"])
            assert isinstance(ghs, GeometricHealthScore)
            assert 0.0 <= ghs.composite_score <= 1.0
            
            # Test edge case bounds
            for vs, se, ns in [(0.9, 1.0, 0.05), (0.1, 7.0, 0.9), (0.5, 4.0, 0.5)]:
                snap_edge = _make_snapshot(viability=vs, spectral_entropy=se)
                snap_edge.noise_sensitivity = ns
                mock_snap.return_value = snap_edge
                ghs_edge = compute_geometric_health(mock_model, MagicMock(), ["prompt"])
                assert 0.0 <= ghs_edge.composite_score <= 1.0

    def test_layer_entropy_profile_roundtrip(self):
        lp = LayerEntropyProfile(layer_idx=7, spectral_entropy=2.3, effective_dimension=128.0,
                                  viability_score=0.63, fisher_curvature=0.9)
        lp2 = LayerEntropyProfile.from_dict(lp.to_dict())
        assert lp2.layer_idx == 7
        assert abs(lp2.spectral_entropy - 2.3) < 1e-9
