"""PRISM telemetry schemas.

Pure dataclasses — no business logic.
These can be used to serialize/deserialize PRISM geometric states.
"""

import math
import time
from dataclasses import dataclass, field, asdict
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class MetricSummary:
    """Repeated-run summary for a scalar metric."""

    mean: float
    variance: float
    std_dev: float
    ci_low: float
    ci_high: float
    n_samples: int
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[float],
        confidence_level: float = 0.95,
    ) -> "MetricSummary":
        values = [float(v) for v in samples if v is not None]
        if not values:
            return cls(
                mean=0.0,
                variance=0.0,
                std_dev=0.0,
                ci_low=0.0,
                ci_high=0.0,
                n_samples=0,
                confidence_level=confidence_level,
            )

        mean = float(sum(values) / len(values))
        if len(values) > 1:
            variance = float(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        else:
            variance = 0.0
        std_dev = math.sqrt(variance)
        if len(values) > 1 and confidence_level > 0.0:
            z = NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)
            margin = z * (std_dev / math.sqrt(len(values)))
        else:
            margin = 0.0
        return cls(
            mean=mean,
            variance=variance,
            std_dev=std_dev,
            ci_low=mean - margin,
            ci_high=mean + margin,
            n_samples=len(values),
            confidence_level=confidence_level,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricSummary":
        return cls(
            mean=float(d["mean"]),
            variance=float(d["variance"]),
            std_dev=float(d["std_dev"]),
            ci_low=float(d["ci_low"]),
            ci_high=float(d["ci_high"]),
            n_samples=int(d["n_samples"]),
            confidence_level=float(d.get("confidence_level", 0.95)),
        )


@dataclass
class SpectralMetricSummary:
    """Repeated-run summary for the two core spectral metrics."""

    spectral_entropy: Optional[MetricSummary] = None
    effective_dimension: Optional[MetricSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spectral_entropy": self.spectral_entropy.to_dict() if self.spectral_entropy else None,
            "effective_dimension": self.effective_dimension.to_dict() if self.effective_dimension else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectralMetricSummary":
        spectral_entropy = d.get("spectral_entropy")
        effective_dimension = d.get("effective_dimension")
        return cls(
            spectral_entropy=MetricSummary.from_dict(spectral_entropy) if spectral_entropy else None,
            effective_dimension=MetricSummary.from_dict(effective_dimension) if effective_dimension else None,
        )


@dataclass
class CausalProvenanceReport:
    """Residual-stream attribution summary tying layer metrics together."""

    layer_idx: int
    pre_state_rank: float
    post_attention_rank: float
    post_mlp_rank: float
    attention_rank_delta: float
    mlp_rank_delta: float
    pre_to_post_attention_cosine: float
    pre_to_post_mlp_cosine: float
    route_confidence: float
    method: str = "residual_route_attribution"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalProvenanceReport":
        return cls(
            layer_idx=int(d["layer_idx"]),
            pre_state_rank=float(d["pre_state_rank"]),
            post_attention_rank=float(d["post_attention_rank"]),
            post_mlp_rank=float(d["post_mlp_rank"]),
            attention_rank_delta=float(d["attention_rank_delta"]),
            mlp_rank_delta=float(d["mlp_rank_delta"]),
            pre_to_post_attention_cosine=float(d["pre_to_post_attention_cosine"]),
            pre_to_post_mlp_cosine=float(d["pre_to_post_mlp_cosine"]),
            route_confidence=float(d["route_confidence"]),
            method=str(d.get("method", "residual_route_attribution")),
        )


@dataclass
class CircuitReport:
    """Shared interpretability artifact for scan and discovery surfaces."""

    kind: str
    prompt: str = ""
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, Any] = field(default_factory=dict)
    telemetry: Optional["EntropySnapshot"] = None
    provenance: Optional[CausalProvenanceReport] = None
    spectral_summary: Optional[SpectralMetricSummary] = None
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "kind": self.kind,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "metadata": self.metadata,
            "generated_at": self.generated_at,
        }
        result.update(self.sections)
        if self.telemetry is not None:
            result["telemetry"] = self.telemetry.to_dict()
        if self.provenance is not None:
            result["causal_provenance"] = self.provenance.to_dict()
        if self.spectral_summary is not None:
            result["spectral_summary"] = self.spectral_summary.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CircuitReport":
        d = dict(d)
        telemetry = d.pop("telemetry", None)
        provenance = d.pop("causal_provenance", None)
        spectral_summary = d.pop("spectral_summary", None)
        generated_at = float(d.pop("generated_at", time.time()))
        kind = str(d.pop("kind", ""))
        prompt = str(d.pop("prompt", ""))
        model_name = str(d.pop("model_name", ""))
        metadata = dict(d.pop("metadata", {}))
        return cls(
            kind=kind,
            prompt=prompt,
            model_name=model_name,
            metadata=metadata,
            sections=d,
            telemetry=EntropySnapshot.from_dict(telemetry) if telemetry else None,
            provenance=CausalProvenanceReport.from_dict(provenance) if provenance else None,
            spectral_summary=SpectralMetricSummary.from_dict(spectral_summary) if spectral_summary else None,
            generated_at=generated_at,
        )

@dataclass
class LayerEntropyProfile:
    """Per-layer geometric state captured by PRISM at a single point in time."""
    layer_idx: int
    spectral_entropy: float     # Shannon entropy of the eigenvalue distribution
    effective_dimension: float  # Shannon effective rank = exp(spectral_entropy)
    viability_score: float      # effective_dimension / hidden_dim  ∈ [0, 1]; higher = healthier
    fisher_curvature: float     # Trace FIM approximation (sum of activation variances)
    spectral_summary: Optional[SpectralMetricSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.spectral_summary is not None:
            d["spectral_summary"] = self.spectral_summary.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerEntropyProfile":
        spectral_summary = d.get("spectral_summary")
        return cls(
            layer_idx=int(d["layer_idx"]),
            spectral_entropy=float(d["spectral_entropy"]),
            effective_dimension=float(d["effective_dimension"]),
            viability_score=float(d["viability_score"]),
            fisher_curvature=float(d["fisher_curvature"]),
            spectral_summary=SpectralMetricSummary.from_dict(spectral_summary) if spectral_summary else None,
        )

@dataclass
class EntropySnapshot:
    """Complete geometric state of a model at a point in time."""
    timestamp: float = field(default_factory=time.time)
    model_name: str = ""
    generation_idx: int = -1    # generation index; -1 if standalone
    regime: str = ""            # regime label or ""

    # Aggregate spectral metrics
    mean_spectral_entropy: float = 0.0
    mean_effective_dimension: float = 0.0
    mean_viability_score: float = 0.0
    mean_fisher_curvature: float = 0.0
    spectral_summary: Optional[SpectralMetricSummary] = None

    # Per-layer breakdown
    layer_profiles: List[LayerEntropyProfile] = field(default_factory=list)

    # Phase coherence: mean PLV between first and last sampled layer
    mean_phase_coherence: float = 0.0

    # Representational noise sensitivity
    noise_sensitivity: float = 0.0

    # Number of eval prompts
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layer_profiles"] = [lp.to_dict() for lp in self.layer_profiles]
        if self.spectral_summary is not None:
            d["spectral_summary"] = self.spectral_summary.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntropySnapshot":
        d = dict(d)
        layer_profiles = [LayerEntropyProfile.from_dict(lp) for lp in d.pop("layer_profiles", [])]
        spectral_summary = d.pop("spectral_summary", None)
        return cls(
            **d,
            layer_profiles=layer_profiles,
            spectral_summary=SpectralMetricSummary.from_dict(spectral_summary) if spectral_summary else None,
        )


@dataclass
class EntropyDeltaProof:
    """Proof of entropy reduction between two model states."""
    snapshot_before: EntropySnapshot
    snapshot_after: EntropySnapshot

    delta_spectral_entropy: float = 0.0       # < 0 means entropy decreased 
    delta_effective_dimension: float = 0.0    # > 0 means more dimensions utilized 
    delta_viability_score: float = 0.0        # > 0 means healthier geometry 
    delta_phase_coherence: float = 0.0        # > 0 means more coherent 

    epsilon_threshold: float = 0.01           # Minimum required reduction |ΔS| > ε
    reduction_verified: bool = False          # True iff delta_spectral_entropy < −epsilon
    cka_drift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_before": self.snapshot_before.to_dict(),
            "snapshot_after": self.snapshot_after.to_dict(),
            "delta_spectral_entropy": self.delta_spectral_entropy,
            "delta_effective_dimension": self.delta_effective_dimension,
            "delta_viability_score": self.delta_viability_score,
            "delta_phase_coherence": self.delta_phase_coherence,
            "epsilon_threshold": self.epsilon_threshold,
            "reduction_verified": self.reduction_verified,
            "cka_drift": self.cka_drift,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntropyDeltaProof":
        d = dict(d)
        snapshot_before = EntropySnapshot.from_dict(d.pop("snapshot_before"))
        snapshot_after = EntropySnapshot.from_dict(d.pop("snapshot_after"))
        return cls(snapshot_before=snapshot_before, snapshot_after=snapshot_after, **d)


@dataclass
class GeometricHealthScore:
    """Summary of a model's geometric health."""
    viability_score: float      
    spectral_entropy: float     
    noise_sensitivity: float    
    composite_score: float      

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeometricHealthScore":
        return cls(**d)
