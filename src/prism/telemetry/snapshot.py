import logging
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import torch

from prism.analysis import compute_spectral_metrics
from prism.geometry.viability import GeometricViability
from prism.phase.coherence import PhaseAnalyzer
from .schemas import (
    EntropySnapshot,
    EntropyDeltaProof,
    GeometricHealthScore,
    LayerEntropyProfile,
    MetricSummary,
    SpectralMetricSummary,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch.nn as nn

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_summary(values: List[float], confidence_level: float) -> Optional[MetricSummary]:
    return MetricSummary.from_samples(values, confidence_level=confidence_level) if values else None

def take_snapshot(
    model: "nn.Module",
    tokenizer: Any,
    eval_prompts: List[str],
    generation_idx: int = -1,
    regime: str = "",
    layers_to_profile: Optional[List[int]] = None,
    n_samples: int = 10,
    device: Optional[str] = None,
    confidence_level: float = 0.95,
) -> EntropySnapshot:
    """Capture the geometric state of a model across a sample of eval prompts."""

    model.eval()

    if device is None:
        device = str(next(model.parameters()).device) if list(model.parameters()) else "cpu"

    num_layers = getattr(model.config, "num_hidden_layers", 12)
    hidden_dim = getattr(model.config, "hidden_size", 768)
    model_name = getattr(model.config, "_name_or_path", "")

    if layers_to_profile is None:
        step = max(1, num_layers // 8)
        layers_to_profile = list(range(1, num_layers + 1, step))[:8]

    prompts = eval_prompts[:n_samples]

    all_layer_metrics: Dict[int, List[Dict[str, float]]] = {li: [] for li in layers_to_profile}
    all_phase_plv: List[float] = []
    all_noise_sensitivity: List[float] = []
    prompt_spectral_entropy: List[float] = []
    prompt_effective_dimension: List[float] = []
    prompt_viability_score: List[float] = []
    prompt_fisher_curvature: List[float] = []

    geo_viability = GeometricViability(model)
    phase_analyzer = PhaseAnalyzer(model)

    with torch.no_grad():
        for prompt in prompts:
            try:
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)

                outputs = model(**enc, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                prompt_entropy_values: List[float] = []
                prompt_effective_values: List[float] = []
                prompt_viability_values: List[float] = []
                prompt_fisher_values: List[float] = []

                for li in layers_to_profile:
                    if li >= len(hidden_states):
                        continue
                    h = hidden_states[li][0].float()
                    try:
                        spectral_entropy, effective_dim = compute_spectral_metrics(h)
                        viability = geo_viability.compute_viability_score(effective_dim, hidden_dim)
                        fisher = geo_viability.fisher_information_curvature(h)
                        all_layer_metrics[li].append({
                            "spectral_entropy": float(spectral_entropy),
                            "effective_dimension": float(effective_dim),
                            "viability_score": float(viability),
                            "fisher_curvature": float(fisher),
                        })
                        prompt_entropy_values.append(float(spectral_entropy))
                        prompt_effective_values.append(float(effective_dim))
                        prompt_viability_values.append(float(viability))
                        prompt_fisher_values.append(float(fisher))
                    except Exception as e:
                        logger.debug(f"prism_telemetry: layer {li} metric failed: {e}")

                first_li = layers_to_profile[0]
                last_li = layers_to_profile[-1]
                if first_li < len(hidden_states) and last_li < len(hidden_states):
                    try:
                        h_first = hidden_states[first_li][0].float()
                        h_last = hidden_states[last_li][0].float()
                        phase_first = phase_analyzer.extract_hilbert_phase(h_first)
                        phase_last = phase_analyzer.extract_hilbert_phase(h_last)
                        plv = phase_analyzer.compute_plv(phase_first, phase_last)
                        all_phase_plv.append(float(plv))
                    except Exception as e:
                        logger.debug(f"prism_telemetry: PLV failed: {e}")

                if last_li < len(hidden_states):
                    try:
                        h_last = hidden_states[last_li][0].float()
                        ns = geo_viability.representational_noise_sensitivity(h_last)
                        all_noise_sensitivity.append(float(ns))
                    except Exception as e:
                        logger.debug(f"prism_telemetry: noise sensitivity failed: {e}")

                if prompt_entropy_values:
                    prompt_spectral_entropy.append(_mean(prompt_entropy_values))
                    prompt_effective_dimension.append(_mean(prompt_effective_values))
                    prompt_viability_score.append(_mean(prompt_viability_values))
                    prompt_fisher_curvature.append(_mean(prompt_fisher_values))

            except Exception as e:
                logger.warning(f"prism_telemetry: prompt eval failed, skipping: {e}")
                continue

    layer_profiles: List[LayerEntropyProfile] = []
    for li in layers_to_profile:
        metrics_list = all_layer_metrics[li]
        if not metrics_list:
            continue
        layer_profiles.append(LayerEntropyProfile(
            layer_idx=li,
            spectral_entropy=_mean([m["spectral_entropy"] for m in metrics_list]),
            effective_dimension=_mean([m["effective_dimension"] for m in metrics_list]),
            viability_score=_mean([m["viability_score"] for m in metrics_list]),
            fisher_curvature=_mean([m["fisher_curvature"] for m in metrics_list]),
            spectral_summary=SpectralMetricSummary(
                spectral_entropy=_metric_summary([m["spectral_entropy"] for m in metrics_list], confidence_level),
                effective_dimension=_metric_summary([m["effective_dimension"] for m in metrics_list], confidence_level),
            ),
        ))

    overall_spectral_summary = SpectralMetricSummary(
        spectral_entropy=_metric_summary(prompt_spectral_entropy, confidence_level),
        effective_dimension=_metric_summary(prompt_effective_dimension, confidence_level),
    )

    mean_spectral_entropy = overall_spectral_summary.spectral_entropy.mean if overall_spectral_summary.spectral_entropy else 0.0
    mean_effective_dimension = overall_spectral_summary.effective_dimension.mean if overall_spectral_summary.effective_dimension else 0.0
    mean_viability_score = _mean(prompt_viability_score)
    mean_fisher_curvature = _mean(prompt_fisher_curvature)

    return EntropySnapshot(
        model_name=model_name,
        generation_idx=generation_idx,
        regime=regime,
        mean_spectral_entropy=mean_spectral_entropy,
        mean_effective_dimension=mean_effective_dimension,
        mean_viability_score=mean_viability_score,
        mean_fisher_curvature=mean_fisher_curvature,
        layer_profiles=layer_profiles,
        spectral_summary=overall_spectral_summary,
        mean_phase_coherence=_mean(all_phase_plv) if all_phase_plv else 0.0,
        noise_sensitivity=_mean(all_noise_sensitivity) if all_noise_sensitivity else 0.0,
        n_samples=len(prompts),
    )


def compute_entropy_delta(
    snapshot_before: EntropySnapshot,
    snapshot_after: EntropySnapshot,
    epsilon: float = 0.01,
) -> EntropyDeltaProof:
    """Compute the entropy delta between two model states."""
    delta_se = snapshot_after.mean_spectral_entropy - snapshot_before.mean_spectral_entropy
    delta_ed = snapshot_after.mean_effective_dimension - snapshot_before.mean_effective_dimension
    delta_vs = snapshot_after.mean_viability_score - snapshot_before.mean_viability_score
    delta_pc = snapshot_after.mean_phase_coherence - snapshot_before.mean_phase_coherence

    return EntropyDeltaProof(
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        delta_spectral_entropy=delta_se,
        delta_effective_dimension=delta_ed,
        delta_viability_score=delta_vs,
        delta_phase_coherence=delta_pc,
        epsilon_threshold=epsilon,
        reduction_verified=delta_se < -epsilon,
    )


def compute_geometric_health(
    model: "nn.Module",
    tokenizer: Any,
    eval_prompts: List[str],
    n_samples: int = 5,
) -> GeometricHealthScore:
    """Compute a composite geometric health score for use in GroundingRequest priority."""
    snapshot = take_snapshot(
        model=model,
        tokenizer=tokenizer,
        eval_prompts=eval_prompts,
        n_samples=n_samples,
    )

    v = snapshot.mean_viability_score
    hidden_dim = getattr(model.config, "hidden_size", 768)
    max_entropy = math.log(hidden_dim) if hidden_dim > 1 else 1.0
    inv_entropy = 1.0 - min(snapshot.mean_spectral_entropy / max_entropy, 1.0)
    inv_sensitivity = 1.0 - min(snapshot.noise_sensitivity, 1.0)

    health = 0.4 * v + 0.3 * inv_entropy + 0.3 * inv_sensitivity
    composite_score = float(1.0 - max(0.0, min(health, 1.0)))

    return GeometricHealthScore(
        viability_score=snapshot.mean_viability_score,
        spectral_entropy=snapshot.mean_spectral_entropy,
        noise_sensitivity=snapshot.noise_sensitivity,
        composite_score=composite_score,
    )
