"""Tests for prism.eval.drift_metrics — EarlyWarningDetector.

Covers the behavioral trajectory analysis as well as the geometric
silent-drift extension that reads PRISM spectral_entropy / effective_dimension keys.

Ported from sgt/tests/test_drift_metrics.py and the TestEarlyWarningDetectorGeometric
section of sgt/tests/test_prism_integration.py.
No model / GPU required.
"""

import pytest
from prism.eval import EarlyWarningDetector


# ---------------------------------------------------------------------------
# Behavioural signature detection (basic)
# ---------------------------------------------------------------------------

def test_grounding_fails_first():
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.8, "perplexity": 10.1},  # Acc drops 20%
        {"generation": 2, "accuracy": 0.7, "perplexity": 12.0},  # PPL rises 20%
    ]
    analysis = detector.analyze_trajectory(results)
    assert analysis["signature_detected"] is True
    assert analysis["failure_ordering"] == "grounding_failed_first"
    assert analysis["accuracy_failure_generation"] == 1
    assert analysis["perplexity_failure_generation"] == 2


def test_perplexity_fails_first():
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.95, "perplexity": 15.0},  # PPL rises 50%
        {"generation": 2, "accuracy": 0.8, "perplexity": 20.0},   # Acc drops 20%
    ]
    analysis = detector.analyze_trajectory(results)
    assert analysis["signature_detected"] is False
    assert analysis["failure_ordering"] == "perplexity_failed_first"
    assert analysis["accuracy_failure_generation"] == 2
    assert analysis["perplexity_failure_generation"] == 1


def test_grounding_fails_only():
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.8, "perplexity": 10.0},
        {"generation": 2, "accuracy": 0.5, "perplexity": 10.5},
    ]
    analysis = detector.analyze_trajectory(results)
    assert analysis["signature_detected"] is True
    assert analysis["failure_ordering"] == "grounding_failed_only"
    assert analysis["accuracy_failure_generation"] == 1
    assert analysis["perplexity_failure_generation"] == -1


def test_stable_trajectory():
    detector = EarlyWarningDetector()
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.95, "perplexity": 10.1},
        {"generation": 2, "accuracy": 0.95, "perplexity": 10.2},
    ]
    analysis = detector.analyze_trajectory(results)
    assert analysis["signature_detected"] is False
    assert analysis["failure_ordering"] == "stable"
    assert analysis["accuracy_failure_generation"] == -1
    assert analysis["perplexity_failure_generation"] == -1


# ---------------------------------------------------------------------------
# Geometric (PRISM spectral) extension
# ---------------------------------------------------------------------------

def _make_trajectory(spectral_entropies, effective_dims, base_acc=0.9):
    """Build a mixed behavioral + PRISM metric trajectory."""
    results = []
    for i, (se, ed) in enumerate(zip(spectral_entropies, effective_dims)):
        results.append({
            "generation": i,
            "accuracy": base_acc - i * 0.02,
            "perplexity": 10.0 + i * 0.1,
            "spectral_entropy": se,
            "effective_dimension": ed,
        })
    return results


def test_geometric_drift_detected_rising_entropy():
    """Rising spectral entropy beyond threshold triggers geometric_drift_detected=True."""
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    trajectory = _make_trajectory(
        spectral_entropies=[2.0, 2.5, 3.0],
        effective_dims=[200.0, 190.0, 180.0],
    )
    result = detector.analyze_trajectory(trajectory)
    assert "geometric_drift_detected" in result
    assert result["geometric_drift_detected"] is True
    assert result["geometric_failure_generation"] != -1


def test_geometric_drift_absent_stable_entropy():
    """Stable entropy trajectory → geometric_drift_detected=False."""
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    trajectory = _make_trajectory(
        spectral_entropies=[2.0, 2.01, 2.02],  # < 10% rise
        effective_dims=[200.0, 199.5, 199.0],
    )
    result = detector.analyze_trajectory(trajectory)
    assert result["geometric_drift_detected"] is False
    assert result["geometric_failure_generation"] == -1


def test_geometric_keys_absent_when_no_prism_data():
    """When spectral_entropy is absent from data, geometric keys are NOT added."""
    detector = EarlyWarningDetector()
    standard_trajectory = [
        {"generation": 0, "accuracy": 0.9, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.85, "perplexity": 10.5},
    ]
    result = detector.analyze_trajectory(standard_trajectory)
    assert "geometric_drift_detected" not in result
    assert "spectral_entropy_trajectory" not in result


def test_behavioral_and_geometric_both_reported():
    """When PRISM data is present, both behavioral and geometric keys appear."""
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    trajectory = _make_trajectory(
        spectral_entropies=[2.0, 3.5, 5.0],
        effective_dims=[200.0, 150.0, 100.0],
        base_acc=0.9,
    )
    result = detector.analyze_trajectory(trajectory)
    assert "signature_detected" in result
    assert "failure_ordering" in result
    assert "geometric_drift_detected" in result
    assert "spectral_entropy_trajectory" in result
    assert "effective_dimension_trajectory" in result
    assert len(result["spectral_entropy_trajectory"]) == 3


def test_geometric_trajectory_includes_all_gens():
    """spectral_entropy_trajectory has one entry per generation."""
    detector = EarlyWarningDetector()
    trajectory = _make_trajectory([1.0, 1.1, 1.2, 1.3, 1.4], [100.0] * 5)
    result = detector.analyze_trajectory(trajectory)
    assert len(result["spectral_entropy_trajectory"]) == 5
    assert len(result["effective_dimension_trajectory"]) == 5
