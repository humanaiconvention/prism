"""Tests for prism.eval.temporal — TemporalAnalyzer.

Analyzes generation-over-generation trajectories and classifies failure regimes
(accuracy_first, perplexity_first, synchronized, no_collapse).

Ported from sgt/tests/test_temporal_analyzer.py.
No model / GPU required.
"""

import pytest
from prism.eval import TemporalAnalyzer


@pytest.fixture
def analyzer():
    return TemporalAnalyzer(drop_threshold=0.05, rise_threshold=0.05)


def test_accuracy_first_collapse(analyzer):
    """OOD accuracy drops BEFORE perplexity rises → accuracy_first."""
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.7, "val_perplexity": 10.2},
        {"generation": 2, "arc_easy_accuracy": 0.6, "val_perplexity": 11.5},
    ]
    analysis = analyzer.analyze_run(results)
    assert analysis["T_OOD"] == 1
    assert analysis["T_PPL"] == 2
    assert analysis["delta_t"] == 1  # T_PPL - T_OOD
    assert analysis["regime_classification"] == "accuracy_first"


def test_perplexity_first_collapse(analyzer):
    """Perplexity rises BEFORE OOD accuracy drops → perplexity_first."""
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.79, "val_perplexity": 11.5},
        {"generation": 2, "arc_easy_accuracy": 0.6, "val_perplexity": 12.0},
    ]
    analysis = analyzer.analyze_run(results)
    assert analysis["T_OOD"] == 2
    assert analysis["T_PPL"] == 1
    assert analysis["delta_t"] == -1  # T_PPL - T_OOD
    assert analysis["regime_classification"] == "perplexity_first"


def test_synchronized_collapse(analyzer):
    """Both fail in the same generation → synchronized."""
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.6, "val_perplexity": 12.0},
    ]
    analysis = analyzer.analyze_run(results)
    assert analysis["T_OOD"] == 1
    assert analysis["T_PPL"] == 1
    assert analysis["delta_t"] == 0
    assert analysis["regime_classification"] == "synchronized"


def test_no_collapse(analyzer):
    """No threshold crossed → no_collapse."""
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.79, "val_perplexity": 10.1},
        {"generation": 2, "arc_easy_accuracy": 0.78, "val_perplexity": 10.2},
    ]
    analysis = analyzer.analyze_run(results)
    assert analysis["T_OOD"] == -1
    assert analysis["T_PPL"] == -1
    assert analysis["delta_t"] is None
    assert analysis["regime_classification"] == "no_collapse"


def test_empty_run(analyzer):
    """Empty data returns empty dict."""
    analysis = analyzer.analyze_run([])
    assert analysis == {}


def test_missing_metrics(analyzer):
    """Results with missing OOD/PPL keys return missing_metrics status."""
    results = [{"generation": 0, "other": 1.0}]
    analysis = analyzer.analyze_run(results)
    assert analysis["status"] == "missing_metrics"
