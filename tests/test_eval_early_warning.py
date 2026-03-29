"""Tests for prism.eval.early_warning — EarlyWarningAnalyzer.

Ported from sgt/tests/test_early_warning.py.
No model / GPU / SGT infrastructure required.
"""

import pytest
from prism.eval import EarlyWarningAnalyzer


@pytest.fixture
def analyzer():
    return EarlyWarningAnalyzer(acc_threshold=0.10, ppl_threshold=0.05)


def test_detect_accuracy_first(analyzer):
    """(1) accuracy drops at gen 1, ppl rises at gen 2 -> signature_detected=False."""
    metrics = [
        {"generation": 0, "grounded_arc_accuracy": 0.8, "grounded_arc_perplexity": 10.0},
        {"generation": 1, "grounded_arc_accuracy": 0.6, "grounded_arc_perplexity": 10.2},
        {"generation": 2, "grounded_arc_accuracy": 0.5, "grounded_arc_perplexity": 12.0},
    ]
    result = analyzer.detect(metrics)
    assert result["signature_detected"] is False
    assert result["t_acc"] == 1
    assert result["t_ppl"] == 2


def test_detect_perplexity_first(analyzer):
    """(2) perplexity rises at gen 1, accuracy drops at gen 2 -> signature_detected=True."""
    metrics = [
        {"generation": 0, "grounded_arc_accuracy": 0.8, "grounded_arc_perplexity": 10.0},
        {"generation": 1, "grounded_arc_accuracy": 0.79, "grounded_arc_perplexity": 11.0},
        {"generation": 2, "grounded_arc_accuracy": 0.6, "grounded_arc_perplexity": 12.0},
    ]
    result = analyzer.detect(metrics)
    assert result["signature_detected"] is True
    assert result["t_acc"] == 2
    assert result["t_ppl"] == 1


def test_detect_single_generation(analyzer):
    """(3) only one generation -> no detection (insufficient data)."""
    metrics = [
        {"generation": 0, "grounded_arc_accuracy": 0.8, "grounded_arc_perplexity": 10.0}
    ]
    result = analyzer.detect(metrics)
    assert result["signature_detected"] is False


def test_generate_report(analyzer):
    """(4) generate_report() returns a string containing regime name and signature status."""
    detection_result = {"signature_detected": True}
    report = analyzer.generate_report("Regime-X", detection_result)
    assert isinstance(report, str)
    assert "Regime-X" in report
    assert "signature detected" in report
