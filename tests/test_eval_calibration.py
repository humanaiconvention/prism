import pytest
import numpy as np
from prism.eval.calibration import CalibrationMetrics, DiversityMetrics

class TestCalibrationMetrics:
    def test_calculate_ece_perfect(self):
        """(1) Perfectly calibrated: confidence matches accuracy."""
        # 10 samples with 0.8 confidence, 8 are correct
        probs = [0.8] * 10
        labels = [1] * 8 + [0] * 2
        
        ece = CalibrationMetrics.calculate_ece(probs, labels, n_bins=10)
        # avg_conf = 0.8, accuracy = 0.8, gap = 0.0
        assert pytest.approx(ece, abs=1e-5) == 0.0

    def test_calculate_ece_miscalibrated(self):
        """(1) Maximally miscalibrated: high confidence, zero accuracy."""
        probs = [1.0] * 10
        labels = [0] * 10
        
        ece = CalibrationMetrics.calculate_ece(probs, labels, n_bins=10)
        # avg_conf = 1.0, accuracy = 0.0, gap = 1.0
        assert ece > 0.5
        assert pytest.approx(ece, abs=1e-5) == 1.0

    def test_calculate_ece_empty(self):
        """(4) Empty input handling."""
        assert CalibrationMetrics.calculate_ece([], []) == 0.0

class TestDiversityMetrics:
    def test_ngram_diversity_comparison(self):
        """(2) Compare low vs high diversity sets."""
        low_div = ["the ball is red", "the ball is red", "the ball is red"]
        high_div = ["the ball is red", "a cat sat down", "birds fly high up"]
        
        div_low = DiversityMetrics.calculate_ngram_diversity(low_div, n=2)
        div_high = DiversityMetrics.calculate_ngram_diversity(high_div, n=2)
        
        assert div_high > div_low
        assert div_low < 1.0
        # For low_div: 3 unique bigrams, 9 total bigrams = 0.333
        # For high_div: 9 unique bigrams, 9 total bigrams = 1.0
        assert pytest.approx(div_high) == 1.0

    def test_self_bleu_identical(self):
        """(3) Identical texts should have high self-bleu (near 1.0)."""
        texts = ["identical sentence"] * 10
        score = DiversityMetrics.calculate_self_bleu(texts)
        # Implementation: 1.0 - unique/total = 1.0 - 1/10 = 0.9
        assert score >= 0.9

    def test_self_bleu_different(self):
        """(3) Completely different texts should have low self-bleu (< 0.5)."""
        texts = [f"unique sentence {i}" for i in range(10)]
        score = DiversityMetrics.calculate_self_bleu(texts)
        # Implementation: 1.0 - unique/total = 1.0 - 10/10 = 0.0
        assert score < 0.5

    def test_diversity_empty_inputs(self):
        """(4) Empty input handling."""
        assert DiversityMetrics.calculate_ngram_diversity([], n=2) == 0.0
        assert DiversityMetrics.calculate_self_bleu([]) == 0.0
