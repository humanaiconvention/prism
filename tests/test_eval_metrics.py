"""
Unit tests for the metrics module.
"""

import pytest
from prism.eval.metrics import GroundingMetrics


class TestGroundingMetrics:
    """Tests for GroundingMetrics class."""

    def test_calculate_perfect_match(self):
        """Test calculating metrics with perfect match."""
        expected = ["apple", "banana", "orange"]
        actual = ["apple", "banana", "orange"]

        metrics = GroundingMetrics.calculate(expected, actual)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0
        assert metrics.grounded_concepts == ["apple", "banana", "orange"]
        assert metrics.missing_concepts == []
        assert metrics.extra_concepts == []

    def test_calculate_partial_match(self):
        """Test calculating metrics with partial match."""
        expected = ["apple", "banana", "orange"]
        actual = ["apple", "banana"]

        metrics = GroundingMetrics.calculate(expected, actual)

        assert metrics.precision == 1.0  # All actual are correct
        assert metrics.recall == 2 / 3  # 2 out of 3 expected found
        assert metrics.f1_score == pytest.approx(0.8, rel=0.01)
        assert metrics.grounded_concepts == ["apple", "banana"]
        assert metrics.missing_concepts == ["orange"]
        assert metrics.extra_concepts == []

    def test_calculate_with_extra_concepts(self):
        """Test calculating metrics with extra concepts."""
        expected = ["apple", "banana"]
        actual = ["apple", "banana", "orange", "grape"]

        metrics = GroundingMetrics.calculate(expected, actual)

        assert metrics.precision == 0.5  # 2 out of 4 actual are correct
        assert metrics.recall == 1.0  # All expected found
        assert metrics.f1_score == pytest.approx(0.667, rel=0.01)
        assert metrics.grounded_concepts == ["apple", "banana"]
        assert metrics.missing_concepts == []
        assert metrics.extra_concepts == ["grape", "orange"]

    def test_calculate_no_match(self):
        """Test calculating metrics with no match."""
        expected = ["apple", "banana"]
        actual = ["orange", "grape"]

        metrics = GroundingMetrics.calculate(expected, actual)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.grounded_concepts == []
        assert sorted(metrics.missing_concepts) == ["apple", "banana"]
        assert sorted(metrics.extra_concepts) == ["grape", "orange"]

    def test_calculate_case_insensitive(self):
        """Test calculating metrics with case-insensitive matching."""
        expected = ["Apple", "Banana", "Orange"]
        actual = ["apple", "BANANA", "orange"]

        metrics = GroundingMetrics.calculate(expected, actual, case_sensitive=False)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.grounded_concepts == ["apple", "banana", "orange"]

    def test_calculate_case_sensitive(self):
        """Test calculating metrics with case-sensitive matching."""
        expected = ["Apple", "Banana", "Orange"]
        actual = ["apple", "BANANA", "orange"]

        metrics = GroundingMetrics.calculate(expected, actual, case_sensitive=True)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.grounded_concepts == []

    def test_calculate_empty_actual(self):
        """Test calculating metrics with empty actual groundings."""
        expected = ["apple", "banana"]
        actual = []

        metrics = GroundingMetrics.calculate(expected, actual)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.grounded_concepts == []
        assert sorted(metrics.missing_concepts) == ["apple", "banana"]
        assert metrics.extra_concepts == []

    def test_calculate_empty_expected(self):
        """Test calculating metrics with empty expected groundings."""
        expected = []
        actual = ["apple", "banana"]

        metrics = GroundingMetrics.calculate(expected, actual)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.grounded_concepts == []
        assert metrics.missing_concepts == []
        assert sorted(metrics.extra_concepts) == ["apple", "banana"]

    def test_calculate_handles_whitespace(self):
        """Test that calculation handles whitespace correctly."""
        expected = ["  apple  ", "banana", "orange  "]
        actual = ["apple", "  banana  ", "orange"]

        metrics = GroundingMetrics.calculate(expected, actual)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        expected = ["apple", "banana"]
        actual = ["apple"]

        metrics = GroundingMetrics.calculate(expected, actual)
        result = metrics.to_dict()

        assert result["precision"] == 1.0
        assert result["recall"] == 0.5
        assert result["grounded_concepts"] == ["apple"]
        assert result["missing_concepts"] == ["banana"]
        assert result["extra_concepts"] == []

    def test_extract_concepts_from_text(self):
        """Test extracting concepts from text."""
        text = "red ball, table, window"
        concepts = GroundingMetrics.extract_concepts_from_text(text)

        assert isinstance(concepts, list)
        assert len(concepts) > 0
        # Should extract some meaningful segments
        assert any(len(c) > 2 for c in concepts)
        # Should include the concepts
        assert "red ball" in concepts or " red ball" in concepts

    def test_extract_concepts_empty_text(self):
        """Test extracting concepts from empty text."""
        concepts = GroundingMetrics.extract_concepts_from_text("")
        assert concepts == []

    def test_extract_concepts_filters_short(self):
        """Test that extraction filters very short segments."""
        text = "a, b, c, longer phrase here"
        concepts = GroundingMetrics.extract_concepts_from_text(text)

        # Short segments (a, b, c) should be filtered
        assert all(len(c) > 2 for c in concepts)

    @pytest.mark.parametrize(
        "mode,expected_error",
        [
            ("invalid", "Unsupported mode"),
        ],
    )
    def test_calculate_invalid_mode(self, mode, expected_error):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match=expected_error):
            GroundingMetrics.calculate(["test"], ["test"], mode=mode)

    def test_calculate_embedding_mode_requires_package(self):
        """Test that embedding mode raises ImportError when sentence-transformers is missing."""
        from unittest.mock import patch
        # Force the import to fail regardless of whether the package is installed
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers package is required"):
                GroundingMetrics.calculate(
                    ["test"], ["test"], mode="embedding", embedding_model_name="all-MiniLM-L6-v2"
                )

    def test_calculate_embedding_mode_perfect_match(self):
        """Test embedding mode with identical strings (cosine similarity = 1.0)."""
        result = GroundingMetrics.calculate(
            ["water", "fire"], ["water", "fire"],
            mode="embedding", embedding_model_name="all-MiniLM-L6-v2"
        )
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_calculate_embedding_mode_semantic_match(self):
        """Test embedding mode catches semantic similarity that exact mode would miss."""
        exact_result = GroundingMetrics.calculate(
            ["automobile"], ["car"], mode="exact"
        )
        embed_result = GroundingMetrics.calculate(
            ["automobile"], ["car"],
            mode="embedding", embedding_model_name="all-MiniLM-L6-v2",
            embedding_threshold=0.5
        )
        # Exact mode misses the match; embedding mode should catch it
        assert exact_result.recall == 0.0
        assert embed_result.recall == 1.0

    def test_calculate_embedding_mode_no_match(self):
        """Test embedding mode with completely unrelated concepts."""
        result = GroundingMetrics.calculate(
            ["quantum physics"], ["chocolate cake"],
            mode="embedding", embedding_model_name="all-MiniLM-L6-v2",
            embedding_threshold=0.7
        )
        assert result.recall == 0.0

    def test_calculate_embedding_mode_empty(self):
        """Test embedding mode with empty inputs."""
        result = GroundingMetrics.calculate(
            [], ["something"], mode="embedding", embedding_model_name="all-MiniLM-L6-v2"
        )
        assert result.recall == 0.0
