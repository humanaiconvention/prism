"""
Metrics for evaluating semantic grounding performance.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set
import re


# Cache for sentence-transformers models to avoid reloading
_EMBEDDING_MODEL_CACHE: Dict[str, Any] = {}


@dataclass
class GroundingMetrics:
    """
    Metrics for evaluating semantic grounding quality.

    Attributes:
        precision: Precision of grounded concepts (correct groundings / total groundings)
        recall: Recall of grounded concepts (correct groundings / expected groundings)
        f1_score: F1 score combining precision and recall
        accuracy: Overall accuracy of the grounding
        grounded_concepts: List of concepts that were successfully grounded
        missing_concepts: List of expected concepts that were not grounded
        extra_concepts: List of concepts that were grounded but not expected
    """

    precision: float
    recall: float
    f1_score: float
    accuracy: float
    grounded_concepts: List[str]
    missing_concepts: List[str]
    extra_concepts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "grounded_concepts": self.grounded_concepts,
            "missing_concepts": self.missing_concepts,
            "extra_concepts": self.extra_concepts,
        }

    @staticmethod
    def calculate(
        expected_groundings: List[str],
        actual_groundings: List[str],
        case_sensitive: bool = False,
        mode: str = "exact",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_threshold: float = 0.7,
    ) -> "GroundingMetrics":
        """
        Calculate grounding metrics based on expected and actual groundings.

        Args:
            expected_groundings: List of expected grounding concepts
            actual_groundings: List of actual grounding concepts from the model
            case_sensitive: Whether to perform case-sensitive matching
            mode: Matching mode - "exact" for n-gram matching or "embedding" for semantic similarity
            embedding_model_name: Name of sentence-transformers model (used when mode="embedding")
            embedding_threshold: Cosine similarity threshold for embedding matching (default: 0.7)

        Returns:
            GroundingMetrics object with calculated metrics
        """
        if mode == "embedding":
            return GroundingMetrics._calculate_embedding_mode(
                expected_groundings,
                actual_groundings,
                embedding_model_name,
                embedding_threshold,
            )
        elif mode == "exact":
            return GroundingMetrics._calculate_exact_mode(
                expected_groundings, actual_groundings, case_sensitive
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'exact' or 'embedding'.")

    @staticmethod
    def _calculate_exact_mode(
        expected_groundings: List[str],
        actual_groundings: List[str],
        case_sensitive: bool = False,
    ) -> "GroundingMetrics":
        """Calculate metrics using exact n-gram matching."""
        # Normalize groundings for comparison
        if not case_sensitive:
            expected_set: Set[str] = {g.lower().strip() for g in expected_groundings}
            actual_set: Set[str] = {g.lower().strip() for g in actual_groundings}
        else:
            expected_set = {g.strip() for g in expected_groundings}
            actual_set = {g.strip() for g in actual_groundings}

        # Calculate intersections and differences
        correct_groundings = expected_set.intersection(actual_set)
        missing = expected_set - actual_set
        extra = actual_set - expected_set

        # Calculate metrics
        precision = len(correct_groundings) / len(actual_set) if len(actual_set) > 0 else 0.0
        recall = len(correct_groundings) / len(expected_set) if len(expected_set) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        # Accuracy as the ratio of correct groundings to total unique groundings
        total_unique = len(expected_set.union(actual_set))
        accuracy = len(correct_groundings) / total_unique if total_unique > 0 else 0.0

        return GroundingMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            grounded_concepts=sorted(list(correct_groundings)),
            missing_concepts=sorted(list(missing)),
            extra_concepts=sorted(list(extra)),
        )

    @staticmethod
    def _calculate_embedding_mode(
        expected_groundings: List[str],
        actual_groundings: List[str],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.7,
    ) -> "GroundingMetrics":
        """
        Calculate metrics using embedding-based semantic similarity with greedy matching.

        Uses sentence-transformers to encode phrases and performs greedy matching
        based on cosine similarity threshold. Models are cached to avoid reloading.

        Args:
            expected_groundings: List of expected grounding concepts
            actual_groundings: List of actual grounding concepts from the model
            embedding_model_name: Name of sentence-transformers model
            threshold: Cosine similarity threshold (0.0-1.0)

        Returns:
            GroundingMetrics object with calculated metrics
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.util import cos_sim
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for embedding mode. "
                "Install it with: pip install sentence-transformers>=2.2.2"
            )

        if not expected_groundings or not actual_groundings:
            # Handle empty cases
            if not actual_groundings and not expected_groundings:
                return GroundingMetrics(
                    precision=1.0,
                    recall=1.0,
                    f1_score=1.0,
                    accuracy=1.0,
                    grounded_concepts=[],
                    missing_concepts=[],
                    extra_concepts=[],
                )
            elif not actual_groundings:
                return GroundingMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    accuracy=0.0,
                    grounded_concepts=[],
                    missing_concepts=expected_groundings,
                    extra_concepts=[],
                )
            else:  # not expected_groundings
                return GroundingMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    accuracy=0.0,
                    grounded_concepts=[],
                    missing_concepts=[],
                    extra_concepts=actual_groundings,
                )

        # Load embedding model (with caching)
        if embedding_model_name not in _EMBEDDING_MODEL_CACHE:
            _EMBEDDING_MODEL_CACHE[embedding_model_name] = SentenceTransformer(
                embedding_model_name
            )
        model = _EMBEDDING_MODEL_CACHE[embedding_model_name]

        # Encode all phrases
        expected_embeddings = model.encode(expected_groundings, convert_to_tensor=True)
        actual_embeddings = model.encode(actual_groundings, convert_to_tensor=True)

        # Compute cosine similarity matrix
        similarity_matrix = cos_sim(expected_embeddings, actual_embeddings)

        # Greedy matching: for each expected, find best match in actual if above threshold
        matched_expected = set()
        matched_actual = set()
        grounded_pairs = []  # (expected_idx, actual_idx)

        # Sort all similarities in descending order
        similarities = []
        for i in range(len(expected_groundings)):
            for j in range(len(actual_groundings)):
                similarities.append((similarity_matrix[i][j].item(), i, j))
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Greedy assignment: assign highest similarities first
        for sim, exp_idx, act_idx in similarities:
            if sim < threshold:
                break  # All remaining are below threshold
            if exp_idx not in matched_expected and act_idx not in matched_actual:
                matched_expected.add(exp_idx)
                matched_actual.add(act_idx)
                grounded_pairs.append((exp_idx, act_idx))

        # Build result lists
        grounded_concepts = [expected_groundings[i] for i, _ in grounded_pairs]
        missing_concepts = [
            expected_groundings[i]
            for i in range(len(expected_groundings))
            if i not in matched_expected
        ]
        extra_concepts = [
            actual_groundings[i] for i in range(len(actual_groundings)) if i not in matched_actual
        ]

        # Calculate metrics
        num_correct = len(grounded_pairs)
        precision = num_correct / len(actual_groundings) if len(actual_groundings) > 0 else 0.0
        recall = num_correct / len(expected_groundings) if len(expected_groundings) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        total_unique = len(expected_groundings) + len(actual_groundings) - num_correct
        accuracy = num_correct / total_unique if total_unique > 0 else 0.0

        return GroundingMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            grounded_concepts=sorted(grounded_concepts),
            missing_concepts=sorted(missing_concepts),
            extra_concepts=sorted(extra_concepts),
        )

    @staticmethod
    def extract_concepts_from_text(text: str) -> List[str]:
        """
        Extract potential grounding concepts from text.

        This is a simple heuristic that extracts noun phrases and named entities.
        For production use, consider using more sophisticated NLP techniques.

        Args:
            text: Input text to extract concepts from

        Returns:
            List of extracted concepts
        """
        # Simple extraction: split by common delimiters and filter
        # In production, you might want to use NLP libraries like spaCy
        concepts = []

        # Split by common separators
        segments = re.split(r"[,;.\n]", text)

        for segment in segments:
            # Remove common stop words and extract meaningful phrases
            words = segment.strip().split()
            if len(words) > 0 and len(words) <= 5:  # Limit to short phrases
                concepts.append(segment.strip())

        return [c for c in concepts if c and len(c) > 2]
