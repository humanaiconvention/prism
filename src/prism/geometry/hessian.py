"""
Hessian-based diagnostic system for detecting failure modes in semantic grounding models.

This module implements comprehensive landscape diagnostics to detect:
1. Hallucinations (confident interpolation to nowhere) - high condition number κ
2. Distribution Shift Failures (memorized patterns don't transfer) - high spectral sharpness ε
3. Adversarial Brittleness (small input changes cause large output swings) - negative eigenvalues δ

Research Citations:
- Kaur et al. 2023: Loss landscape analysis for neural network robustness
- Ghorbani et al. 2019: Data shapley and Hessian-based robustness metrics
- OpenReview: Power-law spectral decay in overparameterized networks

RESEARCH GAPS:
- Lack of universal thresholds across model architectures
- Context-dependency of κ, λ_max, eigenvalue signs
- Correlation vs causation issues (flatness ≠ robustness proven)
- Limited literature on Hessian spectral properties in transformer-based semantic grounding
- Need for empirical calibration on QWEN/LLAMA models
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import torch
import numpy as np


class RiskLevel(Enum):
    """Risk level classification."""

    SAFE = "SAFE"
    CAUTION = "CAUTION"
    RISK = "RISK"


@dataclass
class LandscapeMetrics:
    """
    Hessian-based loss landscape metrics for robustness evaluation.

    These metrics extend GroundingMetrics with landscape diagnostics that help identify
    three critical failure modes in semantic grounding models.

    Attributes:
        condition_number: Hessian condition number κ (ratio of max to min eigenvalue)
            High κ indicates ill-conditioned landscape (hallucination risk)
        spectral_sharpness: Maximum eigenvalue λ_max
            High λ_max indicates sharp minima (distribution shift risk)
        min_eigenvalue: Minimum eigenvalue (detect negative values)
            Negative values indicate saddle points (adversarial brittleness)
        eigenvalue_spread: Range of eigenvalues (max - min)
            Larger spread indicates varied sensitivity across directions
        power_law_compliance: Deviation from expected power-law decay (0-1)
            0 = perfect power-law, 1 = maximum deviation
        has_negative_eigenvalues: Saddle point flag
            True if any eigenvalue is significantly negative
        stability_score: Composite robustness score (0-1)
            Higher is better, combines all metrics into single score
        eigenvalue_spectrum: Full list of eigenvalues (optional, for analysis)
            Sorted in descending order
    """

    condition_number: float
    spectral_sharpness: float
    min_eigenvalue: float
    eigenvalue_spread: float
    power_law_compliance: float
    has_negative_eigenvalues: bool
    stability_score: float
    eigenvalue_spectrum: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        result = {
            "condition_number": self.condition_number,
            "spectral_sharpness": self.spectral_sharpness,
            "min_eigenvalue": self.min_eigenvalue,
            "eigenvalue_spread": self.eigenvalue_spread,
            "power_law_compliance": self.power_law_compliance,
            "has_negative_eigenvalues": self.has_negative_eigenvalues,
            "stability_score": self.stability_score,
        }
        if self.eigenvalue_spectrum is not None:
            result["eigenvalue_spectrum"] = self.eigenvalue_spectrum
        return result


class HessianCompute:
    """
    Scalable Hessian computation for large neural networks.

    Implements both Hutchinson's stochastic trace estimator (for large models)
    and exact Hessian computation (for validation sets).
    """

    def __init__(self, model: Optional[Any] = None, device: str = "cpu"):
        """
        Initialize Hessian computer.

        Args:
            model: PyTorch model (optional, can be set later)
            device: Device to run computations on ('cpu', 'cuda', etc.)
        """
        self.model = model
        self.device = device

    def compute_hutchinson_trace(
        self,
        loss_fn: Any,
        num_samples: int = 100,
        use_rademacher: bool = True,
    ) -> float:
        """
        Compute Hessian trace using Hutchinson's stochastic estimator.

        This method provides scalable trace estimation for large models using
        random projections (Rademacher or Gaussian).

        Args:
            loss_fn: Callable that computes loss given parameters
            num_samples: Number of random vectors for estimation
            use_rademacher: Use Rademacher (+1/-1) instead of Gaussian random vectors

        Returns:
            Estimated trace of Hessian

        Research Note:
            Hutchinson's method: E[v^T H v] = tr(H) for random v with E[vv^T] = I
            Convergence rate: O(1/sqrt(num_samples))
        """
        if self.model is None:
            raise ValueError("Model must be set before computing Hessian")

        trace_estimate = 0.0

        for _ in range(num_samples):
            # Generate random vector
            if use_rademacher:
                # Rademacher: uniformly +1 or -1
                v = torch.randint(0, 2, (self._get_param_count(),), device=self.device) * 2.0 - 1.0
            else:
                # Gaussian N(0, 1)
                v = torch.randn(self._get_param_count(), device=self.device)

            # Compute Hessian-vector product (Hv)
            hv = self._hessian_vector_product(loss_fn, v)

            # Estimate: v^T H v
            trace_estimate += torch.dot(v, hv).item()

        return trace_estimate / num_samples

    def compute_exact_hessian(
        self,
        loss_fn: Any,
        max_params: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute exact Hessian matrix (for small models or validation).

        WARNING: This has O(n^2) memory complexity where n is the number of parameters.
        Only use on small models or subsets of parameters.

        Args:
            loss_fn: Callable that computes loss given parameters
            max_params: Maximum number of parameters to compute (truncate if needed)

        Returns:
            Hessian matrix as torch.Tensor

        Research Note:
            For large models (>1M parameters), use Hutchinson or eigenvalue methods instead.
            Memory requirement: ~8 bytes * n^2 (e.g., 8GB for 32k parameters)
        """
        if self.model is None:
            raise ValueError("Model must be set before computing Hessian")

        # Get all parameters as a single vector
        params = self._get_param_vector()
        n_params = len(params)

        if max_params is not None and n_params > max_params:
            raise ValueError(
                f"Model has {n_params} parameters, exceeding max_params={max_params}. "
                "Use Hutchinson's method for large models."
            )

        # Compute gradient
        loss = loss_fn()
        grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)[0]

        # Compute Hessian row by row
        hessian = torch.zeros(n_params, n_params, device=self.device)

        for i in range(n_params):
            # Compute gradient of grad[i] w.r.t. all parameters
            grad2 = torch.autograd.grad(grad[i], params, retain_graph=True, allow_unused=True)[0]
            if grad2 is not None:
                hessian[i] = grad2

        return hessian

    def compute_eigenvalues(
        self,
        loss_fn: Any,
        num_eigenvalues: int = 50,
        method: str = "lanczos",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute top eigenvalues of Hessian using iterative methods.

        More efficient than computing full Hessian for large models.

        Args:
            loss_fn: Callable that computes loss given parameters
            num_eigenvalues: Number of eigenvalues to compute
            method: Eigenvalue computation method ('lanczos' or 'power')

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in descending order
            eigenvectors is None if not computed

        Research Note:
            Lanczos algorithm: Efficient for sparse/large matrices
            Power iteration: Simple but only finds dominant eigenvalue
        """
        if method == "lanczos":
            return self._lanczos_eigenvalues(loss_fn, num_eigenvalues)
        elif method == "power":
            return self._power_iteration_eigenvalues(loss_fn, num_eigenvalues)
        else:
            raise ValueError(f"Unknown eigenvalue method: {method}")

    def compute_condition_number(
        self,
        eigenvalues: Optional[np.ndarray] = None,
        loss_fn: Optional[Any] = None,
    ) -> float:
        """
        Compute Hessian condition number κ = λ_max / λ_min.

        Args:
            eigenvalues: Pre-computed eigenvalues (if available)
            loss_fn: Loss function (if eigenvalues not provided)

        Returns:
            Condition number κ

        Research Note:
            High κ (>1000) indicates ill-conditioning
            Typical range: [1, 10^6] for neural networks
            Vision models: often 10^2 - 10^4
            Language models: often 10^3 - 10^6 (need more research)
        """
        if eigenvalues is None:
            if loss_fn is None:
                raise ValueError("Must provide either eigenvalues or loss_fn")
            eigenvalues, _ = self.compute_eigenvalues(loss_fn)

        max_eig = np.max(np.abs(eigenvalues))
        min_eig = np.min(np.abs(eigenvalues[eigenvalues != 0]))  # Avoid division by zero

        if min_eig == 0:
            return float("inf")

        return max_eig / min_eig

    def get_spectral_sharpness(self, eigenvalues: np.ndarray) -> float:
        """
        Extract maximum eigenvalue λ_max (spectral sharpness).

        Args:
            eigenvalues: Array of eigenvalues

        Returns:
            Maximum eigenvalue

        Research Note:
            Smaller λ_max correlates with flatter minima and better generalization
            Typical ranges:
            - Vision (ResNet): 0.1 - 2.0
            - NLP (BERT): 0.5 - 5.0 (preliminary, needs verification)
        """
        return float(np.max(eigenvalues))

    def detect_negative_eigenvalues(
        self,
        eigenvalues: np.ndarray,
        threshold: float = -0.01,
    ) -> Tuple[bool, int, List[float]]:
        """
        Detect negative eigenvalues indicating saddle points.

        Args:
            eigenvalues: Array of eigenvalues
            threshold: Threshold for significant negativity (default: -0.01)

        Returns:
            Tuple of (has_negative, count, negative_values)

        Research Note:
            Negative eigenvalues are rare in overparameterized networks
            but critical when present (indicate unstable equilibrium)
            Threshold of -0.01 filters numerical noise
        """
        negative_mask = eigenvalues < threshold
        negative_eigenvalues = eigenvalues[negative_mask].tolist()

        return len(negative_eigenvalues) > 0, len(negative_eigenvalues), negative_eigenvalues

    def _get_param_count(self) -> int:
        """Get total number of parameters in model."""
        if self.model is None:
            raise ValueError("Model must be set")
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _get_param_vector(self) -> torch.Tensor:
        """Get all parameters as a single flattened vector."""
        if self.model is None:
            raise ValueError("Model must be set")
        return torch.cat([p.flatten() for p in self.model.parameters() if p.requires_grad])

    def _hessian_vector_product(self, loss_fn: Any, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product efficiently using double backprop.

        Args:
            loss_fn: Callable that computes loss
            v: Vector to multiply with Hessian

        Returns:
            Hv (Hessian-vector product)
        """
        # Compute gradient
        loss = loss_fn()
        params = list(self.model.parameters())
        grad = torch.autograd.grad(loss, params, create_graph=True)

        # Flatten gradient
        grad_vec = torch.cat([g.flatten() for g in grad if g is not None])

        # Compute gradient-vector product
        gv = torch.dot(grad_vec, v)

        # Compute gradient of gv (this is Hv)
        hvs = torch.autograd.grad(gv, params, retain_graph=True)
        hv = torch.cat([h.flatten() for h in hvs if h is not None])

        return hv

    def _lanczos_eigenvalues(
        self,
        loss_fn: Any,
        num_eigenvalues: int,
    ) -> Tuple[np.ndarray, None]:
        """
        Compute eigenvalues using Lanczos algorithm.

        This is a placeholder implementation. Full implementation would use
        scipy.sparse.linalg.eigsh with LinearOperator.

        Args:
            loss_fn: Loss function
            num_eigenvalues: Number of eigenvalues to compute

        Returns:
            (eigenvalues, None)
        """
        # For now, use power iteration as fallback
        # A full implementation would use Lanczos tridiagonalization
        return self._power_iteration_eigenvalues(loss_fn, num_eigenvalues)

    def _power_iteration_eigenvalues(
        self,
        loss_fn: Any,
        num_eigenvalues: int,
        max_iterations: int = 100,
    ) -> Tuple[np.ndarray, None]:
        """
        Compute top eigenvalues using power iteration.

        Args:
            loss_fn: Loss function
            num_eigenvalues: Number of eigenvalues to compute
            max_iterations: Maximum iterations per eigenvalue

        Returns:
            (eigenvalues, None)
        """
        eigenvalues = []

        for _ in range(num_eigenvalues):
            # Random initialization
            v = torch.randn(self._get_param_count(), device=self.device)
            v = v / torch.norm(v)

            # Power iteration
            for _ in range(max_iterations):
                hv = self._hessian_vector_product(loss_fn, v)
                v = hv / torch.norm(hv)

            # Rayleigh quotient
            eigenvalue = torch.dot(v, self._hessian_vector_product(loss_fn, v)).item()
            eigenvalues.append(eigenvalue)

        return np.array(sorted(eigenvalues, reverse=True)), None


class FailureModeClassifier:
    """
    Classifier for identifying failure modes based on Hessian metrics.

    Implements evidence-based thresholds from recent literature, with extensive
    documentation of research gaps and context-dependency.
    """

    # Thresholds based on empirical research (with caveats)
    # See: Kaur et al. 2023, Ghorbani et al. 2019

    # Hallucination risk thresholds (condition number κ)
    CONDITION_NUMBER_SAFE = 100.0
    CONDITION_NUMBER_CAUTION = 1000.0

    # Distribution shift risk thresholds (spectral sharpness λ_max)
    SPECTRAL_SHARPNESS_SAFE = 0.5
    SPECTRAL_SHARPNESS_CAUTION = 1.5

    # Adversarial brittleness thresholds (negative eigenvalues)
    NEGATIVE_EIGENVALUE_THRESHOLD = -0.01
    NEGATIVE_EIGENVALUE_COUNT_CAUTION = 3

    def __init__(self, custom_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize classifier with optional custom thresholds.

        Args:
            custom_thresholds: Dictionary of custom threshold values
                Keys: 'condition_number_safe', 'condition_number_caution',
                      'spectral_sharpness_safe', 'spectral_sharpness_caution',
                      'negative_eigenvalue_threshold', 'negative_eigenvalue_count_caution'
        """
        self.thresholds = {
            "condition_number_safe": self.CONDITION_NUMBER_SAFE,
            "condition_number_caution": self.CONDITION_NUMBER_CAUTION,
            "spectral_sharpness_safe": self.SPECTRAL_SHARPNESS_SAFE,
            "spectral_sharpness_caution": self.SPECTRAL_SHARPNESS_CAUTION,
            "negative_eigenvalue_threshold": self.NEGATIVE_EIGENVALUE_THRESHOLD,
            "negative_eigenvalue_count_caution": self.NEGATIVE_EIGENVALUE_COUNT_CAUTION,
        }

        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def classify_hallucination_risk(
        self,
        condition_number: float,
    ) -> Tuple[RiskLevel, float]:
        """
        Classify hallucination risk based on condition number κ.

        High κ indicates ill-conditioned loss landscape with flat, featureless regions
        where the model may confidently interpolate to nowhere.

        Args:
            condition_number: Hessian condition number κ

        Returns:
            Tuple of (risk_level, confidence)
            confidence: 0.0-1.0 (1.0 = certain classification)

        Research Context:
            - Vision models (ResNet): κ ~ 10^2 - 10^4
            - Language models: Limited data, estimated 10^3 - 10^6
            - NO universal threshold; these are empirical baselines
            - Context-dependent: task, architecture, training procedure
        """
        if condition_number < self.thresholds["condition_number_safe"]:
            return RiskLevel.SAFE, 1.0
        elif condition_number < self.thresholds["condition_number_caution"]:
            # Interpolate confidence
            range_size = (
                self.thresholds["condition_number_caution"]
                - self.thresholds["condition_number_safe"]
            )
            position = condition_number - self.thresholds["condition_number_safe"]
            confidence = 0.5 + 0.5 * (position / range_size)
            return RiskLevel.CAUTION, confidence
        else:
            # High risk: confidence increases with severity
            excess = condition_number - self.thresholds["condition_number_caution"]
            confidence = min(
                1.0, 0.7 + 0.3 * (excess / self.thresholds["condition_number_caution"])
            )
            return RiskLevel.RISK, confidence

    def classify_distribution_shift_risk(
        self,
        spectral_sharpness: float,
    ) -> Tuple[RiskLevel, float]:
        """
        Classify distribution shift risk based on spectral sharpness λ_max.

        High λ_max indicates sharp minima that may not generalize well to
        distribution shifts (memorized patterns don't transfer).

        Args:
            spectral_sharpness: Maximum eigenvalue λ_max

        Returns:
            Tuple of (risk_level, confidence)

        Research Context:
            - Flat minima (low λ_max) correlate with better generalization
            - BUT: correlation ≠ causation (active research area)
            - SAM (Sharpness-Aware Minimization) targets low λ_max
            - Thresholds vary widely by domain and task
        """
        if spectral_sharpness < self.thresholds["spectral_sharpness_safe"]:
            return RiskLevel.SAFE, 1.0
        elif spectral_sharpness < self.thresholds["spectral_sharpness_caution"]:
            range_size = (
                self.thresholds["spectral_sharpness_caution"]
                - self.thresholds["spectral_sharpness_safe"]
            )
            position = spectral_sharpness - self.thresholds["spectral_sharpness_safe"]
            confidence = 0.5 + 0.5 * (position / range_size)
            return RiskLevel.CAUTION, confidence
        else:
            excess = spectral_sharpness - self.thresholds["spectral_sharpness_caution"]
            confidence = min(
                1.0, 0.7 + 0.3 * (excess / self.thresholds["spectral_sharpness_caution"])
            )
            return RiskLevel.RISK, confidence

    def classify_adversarial_brittleness_risk(
        self,
        has_negative_eigenvalues: bool,
        negative_count: int,
        negative_values: List[float],
    ) -> Tuple[RiskLevel, float]:
        """
        Classify adversarial brittleness risk based on negative eigenvalues.

        Negative eigenvalues indicate saddle points where small perturbations
        can cause large output changes (unstable equilibrium).

        Args:
            has_negative_eigenvalues: Whether any negative eigenvalues exist
            negative_count: Number of negative eigenvalues
            negative_values: List of negative eigenvalue values

        Returns:
            Tuple of (risk_level, confidence)

        Research Context:
            - Rare in overparameterized networks (mostly positive semi-definite)
            - When present, indicate unstable critical points
            - Large negative eigenvalues (λ < -0.1) are particularly concerning
            - Connection to adversarial robustness is empirical, not proven
        """
        if not has_negative_eigenvalues:
            return RiskLevel.SAFE, 1.0

        # Count significant negatives
        significant_negatives = [
            v for v in negative_values if v < self.thresholds["negative_eigenvalue_threshold"]
        ]

        if len(significant_negatives) == 0:
            # Only small negatives (likely numerical noise)
            return RiskLevel.SAFE, 0.8

        if len(significant_negatives) <= self.thresholds["negative_eigenvalue_count_caution"]:
            # Few negative eigenvalues
            confidence = 0.6 + 0.2 * (
                len(significant_negatives) / self.thresholds["negative_eigenvalue_count_caution"]
            )
            return RiskLevel.CAUTION, confidence
        else:
            # Many negative eigenvalues
            confidence = min(
                1.0,
                0.8
                + 0.2
                * (
                    len(significant_negatives)
                    / (2 * self.thresholds["negative_eigenvalue_count_caution"])
                ),
            )
            return RiskLevel.RISK, confidence

    def generate_diagnostic_report(
        self,
        landscape_metrics: LandscapeMetrics,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diagnostic report with all failure mode analyses.

        Args:
            landscape_metrics: Computed landscape metrics

        Returns:
            Dictionary with detailed diagnostic information and recommendations

        Report Structure:
            - Overall stability assessment
            - Individual failure mode classifications
            - Risk factors and warnings
            - Actionable recommendations
            - Research gaps and caveats
        """
        # Classify all failure modes
        hallucination_risk, hallucination_conf = self.classify_hallucination_risk(
            landscape_metrics.condition_number
        )
        distribution_risk, distribution_conf = self.classify_distribution_shift_risk(
            landscape_metrics.spectral_sharpness
        )

        # Detect negative eigenvalues from metrics
        has_neg = landscape_metrics.has_negative_eigenvalues
        neg_count = 0
        neg_values = []

        if landscape_metrics.eigenvalue_spectrum:
            neg_values = [e for e in landscape_metrics.eigenvalue_spectrum if e < 0]
            neg_count = len(neg_values)

        adversarial_risk, adversarial_conf = self.classify_adversarial_brittleness_risk(
            has_neg, neg_count, neg_values
        )

        # Overall risk level (worst of the three)
        risk_levels = [hallucination_risk, distribution_risk, adversarial_risk]
        if RiskLevel.RISK in risk_levels:
            overall_risk = RiskLevel.RISK
        elif RiskLevel.CAUTION in risk_levels:
            overall_risk = RiskLevel.CAUTION
        else:
            overall_risk = RiskLevel.SAFE

        # Generate recommendations
        recommendations = self._generate_recommendations(
            hallucination_risk,
            distribution_risk,
            adversarial_risk,
            landscape_metrics,
        )

        # Compile report
        report = {
            "overall_risk": overall_risk.value,
            "stability_score": landscape_metrics.stability_score,
            "failure_modes": {
                "hallucination": {
                    "risk_level": hallucination_risk.value,
                    "confidence": hallucination_conf,
                    "metric": "condition_number",
                    "value": landscape_metrics.condition_number,
                    "description": "Confident interpolation to nowhere (ill-conditioned landscape)",
                },
                "distribution_shift": {
                    "risk_level": distribution_risk.value,
                    "confidence": distribution_conf,
                    "metric": "spectral_sharpness",
                    "value": landscape_metrics.spectral_sharpness,
                    "description": "Memorized patterns don't transfer (sharp minima)",
                },
                "adversarial_brittleness": {
                    "risk_level": adversarial_risk.value,
                    "confidence": adversarial_conf,
                    "metric": "negative_eigenvalues",
                    "value": {"count": neg_count, "has_negative": has_neg},
                    "description": "Small input changes cause large output swings (saddle points)",
                },
            },
            "recommendations": recommendations,
            "research_caveats": [
                "Thresholds are empirical baselines, not universal constants",
                "Limited research on Hessian metrics for transformer-based semantic grounding",
                "Correlation between flatness and robustness is not proven causation",
                "Metrics may vary significantly across model architectures and tasks",
                "Requires empirical calibration for QWEN/LLAMA models in production",
            ],
        }

        return report

    def _generate_recommendations(
        self,
        hallucination_risk: RiskLevel,
        distribution_risk: RiskLevel,
        adversarial_risk: RiskLevel,
        metrics: LandscapeMetrics,
    ) -> List[str]:
        """Generate actionable recommendations based on risk classifications."""
        recommendations = []

        if hallucination_risk == RiskLevel.RISK:
            recommendations.append(
                f"HIGH CONDITION NUMBER (κ={metrics.condition_number:.1f}): "
                "Model may hallucinate in flat landscape regions. "
                "Consider: (1) Regularization techniques, (2) Increase training data diversity, "
                "(3) Monitor confidence calibration on out-of-distribution inputs"
            )
        elif hallucination_risk == RiskLevel.CAUTION:
            recommendations.append(
                f"MODERATE CONDITION NUMBER (κ={metrics.condition_number:.1f}): "
                "Exercise caution with novel inputs outside training distribution"
            )

        if distribution_risk == RiskLevel.RISK:
            recommendations.append(
                f"HIGH SPECTRAL SHARPNESS (λ_max={metrics.spectral_sharpness:.2f}): "
                "Model may fail under distribution shift. "
                "Consider: (1) Sharpness-Aware Minimization (SAM), "
                "(2) Increase robustness training, (3) Evaluate on diverse test sets"
            )
        elif distribution_risk == RiskLevel.CAUTION:
            recommendations.append(
                f"MODERATE SPECTRAL SHARPNESS (λ_max={metrics.spectral_sharpness:.2f}): "
                "Test model performance on domain-shifted data"
            )

        if adversarial_risk == RiskLevel.RISK:
            recommendations.append(
                "NEGATIVE EIGENVALUES DETECTED: Model at saddle point (unstable). "
                "Consider: (1) Continue training, (2) Adversarial training, "
                "(3) Verify with gradient checks"
            )
        elif adversarial_risk == RiskLevel.CAUTION:
            recommendations.append(
                "SMALL NEGATIVE EIGENVALUES: May indicate near-saddle configuration. "
                "Monitor for training instability"
            )

        if not recommendations:
            recommendations.append(
                "All metrics within safe ranges. Continue monitoring during deployment."
            )

        return recommendations


def compute_landscape_metrics(
    eigenvalues: np.ndarray,
    condition_number: Optional[float] = None,
) -> LandscapeMetrics:
    """
    Compute comprehensive landscape metrics from eigenvalue spectrum.

    This is a utility function that computes all landscape metrics from
    eigenvalues, handling edge cases and numerical stability.

    Args:
        eigenvalues: Array of eigenvalues (should be sorted descending)
        condition_number: Pre-computed condition number (optional)

    Returns:
        LandscapeMetrics object with all computed metrics

    Example:
        >>> eigenvalues = np.array([2.0, 1.5, 0.8, 0.3, 0.1, -0.05])
        >>> metrics = compute_landscape_metrics(eigenvalues)
        >>> print(metrics.spectral_sharpness)  # 2.0
        >>> print(metrics.has_negative_eigenvalues)  # True
    """
    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # Spectral sharpness (maximum eigenvalue)
    spectral_sharpness = float(np.max(sorted_eigenvalues))

    # Minimum eigenvalue
    min_eigenvalue = float(np.min(sorted_eigenvalues))

    # Eigenvalue spread
    eigenvalue_spread = spectral_sharpness - min_eigenvalue

    # Condition number
    if condition_number is None:
        max_abs = np.max(np.abs(sorted_eigenvalues))
        min_abs = np.min(np.abs(sorted_eigenvalues[sorted_eigenvalues != 0]))
        if min_abs == 0 or min_abs < 1e-10:
            condition_number = float("inf")
        else:
            condition_number = max_abs / min_abs

    # Detect negative eigenvalues
    has_negative_eigenvalues = bool(np.any(sorted_eigenvalues < -1e-6))

    # Power-law compliance (measure deviation from expected distribution)
    power_law_compliance = _compute_power_law_compliance(sorted_eigenvalues)

    # Stability score (composite metric, 0-1, higher is better)
    stability_score = _compute_stability_score(
        condition_number,
        spectral_sharpness,
        has_negative_eigenvalues,
        power_law_compliance,
    )

    return LandscapeMetrics(
        condition_number=condition_number,
        spectral_sharpness=spectral_sharpness,
        min_eigenvalue=min_eigenvalue,
        eigenvalue_spread=eigenvalue_spread,
        power_law_compliance=power_law_compliance,
        has_negative_eigenvalues=has_negative_eigenvalues,
        stability_score=stability_score,
        eigenvalue_spectrum=sorted_eigenvalues.tolist(),
    )


def _compute_power_law_compliance(eigenvalues: np.ndarray) -> float:
    """
    Compute deviation from expected power-law distribution.

    In overparameterized networks, eigenvalues typically follow a power-law:
    - Most eigenvalues cluster near zero (flat directions)
    - Few eigenvalues dominate (sharp directions)
    - Top-k eigenvalues account for ~80% of total trace

    Args:
        eigenvalues: Sorted eigenvalues (descending)

    Returns:
        Compliance score (0 = perfect power-law, 1 = maximum deviation)
    """
    if len(eigenvalues) < 10:
        return 0.5  # Not enough data to judge

    # Compute cumulative sum
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if len(positive_eigenvalues) == 0:
        return 1.0  # Pathological case

    total_trace = np.sum(positive_eigenvalues)
    cumsum = np.cumsum(positive_eigenvalues)

    # Expected: top 20% of eigenvalues account for ~80% of trace
    top_20_idx = max(1, len(positive_eigenvalues) // 5)
    top_20_contribution = cumsum[top_20_idx] / total_trace

    # Ideal is 0.8, measure deviation
    deviation = abs(top_20_contribution - 0.8)

    # Normalize to [0, 1]
    compliance = min(1.0, deviation / 0.5)

    return compliance


def _compute_stability_score(
    condition_number: float,
    spectral_sharpness: float,
    has_negative_eigenvalues: bool,
    power_law_compliance: float,
) -> float:
    """
    Compute composite stability score (0-1, higher is better).

    Combines all metrics into a single robustness indicator.

    Args:
        condition_number: Hessian condition number
        spectral_sharpness: Maximum eigenvalue
        has_negative_eigenvalues: Whether negative eigenvalues exist
        power_law_compliance: Deviation from power-law (0-1)

    Returns:
        Stability score (0-1)
    """
    # Component scores (0-1, higher is better)

    # Condition number score
    if condition_number == float("inf"):
        cond_score = 0.0
    elif condition_number < 100:
        cond_score = 1.0
    elif condition_number < 1000:
        cond_score = 0.5
    else:
        cond_score = max(0.0, 0.5 - (condition_number - 1000) / 10000)

    # Spectral sharpness score
    if spectral_sharpness < 0.5:
        sharp_score = 1.0
    elif spectral_sharpness < 1.5:
        sharp_score = 0.5
    else:
        sharp_score = max(0.0, 0.5 - (spectral_sharpness - 1.5) / 5)

    # Negative eigenvalue penalty
    neg_score = 0.0 if has_negative_eigenvalues else 1.0

    # Power-law score (invert compliance since lower is better)
    power_law_score = 1.0 - power_law_compliance

    # Weighted combination
    stability_score = 0.3 * cond_score + 0.3 * sharp_score + 0.2 * neg_score + 0.2 * power_law_score

    return stability_score
