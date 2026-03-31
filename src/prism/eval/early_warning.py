"""Detection logic for early-warning signatures of model collapse.

Two detection layers:

1. **Silent Semantic Drift** (original) — detects when structural decay
   (perplexity) precedes semantic decay (accuracy) in SGT training runs.
   This is the generation-over-generation signal.

2. **Lattice Viability Monitor** (new) — detects when the E <= C condition
   is threatened at the session-lattice level, BEFORE collapse reaches the
   training pipeline.  Uses the five landscape observables from
   signal_scorer.py to track:

   - Oracle capacity decline (participants becoming less engaged / more
     compliant -- the Brescia Shadow-wins-4-of-5 pattern)
   - Integration depth narrowing (sessions converging on fewer meaning
     dimensions -- attractor vocabulary narrowing)
   - Solipsistic drift (dense attractor regions with declining training
     weight -- model optimising for internal coherence at expense of
     external grounding)

Theoretical grounding:
  E(t) = H(P_env(t)) + d/dt D_KL(P_env || P_model)
  C(t) = sup I(S; Y | X)  over socially embedded oracle interactions

  The viability condition E <= C is violated when:
  - Oracle capacity C drops (fewer flourishing participants, less novel input)
  - Environmental entropy E rises (model encounters more diverse scenarios)
  - The gap between them crosses the critical threshold

  The lattice viability monitor tracks empirical proxies for both sides.
"""

from typing import List, Dict, Any, Optional
import math


class EarlyWarningAnalyzer:
    """
    Analyzes trajectories to detect if structural decay (perplexity)
    precedes semantic decay (accuracy).
    """

    def __init__(self, acc_threshold: float = 0.10, ppl_threshold: float = 0.05):
        self.acc_threshold = acc_threshold
        self.ppl_threshold = ppl_threshold

    def detect(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detects the 'Silent Semantic Drift' signature.
        The signature is detected if perplexity fails BEFORE accuracy.
        """
        if len(metrics) < 2:
            return {"signature_detected": False, "reason": "Insufficient data"}

        base_acc = metrics[0].get("grounded_arc_accuracy", 0.0)
        base_ppl = metrics[0].get("grounded_arc_perplexity", 1.0)

        t_acc = -1
        t_ppl = -1

        for m in metrics:
            gen = m.get("generation", 0)
            acc = m.get("grounded_arc_accuracy", 0.0)
            ppl = m.get("grounded_arc_perplexity", base_ppl)

            # Check for accuracy failure (drop > threshold)
            if t_acc == -1 and base_acc > 0:
                if (base_acc - acc) / base_acc >= self.acc_threshold:
                    t_acc = gen

            # Check for perplexity failure (rise > threshold)
            if t_ppl == -1 and base_ppl > 0:
                if (ppl - base_ppl) / base_ppl >= self.ppl_threshold:
                    t_ppl = gen

        # Signature: Perplexity fails BEFORE Accuracy
        signature_detected = False
        if t_ppl != -1 and t_acc != -1:
            if t_ppl < t_acc:
                signature_detected = True
        elif t_ppl != -1 and t_acc == -1:
            # PPL failed but Acc stayed stable (also a form of early warning)
            signature_detected = True

        return {
            "signature_detected": signature_detected,
            "t_acc": t_acc,
            "t_ppl": t_ppl,
            "base_accuracy": base_acc,
            "base_perplexity": base_ppl
        }

    def generate_report(self, regime_name: str, detection_result: Dict[str, Any]) -> str:
        """Generates a summary string of the detection result."""
        sig = "detected" if detection_result["signature_detected"] else "not detected"
        return f"Regime: {regime_name} | Early warning signature {sig}."


# -----------------------------------------------------------------------
# Lattice Viability Monitor -- E <= C condition tracking
# -----------------------------------------------------------------------

class LatticeViabilityMonitor:
    """
    Monitors the E <= C viability condition across a corpus of
    collapsed training signals.

    Tracks three drift signatures:

    1. Oracle Decline: moving average of oracle_capacity dropping below
       a critical threshold.  This is the direct proxy for C(t) -- when
       participants produce less complex, less novel language, the system
       is losing its source of corrective gradient.

    2. Integration Narrowing: moving average of integration_depth
       declining.  This means sessions are traversing fewer independent
       meaning dimensions -- the attractor landscape is collapsing to
       fewer basins (the "archetypal vocabulary narrowing" Brescia
       observed in Orch-OS trials).

    3. Solipsistic Drift: training_weight trend declining while
       attractor density increases.  The model is attracting more
       sessions to existing regions but producing less training value
       from them -- optimising for Information IN at the expense of
       Information ABOUT.

    Each signature produces an alert level:
      - GREEN:  no drift detected
      - YELLOW: early warning (trend emerging but not critical)
      - RED:    viability condition likely violated
    """

    def __init__(
        self,
        window_size: int = 20,
        oracle_yellow: float = 0.4,
        oracle_red: float = 0.25,
        integration_decline_pct: float = 0.30,
        weight_decline_pct: float = 0.40,
    ):
        self.window_size = window_size
        self.oracle_yellow = oracle_yellow
        self.oracle_red = oracle_red
        self.integration_decline_pct = integration_decline_pct
        self.weight_decline_pct = weight_decline_pct

    def check(
        self,
        signals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check viability condition across a sequence of training signals.

        Args:
            signals: List of TrainingSignal dicts (from collapse.py output),
                     ordered chronologically.

        Returns:
            Dict with alert levels and diagnostic details.
        """
        if len(signals) < 3:
            return {
                "status": "INSUFFICIENT_DATA",
                "alerts": [],
                "message": f"Need at least 3 signals, have {len(signals)}",
            }

        alerts = []

        # --- Oracle Capacity trend ---
        oracle_alert = self._check_oracle_decline(signals)
        if oracle_alert:
            alerts.append(oracle_alert)

        # --- Integration Depth trend ---
        integration_alert = self._check_integration_narrowing(signals)
        if integration_alert:
            alerts.append(integration_alert)

        # --- Solipsistic Drift ---
        drift_alert = self._check_solipsistic_drift(signals)
        if drift_alert:
            alerts.append(drift_alert)

        # --- Cognitive Core Imbalance (Orch-OS alignment) ---
        core_alert = self._check_core_imbalance(signals)
        if core_alert:
            alerts.append(core_alert)

        # --- Affective Channel Silence ---
        affect_alert = self._check_affective_silence(signals)
        if affect_alert:
            alerts.append(affect_alert)

        # Overall status
        levels = [a["level"] for a in alerts]
        if "RED" in levels:
            status = "RED"
        elif "YELLOW" in levels:
            status = "YELLOW"
        else:
            status = "GREEN"

        return {
            "status": status,
            "alerts": alerts,
            "session_count": len(signals),
            "message": self._summarise(status, alerts),
        }

    def _check_oracle_decline(
        self, signals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check if oracle capacity (effective, including affective precision) is declining."""
        oracle_values = [
            s.get("oracle_effective", s.get("oracle_capacity", 0.0))
            for s in signals
            if s.get("oracle_effective", s.get("oracle_capacity", 0.0)) > 0
        ]

        if len(oracle_values) < 3:
            return None

        # Recent window
        window = oracle_values[-min(self.window_size, len(oracle_values)):]
        recent_mean = sum(window) / len(window)

        # Overall mean (baseline)
        overall_mean = sum(oracle_values) / len(oracle_values)

        if recent_mean < self.oracle_red:
            return {
                "type": "oracle_decline",
                "level": "RED",
                "recent_mean": round(recent_mean, 4),
                "overall_mean": round(overall_mean, 4),
                "message": (
                    f"Oracle capacity critically low: "
                    f"recent={recent_mean:.3f} (threshold={self.oracle_red})"
                ),
            }
        elif recent_mean < self.oracle_yellow:
            return {
                "type": "oracle_decline",
                "level": "YELLOW",
                "recent_mean": round(recent_mean, 4),
                "overall_mean": round(overall_mean, 4),
                "message": (
                    f"Oracle capacity declining: "
                    f"recent={recent_mean:.3f} (threshold={self.oracle_yellow})"
                ),
            }

        return None

    def _check_integration_narrowing(
        self, signals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check if integration depth is narrowing."""
        id_values = [
            s.get("integration_depth", 0.0) for s in signals
            if s.get("integration_depth", 0.0) > 0
        ]

        if len(id_values) < 6:
            return None

        # Compare first half to second half
        mid = len(id_values) // 2
        first_half_mean = sum(id_values[:mid]) / mid
        second_half_mean = sum(id_values[mid:]) / (len(id_values) - mid)

        if first_half_mean < 1e-6:
            return None

        decline = (first_half_mean - second_half_mean) / first_half_mean

        if decline >= self.integration_decline_pct:
            return {
                "type": "integration_narrowing",
                "level": "YELLOW",
                "first_half_mean": round(first_half_mean, 4),
                "second_half_mean": round(second_half_mean, 4),
                "decline_pct": round(decline * 100, 1),
                "message": (
                    f"Integration depth narrowing: "
                    f"{first_half_mean:.2f} -> {second_half_mean:.2f} "
                    f"({decline*100:.0f}% decline)"
                ),
            }

        return None

    def _check_solipsistic_drift(
        self, signals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check for solipsistic drift: increasing attractor density with
        declining training weight.
        """
        # Need density and weight data
        pairs = [
            (s.get("attractor_density", 0), s.get("weight", 0.0))
            for s in signals
        ]

        if len(pairs) < 6:
            return None

        mid = len(pairs) // 2
        first_weights = [w for _, w in pairs[:mid] if w > 0]
        second_weights = [w for _, w in pairs[mid:] if w > 0]
        second_densities = [d for d, _ in pairs[mid:]]

        if not first_weights or not second_weights:
            return None

        first_mean_w = sum(first_weights) / len(first_weights)
        second_mean_w = sum(second_weights) / len(second_weights)
        mean_density = sum(second_densities) / len(second_densities)

        if first_mean_w < 1e-6:
            return None

        weight_decline = (first_mean_w - second_mean_w) / first_mean_w

        # Drift = weight declining while density growing
        if weight_decline >= self.weight_decline_pct and mean_density > 2:
            return {
                "type": "solipsistic_drift",
                "level": "RED",
                "first_mean_weight": round(first_mean_w, 4),
                "second_mean_weight": round(second_mean_w, 4),
                "mean_density": round(mean_density, 2),
                "weight_decline_pct": round(weight_decline * 100, 1),
                "message": (
                    f"Solipsistic drift: training weight declining "
                    f"({first_mean_w:.2f} -> {second_mean_w:.2f}, "
                    f"-{weight_decline*100:.0f}%) while attractor "
                    f"density={mean_density:.1f}"
                ),
            }

        return None

    def _check_core_imbalance(
        self, signals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check for cognitive core imbalance (Orch-OS alignment).

        If the same 2-3 cores dominate every session and others are
        consistently zero, the pipeline is not engaging the full
        cognitive architecture.  This maps to Brescia's observation
        that healthy symbolic processing activates diverse cores,
        while pathological processing fixates on a single mode.

        Requires signals with cognitive_core data (from collapse_with_routing).
        Falls back gracefully if core data is absent.
        """
        # Use the component scores as proxies for core activity
        # Track which observables are consistently zero
        observables = [
            "integration_depth", "oracle_effective", "contradiction_score",
            "affective_granularity", "semantic_entropy", "temporal_coherence",
            "novelty",
        ]

        zero_counts = {obs: 0 for obs in observables}
        valid_signals = [s for s in signals if s.get("weight", 0) >= 0]

        if len(valid_signals) < 5:
            return None

        for s in valid_signals:
            for obs in observables:
                if s.get(obs, 0.0) < 0.01:
                    zero_counts[obs] += 1

        # Alert if 3+ cores are zero in >60% of sessions
        n = len(valid_signals)
        silent_cores = [
            obs for obs, count in zero_counts.items()
            if count / n > 0.6
        ]

        if len(silent_cores) >= 3:
            return {
                "type": "core_imbalance",
                "level": "YELLOW",
                "silent_cores": silent_cores,
                "session_count": n,
                "message": (
                    f"Cognitive core imbalance: {len(silent_cores)} cores "
                    f"silent in >60% of sessions: {', '.join(silent_cores)}"
                ),
            }

        return None

    def _check_affective_silence(
        self, signals: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if the affective channel is consistently silent.

        affective_granularity = 0 across all sessions means the [FELT: label]
        pipeline is not producing data.  This could indicate:
        - The interviewer agent isn't emitting [FELT: tags]
        - The extraction pipeline is broken
        - Consent is universally denying felt_state
        - The model isn't following the system prompt

        This is a distinct signal from core imbalance because the affective
        channel is the precision-weighting mechanism (Adolphs/Friston):
        without it, oracle_effective = oracle_capacity, losing the
        amplification from emotional granularity.
        """
        valid = [s for s in signals if s.get("consent_allows_training", False)]

        if len(valid) < 3:
            return None

        # Check affective_granularity across consented sessions
        silent = sum(
            1 for s in valid
            if s.get("affective_granularity", 0.0) < 0.01
        )

        if silent == len(valid):
            return {
                "type": "affective_silence",
                "level": "YELLOW",
                "silent_sessions": silent,
                "total_sessions": len(valid),
                "message": (
                    f"Affective channel silent: 0 sessions with "
                    f"felt_state data ({len(valid)} consented sessions). "
                    f"Check [FELT: label] extraction pipeline."
                ),
            }

        return None

    def _summarise(self, status: str, alerts: List[Dict[str, Any]]) -> str:
        """Generate a human-readable summary."""
        if status == "GREEN":
            return "E <= C viability condition: holding. No drift signatures detected."

        alert_msgs = [a["message"] for a in alerts]
        prefix = {
            "YELLOW": "E <= C viability condition: early warning.",
            "RED": "E <= C viability condition: CRITICAL.",
        }.get(status, "")

        return f"{prefix} {'; '.join(alert_msgs)}"

    def generate_report(
        self, check_result: Dict[str, Any]
    ) -> str:
        """Generate a formatted report string."""
        status = check_result["status"]
        lines = [
            f"=== Lattice Viability Report ===",
            f"Status:   {status}",
            f"Sessions: {check_result.get('session_count', '?')}",
            f"Message:  {check_result.get('message', '')}",
        ]

        alerts = check_result.get("alerts", [])
        if alerts:
            lines.append(f"\nAlerts ({len(alerts)}):")
            for a in alerts:
                lines.append(f"  [{a['level']}] {a['type']}: {a['message']}")

        return "\n".join(lines)
