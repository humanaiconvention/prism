"""Metrics for detecting early-warning signatures of semantic drift."""

from typing import List, Dict, Any

class EarlyWarningDetector:
    """
    Detects whether grounded performance declines before perplexity.
    This is the key signature of semantic drift in recursive learning.
    """
    
    def __init__(self, accuracy_threshold: float = 0.10, perplexity_threshold: float = 0.10):
        # Drop relative to baseline required to be considered a "failure"
        self.accuracy_threshold = accuracy_threshold
        # Increase relative to baseline required to be considered a "failure"
        self.perplexity_threshold = perplexity_threshold

    def analyze_trajectory(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes a generation-over-generation trajectory.
        Expected keys in results dict: 'generation', 'accuracy', 'perplexity'
        """
        if not results:
            return {"status": "error", "message": "No results provided"}
            
        g0 = results[0]
        base_acc = g0.get("accuracy", 1.0)
        base_ppl = g0.get("perplexity", 1.0)
        
        acc_failure_gen = -1
        ppl_failure_gen = -1
        
        early_warning_gap = []
        
        for r in results:
            gen = r["generation"]
            acc = r.get("accuracy", 0.0)
            ppl = r.get("perplexity", base_ppl)
            
            # Record gaps
            acc_drop = (base_acc - acc) / (base_acc if base_acc > 0 else 1.0)
            ppl_rise = (ppl - base_ppl) / (base_ppl if base_ppl > 0 else 1.0)
            
            early_warning_gap.append({
                "generation": gen,
                "acc_drop_pct": acc_drop,
                "ppl_rise_pct": ppl_rise,
                "gap": acc_drop - ppl_rise # Positive means accuracy dropped more than PPL rose
            })
            
            # Detect first failure
            if acc_failure_gen == -1 and acc_drop >= self.accuracy_threshold:
                acc_failure_gen = gen
                
            if ppl_failure_gen == -1 and ppl_rise >= self.perplexity_threshold:
                ppl_failure_gen = gen
                
        # Determine ordering
        ordering = "stable"
        if acc_failure_gen != -1 and ppl_failure_gen != -1:
            if acc_failure_gen < ppl_failure_gen:
                ordering = "grounding_failed_first" # The predicted signature
            elif acc_failure_gen > ppl_failure_gen:
                ordering = "perplexity_failed_first"
            else:
                ordering = "simultaneous_failure"
        elif acc_failure_gen != -1:
            ordering = "grounding_failed_only" # Also a form of the signature
        elif ppl_failure_gen != -1:
            ordering = "perplexity_failed_only"
            
        result = {
            "baseline_accuracy": base_acc,
            "baseline_perplexity": base_ppl,
            "accuracy_failure_generation": acc_failure_gen,
            "perplexity_failure_generation": ppl_failure_gen,
            "failure_ordering": ordering,
            "signature_detected": ordering in ["grounding_failed_first", "grounding_failed_only"],
            "trajectory_gaps": early_warning_gap
        }

        # PRISM geometric extension: if spectral_entropy and effective_dimension keys are
        # present in the trajectory, detect geometric silent drift alongside the behavioral signal.
        if results and all("spectral_entropy" in r for r in results):
            result.update(self._analyze_geometric_trajectory(results))

        return result

    def _analyze_geometric_trajectory(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect geometric silent drift using PRISM spectral metrics.

        Geometric silent drift: effective_dimension collapses or spectral_entropy rises
        *before* the behavioral accuracy warning fires.  This is the mechanistic
        precursor to the silent semantic drift signature.

        Args:
            results: Same trajectory list passed to analyze_trajectory().
                     Must contain 'spectral_entropy' and 'effective_dimension' keys.

        Returns:
            Dict with keys:
                geometric_failure_generation   int  (-1 if no geometric failure detected)
                geometric_drift_detected       bool
                spectral_entropy_trajectory    List[float]
                effective_dimension_trajectory List[float]
        """
        entropies = [r.get("spectral_entropy", 0.0) for r in results]
        dims = [r.get("effective_dimension", 0.0) for r in results]

        base_e = entropies[0] if entropies else 0.0
        base_d = dims[0] if dims else 0.0

        geo_failure_gen = -1
        for i, r in enumerate(results[1:], 1):
            e_rise = (r.get("spectral_entropy", base_e) - base_e) / (abs(base_e) + 1e-9)
            d_drop = (base_d - r.get("effective_dimension", base_d)) / (abs(base_d) + 1e-9)
            if e_rise >= self.perplexity_threshold or d_drop >= self.accuracy_threshold:
                geo_failure_gen = r.get("generation", i)
                break

        return {
            "geometric_failure_generation": geo_failure_gen,
            "geometric_drift_detected": geo_failure_gen != -1,
            "spectral_entropy_trajectory": entropies,
            "effective_dimension_trajectory": dims,
        }
