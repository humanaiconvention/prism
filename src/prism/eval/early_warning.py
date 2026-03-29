"""Detection logic for early-warning signatures of model collapse."""

from typing import List, Dict, Any

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
