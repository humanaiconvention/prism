"""Statistical calculation of temporal signatures and failure regimes."""

import json
import os
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

class TemporalAnalyzer:
    """Analyzes trajectories to compute T_OOD, T_PPL, and delta_t."""
    
    def __init__(self, drop_threshold: float = 0.05, rise_threshold: float = 0.05):
        self.drop_threshold = drop_threshold
        self.rise_threshold = rise_threshold
        
    def analyze_run(self, results: List[Dict[str, Any]], ood_key: str = "arc_easy_accuracy", ppl_key: str = "val_perplexity") -> Dict[str, Any]:
        """Calculates temporal metrics for a single seed run."""
        if not results: return {}
        
        df = pd.DataFrame(results).sort_values("generation")
        
        if ood_key not in df.columns or ppl_key not in df.columns:
            return {"status": "missing_metrics"}
            
        base_ood = df.iloc[0][ood_key]
        base_ppl = df.iloc[0][ppl_key]
        
        t_ood = -1
        t_ppl = -1
        
        for idx, row in df.iterrows():
            gen = row["generation"]
            
            # OOD Drop (>5% relative or absolute? Let's use relative for PPL, absolute/relative for ACC)
            # We'll use relative to be consistent. If base_ood is 0, we can't drop.
            if t_ood == -1 and base_ood > 0:
                rel_drop = (base_ood - row[ood_key]) / base_ood
                if rel_drop >= self.drop_threshold:
                    t_ood = gen
                    
            # PPL Rise (>5% relative)
            if t_ppl == -1 and base_ppl > 0:
                rel_rise = (row[ppl_key] - base_ppl) / base_ppl
                if rel_rise >= self.rise_threshold:
                    t_ppl = gen
                    
        # Calculate Delta T
        # delta_t = T_PPL - T_OOD
        # If delta_t > 0: PPL rose AFTER OOD dropped (accuracy_first)
        # If delta_t < 0: PPL rose BEFORE OOD dropped (perplexity_first)
        delta_t = None
        if t_ood != -1 and t_ppl != -1:
            delta_t = t_ppl - t_ood
            
        # Classify Regime
        regime = "no_collapse"
        if t_ood != -1 and t_ppl != -1:
            if delta_t > 0: regime = "accuracy_first"
            elif delta_t < 0: regime = "perplexity_first"
            else: regime = "synchronized"
        elif t_ood != -1:
            regime = "accuracy_first" # Only accuracy collapsed
        elif t_ppl != -1:
            regime = "perplexity_first" # Only perplexity collapsed
            
        return {
            "baseline_ood": base_ood,
            "baseline_ppl": base_ppl,
            "T_OOD": t_ood,
            "T_PPL": t_ppl,
            "delta_t": delta_t,
            "regime_classification": regime
        }
