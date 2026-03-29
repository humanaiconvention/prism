"""Calibration and diversity metrics for recursive evaluation."""

import torch
import numpy as np
from typing import List, Dict, Any

class CalibrationMetrics:
    """Calculates Brier Score and Expected Calibration Error (ECE)."""
    
    @staticmethod
    def calculate_brier_score(probabilities: List[float], labels: List[int]) -> float:
        """
        Brier Score: mean((prob - label)^2)
        Lower is better.
        """
        probs = np.array(probabilities)
        targets = np.array(labels)
        return np.mean((probs - targets)**2)

    @staticmethod
    def calculate_ece(probabilities: List[float], labels: List[int], n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE).
        Measures the gap between confidence and accuracy.
        """
        if not probabilities:
            return 0.0
            
        confidences = np.array(probabilities)
        accuracies = np.array(labels)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Items in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

class DiversityMetrics:
    """Calculates n-gram diversity and semantic variance."""
    
    @staticmethod
    def calculate_ngram_diversity(texts: List[str], n: int = 2) -> float:
        """
        Dist-n metric: Unique n-grams / Total n-grams.
        Lower values suggest mode collapse or repetitive generation.
        """
        if not texts:
            return 0.0
            
        ngrams = set()
        total_ngrams = 0
        
        for text in texts:
            tokens = text.lower().split()
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams.add(ngram)
                total_ngrams += 1
                
        if total_ngrams == 0:
            return 0.0
        return len(ngrams) / total_ngrams

    @staticmethod
    def calculate_self_bleu(texts: List[str]) -> float:
        """
        Simplified proxy for Self-BLEU to measure intra-generation redundancy.
        Higher values mean more redundancy (lower diversity).
        """
        if not texts:
            return 0.0
        unique_texts = set(texts)
        # Ratio of non-unique texts as a proxy for redundancy
        return 1.0 - (len(unique_texts) / len(texts))
