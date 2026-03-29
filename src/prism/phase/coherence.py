"""Hilbert phase extraction, PLV, and FFT telemetry."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.signal import hilbert

class PhaseAnalyzer:
    """Analyzes phase synchronization and spectral coherence across layers."""
    
    def __init__(self, model: nn.Module):
        self.model = model

    def extract_hilbert_phase(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extracts instantaneous phase using Hilbert transform along the token axis.
        hidden_states: (batch, seq_len, hidden_dim)
        Returns: (batch, seq_len, hidden_dim) phase angles in radians.
        """
        # scipy.signal.hilbert operates on the last axis by default, 
        # so we transpose to (batch, hidden_dim, seq_len)
        x = hidden_states.float().cpu().numpy()
        x_trans = x.transpose(0, 2, 1)
        
        analytic_signal = hilbert(x_trans)
        phase = np.angle(analytic_signal)
        
        # Transpose back to (batch, seq_len, hidden_dim)
        return torch.from_numpy(phase.transpose(0, 2, 1))

    def compute_plv(self, phase1: torch.Tensor, phase2: torch.Tensor) -> float:
        """
        Computes Phase Locking Value (PLV) between two layer phases.
        PLV = |1/T * sum(exp(i * (phi1 - phi2)))|
        """
        diff = phase1 - phase2
        complex_phase_diff = torch.complex(torch.cos(diff), torch.sin(diff))
        plv = torch.abs(complex_phase_diff.mean(dim=1)).mean().item()
        return plv

    def detect_phase_resets(self, phase_trajectory: torch.Tensor, threshold: float = 2.0) -> List[int]:
        """
        Identifies token positions where the phase relationship undergoes abrupt discontinuities.
        phase_trajectory: (seq_len, hidden_dim)
        """
        resets = []
        # Compute circular distance between consecutive steps
        for t in range(1, phase_trajectory.size(0)):
            diff = phase_trajectory[t] - phase_trajectory[t-1]
            # Wrap to [-pi, pi]
            dist = torch.atan2(torch.sin(diff), torch.cos(diff)).abs().mean().item()
            if dist > threshold:
                resets.append(t)
        return resets

    def fft_telemetry(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """Computes DFT across the token dimension to find dominant spatial frequencies."""
        # hidden_states: (batch, seq_len, hidden_dim)
        x = hidden_states.float()
        fft_res = torch.fft.fft(x, dim=1)
        magnitudes = torch.abs(fft_res)
        
        return {
            "magnitudes": magnitudes.mean(dim=0).mean(dim=-1).tolist(),
            "dominant_freq": torch.argmax(magnitudes.mean(dim=(0, 2))).item()
        }

    def phase_clustering_index(self, phases: torch.Tensor) -> float:
        """Measures circular variance of phases across the token batch."""
        # Low circular variance = high clustering
        sin_sum = torch.sin(phases).mean()
        cos_sum = torch.cos(phases).mean()
        r = torch.sqrt(sin_sum**2 + cos_sum**2)
        return (1 - r).item()
