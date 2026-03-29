"""Linear probing and concept direction extraction."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class LinearProbe(nn.Module):
    """Simple logistic regression probe."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.net(x))

class ConceptProber:
    """Trains linear probes on residual stream states to identify concept directions."""
    
    def __init__(self, hidden_size: int, device: str = "cpu"):
        self.hidden_size = hidden_size
        self.device = device
        self.probes = {} # layer_idx -> LinearProbe

    def train_layer_probe(self, 
                          layer_idx: int, 
                          hidden_states: torch.Tensor, 
                          labels: torch.Tensor, 
                          epochs: int = 50, 
                          lr: float = 1e-2) -> float:
        """
        Trains a probe on states for a specific layer.
        hidden_states: (N, hidden_size)
        labels: (N,) binary 0 or 1
        """
        probe = LinearProbe(self.hidden_size).to(self.device)
        optimizer = optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        X = hidden_states.to(self.device).float()
        y = labels.to(self.device).float().unsqueeze(1)
        
        probe.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = probe(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        self.probes[layer_idx] = probe
        
        # Calculate accuracy
        probe.eval()
        with torch.no_grad():
            preds = (probe(X) > 0.5).float()
            acc = (preds == y).float().mean().item()
            
        return acc

    def get_concept_direction(self, layer_idx: int) -> torch.Tensor:
        """Returns the normal vector of the decision boundary (the 'concept direction')."""
        if layer_idx not in self.probes:
            raise ValueError(f"No probe trained for layer {layer_idx}")
        
        from prism.geometry.core import unit_vector
        # The weights of the linear layer represent the direction
        direction = self.probes[layer_idx].net[0].weight.data.detach().clone()
        # Normalize
        direction = unit_vector(direction, eps=1e-8)
        return direction.squeeze()

    def compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Computes Centered Kernel Alignment (CKA) between two sets of representations.
        X, Y shapes: (N, dim)
        """
        X = X.float()
        Y = Y.float()
        
        def centering(K):
            n = K.shape[0]
            unit = torch.ones(n, n, device=K.device)
            I = torch.eye(n, device=K.device)
            H = I - unit / n
            return H @ K @ H

        def linear_kernel(A):
            return A @ A.t()

        K = linear_kernel(X)
        L = linear_kernel(Y)
        
        Kc = centering(K)
        Lc = centering(L)
        
        hsic = (Kc * Lc).sum()
        norm_x = torch.sqrt((Kc * Kc).sum())
        norm_y = torch.sqrt((Lc * Lc).sum())
        
        return (hsic / (norm_x * norm_y)).item()

class SteeringVectorExtractor:
    """Constructs causal inner product (CIP) steering vectors."""
    
    def extract_cip_vector(self, 
                           clean_states: torch.Tensor, 
                           corrupt_states: torch.Tensor) -> torch.Tensor:
        """
        Implements a simplified version of Causal Inner Product extraction.
        Essentially finds the direction that best explains the counterfactual difference.
        """
        from prism.geometry.core import unit_vector
        # delta = mean(clean) - mean(corrupt)
        diffs = clean_states.float() - corrupt_states.float()
        direction = diffs.mean(dim=0)
        
        # Normalize
        direction = unit_vector(direction, eps=1e-8)
        return direction
