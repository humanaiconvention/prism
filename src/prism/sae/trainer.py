"""TopK Sparse Autoencoder (SAE) training module."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional

class TopKSAE(nn.Module):
    """Minimal TopK SAE implementation."""
    def __init__(self, hidden_size: int, dict_size: int, k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.k = k
        
        # Encoder: Linear + Bias
        self.encoder = nn.Linear(hidden_size, dict_size)
        self.encoder.bias.data.zero_()
        
        # Decoder: Linear (Normalized)
        self.decoder = nn.Linear(dict_size, hidden_size, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().t()
        self._normalize_decoder()

    def _normalize_decoder(self):
        """Forces decoder weights to unit norm."""
        with torch.no_grad():
            norm = torch.norm(self.decoder.weight, dim=0, keepdim=True)
            self.decoder.weight.div_(norm + 1e-8)

    def forward(self, x: torch.Tensor):
        # 1. Encode
        activations = self.encoder(x)
        
        # 2. TopK Sparsity
        # Select only the top k activations, zero the rest
        topk_vals, topk_indices = torch.topk(activations, self.k, dim=-1)
        
        # Create a sparse representation
        sparse_acts = torch.zeros_like(activations)
        sparse_acts.scatter_(-1, topk_indices, topk_vals)
        
        # 3. Decode
        reconstruction = self.decoder(sparse_acts)
        
        return reconstruction, sparse_acts

class SAETrainer:
    """Trains a TopK SAE on captured residual stream activations."""
    
    def __init__(self, hidden_size: int, dict_size: int, k: int = 32, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = TopKSAE(hidden_size, dict_size, k).to(self.device)
        self.optimizer = optim.Adam(self.sae.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """Performs a single training step on a batch of activations."""
        self.sae.train()
        x = x.to(self.device)
        
        # Forward
        reconstruction, sparse_acts = self.sae(x)
        loss = self.criterion(reconstruction, x)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Keep decoder normalized
        self.sae._normalize_decoder()
        
        # Metrics
        with torch.no_grad():
            l0 = (sparse_acts > 0).float().sum(-1).mean().item()
            mse = loss.item()
            
        return {"loss": mse, "l0": l0}
        
    def save_dictionary(self, path: str):
        torch.save({
            "state_dict": self.sae.state_dict(),
            "config": {
                "hidden_size": self.sae.hidden_size,
                "dict_size": self.sae.dict_size,
                "k": self.sae.k
            }
        }, path)
