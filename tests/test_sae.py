import pytest
import torch
import torch.nn as nn
from prism.sae.trainer import SAETrainer, TopKSAE

class DummyModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        return self.fc(x)

def test_sae_trainer():
    # Setup
    hidden_size = 16
    dictionary_size = 32
    batch_size = 4
    seq_len = 10
    
    # Init SAE
    sae = TopKSAE(hidden_size=hidden_size, dict_size=dictionary_size, k=4)
    trainer = SAETrainer(hidden_size=hidden_size, dict_size=dictionary_size, k=4, lr=0.01)
    
    # Train step
    states = torch.randn(batch_size, seq_len, hidden_size)
    loss = trainer.train_step(states)
    
    # Assertions
    assert isinstance(loss, dict)
    assert "loss" in loss
    assert loss["loss"] > 0.0
    
    # Forward pass
    reconstruction, features = sae(states)
    assert reconstruction.shape == states.shape
    assert features.shape == (batch_size, seq_len, dictionary_size)
    
    # Top K sparsity check
    # Non-zero elements should be exactly K per vector
    non_zeros = (features > 0).float().sum(dim=-1)
    assert torch.all(non_zeros <= 4)
