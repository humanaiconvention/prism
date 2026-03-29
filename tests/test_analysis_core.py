import pytest
import torch
from prism.analysis import (
    compute_spectral_metrics,
    compute_shannon_effective_rank,
    compute_top_eigenvalues,
    compute_top_head_idx
)

def test_compute_spectral_metrics():
    # Setup dummy data
    batch_size = 2
    seq_len = 10
    hidden_dim = 16
    
    # 1. Batched case
    x_batched = torch.randn(batch_size, seq_len, hidden_dim)
    entropy_b, eff_dim_b = compute_spectral_metrics(x_batched)
    
    assert len(entropy_b) == batch_size
    assert len(eff_dim_b) == batch_size
    assert all(isinstance(e, float) for e in entropy_b)
    assert all(isinstance(d, float) for d in eff_dim_b)
    
    # 2. Unbatched case
    x_single = torch.randn(seq_len, hidden_dim)
    entropy_s, eff_dim_s = compute_spectral_metrics(x_single)
    
    assert isinstance(entropy_s, float)
    assert isinstance(eff_dim_s, float)

def test_compute_shannon_effective_rank():
    seq_len = 10
    hidden_dim = 16
    
    # Random orthogonal matrix should have max rank (i.e. close to hidden_dim)
    x = torch.randn(seq_len, hidden_dim)
    rank = compute_shannon_effective_rank(x)
    
    assert isinstance(rank, float)
    assert rank > 1.0

def test_compute_top_eigenvalues():
    batch_size = 2
    seq_len = 10
    hidden_dim = 16
    k = 3
    
    # 1. Batched
    x_batched = torch.randn(batch_size, seq_len, hidden_dim)
    top_k_b = compute_top_eigenvalues(x_batched, k)
    
    assert len(top_k_b) == batch_size
    assert len(top_k_b[0]) == k
    assert all(isinstance(val, float) for val in top_k_b[0])
    
    # 2. Unbatched
    x_single = torch.randn(seq_len, hidden_dim)
    top_k_s = compute_top_eigenvalues(x_single, k)
    
    assert len(top_k_s) == k
    assert all(isinstance(val, float) for val in top_k_s)

def test_compute_top_head_idx():
    num_layers = 2
    batch_size = 1
    num_heads = 4
    seq_len = 5
    
    # List of attention tensors per layer
    attentions = [
        torch.randn(batch_size, num_heads, seq_len, seq_len) for _ in range(num_layers)
    ]
    
    top_head = compute_top_head_idx(attentions)
    
    assert isinstance(top_head, str)
    assert top_head.startswith("L")
    assert "_H" in top_head
