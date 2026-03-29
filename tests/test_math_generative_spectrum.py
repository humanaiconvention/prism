import pytest
import torch

# --- Core Math copied from Genesis Phase 21 for PRISM testing ---

def compute_entropy(probs):
    return float((-(probs * torch.log(probs + 1e-10)).sum()).item())

def compute_kl_div(p, q):
    p = p + 1e-10
    q = q + 1e-10
    return float(torch.sum(p * torch.log(p / q)).item())

# --- Pytest Tests ---

def test_compute_entropy():
    # Uniform distribution over 2 outcomes -> entropy is ln(2)
    probs = torch.tensor([0.5, 0.5])
    expected_entropy = torch.log(torch.tensor(2.0)).item()
    assert abs(compute_entropy(probs) - expected_entropy) < 1e-5

    # Certain distribution -> entropy is 0
    probs = torch.tensor([1.0, 0.0])
    assert abs(compute_entropy(probs)) < 1e-5

    # Uniform over 4 -> ln(4)
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    expected_entropy = torch.log(torch.tensor(4.0)).item()
    assert abs(compute_entropy(probs) - expected_entropy) < 1e-5

def test_compute_kl_div():
    # Identical distributions -> KL div is 0
    p = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.5, 0.5])
    assert abs(compute_kl_div(p, q)) < 1e-5

    # Different distributions
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.5, 0.5])
    # p*ln(p/q) = 1*ln(1/0.5) = ln(2)
    expected_kl = torch.log(torch.tensor(2.0)).item()
    assert abs(compute_kl_div(p, q) - expected_kl) < 1e-5

    # Near certain p vs highly unlikely q 
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.01, 0.99])
    # KL will be bounded arbitrarily by the 1e-10 addition, but we just verify it calculates without error
    kl = compute_kl_div(p, q)
    assert kl > 0.0
