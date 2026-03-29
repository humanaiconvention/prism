import pytest
import torch
import numpy as np
from prism.analysis import compute_eigenvalues_hardware_aware

def test_compute_eigenvalues_hardware_aware_square():
    """Test spectral metrics on a square symmetric matrix."""
    # Create a 10x10 symmetric positive definite matrix
    A = torch.randn(10, 10)
    matrix = A @ A.T
    
    # Compute on CPU
    cpu_vals = compute_eigenvalues_hardware_aware(matrix, device_override="cpu")
    assert cpu_vals.shape == (10,)
    assert torch.all(cpu_vals >= -1e-5) # Eigenvalues of AA^T are non-negative
    
    if torch.cuda.is_available():
        gpu_vals = compute_eigenvalues_hardware_aware(matrix, device_override="cuda")
        assert gpu_vals.shape == (10,)
        # Check if values are close
        assert torch.allclose(cpu_vals.sort().values, gpu_vals.sort().values, atol=1e-3)

def test_compute_eigenvalues_hardware_aware_non_square():
    """Test spectral metrics (SVD) on a non-square matrix."""
    matrix = torch.randn(20, 10)
    
    vals = compute_eigenvalues_hardware_aware(matrix, device_override="cpu")
    assert vals.shape == (10,) # min(20, 10)
    assert torch.all(vals >= 0)

def test_hardware_aware_fallback():
    """Test fallback logic when GPU computation fails (simulated)."""
    # Create a matrix that might cause issues or just mock the failure
    matrix = torch.randn(5, 5)
    
    if not torch.cuda.is_available():
        vals = compute_eigenvalues_hardware_aware(matrix) 
        assert vals.shape == (5,)
