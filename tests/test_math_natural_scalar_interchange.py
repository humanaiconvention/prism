import pytest
import torch
import numpy as np

from prism.geometry.core import unit_vector, compute_coeff

def test_unit_vector():
    vec = torch.tensor([3.0, 4.0]) # norm is 5
    u_vec = unit_vector(vec)
    
    assert torch.allclose(u_vec, torch.tensor([0.6, 0.8]))
    assert np.isclose(torch.norm(u_vec).item(), 1.0)
    
    # Test zero vector fallback
    zero_vec = torch.tensor([0.0, 0.0])
    u_zero = unit_vector(zero_vec, eps=1e-5)
    
    # Should divide by eps = 1e-5
    assert torch.allclose(u_zero, torch.tensor([0.0, 0.0]))

def test_compute_coeff():
    activation = torch.tensor([2.0, 3.0, 4.0])
    # direction along the y axis
    direction = torch.tensor([0.0, 1.0, 0.0])
    
    coeff = compute_coeff(activation, direction)
    
    # projection of [2,3,4] onto [0,1,0] is 3.0
    assert np.isclose(coeff, 3.0)
