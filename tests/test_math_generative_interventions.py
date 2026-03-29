import pytest
import torch
import numpy as np

from prism.geometry.core import (
    project_onto_basis,
    project_out_basis,
    compute_mean_cosine_to_ref,
    apply_givens_rotations
)

# --- Pytest Tests ---

def test_project_onto_and_out_of_basis():
    # 3D vector space
    tensor = torch.tensor([[1.0, 2.0, 3.0]])
    
    # Basis spanning the x and y axes
    basis = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    
    proj_in = project_onto_basis(tensor, basis)
    proj_out = project_out_basis(tensor, basis)
    
    # The projection onto XY plane should be [1.0, 2.0, 0.0]
    assert torch.allclose(proj_in, torch.tensor([[1.0, 2.0, 0.0]]))
    
    # The projection out of XY plane (i.e. Z axis) should be [0.0, 0.0, 3.0]
    assert torch.allclose(proj_out, torch.tensor([[0.0, 0.0, 3.0]]))
    
    # Sum of projected_in and projected_out should equal original
    assert torch.allclose(proj_in + proj_out, tensor)

def test_compute_mean_cosine_to_ref():
    s1 = torch.tensor([1.0, 0.0, 0.0])
    s2 = torch.tensor([0.0, 1.0, 0.0])
    s3 = torch.tensor([1.0, 1.0, 0.0]) # 45 degrees, cosine is 1/sqrt(2) ~ 0.707
    
    states = [s1, s2, s3]
    
    mean_cos = compute_mean_cosine_to_ref(states, ref_idx=0)
    
    # cos(s1, s1) = 1.0
    # cos(s2, s1) = 0.0
    # cos(s3, s1) = 0.707106
    # Mean ~ 0.569
    assert np.isclose(mean_cos, (1.0 + 0.0 + 1.0 / np.sqrt(2)) / 3.0)

def test_apply_givens_rotations():
    rng = np.random.default_rng(42)
    weight = torch.eye(4) # 4x4 identity matrix
    
    # 1 rotation of pi/2 -> swaps two orthogonal vectors and flips a sign
    rotated = apply_givens_rotations(weight, rng, rotations=1, angle=np.pi/2)
    
    # Norms of the columns should be preserved since Givens rotations are orthogonal
    for col in range(4):
        assert np.isclose(torch.norm(rotated[:, col]).item(), 1.0)
    
    # Weight should be orthogonal (W^T W = I)
    assert torch.allclose(rotated.T @ rotated, torch.eye(4), atol=1e-6)
