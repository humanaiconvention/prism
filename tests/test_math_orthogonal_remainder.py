import pytest
import torch
import numpy as np

from prism.geometry.core import (
    orthogonal_residual,
    fit_pca_bank,
    make_random_orthogonal_subspace,
    project_state
)

# --- Pytest Tests ---

def test_orthogonal_residual():
    state = torch.tensor([2.0, 3.0, 4.0])
    # normalized x axis direction
    direction = torch.tensor([1.0, 0.0, 0.0])
    
    residual = orthogonal_residual(state, direction)
    
    # x component should be zeroed
    assert torch.allclose(residual, torch.tensor([0.0, 3.0, 4.0]))

def test_fit_pca_bank():
    # 3 points forming a line passing through (1, 1, 1) in direction (1, 1, 0)
    states = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 2.0, 1.0])
    ]
    
    bank = fit_pca_bank(states, requested_rank=1)
    
    # mean should be (1, 1, 1)
    assert np.allclose(bank["mean"], np.array([1.0, 1.0, 1.0]))
    
    # basis should be parallel to (1, 1, 0)
    # Unit vector is (1/sqrt(2), 1/sqrt(2), 0)
    assert bank["effective_rank"] == 1
    basis_vec = bank["basis"][:, 0]
    expected = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0])
    
    # PCA basis can be flipped by -1
    assert np.allclose(basis_vec, expected) or np.allclose(basis_vec, -expected)

def test_make_random_orthogonal_subspace():
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    
    basis = make_random_orthogonal_subspace(direction, rank=2, seed=42)
    
    assert basis.shape == (4, 2)
    
    # Basis vectors should be orthogonal to the direction
    assert np.isclose(np.dot(basis[:, 0], direction), 0.0, atol=1e-8)
    assert np.isclose(np.dot(basis[:, 1], direction), 0.0, atol=1e-8)
    
    # Basis vectors should be orthogonal to each other
    assert np.isclose(np.dot(basis[:, 0], basis[:, 1]), 0.0, atol=1e-8)
    
    # Basis vectors should be unit vectors
    assert np.isclose(np.linalg.norm(basis[:, 0]), 1.0)
    assert np.isclose(np.linalg.norm(basis[:, 1]), 1.0)

def test_project_state():
    # 3D space
    basis = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    mean = np.array([0.0, 0.0, 1.0]) # Translate along Z
    
    state = np.array([3.0, 4.0, 1.0]) # Lies perfectly in the translated plane
    
    proj = project_state(state, mean, basis)
    
    # 100% of the energy (relative to mean) should be captured
    assert np.isclose(proj["projection_fraction"], 1.0)
    # Norm of offset is sqrt(3^2 + 4^2 + 0^2) = 5
    assert np.isclose(proj["projection_norm"], 5.0)
