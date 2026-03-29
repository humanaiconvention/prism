import pytest
import torch

from prism.geometry import compute_cosine, split_vector_by_direction

# --- Pytest Tests ---

def test_compute_cosine():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    assert compute_cosine(a, b) == 0.0

    a = torch.tensor([1.0, 1.0, 0.0])
    b = torch.tensor([1.0, 1.0, 0.0])
    # Expect 1.0, account for floating point errors
    assert abs(compute_cosine(a, b) - 1.0) < 1e-6

    a = torch.tensor([1.0, 1.0, 0.0])
    b = torch.tensor([-1.0, -1.0, 0.0])
    assert abs(compute_cosine(a, b) + 1.0) < 1e-6

    # Test near-zero vector falls back to 0.0
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    assert compute_cosine(a, b) == 0.0

def test_split_vector_by_direction():
    vec = torch.tensor([3.0, 4.0, 5.0])
    direction = torch.tensor([1.0, 0.0, 0.0])

    parallel, orthogonal = split_vector_by_direction(vec, direction)
    assert torch.allclose(parallel, torch.tensor([3.0, 0.0, 0.0]))
    assert torch.allclose(orthogonal, torch.tensor([0.0, 4.0, 5.0]))
    
    # Verify reconstruction
    assert torch.allclose(parallel + orthogonal, vec)

    # Verify orthogonality of the split (dot product approximately zero)
    assert abs(torch.dot(parallel, orthogonal).item()) < 1e-6
    
    # Test near-zero direction vector yields zero parallel component and original vec as orthogonal
    vec = torch.tensor([3.0, 4.0, 5.0])
    zero_dir = torch.tensor([0.0, 0.0, 0.0])
    parallel, orthogonal = split_vector_by_direction(vec, zero_dir)
    assert torch.allclose(parallel, torch.zeros_like(vec))
    assert torch.allclose(orthogonal, vec)
