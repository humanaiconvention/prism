import pytest
import torch

from prism.geometry import (
    VALID_SPECTRAL_MODES,
    parse_spectral_modes,
    project_noise_to_component
)

# --- Pytest Tests ---

def test_parse_spectral_modes():
    modes = parse_spectral_modes("full_spectrum, principal_subspace")
    assert modes == ["full_spectrum", "principal_subspace"]
    
    with pytest.raises(ValueError, match="Unknown spectral modes"):
        parse_spectral_modes("full_spectrum, invalid_mode")
        
    with pytest.raises(ValueError, match="At least one"):
        parse_spectral_modes("")

def test_project_noise_to_component():
    torch.manual_seed(42)
    # 2 samples, 3 features
    noise = torch.randn(2, 3) 
    
    # Let basis be just the first coordinate (1, 0, 0)
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    
    # 1. Full spectrum
    res_full = project_noise_to_component(noise, basis, "full_spectrum")
    assert torch.allclose(res_full, noise)
    
    # 2. Principal subspace
    res_prin = project_noise_to_component(noise, basis, "principal_subspace")
    assert res_prin.shape == noise.shape
    # Only the first feature should be non-zero
    assert not torch.allclose(res_prin[:, 0], torch.zeros_like(res_prin[:, 0]))
    assert torch.allclose(res_prin[:, 1:], torch.zeros_like(res_prin[:, 1:]))
    assert torch.allclose(res_prin[:, 0], noise[:, 0])

    # 3. Orthogonal complement
    res_orth = project_noise_to_component(noise, basis, "orthogonal_complement")
    assert res_orth.shape == noise.shape
    # First feature should be exactly zeroed out
    assert torch.allclose(res_orth[:, 0], torch.zeros_like(res_orth[:, 0]))
    assert torch.allclose(res_orth[:, 1:], noise[:, 1:])
    
    # Check that principal + orthogonal = original
    assert torch.allclose(res_prin + res_orth, noise)

def test_project_noise_to_component_no_basis():
    noise = torch.randn(2, 3) 
    with pytest.raises(ValueError, match="basis is required"):
        project_noise_to_component(noise, None, "principal_subspace")
