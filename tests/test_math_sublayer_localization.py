import pytest
import torch
import torch.nn.functional as F
import numpy as np

from prism.geometry.core import project_orthogonal_noise, fit_principal_basis
from prism.entropy.lens import score_choice_logits

# --- Pytest Tests ---

def test_score_choice_logits():
    # logits shape: (batch_size, seq_len, vocab_size)
    # math token id = 0, creative token id = 1
    # raw logits: math=2.0, creative=1.0, other=0.0
    logits = torch.tensor([[
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 0.0]
    ]])
    
    # 1. Positive label sign (Math is correct)
    item_pos = {
        "math_token_id": 0,
        "creative_token_id": 1,
        "label_sign": 1.0
    }
    res_pos = score_choice_logits(logits, item_pos)
    
    # Margin = 2.0 - 1.0 = 1.0
    assert np.isclose(res_pos["signed_label_margin"], 1.0)
    assert res_pos["label_accuracy"] == 1.0
    
    # Math prob vs creative prob ratio
    # e^2 / (e^2 + e^1) = e / (e + 1)
    expected_prob = np.exp(2.0) / (np.exp(2.0) + np.exp(1.0))
    assert np.isclose(res_pos["label_target_pairwise_prob"], expected_prob)
    
    # 2. Negative label sign (Creative is correct)
    item_neg = {
        "math_token_id": 0,
        "creative_token_id": 1,
        "label_sign": -1.0
    }
    res_neg = score_choice_logits(logits, item_neg)
    
    # Margin = - (2.0 - 1.0) = -1.0
    assert np.isclose(res_neg["signed_label_margin"], -1.0)
    assert res_neg["label_accuracy"] == 0.0 # because creative probability is lower
    assert np.isclose(res_neg["label_target_pairwise_prob"], 1.0 - expected_prob)

def test_project_orthogonal_noise():
    # 3D space, basis spans x and y axes
    basis = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    
    # Noise along x, y, and z
    noise = torch.tensor([[1.0, 2.0, 3.0]])
    
    # Projection should remove x and y components, leaving only z
    orth_noise = project_orthogonal_noise(noise, basis)
    assert torch.allclose(orth_noise, torch.tensor([[0.0, 0.0, 3.0]]))
    
    # Test batching support
    noise_batch = torch.tensor([
        [[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]]
    ])
    orth_noise_batch = project_orthogonal_noise(noise_batch, basis)
    assert torch.allclose(orth_noise_batch, torch.tensor([
        [[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]
    ]))

def test_fit_principal_basis():
    # 4 points in 3D, spanning mostly a 2D plane
    chunks = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 2.0, 0.0]]),
        np.array([[-1.0, 0.0, 0.0]]),
        np.array([[0.0, -2.0, 0.0]])
    ]
    
    fit = fit_principal_basis(chunks, top_k=2)
    
    # PCA should identify the 2D plane (x/y) over 3D space
    assert fit["topk_cumulative_explained_variance_ratio"] > 0.99
    assert fit["orthogonal_complement_variance_ratio"] < 0.01
    
    basis = fit["basis"]
    assert basis.shape == (3, 2)
    
    # The basis should span the x and y axes, so projection of [0, 0, 1] into it should be 0
    test_vec = torch.tensor([[0.0, 0.0, 1.0]])
    proj = (test_vec @ basis) @ basis.T
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-6)
