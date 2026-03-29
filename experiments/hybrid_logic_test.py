import torch
import numpy as np
from prism.arch.hybrid import HybridDiagnostics
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_hybrid_logic():
    logging.info("Starting Hybrid Architecture Logic Test (Synthetic)...")
    
    # Create a mock model object to satisfy the constructor
    class MockConfig:
        hidden_size = 512
    class MockModel:
        config = MockConfig()
    
    diag = HybridDiagnostics(MockModel())
    
    # 1. Compare Attention Entropy (FoX-like vs GLA-like)
    logging.info("\n--- 1. Testing Attention Entropy Comparison ---")
    
    # Simulate Softmax Attention Scores (Focused vs Diffuse)
    # Focused: Low entropy, attending to few tokens
    focused_attn = torch.zeros(1, 8, 32, 32)
    focused_attn[:, :, :, 0] = 1.0 # Every token attends only to the first token
    
    # Diffuse: High entropy, attending to all tokens equally
    diffuse_attn = torch.full((1, 8, 32, 32), 1.0 / 32.0)
    
    # Simulate Linear Recurrent State (Compressed vs High-Rank)
    # High-Rank state
    high_rank_state = torch.randn(1, 64, 64)
    # Low-Rank state (rank 1)
    low_rank_state = torch.ones(1, 64, 1) @ torch.ones(1, 1, 64)
    
    res_focused = diag.compare_attention_entropy(focused_attn, low_rank_state)
    res_diffuse = diag.compare_attention_entropy(diffuse_attn, high_rank_state)
    
    logging.info(f"Focused Softmax Entropy: {res_focused['softmax_entropy']:.4f}")
    logging.info(f"Diffuse Softmax Entropy: {res_diffuse['softmax_entropy']:.4f}")
    logging.info(f"Low-Rank Linear Spectral Entropy: {res_focused['linear_spectral_entropy']:.4f}")
    logging.info(f"High-Rank Linear Spectral Entropy: {res_diffuse['linear_spectral_entropy']:.4f}")

    # 2. Track Recurrent Attractors
    logging.info("\n--- 2. Testing Recurrent Attractor Tracking ---")
    
    # Scenario A: Converging Trajectory (Locked Attractor)
    base_state = torch.randn(64, 64)
    u, s, v = torch.linalg.svd(base_state)
    # Force it to be low-rank for clearer attractor detection
    base_low_rank = u[:, :4] @ torch.diag(s[:4]) @ v[:4, :]
    
    converge_traj = []
    for i in range(10):
        # Extremely fast noise decay to simulate absolute convergence
        noise = torch.randn(64, 64) * (0.1 ** i) 
        converge_traj.append(base_low_rank + noise)
        
    res_converge = diag.track_recurrent_attractors(converge_traj)
    logging.info(f"Converging Trajectory - Final Angle: {res_converge['final_angle']:.2f}°")
    logging.info(f"Converging Trajectory - Saturated? {res_converge['is_saturated']}")

    # Scenario B: Random Trajectory (No Attractor)
    random_traj = [torch.randn(64, 64) for _ in range(10)]
    res_random = diag.track_recurrent_attractors(random_traj)
    logging.info(f"Random Trajectory - Final Angle: {res_random['final_angle']:.2f}°")
    logging.info(f"Random Trajectory - Saturated? {res_random['is_saturated']}")

    assert res_converge['is_saturated'] == True
    assert res_random['is_saturated'] == False
    logging.info("\n[+] Hybrid diagnostics logic verified successfully.")

if __name__ == "__main__":
    test_hybrid_logic()
