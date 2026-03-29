import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.arch.hybrid import HybridDiagnostics
from prism.causal.patching import ActivationPatcher
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_hybrid_diagnostics():
    model_name = "guiferrarib/genesis-152m-instruct"
    logging.info(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        output_hidden_states=True
    )
    
    diag = HybridDiagnostics(model)
    patcher = ActivationPatcher(model)
    
    # 1. Compare Linear vs Softmax Entropy
    # Layer 3 is FoX (Softmax), Layer 0 is GLA (Linear)
    fox_layer_idx = 3
    gla_layer_idx = 0
    
    prompt = "Mechanistic interpretability is the study of"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    logging.info("\n--- Comparing Attention Entropy (FoX vs GLA) ---")
    
    # We need to capture the attention weights for FoX
    # And the recurrent state for GLA
    # NOTE: Direct attention score capture requires model-specific hooks.
    # For this smoke test, we'll simulate the inputs to demonstrate the logic.
    
    mock_attn_scores = torch.softmax(torch.randn(1, 9, 10, 10), dim=-1)
    mock_gla_state = torch.randn(1, 64, 64)
    
    entropy_results = diag.compare_attention_entropy(mock_attn_scores, mock_gla_state)
    logging.info(f"Softmax (FoX) Attention Entropy: {entropy_results['softmax_entropy']:.4f}")
    logging.info(f"Linear (GLA) Spectral Entropy: {entropy_results['linear_spectral_entropy']:.4f}")

    # 2. Track Recurrent Attractors
    logging.info("\n--- Tracking Recurrent Attractors (GLA) ---")
    # Simulate a GLA state trajectory over 10 steps
    # We'll make it slowly converge to a specific direction
    base_state = torch.randn(64, 64)
    trajectory = []
    for i in range(10):
        # State = Base + decreasing noise
        noise_scale = 1.0 / (i + 1)
        step_state = base_state + torch.randn(64, 64) * noise_scale
        trajectory.append(step_state)
        
    attractor_results = diag.track_recurrent_attractors(trajectory)
    logging.info(f"Mean Rotation Angle: {attractor_results['mean_rotation_angle']:.2f}°")
    logging.info(f"Final Step Angle: {attractor_results['final_angle']:.2f}°")
    logging.info(f"Is Saturation Detected? {attractor_results['is_saturated']}")

if __name__ == "__main__":
    test_hybrid_diagnostics()
