import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism import SpectralMicroscope

@pytest.fixture(scope="module")
def small_model_setup():
    # Use a very small model for fast integration testing
    model_name = "HuggingFaceTB/SmolLM2-135M" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cpu", # Force CPU for reliable CI
        torch_dtype=torch.float32, 
        attn_implementation="eager"
    )
    return model, tokenizer

def test_full_automated_scan(small_model_setup):
    model, tokenizer = small_model_setup
    
    # Initialize the Microscope
    microscope = SpectralMicroscope()
    
    prompt = "The quick brown fox jumps over the lazy dog"
    
    # SmolLM2-135M has fewer layers, so target an earlier layer than 12
    target_layer = min(5, model.config.num_hidden_layers - 1)
    
    report = microscope.full_scan(model, tokenizer, prompt, target_layer=target_layer)
    
    # 1. Logit Lens Checks
    assert "logit_lens" in report
    assert "top_predictions" in report["logit_lens"]
    assert len(report["logit_lens"]["top_predictions"]) > 0
    assert "entropy_trajectory" in report["logit_lens"]
    
    # 2. Rank Profile
    assert "rank_profile" in report
    rp = report["rank_profile"]
    assert "input_rank" in rp
    assert "attn_impact" in rp
    assert "mlp_impact" in rp
    assert "net_impact" in rp
    
    # 3. Static Circuits
    assert "static_circuits" in report
    assert len(report["static_circuits"]) > 0
    sc = report["static_circuits"][0]
    assert "ov_rank" in sc
    assert "concordance" in sc
    assert "type" in sc
    
    # 4. Positional Sensitivity
    assert "positional_sensitivity" in report
    ps = report["positional_sensitivity"]
    assert "positional_drift" in ps
    assert "rank_collapse_ratio" in ps
