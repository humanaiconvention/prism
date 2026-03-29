import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.causal.patching import AttributionPatcher
from prism.lens.tuned import TunedLens, TunedLensTrainer
from prism.attention.circuits import AttentionAnalyzer

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

def test_induction_head_detection(small_model_setup):
    model, tokenizer = small_model_setup
    analyzer = AttentionAnalyzer(model)
    induction_scores = analyzer.detect_induction_heads(tokenizer, pattern_length=4)
    
    # Assert we get scores for all layers
    assert len(induction_scores) == model.config.num_hidden_layers
    # Assert scores are lists of floats (one for each head)
    for layer_scores in induction_scores.values():
        assert len(layer_scores) == model.config.num_attention_heads
        assert all(isinstance(score, float) for score in layer_scores)

def test_attribution_patching(small_model_setup):
    model, tokenizer = small_model_setup
    atp = AttributionPatcher(model)
    
    clean_text = "The Eiffel Tower is in Paris"
    corrupt_text = "The Eiffel Tower is in Rome"
    
    clean_inputs = tokenizer(clean_text, return_tensors="pt").to(model.device)
    corrupt_inputs = tokenizer(corrupt_text, return_tensors="pt").to(model.device)
    
    paris_id = tokenizer.encode(" Paris")[-1]
    rome_id = tokenizer.encode(" Rome")[-1]
    
    def metric_fn(logits):
        return logits[0, -1, paris_id] - logits[0, -1, rome_id]

    target_modules = {f"layer_{i}": model.model.layers[i] for i in range(min(3, model.config.num_hidden_layers))}
    attributions = atp.compute_attribution(clean_inputs, corrupt_inputs, target_modules, metric_fn)
    
    assert len(attributions) == len(target_modules)
    for name, attr in attributions.items():
        assert isinstance(attr, torch.Tensor)
        # Attribution shape should match output of the module
        assert attr.shape[0] == 1 # batch
        assert attr.shape[1] == clean_inputs.input_ids.shape[1] # seq len

def test_tuned_lens_training(small_model_setup):
    model, tokenizer = small_model_setup
    num_layers = model.config.num_hidden_layers
    lens = TunedLens(model.config.hidden_size, num_layers).to(model.device)
    trainer = TunedLensTrainer(model, lens)
    
    clean_text = "Testing the tuned lens"
    clean_inputs = tokenizer(clean_text, return_tensors="pt").to(model.device)
    
    loss = trainer.train_step(clean_inputs)
    assert isinstance(loss, float)
    assert loss > 0
