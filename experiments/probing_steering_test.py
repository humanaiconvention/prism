import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.causal.patching import ActivationPatcher
from prism.lens.logit import LogitLens
from prism.probing.linear import ConceptProber, SteeringVectorExtractor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_probing_and_steering():
    model_name = "Qwen/Qwen2.5-0.5B"
    logging.info(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        output_hidden_states=True
    )
    
    # 1. DATA COLLECTION
    # We'll compare states for "Paris" vs "Berlin"
    prompts_a = ["Paris is the capital of", "The city of Paris is in"]
    prompts_b = ["Berlin is the capital of", "The city of Berlin is in"]
    
    patcher = ActivationPatcher(model)
    target_layer_idx = 12
    target_layer = model.model.layers[target_layer_idx]
    
    states_a = []
    states_b = []
    
    logging.info("Collecting activations for France vs Germany...")
    for p in prompts_a:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with patcher.trace():
            patcher.cache_activation(target_layer, "act")
            with torch.no_grad():
                _ = model(**inputs)
            states_a.append(patcher._cache["act"][:, -1, :].float()) # Last token
            
    for p in prompts_b:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with patcher.trace():
            patcher.cache_activation(target_layer, "act")
            with torch.no_grad():
                _ = model(**inputs)
            states_b.append(patcher._cache["act"][:, -1, :].float())
            
    X_a = torch.cat(states_a, dim=0)
    X_b = torch.cat(states_b, dim=0)
    
    # 2. PROBING
    logging.info("Training Linear Probe...")
    X = torch.cat([X_a, X_b], dim=0)
    y = torch.tensor([1, 1, 0, 0]) # 1 for France, 0 for Germany
    
    prober = ConceptProber(hidden_size=model.config.hidden_size, device=str(model.device))
    acc = prober.train_layer_probe(target_layer_idx, X, y)
    logging.info(f"Probe Accuracy: {acc*100:.1f}%")
    
    concept_dir = prober.get_concept_direction(target_layer_idx)
    
    # 3. STEERING
    # Neutral prompt
    neutral_prompt = "The capital of this country is"
    inputs = tokenizer(neutral_prompt, return_tensors="pt").to(model.device)
    
    lens = LogitLens(model)
    
    # Baseline Prediction
    with torch.no_grad():
        out_base = model(**inputs)
    top_base = lens.decode_top_k(out_base.hidden_states[-1], tokenizer, k=3)
    logging.info(f"Baseline Prediction: {top_base[0][0]} ({top_base[0][1]:.3f})")
    
    # Steered Prediction
    # We'll manually inject the 'France' direction into the patcher's cache
    # And then use the 'inject_activation' hook
    patcher._cache["steer_vector"] = concept_dir.unsqueeze(0).unsqueeze(0).to(model.dtype) * 10.0 # Scale up for effect
    
    # We need a custom hook for addition rather than replacement for steering
    def steering_hook(mod, inp, out):
        state = out[0] if isinstance(out, tuple) else out
        # Add steering vector to the last token
        state[:, -1, :] += patcher._cache["steer_vector"][:, 0, :]
        return out

    handle = target_layer.register_forward_hook(steering_hook)
    
    with torch.no_grad():
        out_steered = model(**inputs)
    handle.remove()
    
    top_steered = lens.decode_top_k(out_steered.hidden_states[-1], tokenizer, k=3)
    logging.info(f"Steered Prediction (+France): {top_steered[0][0]} ({top_steered[0][1]:.3f})")

if __name__ == "__main__":
    test_probing_and_steering()
