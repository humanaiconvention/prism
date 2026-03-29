import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.attention.circuits import AttentionAnalyzer
from prism.mlp.memory import MLPAnalyzer
from prism.causal.patching import ActivationPatcher
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_mlp_and_attention_heatmap():
    model_name = "Qwen/Qwen2.5-0.5B"
    logging.info(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    
    # --- 1. Test FFN Key-Value Mapping ---
    logging.info("\n--- Testing FFN Key-Value Mapping ---")
    mlp_analyzer = MLPAnalyzer(model)
    layer_idx = 12
    neuron_idx = 452 # Randomly chosen neuron
    
    # 1a. Value direction to Vocabulary
    logging.info(f"Mapping Neuron {neuron_idx} in Layer {layer_idx} to vocabulary...")
    top_tokens = mlp_analyzer.map_neuron_to_vocabulary(
        layer_idx=layer_idx,
        neuron_idx=neuron_idx,
        lm_head=model.lm_head,
        final_norm=model.model.norm,
        tokenizer=tokenizer,
        k=5
    )
    token_str = ", ".join([f"'{t}' ({p:.4f})" for t, p in top_tokens])
    logging.info(f"Neuron {neuron_idx} promotes: {token_str}")

    # 1b. Finding Activating Tokens (Key Detector)
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    patcher = ActivationPatcher(model)
    
    with patcher.trace():
        # Hook the 'gate_proj' output which serves as the key detector in SwiGLU
        patcher.cache_activation(model.model.layers[layer_idx].mlp.gate_proj, "gate_acts")
        with torch.no_grad():
            _ = model(**inputs)
        gate_activations = patcher._cache["gate_acts"][0] # (seq, intermediate_size)
        
    logging.info(f"Scanning prompt for activations of Neuron {neuron_idx}...")
    activating_tokens = mlp_analyzer.find_activating_tokens(
        layer_idx=layer_idx,
        neuron_idx=neuron_idx,
        token_activations=gate_activations,
        tokenizer=tokenizer,
        tokens=inputs["input_ids"],
        k=3
    )
    act_str = ", ".join([f"'{t}' ({v:.2f})" for t, v in activating_tokens])
    logging.info(f"Top activating tokens for Neuron {neuron_idx}: {act_str}")

    # --- 2. Test Attention Entropy Heatmap ---
    logging.info("\n--- Testing Attention Entropy Heatmap ---")
    attn_analyzer = AttentionAnalyzer(model)
    
    # Capture attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Layer 12 attention weights
    attn_weights = outputs.attentions[layer_idx] # (batch, head, query, key)
    entropy_map = attn_analyzer.compute_attention_entropy_map(attn_weights) # (query, head)
    
    logging.info(f"Attention Entropy Map Shape: {entropy_map.shape}")
    logging.info(f"Mean Attention Entropy across heads at pos 0: {entropy_map[0].mean().item():.4f}")
    logging.info(f"Mean Attention Entropy across heads at pos -1: {entropy_map[-1].mean().item():.4f}")

if __name__ == "__main__":
    test_mlp_and_attention_heatmap()
