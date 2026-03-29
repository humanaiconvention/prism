import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.lens.logit import LogitLens
from prism.causal.patching import ActivationPatcher
from prism.attention.circuits import AttentionAnalyzer
from prism.mlp.memory import MLPAnalyzer
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_qwen_baseline():
    model_name = "Qwen/Qwen2.5-0.5B" # Small, fast standard transformer
    logging.info(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        output_hidden_states=True
    )
    
    # --- 1. Static Weight-Space Analysis ---
    logging.info("\n--- Testing Attention Analyzer (Static Weights) ---")
    analyzer = AttentionAnalyzer(model)
    
    mid_layer_idx = (model.config.num_hidden_layers // 2)
    logging.info(f"Generating circuit report for Layer {mid_layer_idx}...")
    
    report = analyzer.generate_circuit_report(mid_layer_idx)
    df_report = pd.DataFrame(report)
    
    # Summarize the layer's geometric personality
    mean_rank = df_report['ov_rank'].mean()
    concordant_heads = df_report[df_report['type'] == 'concordant'].shape[0]
    
    logging.info(f"Layer {mid_layer_idx} Mean Head Rank: {mean_rank:.2f}")
    logging.info(f"Concordant vs Discordant: {concordant_heads} / {analyzer.n_heads}")

    # --- 2. Dynamic Rank Restoration Profile ---
    logging.info("\n--- Testing MLP Analyzer (Dynamic Capture) ---")
    mlp_analyzer = MLPAnalyzer(model)
    patcher = ActivationPatcher(model)
    
    # Target Layer components
    target_layer = model.model.layers[mid_layer_idx]
    
    prompt = "The capital city of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with patcher.trace():
        # Hook pre-attn (input to block), post-attn, and post-mlp (block output)
        # Note: in standard Llama/Qwen blocks, post-attn is the input to the MLP
        patcher.cache_activation(target_layer, "pre_attn") # Module output hook captures block output
        
        # We need to hook the sub-modules directly for internal states
        patcher.cache_activation(target_layer.self_attn, "post_attn")
        patcher.cache_activation(target_layer.mlp, "post_mlp")
        
        with torch.no_grad():
            _ = model(**inputs)
            
        # For 'pre_attn', we actually need the input to the block. 
        # But we can also just use the output of the PREVIOUS layer as a proxy.
        # Let's just grab the hidden_states from the outputs instead for simplicity.
        pass

    # Re-run with hidden states from outputs
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Qwen hidden_states includes the state *after* each full block
    # hidden_states[mid_layer_idx] is input to L12
    # hidden_states[mid_layer_idx+1] is output of L12 (post-FFN)
    
    # We still need the internal post-attention state. Let's use hooks again for just that.
    with patcher.trace():
        patcher.cache_activation(target_layer.self_attn, "post_attn")
        with torch.no_grad():
            _ = model(**inputs)
        post_attn_state = patcher._cache["post_attn"]

    pre_block_state = outputs.hidden_states[mid_layer_idx]
    post_block_state = outputs.hidden_states[mid_layer_idx + 1]
    
    profile = mlp_analyzer.rank_restoration_profile(
        pre_attn_state=pre_block_state,
        post_attn_state=post_attn_state,
        post_mlp_state=post_block_state
    )
    
    logging.info(f"Rank Restoration Profile (Layer {mid_layer_idx}):")
    logging.info(f"  Input Rank: {profile['input_rank']:.2f}")
    logging.info(f"  Post-Attention Rank: {profile['post_attn_rank']:.2f} (Delta: {profile['attn_impact']:.2f})")
    logging.info(f"  Post-MLP Rank: {profile['post_mlp_rank']:.2f} (Delta: {profile['mlp_impact']:.2f})")
    logging.info(f"  Net Rank Change: {profile['net_impact']:.2f}")

    # --- 3. Iterative Inference Analysis ---
    logging.info("\n--- Testing Logit Lens ---")
    lens = LogitLens(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    logging.info(f"Running prompt: '{prompt}'")
    
    # 1. Base Forward Pass (to extract hidden states)
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden_states = outputs.hidden_states # Tuple of (embeddings, layer_1, ..., layer_n)
    
    # 2. Test Logit Lens
    logging.info("\n--- Testing Logit Lens ---")
    lens = LogitLens(model)
    
    # Let's see what the model "thinks" the next word is at different depths
    # We'll sample 5 equidistant layers
    num_layers = len(hidden_states) - 1
    sample_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers]
    
    for l_idx in sample_layers:
        top_k = lens.decode_top_k(hidden_states[l_idx], tokenizer, k=3, position_idx=-1)
        predictions = ", ".join([f"'{token}' ({prob:.2f})" for token, prob in top_k])
        logging.info(f"Layer {l_idx:02d} Top Predictions: {predictions}")
        
    entropies = lens.get_prediction_entropy_trajectory(hidden_states)
    logging.info(f"Layer 0 Prediction Entropy: {entropies[0]:.2f} (Uncertain)")
    logging.info(f"Layer {num_layers} Prediction Entropy: {entropies[-1]:.2f} (Certain)")
    
    # 3. Test Activation Patcher (Caching feature)
    logging.info("\n--- Testing Activation Patcher ---")
    patcher = ActivationPatcher(model)
    
    target_layer = model.model.layers[num_layers // 2]
    
    with patcher.trace():
        # Hook it to cache the output of the middle layer
        patcher.cache_activation(target_layer, "mid_layer_cache")
        
        logging.info("Running forward pass with trace active...")
        with torch.no_grad():
            _ = model(**inputs)
            
        cached_tensor = patcher._cache.get("mid_layer_cache")
        if cached_tensor is not None:
            logging.info(f"Successfully cached activation. Shape: {cached_tensor.shape}")
        else:
            logging.error("Failed to cache activation.")

if __name__ == "__main__":
    test_qwen_baseline()
