import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.causal.patching import ActivationPatcher
from prism.sae.trainer import SAETrainer
from prism.sae.features import FeatureAnalyzer
from prism.lens.logit import LogitLens # to get norm/lm_head handles
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_sae_pipeline():
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
    
    # 1. CAPTURE DATA
    patcher = ActivationPatcher(model)
    target_layer_idx = 8
    target_layer = model.model.layers[target_layer_idx]
    
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "A black hole is a region of spacetime where gravity is so strong.",
        "To be, or not to be, that is the question.",
        "Python is a high-level, interpreted, general-purpose programming language."
    ]
    
    all_activations = []
    logging.info("Capturing activations from Layer 8...")
    
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with patcher.trace():
            patcher.cache_activation(target_layer, "act")
            with torch.no_grad():
                _ = model(**inputs)
            all_activations.append(patcher._cache["act"].float())
            
    # Flatten activations: (batch * seq, hidden_size)
    activations_tensor = torch.cat(all_activations, dim=1).view(-1, model.config.hidden_size)
    logging.info(f"Captured activation buffer: {activations_tensor.shape}")

    # 2. TRAIN SAE
    # Dict size 4x hidden dim (3584), TopK=32
    hidden_size = model.config.hidden_size
    trainer = SAETrainer(hidden_size=hidden_size, dict_size=hidden_size*4, k=32)
    
    logging.info("Training TopK SAE for 50 steps...")
    for i in range(50):
        # In a real run we'd shuffle and batch, but for smoke test we'll just use the full buffer
        metrics = trainer.train_step(activations_tensor)
        if i % 10 == 0:
            logging.info(f"Step {i:02d}: Loss {metrics['loss']:.4f}, L0 {metrics['l0']:.1f}")

    # 3. ANALYZE FEATURES
    logging.info("\n--- Analyzing Learned Features ---")
    analyzer = FeatureAnalyzer(trainer.sae)
    lens = LogitLens(model) # Use lens to get model head/norm handles
    
    # Check the top tokens for the first 3 features
    for f_idx in range(3):
        top_tokens = analyzer.get_top_tokens_for_feature(
            feature_idx=f_idx,
            lm_head=lens.lm_head,
            final_norm=lens.final_norm,
            tokenizer=tokenizer,
            k=5
        )
        token_str = ", ".join([f"'{t}' ({p:.3f})" for t, p in top_tokens])
        logging.info(f"Feature {f_idx:02d} promotes: {token_str}")

if __name__ == "__main__":
    test_sae_pipeline()
