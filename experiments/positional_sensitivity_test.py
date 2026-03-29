import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.arch.hybrid import HybridDiagnostics
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_positional_sensitivity():
    model_name = "Qwen/Qwen2.5-0.5B"
    logging.info(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
    
    diag = HybridDiagnostics(model)
    
    # Prompt with clear spatial structure
    prompt = "The first word is Apple. The second word is Banana. The third word is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    logging.info("\n--- Measuring Positional Sensitivity (RoPE) ---")
    
    # Measure sensitivity in early, middle, and late layers
    num_layers = model.config.num_hidden_layers
    for l_idx in [2, num_layers//2, num_layers-2]:
        results = diag.measure_positional_sensitivity(inputs, l_idx)
        logging.info(f"Layer {l_idx:02d}: Drift {results['positional_drift']:.4f}, Rank Collapse {results['rank_collapse_ratio']:.4f}")
        if results['is_spatially_rigid']:
            logging.info(f"  [!] Layer {l_idx} is Spatially Rigid (Highly dependent on RoPE)")

if __name__ == "__main__":
    test_positional_sensitivity()
