import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import json
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

class WelfordCovariance:
    def __init__(self, d):
        self.d = d
        self.n = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros((d, d), dtype=np.float64)
    
    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def get_covariance(self):
        if self.n < 2:
            return np.zeros((self.d, self.d))
        return self.M2 / (self.n - 1)

class L29CaptureHook:
    def __init__(self):
        self.handle = None
        self.captured_acts = []

    def _make_capture_hook(self):
        def hook_fn(module, inp):
            x = inp[0]
            # Capture sequence tokens [1:] to skip BOS if desired, 
            # or just take the mean over the sequence length. 
            # We want bulk statistics over the generation sequence or prompt tokens.
            # Let's capture all sequence tokens.
            for i in range(x.shape[1]):
                self.captured_acts.append(x[0, i, :].detach().float().cpu().numpy())
            return None
        return hook_fn

    def attach_capture(self, model, layer=29):
        self.handle = model.blocks[layer].attn.register_forward_pre_hook(self._make_capture_hook())

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

def get_category_prompts(prompts_list, category):
    if category == "Mathematical":
        subset = prompts_list[0:10] + prompts_list[60:90]
    elif category == "Creative":
        subset = prompts_list[10:20] + prompts_list[90:120]
    else:
        subset = []
    return [p["text"] if isinstance(p, dict) else p for p in subset]

def compute_domain_covariance(model, tokenizer, config, prompts, layer=29):
    device = next(model.parameters()).device
    cov = WelfordCovariance(config.n_embd)
    
    hook = L29CaptureHook()
    hook.attach_capture(model, layer)
    
    for prompt in tqdm(prompts, desc="Welford Covariance", leave=False):
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        with torch.no_grad():
            model(input_ids)
            
        # process captured activations
        for act in hook.captured_acts:
            cov.update(act)
            
        hook.captured_acts = [] # Reset for next prompt
        
    hook.remove()
    covariance = cov.get_covariance()
    evals, evecs = np.linalg.eigh(covariance)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return cov, evecs, evals

def compute_principal_angles(subspace1, subspace2):
    """
    Computes principal angles between two subspaces defined by orthonormal bases.
    Returns angles in degrees.
    """
    # scipy.linalg.subspace_angles gives radians
    angles_rad = scipy.linalg.subspace_angles(subspace1, subspace2)
    angles_deg = np.degrees(angles_rad)
    return angles_deg

def main():
    print("Loading model and prompts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    
    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_data = json.load(f)["prompts"]
        
    math_prompts = get_category_prompts(prompts_data, "Mathematical")
    creative_prompts = get_category_prompts(prompts_data, "Creative")
    
    print(f"\nComputing L29 Welford covariance for Math domain ({len(math_prompts)} prompts)...")
    _, evecs_math, evals_math = compute_domain_covariance(model, tokenizer, config, math_prompts, layer=29)
    math_er = np.exp(-np.sum( (evals_math[evals_math>0]/evals_math.sum()) * np.log(evals_math[evals_math>0]/evals_math.sum()) ))
    print(f"Math ER: {math_er:.2f}")

    print(f"\nComputing L29 Welford covariance for Creative domain ({len(creative_prompts)} prompts)...")
    _, evecs_creative, evals_creative = compute_domain_covariance(model, tokenizer, config, creative_prompts, layer=29)
    creative_er = np.exp(-np.sum( (evals_creative[evals_creative>0]/evals_creative.sum()) * np.log(evals_creative[evals_creative>0]/evals_creative.sum()) ))
    print(f"Creative ER: {creative_er:.2f}")

    # Use k=185 as rank based on the previously found asymptotic dimension limit
    k = 185
    print(f"\nExtracting top k={k} eigenvectors from each domain...")
    basis_math = evecs_math[:, :k]       # Shape: [d, k]
    basis_creative = evecs_creative[:, :k] # Shape: [d, k]
    
    print("Computing principal angles between the two rank-185 subspaces...")
    angles = compute_principal_angles(basis_math, basis_creative)
    
    print("\n--- Diagnostic Results ---")
    print(f"Smallest  5 Angles: {angles[:5].round(2)} degrees")
    print(f"Largest   5 Angles: {angles[-5:].round(2)} degrees")
    print(f"Mean Principal Angle: {np.mean(angles):.2f} degrees")
    print(f"Median Principal Angle: {np.median(angles):.2f} degrees")
    
    # Analyze orthogonality
    num_orthogonal = np.sum(angles > 80.0)
    percent_orthogonal = (num_orthogonal / k) * 100
    print(f"Dimensions > 80° apart: {num_orthogonal}/{k} ({percent_orthogonal:.1f}%)")
    
    num_aligned = np.sum(angles < 30.0)
    print(f"Dimensions < 30° apart (shared base): {num_aligned}/{k}")
    
    # Save results
    os.makedirs("measurements", exist_ok=True)
    with open("measurements/principal_angles_L29.txt", "w") as f:
        f.write("L29 Domain Subspace Principal Angles (Math vs Creative)\n")
        f.write("====================================================\n")
        f.write(f"Math ER: {math_er:.2f}\n")
        f.write(f"Creative ER: {creative_er:.2f}\n")
        f.write(f"k = {k}\n\n")
        f.write(f"Mean Angle: {np.mean(angles):.2f} degrees\n")
        f.write(f"Median Angle: {np.median(angles):.2f} degrees\n")
        f.write(f"Dimensions > 80 degrees apart: {num_orthogonal}/{k} ({percent_orthogonal:.1f}%)\n")
        f.write(f"Dimensions < 30 degrees apart: {num_aligned}/{k}\n\n")
        f.write("All Angles (Degrees):\n")
        f.write(np.array2string(angles, precision=2, max_line_width=120))
        
    print("\nSaved comprehensive angle data to measurements/principal_angles_L29.txt")

if __name__ == "__main__":
    main()
