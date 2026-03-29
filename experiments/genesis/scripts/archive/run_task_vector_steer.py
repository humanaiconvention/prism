r"""L15-H3 Task-Vector Causal Steering — Phase 5 Experiment 1.

Goal: Extract the orthogonal Math - Creative task vector at L15 pre-mixer,
and add \lambda \delta_\perp to creative prompts to steer them into the
analytic subspace suppressed by H3.

Mathematical Protocol (per Perplexity Sonnet audit):
1. Extract final-token pre-mixer activations (L15).
2. Compute Welford covariance bulk eigenvectors for L15 pre-mixer.
3. delta = \bar{h}_{math} - \bar{h}_{creative}
4. delta_\perp = delta - \sum (\delta \cdot e_i) e_i (project out the top-k bulk)
5. Steer: h' = h + \lambda (\delta_\perp / ||\delta_\perp||)
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import (
    load_genesis_model, format_chatml_prompt,
)

class WelfordCovariance:
    """Online covariance estimator using Welford's algorithm."""
    
    def __init__(self, d):
        self.d = d
        self.n = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros((d, d), dtype=np.float64)
    
    def update(self, x):
        """Add a single d-dimensional sample."""
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def get_covariance(self):
        """Returns the sample covariance matrix (d×d)."""
        if self.n < 2:
            return np.zeros((self.d, self.d))
        return self.M2 / (self.n - 1)

class L15SteeringHook:
    def __init__(self, steering_vector: torch.Tensor = None, lambda_scale: float = 0.0):
        self.steering_vector = steering_vector
        self.lambda_scale = lambda_scale
        self.handle = None
        self.captured_acts = []

    def _make_capture_hook(self):
        def hook_fn(module, inp):
            # Input to the mixer block at L15
            x = inp[0]
            # Capture the last token
            self.captured_acts.append(x[:, -1, :].detach().float().cpu().numpy())
            return None
        return hook_fn

    def _make_steer_hook(self):
        def hook_fn(module, inp):
            if self.steering_vector is None or self.lambda_scale == 0.0:
                return None
            x = inp[0]
            # Add the steering vector only to the last token (or all tokens?)
            # Usually causal steering on the generation token is best done 
            # at every forward pass step during generation on the sequence.
            # Steer the [-1] token:
            x[:, -1, :] = x[:, -1, :] + (self.lambda_scale * self.steering_vector)
            return (x,)
        return hook_fn

    def attach_capture(self, model, layer=15):
        # We hook the input to the attention/FFN mixer block for pre-mixer.
        # Actually hooking the block's `attn` module input gives the pre-mixer state.
        self.handle = model.blocks[layer].attn.register_forward_pre_hook(self._make_capture_hook())

    def attach_steer(self, model, layer=15):
        self.handle = model.blocks[layer].attn.register_forward_pre_hook(self._make_steer_hook())

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

def get_category_prompts(prompts_list, category):
    # Categories: mathematical (0-9, 60-89), creative (10-19, 90-119)
    if category == "Mathematical":
        subset = prompts_list[0:10] + prompts_list[60:90]
    elif category == "Creative":
        subset = prompts_list[10:20] + prompts_list[90:120]
    else:
        subset = []
    return [p["text"] if isinstance(p, dict) else p for p in subset]

def compute_welford_eigenvectors(model, tokenizer, config, prompts, layer=15):
    device = next(model.parameters()).device
    cov = WelfordCovariance(config.n_embd)
    
    hook = L15SteeringHook()
    hook.attach_capture(model, layer)
    
    for prompt in tqdm(prompts, desc="Welford Bulk Covariance"):
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        with torch.no_grad():
            model(input_ids) # Just process the prompt, capture the last token at L15
            
        x = hook.captured_acts[-1][0] # shape (d,)
        cov.update(x)
        
    hook.remove()
    mean = cov.mean
    covariance = cov.get_covariance()
    
    # Compute Top-k eigenvectors
    evals, evecs = np.linalg.eigh(covariance)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    return mean, evecs, evals

def extract_mean_activation(model, tokenizer, prompts, layer=15):
    device = next(model.parameters()).device
    hook = L15SteeringHook()
    hook.attach_capture(model, layer)
    
    for prompt in tqdm(prompts, desc="Extracting Means"):
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        with torch.no_grad():
            model(input_ids)
            
    hook.remove()
    acts = np.stack(hook.captured_acts).squeeze() # [N, d]
    return np.mean(acts, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--lambda-scale", type=float, default=5.0)
    args = parser.parse_args()
    
    import json
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)["prompts"]
        
    math_prompts = get_category_prompts(prompts_data, "Mathematical")
    creative_prompts = get_category_prompts(prompts_data, "Creative")
    
    # We need all prompts to build the background covariance
    all_prompts = [p["text"] if isinstance(p, dict) else p for p in prompts_data]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    
    print("\n1. Computing pre-mixer background covariance (L15)...")
    _, evecs, evals = compute_welford_eigenvectors(model, tokenizer, config, all_prompts, layer=15)
    
    # We want to project out the "bulk". Based on ER~185-207, let's project out the top 70.
    k_bulk = 70
    E_bulk = evecs[:, :k_bulk] # shape (d, k)
    
    print("\n2. Extracting category task vectors...")
    mean_math = extract_mean_activation(model, tokenizer, math_prompts, layer=15)
    mean_creative = extract_mean_activation(model, tokenizer, creative_prompts, layer=15)
    
    delta = mean_math - mean_creative
    print(f"Raw delta norm: {np.linalg.norm(delta):.3f}")
    
    # 3. Orthogonal projection
    # delta_\perp = delta - \sum (\delta \cdot e_i) e_i
    proj_components = E_bulk @ (E_bulk.T @ delta)
    delta_perp = delta - proj_components
    
    delta_perp_norm = np.linalg.norm(delta_perp)
    print(f"Orthogonal delta norm: {delta_perp_norm:.3f} (Projected out {k_bulk} bulk dims)")
    
    # Normalize to unit variance unit
    delta_steering = torch.tensor(delta_perp / delta_perp_norm, device=device, dtype=torch.float32)
    
    print(f"\n4. Preparing causal steering (Lambda = {args.lambda_scale})...")
    
    test_prompt = "Write a highly creative poem about the ocean."
    print(f"Test Prompt: {test_prompt}")
    
    def generate_with_steering(l_scale):
        steer_hook = L15SteeringHook(steering_vector=delta_steering, lambda_scale=l_scale)
        steer_hook.attach_steer(model, layer=15)
        
        chat_input = format_chatml_prompt(test_prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=32, temperature=0.7)
            
        steer_hook.remove()
        
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:].tolist())
        return response
        
    print("\n--- BASELINE (lambda = 0.0) ---")
    print(generate_with_steering(0.0))
    
    print(f"\n--- STEERED (lambda = {args.lambda_scale}) ---")
    print(generate_with_steering(args.lambda_scale))
    print(f"\n--- OVER-STEERED (lambda = {args.lambda_scale * 2.5}) ---")
    print(generate_with_steering(args.lambda_scale * 2.5))

if __name__ == "__main__":
    main()
