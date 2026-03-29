import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import json
from pathlib import Path
import numpy as np
import scipy.spatial.distance as distance
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

def calculate_shannon_er(distribution):
    p = distribution[distribution > 0]
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

class L15SteeringHook:
    def __init__(self, steer_vector):
        self.handle = None
        self.steer_vector = torch.tensor(steer_vector, dtype=torch.float32)
        self.lambda_scale = 0.0

    def _make_capture_hook(self):
        def hook_fn(module, inp):
            x = inp[0]
            self.captured_acts.append(x[:, -1, :].detach().float().cpu().numpy())
            return None
        return hook_fn

    def _make_steer_hook(self):
        def hook_fn(module, inp):
            if self.steer_vector is None or self.lambda_scale == 0.0:
                return None
            x = inp[0]
            x[:, -1, :] = x[:, -1, :] + (self.lambda_scale * self.steer_vector.to(x.device))
            return (x,)
        return hook_fn

    def attach_capture(self, model, layer=15):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_capture_hook())

    def attach_steer(self, model, layer=15):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_steer_hook())

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

class L29CaptureHook:
    def __init__(self):
        self.handle = None
        self.captured_acts = []

    def _make_capture_hook(self):
        def hook_fn(module, inp):
            x = inp[0]
            # Capture generated tokens
            for i in range(x.shape[1]):
                self.captured_acts.append(x[:, i, :].detach().cpu().numpy())
            return None
        return hook_fn

    def attach_capture(self, model, layer=29):
        self.handle = model.blocks[layer].register_forward_pre_hook(self._make_capture_hook())

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

def extract_mean_activation(model, tokenizer, prompts, layer, hook_class):
    device = next(model.parameters()).device
    hook = hook_class()
    hook.captured_acts = []
    hook.attach_capture(model, layer)
    
    for prompt in tqdm(prompts, desc=f"Extracting Means (L{layer})", leave=False):
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        with torch.no_grad():
            model(input_ids)
            
    hook.remove()
    acts = np.stack(hook.captured_acts).squeeze()
    return np.mean(acts, axis=0)

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

def compute_welford_eigenvectors(model, tokenizer, config, prompts, layer=15):
    device = next(model.parameters()).device
    cov = WelfordCovariance(config.n_embd)
    
    hook = L15SteeringHook(np.zeros(config.n_embd, dtype=np.float32))
    hook.captured_acts = []
    hook.attach_capture(model, layer)
    
    for prompt in tqdm(prompts, desc=f"Welford Covariance (L{layer})", leave=False):
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        with torch.no_grad():
            model(input_ids)
            
        x = hook.captured_acts[-1][0]
        cov.update(x)
        
    hook.remove()
    covariance = cov.get_covariance()
    evals, evecs = np.linalg.eigh(covariance)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return cov.mean, evecs, evals

def get_surface_metrics(text):
    words = text.split()
    if not words: return 0, 0
    ttr = len(set(words)) / len(words)
    sentences = [s for s in text.replace('?','.').replace('!','.').split('.') if len(s.strip()) > 0]
    mean_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else len(words)
    return mean_sent_len, ttr


def main():
    print("Loading model for Lambda Sweep Robustness CI...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)

    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_list = json.load(f)["prompts"]
        
    math_prompts = get_category_prompts(prompts_list, "Mathematical")
    creative_prompts = get_category_prompts(prompts_list, "Creative")
    all_prompts = [p["text"] if isinstance(p, dict) else p for p in prompts_list]
    
    print("1. Extracting L15 Base Centroids...")
    mean_math_L15_for_cos = extract_mean_activation(
        model,
        tokenizer,
        math_prompts,
        layer=15,
        hook_class=lambda: L15SteeringHook(np.zeros(config.n_embd, dtype=np.float32)),
    )
    math_centroid_L15 = mean_math_L15_for_cos / np.linalg.norm(mean_math_L15_for_cos)
    
    print("2. Computing L15 Orthogonal Steer Vector...")
    _, evecs_L15, _ = compute_welford_eigenvectors(model, tokenizer, config, all_prompts, layer=15)
    mean_math_L15 = extract_mean_activation(
        model,
        tokenizer,
        math_prompts,
        layer=15,
        hook_class=lambda: L15SteeringHook(np.zeros(config.n_embd, dtype=np.float32)),
    )
    mean_creative_L15 = extract_mean_activation(
        model,
        tokenizer,
        creative_prompts,
        layer=15,
        hook_class=lambda: L15SteeringHook(np.zeros(config.n_embd, dtype=np.float32)),
    )
    
    delta = mean_math_L15 - mean_creative_L15
    k_bulk = 70
    E_bulk = evecs_L15[:, :k_bulk]
    proj_components = E_bulk @ (E_bulk.T @ delta)
    delta_perp = delta - proj_components
    delta_steering = delta_perp / np.linalg.norm(delta_perp)
    
    # Use 15 creative prompts for testing intervention stability
    test_creative = [p["text"] if isinstance(p, dict) else p for p in prompts_list[10:25]]
    
    lambdas_to_test = [0.0, 5.0, 12.5]
    
    results = {lam: {'cos': [], 'sent_len': [], 'ttr': []} for lam in lambdas_to_test}

    steer_hook = L15SteeringHook(delta_steering)
    steer_hook.attach_steer(model, layer=15)
    
    capture_hook = L15SteeringHook(np.zeros(config.n_embd, dtype=np.float32))
    capture_hook.captured_acts = []
    capture_hook.attach_capture(model, layer=15)

    for prompt in tqdm(test_creative, desc="Creative Prompts CI"):
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        
        for lam in lambdas_to_test:
            steer_hook.lambda_scale = lam
            capture_hook.captured_acts = [] # Reset
            
            with torch.no_grad():
                output_ids = model.generate(input_ids, max_new_tokens=48, temperature=0.7)
                
            gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:].tolist())
            
            # Tier 3
            sent_len, ttr = get_surface_metrics(gen_text)
            
            # Tier 1
            acts_matrix = np.array([x[0] for x in capture_hook.captured_acts])
            prompt_centroid = np.mean(acts_matrix, axis=0) # [d_model]
            prompt_centroid = np.asarray(prompt_centroid).reshape(-1)
            math_centroid_vec = np.asarray(math_centroid_L15).reshape(-1)
            cos_dist = distance.cosine(prompt_centroid, math_centroid_vec)
            cos_sim = 1.0 - cos_dist
                
            results[lam]['cos'].append(cos_sim)
            results[lam]['sent_len'].append(sent_len)
            results[lam]['ttr'].append(ttr)

    steer_hook.remove()
    capture_hook.remove()

    print("\n--- Lambda Sweep Confidence Intervals (N=15 Prompts) ---")
    print("Lambda | Cosine to Math L15 | Mean Sent Len | TTR")
    print("-" * 65)
    
    out_lines = ["Lambda Sweep Robustness CI (N=15)"]
    out_lines.append("Lambda | Cosine to Math L15 | Mean Sent Len | TTR")
    
    for lam in lambdas_to_test:
        c_mean, c_std = np.mean(results[lam]['cos']), np.std(results[lam]['cos'])
        s_mean, s_std = np.mean(results[lam]['sent_len']), np.std(results[lam]['sent_len'])
        t_mean, t_std = np.mean(results[lam]['ttr']), np.std(results[lam]['ttr'])
        
        line = f"{lam:6.1f} | {c_mean:.3f} ± {c_std:.3f}   | {s_mean:5.1f} ± {s_std:4.1f}  | {t_mean:.3f} ± {t_std:.3f}"
        print(line)
        out_lines.append(line)
        
    with open("measurements/lambda_sweep_ci.txt", "w") as f:
        f.write("\n".join(out_lines))
        
if __name__ == "__main__":
    main()
