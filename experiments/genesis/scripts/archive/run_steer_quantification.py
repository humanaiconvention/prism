import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
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

class L15SteeringHook:
    def __init__(self, steering_vector: torch.Tensor = None, lambda_scale: float = 0.0):
        self.steering_vector = steering_vector
        self.lambda_scale = lambda_scale
        self.handle = None
        self.captured_acts = []

    def _make_capture_hook(self):
        def hook_fn(module, inp):
            x = inp[0]
            self.captured_acts.append(x[:, -1, :].detach().float().cpu().numpy())
            return None
        return hook_fn

    def _make_steer_hook(self):
        def hook_fn(module, inp):
            if self.steering_vector is None or self.lambda_scale == 0.0:
                return None
            x = inp[0]
            x[:, -1, :] = x[:, -1, :] + (self.lambda_scale * self.steering_vector)
            return (x,)
        return hook_fn

    def attach_capture(self, model, layer=15):
        self.handle = model.blocks[layer].attn.register_forward_pre_hook(self._make_capture_hook())

    def attach_steer(self, model, layer=15):
        self.handle = model.blocks[layer].attn.register_forward_pre_hook(self._make_steer_hook())

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

# We also need a capture hook for L29 to get the output representations
class L29CaptureHook:
    def __init__(self):
        self.handle = None
        self.captured_acts = []

    def _make_capture_hook(self):
        def hook_fn(module, inp):
            x = inp[0] # pre-mixer
            self.captured_acts.append(x[:, -1, :].detach().float().cpu().numpy())
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

def extract_mean_activation(model, tokenizer, prompts, layer, hook_class):
    device = next(model.parameters()).device
    hook = hook_class()
    hook.attach_capture(model, layer)
    
    for prompt in tqdm(prompts, desc=f"Extracting Means (L{layer})", leave=False):
        chat_input = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
        with torch.no_grad():
            model(input_ids)
            
    hook.remove()
    acts = np.stack(hook.captured_acts).squeeze()
    return np.mean(acts, axis=0)

def compute_welford_eigenvectors(model, tokenizer, config, prompts, layer=15):
    device = next(model.parameters()).device
    cov = WelfordCovariance(config.n_embd)
    
    hook = L15SteeringHook()
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

def get_surface_metrics(text, tokenizer):
    tokens = tokenizer.encode(text)
    ttr = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
    
    # Sentence length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    mean_sentence_len = np.mean([len(s.split()) for s in sentences]) if len(sentences) > 0 else len(text.split())
    
    # Enumeration density
    enum_matches = re.findall(r'(\d+\.|-|\*|•)', text)
    enum_density = len(enum_matches) / len(tokens) if len(tokens) > 0 else 0
    
    return {
        "ttr": ttr,
        "mean_sentence_len": mean_sentence_len,
        "enum_density": enum_density
    }

def calculate_er(evals):
    evals = evals[evals > 0]
    p = evals / evals.sum()
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def main():
    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_data = json.load(f)["prompts"]
        
    math_prompts = get_category_prompts(prompts_data, "Mathematical")
    creative_prompts = get_category_prompts(prompts_data, "Creative")
    all_prompts = [p["text"] if isinstance(p, dict) else p for p in prompts_data]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    
    print("1. Extracting L29 Base Centroids (Math vs Creative)...")
    mean_math_L29 = extract_mean_activation(model, tokenizer, math_prompts, layer=29, hook_class=L29CaptureHook)
    # Target centroid for cosine similarity
    math_centroid_L29 = mean_math_L29 / np.linalg.norm(mean_math_L29)
    
    print("2. Extracting L15 Task Vector...")
    _, evecs_L15, _ = compute_welford_eigenvectors(model, tokenizer, config, all_prompts, layer=15)
    mean_math_L15 = extract_mean_activation(model, tokenizer, math_prompts, layer=15, hook_class=L15SteeringHook)
    mean_creative_L15 = extract_mean_activation(model, tokenizer, creative_prompts, layer=15, hook_class=L15SteeringHook)
    
    delta = mean_math_L15 - mean_creative_L15
    k_bulk = 70
    E_bulk = evecs_L15[:, :k_bulk]
    
    # Orthogonal projection
    proj_components = E_bulk @ (E_bulk.T @ delta)
    delta_perp = delta - proj_components
    delta_perp_norm = np.linalg.norm(delta_perp)
    delta_steering = torch.tensor(delta_perp / delta_perp_norm, device=device, dtype=torch.float32)
    
    test_prompts = [
        "Write a highly creative poem about the ocean.",
        "A magical journey into the dark forest reveals",
        "Compose a romantic verse about the moonlit sky:"
    ]
    
    lambdas = [0.0, 5.0, 12.5]
    results = []
    
    print("\n3. Executing Lambda-Sweep Generation...")
    
    for l_scale in lambdas:
        print(f"\nEvaluating Lambda = {l_scale}")
        
        steer_hook = L15SteeringHook(steering_vector=delta_steering, lambda_scale=l_scale)
        capture_hook = L29CaptureHook()
        
        steer_hook.attach_steer(model, layer=15)
        capture_hook.attach_capture(model, layer=29)
        
        all_entropies = []
        all_centroids = []
        all_metrics = []
        
        cov = WelfordCovariance(config.n_embd)
        
        for prompt in test_prompts:
            chat_input = format_chatml_prompt(prompt)
            input_ids = torch.tensor([tokenizer.encode(chat_input)], device=device)
            
            gen_steps = 48
            past_key_values = None
            curr_ids = input_ids
            
            entropies = []
            
            with torch.no_grad():
                for step in tqdm(range(gen_steps), desc=f"Gen L={l_scale}", leave=False):
                    if past_key_values is not None:
                        idx_cond = curr_ids[:, -1:]
                    else:
                        idx_cond = curr_ids
                        
                    # We need to clear the captured acts so we only get the current token
                    capture_hook.captured_acts = []
                        
                    logits, loss, metrics, past_key_values = model(
                        idx_cond,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    # Logit Entropy
                    probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    entropies.append(entropy)
                    
                    # Update Covariance & Centroid with the latest L29 activation
                    act_L29 = capture_hook.captured_acts[-1][0]
                    cov.update(act_L29)
                    all_centroids.append(act_L29)
                    
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Surface metrics
            generated_text = tokenizer.decode(curr_ids[0][input_ids.shape[1]:].tolist())
            surf_metrics = get_surface_metrics(generated_text, tokenizer)
            all_metrics.append(surf_metrics)
            all_entropies.append(np.mean(entropies))
            
        steer_hook.remove()
        capture_hook.remove()
        
        # Calculate aggregate metrics for this lambda
        evals, _ = np.linalg.eigh(cov.get_covariance())
        output_er = calculate_er(evals)
        
        mean_act = np.mean(all_centroids, axis=0)
        cosine_sim = np.dot(mean_act, math_centroid_L29) / (np.linalg.norm(mean_act) + 1e-10)
        
        mean_entropy = np.mean(all_entropies)
        mean_ttr = np.mean([m["ttr"] for m in all_metrics])
        mean_len = np.mean([m["mean_sentence_len"] for m in all_metrics])
        mean_enum = np.mean([m["enum_density"] for m in all_metrics])
        
        results.append({
            "Lambda": l_scale,
            "L29_ER": output_er,
            "Cosine_Math": cosine_sim,
            "Logit_Entropy": mean_entropy,
            "TTR": mean_ttr,
            "Sentence_Len": mean_len,
            "Enum_Density": mean_enum
        })
        
    df = pd.DataFrame(results)
    
    print("\n--- LAMBDA SWEEP RESULTS ---")
    print(df.to_string(index=False))
    
    os.makedirs("measurements", exist_ok=True)
    df.to_csv("measurements/lambda_sweep.csv", index=False)
    print("Saved to measurements/lambda_sweep.csv")

if __name__ == "__main__":
    main()
