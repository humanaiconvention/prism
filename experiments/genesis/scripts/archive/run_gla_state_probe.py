import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import svd

# Ensure TRITON_INTERPRET is set for Windows compatibility
os.environ.setdefault("TRITON_INTERPRET", "1")

# Add the parent directory to the path so we can import scripts.genesis_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

def get_subspace_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces defined by orthonormal columns.
    Returns angles in degrees.
    """
    # Compute SVD of the cross-covariance matrix U1^T U2
    # The singular values are the cosines of the principal angles
    M = U1.T @ U2
    _, S, _ = svd(M)
    
    # Clip to [-1.0, 1.0] to avoid numerical issues with arccos
    S = np.clip(S, -1.0, 1.0)
    
    # Angles in radians, then convert to degrees
    angles = np.arccos(S) * (180.0 / np.pi)
    return angles

import argparse

def run_diagnostic():
    parser = argparse.ArgumentParser(description="Probe GLA Recurrent State")
    parser.add_argument("--layer", type=int, default=14, help="Target GLA layer to probe (e.g., 5, 14, 22)")
    args = parser.parse_args()

    print("Loading Genesis-152M model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device, dtype=torch.float32)
    
    # We will probe the layer specified by the user
    probe_layer = args.layer
    print(f"Probing GLA Recurrent State at Layer {probe_layer}...")

    # A prompt designed to be slightly out-of-distribution to test robustness,
    # or just a standard creative prompt that requires generation.
    prompt = "Write a highly creative story about a clockwork bird that learns to sing."
    formatted = format_chatml_prompt(prompt)
    input_ids = torch.tensor([tokenizer.encode(formatted)], device=device)
    prompt_len = input_ids.shape[1]
    
    gen_steps = 64
    r = 16 # Subspace tracking rank
    tau = 10 # Tracking interval for Theta(t, tau)
    
    print(f"Generating {gen_steps} tokens, tracking top {r} singular vectors of S_t.")
    
    # Initialize structures for state tracking
    S_t_history = []
    
    # Custom generation loop to extract state at each step
    past_key_values = None
    curr_ids = input_ids
    
    with torch.no_grad():
        for step in tqdm(range(gen_steps), desc="Generating & Extracting S_t"):
            if past_key_values is not None:
                # Only feed the last token
                idx_cond = curr_ids[:, -1:]
            else:
                idx_cond = curr_ids
                
            logits, loss, metrics, past_key_values = model(
                idx_cond,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Extract state for the target layer
            # state shape: [B, n_head, head_qk_dim, head_v_dim]
            # We'll take Head 0 for simplicity, or we can average/track a specific head
            # Let's track Head 0.
            states = model.get_segment_states()
            if probe_layer in states:
                state_L14 = states[probe_layer][0, 0].cpu().numpy() # [64, 64]
                S_t_history.append(state_L14.copy())
            else:
                print(f"Error: Layer {probe_layer} state not found! Layer type check needed.")
                return

            # Sample next token greedily
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
    # Now analyze the temporal sequence of S_t
    print("\nAnalyzing Subspace Dynamics...")
    U_history = []
    for t in range(gen_steps):
        S = S_t_history[t]
        U, _, _ = svd(S, full_matrices=False)
        U_history.append(U[:, :r]) # Top r left singular vectors
        
    results = []
    for t in range(gen_steps):
        # 1. Subspace tracking angle Theta(t, tau)
        if t + tau < gen_steps:
            angles = get_subspace_angles(U_history[t], U_history[t+tau])
            theta_mean = np.mean(angles)
        else:
            theta_mean = None
            
        # 2. Cumulative angle divergence Phi(t)
        phi_angles = get_subspace_angles(U_history[0], U_history[t])
        phi_mean = np.mean(phi_angles)
        
        # 3. Innovation ratio
        S = S_t_history[t]
        if t > 0:
            S_prev = S_t_history[t-1]
            diff = np.linalg.norm(S - S_prev, 'fro')
            norm_val = np.linalg.norm(S, 'fro')
            innovation = diff / norm_val if norm_val > 0 else 0
        else:
            innovation = 0.0
            
        results.append({
            "t": t,
            "Theta_tau10": theta_mean,
            "Phi": phi_mean,
            "InnovationRatio": innovation
        })
        
    df = pd.DataFrame(results)
    
    print("\n--- Diagnostic Results ---")
    
    # Determine the status based on NotebookLM/Perplexity thresholds
    # Saturated Limit-Cycle: Theta(t, 10) < 10 degrees for t > 30
    # Dynamically Orthogonal: Theta(t, 10) > 30 degrees persistently
    if gen_steps > 30 + tau:
        late_thetas = df.loc[(df['t'] > 30) & df['Theta_tau10'].notna(), 'Theta_tau10']
        if len(late_thetas) > 0:
            mean_late_theta = late_thetas.mean()
            print(f"Mean Subspace Tracking Angle (t>30, tau=10): {mean_late_theta:.2f}°")
            
            if mean_late_theta < 10.0:
                print("CONCLUSION: Saturated Limit-Cycle Attractor DETECTED.")
                print("The GLA memory basis locks in and stops rotating; pathogenic variance can cause Generative Perseveration.")
            elif mean_late_theta > 30.0:
                print("CONCLUSION: Dynamically Orthogonal Memory Basis DETECTED.")
                print("The GLA state maintains robust rotation, escaping the Temporal Paradox.")
            else:
                print("CONCLUSION: Intermediate rotation state.")
    else:
        print("Sequence too short for definitive limit cycle test.")
        
    out_path = f"measurements/gla_state_probe_L{probe_layer}.csv"
    os.makedirs("measurements", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved temporal metrics to {out_path}")

if __name__ == "__main__":
    run_diagnostic()
