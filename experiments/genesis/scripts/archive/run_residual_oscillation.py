"""Experiment 5: Residual Stream Oscillation
Tests for alternating representational phases in the residual stream across layers.

Validates using 5 independent signals:
1. Cosine similarity between successive layers
2. FFT Frequency spectrum of residual change (deltas)
3. PCA representation trajectory
4. Sublayer norm contributions (Attention vs FFN)
5. Eigenvalue spectrum of the effective linear layer-to-layer operator.
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")

import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model, format_chatml_prompt, HiddenStateHook

class SublayerNormHook:
    def __init__(self):
        self.handles = []
        self.data = [] # List of tuples (layer, sublayer_type, pos, norm)
        
    def _make_hook(self, layer, sublayer_type, pos):
        """pos is 'pre' or 'post'"""
        def hook_fn(module, inp, output=None):
            x = output[0] if output is not None and isinstance(output, tuple) else (output if output is not None else inp[0])
            norm = torch.norm(x[:, -1, :].detach().float(), p=2, dim=-1).item()
            self.data.append({"layer": layer, "sublayer": sublayer_type, "pos": pos, "norm": norm})
            return None
        return hook_fn

    def attach(self, model):
        for i, block in enumerate(model.blocks):
            # Pre/Post Attention
            if hasattr(block, 'attn_norm'):
                self.handles.append(block.attn_norm.register_forward_pre_hook(self._make_hook(i, 'attn', 'pre')))
            self.handles.append(block.attn.register_forward_hook(self._make_hook(i, 'attn', 'post')))
            
            # Pre/Post FFN
            if hasattr(block, 'ffn_norm'):
                self.handles.append(block.ffn_norm.register_forward_pre_hook(self._make_hook(i, 'ffn', 'pre')))
            self.handles.append(block.ffn.register_forward_hook(self._make_hook(i, 'ffn', 'post')))
            
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        
    def clear(self):
        self.data = []

def main():
    print("Loading Genesis-152M for Residual Oscillation Test (5-Part Analysis)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)

    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_list = json.load(f)["prompts"]
        
    # Use 30 diverse prompts for general similarity & operator stats
    test_prompts = [p["text"] if isinstance(p, dict) else p for p in prompts_list[:15] + prompts_list[100:115]]
    
    # Trace specific prompt for visual single-trajectory analysis
    trace_prompt = "Intelligence emerges from structure."
    
    hooks = HiddenStateHook.attach_to_model(model)
    
    print("\n1. Collecting residual stream states & learning linear operator...")
    X_in_list = []
    Y_out_list = []
    
    all_sims = {l: [] for l in range(config.n_layer - 1)}
    trace_residuals = None
    
    for prompt in tqdm(test_prompts + [trace_prompt], desc="Prompts"):
        inp = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(inp)], device=device)
        
        hooks.clear()
        with torch.no_grad():
            model(input_ids)
            
        states = hooks.get_hidden_states() # List of length n_layer tensors
        seq_len = states[0].shape[1]
        
        # We only look at the final generated token conceptually, or the whole sequence?
        # Let's take the last token's representation across layers
        residuals = [s.numpy()[0] for s in states] # list of [d_model] arrays
        
        if prompt == trace_prompt:
            trace_residuals = residuals
            
        # Collect for Linear Operator & Similarities
        for l in range(config.n_layer - 1):
            s1 = residuals[l]
            s2 = residuals[l+1]
            if prompt != trace_prompt:
                all_sims[l].append(1.0 - cosine(s1, s2))
                X_in_list.append(s1)
                Y_out_list.append(s2)
            
    hooks.remove_all()
    
    mean_sims = [np.mean(all_sims[l]) for l in range(config.n_layer - 1)]
    
    print("\n2. Computing Linear Operator (h_l+1 = A h_l)...")
    X_mat = np.array(X_in_list)
    Y_mat = np.array(Y_out_list)
    A_T, residuals, rank, s = np.linalg.lstsq(X_mat, Y_mat, rcond=None)
    A = A_T.T
    
    evals, evecs = np.linalg.eig(A)
    
    print("\n3. Analyzing single trajectory trace (FFT, PCA)...")
    # PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(trace_residuals)
    
    # FFT of layer deltas
    deltas = []
    for i in range(config.n_layer - 1):
        diff = trace_residuals[i+1] - trace_residuals[i]
        deltas.append(np.linalg.norm(diff))
        
    fft_res = np.fft.rfft(deltas)
    power = np.abs(fft_res)
    freq = np.fft.rfftfreq(len(deltas))
    
    print("\n4. Sublayer Hook Test (Attn vs MLP)...")
    norm_hook = SublayerNormHook()
    norm_hook.attach(model)
    norm_hook.clear()
    
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(format_chatml_prompt(trace_prompt))], device=device)
        model(input_ids)
        
    df_norms = pd.DataFrame(norm_hook.data)
    norm_hook.remove()
    
    attn_deltas = []
    ffn_deltas = []
    
    for l in range(config.n_layer):
        # some layers might be exclusively GLA with different names, but we wrapped them as 'attn' and 'ffn' if they triggered hooks
        # Let's handle missing gracefully
        attn_p_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'attn') & (df_norms['pos'] == 'pre')]['norm'].values
        attn_po_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'attn') & (df_norms['pos'] == 'post')]['norm'].values
        ffn_p_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'ffn') & (df_norms['pos'] == 'pre')]['norm'].values
        ffn_po_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'ffn') & (df_norms['pos'] == 'post')]['norm'].values
        
        if len(attn_p_rows) > 0 and len(attn_po_rows) > 0:
            attn_deltas.append(attn_po_rows[0] - attn_p_rows[0])
        else:
            attn_deltas.append(0.0)
            
        if len(ffn_p_rows) > 0 and len(ffn_po_rows) > 0:
            ffn_deltas.append(ffn_po_rows[0] - ffn_p_rows[0])
        else:
            ffn_deltas.append(0.0)

    print("\n5. Plotting 5-Panel Display...")
    os.makedirs("figures", exist_ok=True)
    os.makedirs("measurements", exist_ok=True)
    
    fig = plt.figure(figsize=(24, 10))
    grid = plt.GridSpec(2, 3, figure=fig)
    
    # Plot 1: Cosine Similarity
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(range(config.n_layer - 1), mean_sims, marker='o', color='purple')
    ax1.set_title('A. Residual Cosine Similarity (Layer L vs L+1)')
    ax1.set_xlabel('Layer L')
    ax1.set_ylabel('Cosine Similarity')
    
    # Plot 2: FFT Power Spectrum
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.plot(freq, power, marker='v', color='darkorange')
    ax2.set_title('B. Frequency Spectrum of Residual Deltas (FFT)')
    ax2.set_xlabel('Frequency (cycles/layer)')
    ax2.set_ylabel('Power')
    
    # Plot 3: Sublayer Norm Deltas (Push/Pull)
    ax3 = fig.add_subplot(grid[0, 2])
    x_layers = np.arange(config.n_layer)
    width = 0.4
    ax3.bar(x_layers - width/2, attn_deltas, width, label='Attention $\Delta$', color='blue')
    ax3.bar(x_layers + width/2, ffn_deltas, width, label='FFN $\Delta$', color='orange')
    ax3.set_title('C. Sublayer Norm Push/Pull')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('L2 Norm Delta (Post - Pre)')
    ax3.legend()
    
    # Plot 4: PCA Trajectory
    ax4 = fig.add_subplot(grid[1, 0:2])
    ax4.plot(proj[:, 0], proj[:, 1], marker='o', linestyle='-', color='teal', alpha=0.6)
    for i in range(config.n_layer):
        if i % 3 == 0 or i == config.n_layer - 1:
            ax4.annotate(f"L{i}", (proj[i, 0], proj[i, 1]))
    ax4.set_title('D. Residual Stream Trajectory (PCA)')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    
    # Plot 5: Matrix A Eigenvalues
    ax5 = fig.add_subplot(grid[1, 2])
    # Draw unit circle
    t = np.linspace(0, 2*np.pi, 100)
    ax5.plot(np.cos(t), np.sin(t), color='black', alpha=0.2)
    # Plot eigenvalues
    real_parts = np.real(evals)
    imag_parts = np.imag(evals)
    ax5.scatter(real_parts, imag_parts, color='red', s=5, alpha=0.5)
    ax5.set_title('E. Eigenvalues of Operator A (Complex Plane)')
    ax5.set_xlabel('Re')
    ax5.set_ylabel('Im')
    ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("figures/residual_oscillation_5part.png", dpi=150)
    
    # Identify dominant frequency
    max_idx = np.argmax(power)
    dom_freq = freq[max_idx]
    
    # Count complex eigenvalues
    complex_count = np.sum(np.abs(imag_parts) > 1e-4)
    
    print("\n--- MEASUREMENT SUMMARY ---")
    print(f"Mean similarity: {np.mean(mean_sims):.3f}")
    print(f"Dominant FFT Frequency: {dom_freq:.3f} cycles/layer")
    print(f"Operator Eigenvalues: {len(evals)} total, {complex_count} complex (rotational)")
    print(f"Complex fraction: {complex_count/len(evals)*100:.1f}%")
    max_radius = np.max(np.abs(evals))
    print(f"Spectral Radius of A: {max_radius:.3f}")
    print("\nSaved 5-panel plot to figures/residual_oscillation_5part.png")

if __name__ == "__main__":
    main()
