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
        self.data = []
        
    def _make_hook(self, layer, sublayer_type, pos):
        def hook_fn(module, inp, output=None):
            x = output[0] if output is not None and isinstance(output, tuple) else (output if output is not None else inp[0])
            norm = torch.norm(x[:, -1, :].detach().float(), p=2, dim=-1).item()
            self.data.append({"layer": layer, "sublayer": sublayer_type, "pos": pos, "norm": norm})
            return None
        return hook_fn

    def attach(self, model):
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn_norm'):
                self.handles.append(block.attn_norm.register_forward_pre_hook(self._make_hook(i, 'attn', 'pre')))
            if hasattr(block, 'attn'):
                self.handles.append(block.attn.register_forward_hook(self._make_hook(i, 'attn', 'post')))
            if hasattr(block, 'ffn_norm'):
                self.handles.append(block.ffn_norm.register_forward_pre_hook(self._make_hook(i, 'ffn', 'pre')))
            if hasattr(block, 'ffn'):
                self.handles.append(block.ffn.register_forward_hook(self._make_hook(i, 'ffn', 'post')))
            
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        
    def clear(self):
        self.data = []

def main():
    print("Loading Genesis-152M for Phase 7B Oscillation Confirmation Suite...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)

    with open("prompts/prompts_200.json", "r", encoding="utf-8") as f:
        prompts_list = json.load(f)["prompts"]
        
    test_prompts = [p["text"] if isinstance(p, dict) else p for p in prompts_list[:30]]
    
    hooks = HiddenStateHook.attach_to_model(model)
    
    all_sims = {l: [] for l in range(config.n_layer - 1)}
    all_fox_resets = {b: [] for b in range(config.n_layer // 4)}
    all_deltas = []
    
    # We will average the PCA trajectory across prompts to denoise it
    all_pca_trajs = []
    
    # Sublayer Norm collection
    norm_hook = SublayerNormHook()
    norm_hook.attach(model)
    norm_data_all = []

    print(f"\nProcessing {len(test_prompts)} prompts...")
    for pidx, prompt in enumerate(tqdm(test_prompts)):
        inp = format_chatml_prompt(prompt)
        input_ids = torch.tensor([tokenizer.encode(inp)], device=device)
        
        hooks.clear()
        norm_hook.clear()
        with torch.no_grad():
            model(input_ids)
            
        states = hooks.get_hidden_states()
        residuals = [s.numpy()[0] for s in states] 
        
        # 1. Cosine Similarities (Layer to Layer + FoX Resets)
        for l in range(config.n_layer - 1):
            all_sims[l].append(1.0 - cosine(residuals[l], residuals[l+1]))
            
        # FoX Resets: distance between h_n (block entry) and h_n+3 (FoX output)
        for b in range(config.n_layer // 4):
            n = b * 4
            if n + 3 < config.n_layer:
                all_fox_resets[b].append(1.0 - cosine(residuals[n], residuals[n+3]))
        
        # 2. Layer Deltas for FFT
        prompt_deltas = []
        for i in range(config.n_layer - 1):
            diff = residuals[i+1] - residuals[i]
            prompt_deltas.append(np.linalg.norm(diff))
        all_deltas.append(prompt_deltas)
        
        # 3. PCA Trajectory
        pca = PCA(n_components=2)
        proj = pca.fit_transform(residuals)
        all_pca_trajs.append(proj)
        
        # 4. Norm hook data
        for d in norm_hook.data:
            d['prompt_idx'] = pidx
            norm_data_all.append(d)
            
    hooks.remove_all()
    norm_hook.remove()
    
    mean_sims = [np.mean(all_sims[l]) for l in range(config.n_layer - 1)]
    mean_fox_resets = [np.mean(all_fox_resets[b]) for b in range(config.n_layer // 4) if all_fox_resets[b]]
    print("\n[Phase 7A Test 2] Mean FoX Reset Cosine (h_n vs h_{n+3}) per 4-layer block:")
    for b, reset_val in enumerate(mean_fox_resets):
        print(f"  Block {b} (L{b*4}->L{b*4+3}): {reset_val:.4f}")
    
    # Aggregate PCA (using the first prompt as the visual trace to preserve dynamics)
    trace_proj = all_pca_trajs[0]
    
    # Average FFT
    mean_deltas = np.mean(all_deltas, axis=0)
    fft_res = np.fft.rfft(mean_deltas)
    power = np.abs(fft_res)
    freq = np.fft.rfftfreq(len(mean_deltas))
    
    # Aggregate Norms
    df_norms = pd.DataFrame(norm_data_all)
    attn_deltas = []
    ffn_deltas = []
    
    for l in range(config.n_layer):
        attn_p_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'attn') & (df_norms['pos'] == 'pre')].groupby('prompt_idx')['norm'].mean()
        attn_po_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'attn') & (df_norms['pos'] == 'post')].groupby('prompt_idx')['norm'].mean()
        ffn_p_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'ffn') & (df_norms['pos'] == 'pre')].groupby('prompt_idx')['norm'].mean()
        ffn_po_rows = df_norms[(df_norms['layer'] == l) & (df_norms['sublayer'] == 'ffn') & (df_norms['pos'] == 'post')].groupby('prompt_idx')['norm'].mean()
        
        # align by prompt_idx and subtract
        if len(attn_p_rows) > 0 and len(attn_po_rows) > 0:
            attn_deltas.append((attn_po_rows - attn_p_rows).mean())
        else:
            attn_deltas.append(0.0)
            
        if len(ffn_p_rows) > 0 and len(ffn_po_rows) > 0:
            ffn_deltas.append((ffn_po_rows - ffn_p_rows).mean())
        else:
            ffn_deltas.append(0.0)

    print("\nGenerating 4-Panel Publishable Plot...")
    os.makedirs("figures", exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Residual Cosine Similarity
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(range(config.n_layer - 1), mean_sims, marker='o', color='purple')
    for b in range(config.n_layer // 4):
        if b*4+3 < config.n_layer-1:
            ax1.axvline(x=b*4+3, color='gray', linestyle='--', alpha=0.3) # Mark FoX layers
    ax1.set_title('A. Residual Cosine Similarity (Layer L vs L+1)')
    ax1.set_xlabel('Layer L')
    ax1.set_ylabel('Mean Cosine Similarity')
    ax1.grid(True, alpha=0.2)
    
    # 2. FFT Power Spectrum
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.plot(freq, power, marker='v', color='darkorange', linewidth=2)
    ax2.set_title('B. Frequency Spectrum of Residual Deltas (FFT)')
    ax2.set_xlabel('Frequency (cycles/layer)')
    ax2.set_ylabel('Power')
    ax2.grid(True, alpha=0.2)
    
    # 3. PCA Trajectory (Sample Prompt)
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.plot(trace_proj[:, 0], trace_proj[:, 1], marker='o', linestyle='-', color='teal', alpha=0.6)
    for i in range(config.n_layer):
        if i % 3 == 0 or i == config.n_layer - 1:
            ax3.annotate(f"L{i}", (trace_proj[i, 0], trace_proj[i, 1]))
    ax3.set_title('C. Residual Stream Trajectory (PCA, N=1 Trace)')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.grid(True, alpha=0.2)
    
    # 4. Sublayer Norm Push/Pull
    ax4 = fig.add_subplot(grid[1, 1])
    x_layers = np.arange(config.n_layer)
    width = 0.4
    ax4.bar(x_layers - width/2, attn_deltas, width, label='Attention $\Delta$ (Post-Pre)', color='blue', alpha=0.7)
    ax4.bar(x_layers + width/2, ffn_deltas, width, label='FFN $\Delta$ (Post-Pre)', color='orange', alpha=0.7)
    ax4.set_title('D. Sublayer Magnitude Push/Pull')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Mean L2 Norm Change')
    ax4.legend()
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("figures/oscillation_suite_4part.png", dpi=300)
    print("Saved -> figures/oscillation_suite_4part.png")
    
if __name__ == "__main__":
    main()
