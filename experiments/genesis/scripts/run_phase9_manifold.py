"""Phase 9B: Manifold Mapping.

Analyzes the topological structure of the extracted manifold.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_er(evals):
    evals = evals[evals > 1e-10]
    p = evals / evals.sum()
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def main():
    parser = argparse.ArgumentParser(description="Phase 9B: Manifold Mapping")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/manifold_results.csv")
    args = parser.parse_args()

    results = []
    
    # List files in data dir
    files = [f for f in os.listdir(args.data_dir) if f.startswith("layer_") and f.endswith("_stats.npz")]
    
    print(f"Analyzing {len(files)} layer-wise statistics...")

    for f in sorted(files):
        layer_idx = int(f.split("_")[1])
        data = np.load(os.path.join(args.data_dir, f))
        
        # Calculate ER from full covariance
        cov = data["covariance"]
        evals, _ = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        
        global_er = calculate_er(evals)
        
        # For domain-specific volume, we'd ideally have separate covariance,
        # but for this phase we can approximate by checking projection mags 
        # or assuming the global ER covers the active semantic volume.
        # We'll stick to Global ER as the primary metric.
        
        results.append({
            "Layer": layer_idx,
            "Global_ER": global_er,
            "Utilization_Pct": (global_er / 576.0) * 100.0,
            "N_Samples": data["n_samples"]
        })

    df = pd.DataFrame(results)
    print("\n--- MANIFOLD MAPPING SUMMARY ---")
    print(df.to_string(index=False))
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()
