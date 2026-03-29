"""Phase 9D: Layerwise Semantic Evolution.

Maps the alignment and principal angles of semantic directions 
across the 30-layer stack.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

def calculate_cosine_alignment(v1, v2):
    v1_n = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_n = v2 / (np.linalg.norm(v2) + 1e-10)
    return np.dot(v1_n, v2_n)

def main():
    parser = argparse.ArgumentParser(description="Phase 9D: Layerwise Semantic Evolution")
    parser.add_argument("--vector-dir", type=str, default="logs/phase9/vectors")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/layerwise_alignment.csv")
    args = parser.parse_args()

    # Get sorted list of vector files
    files = [f for f in os.listdir(args.vector_dir) if f.startswith("layer_") and f.endswith("_vector.npz")]
    files.sort(key=lambda x: int(x.split("_")[1]))
    
    layer_indices = [int(f.split("_")[1]) for f in files]
    
    print(f"Analyzing semantic alignment across {len(layer_indices)} layers...")

    alignment_data = []

    for i in range(len(files)):
        layer_curr = layer_indices[i]
        data_curr = np.load(os.path.join(args.vector_dir, files[i]))
        v_curr = data_curr["delta_perp"]
        
        # Alignment with final layer (semantic anchor)
        data_final = np.load(os.path.join(args.vector_dir, files[-1]))
        v_final = data_final["delta_perp"]
        cos_to_final = calculate_cosine_alignment(v_curr, v_final)
        
        # Alignment with previous layer
        if i > 0:
            data_prev = np.load(os.path.join(args.vector_dir, files[i-1]))
            v_prev = data_prev["delta_perp"]
            cos_to_prev = calculate_cosine_alignment(v_curr, v_prev)
        else:
            cos_to_prev = 1.0

        alignment_data.append({
            "Layer": layer_curr,
            "Cos_To_Final": cos_to_final,
            "Cos_To_Prev": cos_to_prev,
            "Perp_Norm": data_curr["perp_norm"],
            "Reduction_Ratio": data_curr["reduction_ratio"]
        })

    df = pd.DataFrame(alignment_data)
    print("\n--- LAYERWISE SEMANTIC ALIGNMENT ---")
    print(df.to_string(index=False))
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()
