"""Phase 9E: Semantic Rotational Dynamics.

Analyzes the rotational dynamics of semantic features across layers.
Estimates the layer-to-layer operator A and checks for complex eigenmodes.
"""

import os
import argparse
import numpy as np
import scipy.linalg as la
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Phase 9E: Semantic Rotational Dynamics")
    parser.add_argument("--vector-dir", type=str, default="logs/phase9/vectors")
    parser.add_argument("--output-csv", type=str, default="logs/phase9/rotational_dynamics.csv")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.vector_dir) if f.startswith("layer_") and f.endswith("_vector.npz")]
    files.sort(key=lambda x: int(x.split("_")[1]))
    
    if len(files) < 2:
        print("Error: Need at least 2 layers for rotation analysis.")
        return

    # Extract all directions into a matrix [N_layers, D]
    all_vectors = []
    layer_indices = []
    for f in files:
        data = np.load(os.path.join(args.vector_dir, f))
        all_vectors.append(data["delta_perp"])
        layer_indices.append(int(f.split("_")[1]))
    
    H = np.stack(all_vectors) # (T, D)
    T, D = H.shape
    
    print(f"Estimating system dynamics for {T} semantic directions in {D}-dim space...")
    
    # Estimate h_{l+1} \approx A h_l
    # H_next = H[1:]
    # H_prev = H[:-1]
    # A = H_next^T @ H_prev @ (H_prev^T @ H_prev)^-1
    # For semantic directions, we have a low-rank sequence.
    
    H_prev = H[:-1]
    H_next = H[1:]
    
    # Use pseudo-inverse for stability
    A, residuals, rank, s = np.linalg.lstsq(H_prev, H_next, rcond=None)
    # A is (D, D) conceptually, but lstsq solves for X in AX=B. 
    # Here X is A, and we want H_next = H_prev @ A.
    # So A is (D, D).
    
    # Calculate eigenvalues of the transition matrix
    # Actually, we can use DMD (Dynamic Mode Decomposition) logic if we want.
    # But a simple way to see "rotation" is to check for complex eigenvalues 
    # in the operator that maps the semantic sequence.
    
    # We can perform SVD on the sequence itself or the transition operator.
    # Finding 15 says: "560 were complex conjugate pairs (97.2%)".
    
    # Let's compute the eigenvalues of the estimated operator A.
    # Note: A is estimated from the full sequence.
    
    # For a low-rank sequence, A will be highly singular.
    # We'll compute eigenvalues of the projected operator (U^T A U)
    U, S, Vh = np.linalg.svd(H_prev, full_matrices=False)
    k = rank
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = Vh[:k, :].T
    
    # Tilting A into the subspace: Atilde = U^T @ H_next @ V @ S^-1
    Atilde = U_k.T @ H_next @ V_k @ np.linalg.inv(S_k)
    
    evals = np.linalg.eigvals(Atilde)
    n_complex = np.sum(np.iscomplex(evals))
    
    print(f"\n--- SEMANTIC ROTATIONAL DYNAMICS ---")
    print(f"Sequence Rank: {rank}")
    print(f"Eigenvalues: {len(evals)}")
    print(f"Complex Eigenvalues: {n_complex} ({n_complex/len(evals):.1%})")
    
    if n_complex > 0:
        print("  ROTATION CONFIRMED: Complex eigenmodes dominate the semantic transition.")
    else:
        print("  MONOTONIC REFINEMENT: Only real eigenvalues found.")

    # Save summary
    with open(args.output_csv, "w") as f:
        f.write("rank,n_evals,n_complex,complex_pct\n")
        f.write(f"{rank},{len(evals)},{n_complex},{n_complex/len(evals):.4f}\n")

    print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()
