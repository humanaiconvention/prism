"""Phase 9C: Semantic Direction Isolation.

Identifies and orthogonalizes semantic task vectors from the
extracted manifolds, with guardrails to avoid annihilating the
semantic direction when projecting out the bulk subspace.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np


def effective_rank(evals):
    """Shannon effective rank for non-negative eigenvalues."""
    clipped = np.clip(np.asarray(evals, dtype=np.float64), 0.0, None)
    total = clipped.sum()
    if total <= 0:
        return 0.0
    probs = clipped / total
    probs = probs[probs > 0]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def project_delta(delta, evecs, k_bulk):
    if k_bulk <= 0:
        return delta.copy()
    e_bulk = evecs[:, :k_bulk]
    return delta - e_bulk @ (e_bulk.T @ delta)


def isolate_semantic_direction(cov, math_centroid, creative_centroid, k_bulk=70, min_retained_fraction=0.10, rank_tol=1e-8, n_samples=None):
    cov = np.asarray(cov, dtype=np.float64)
    math_centroid = np.asarray(math_centroid, dtype=np.float64)
    creative_centroid = np.asarray(creative_centroid, dtype=np.float64)

    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    delta = math_centroid - creative_centroid
    raw_norm = float(np.linalg.norm(delta))

    top_eval = float(max(np.max(np.abs(evals)), 1.0))
    rank_eps = rank_tol * top_eval
    numerical_rank = int(np.sum(evals > rank_eps))
    n_samples = int(n_samples) if n_samples is not None else cov.shape[0]
    sample_rank_cap = int(min(max(n_samples - 1, 0), cov.shape[0] - 1))
    rank_limited_cap = min(k_bulk, max(numerical_rank - 1, 0), sample_rank_cap)

    k_effective = max(rank_limited_cap, 0)
    delta_perp = project_delta(delta, evecs, k_effective)
    perp_norm = float(np.linalg.norm(delta_perp))
    retained_fraction = float(perp_norm / raw_norm) if raw_norm > 0 else 0.0

    while raw_norm > 0 and k_effective > 0 and retained_fraction < min_retained_fraction:
        k_effective -= 1
        delta_perp = project_delta(delta, evecs, k_effective)
        perp_norm = float(np.linalg.norm(delta_perp))
        retained_fraction = float(perp_norm / raw_norm)

    reduction_ratio = float(raw_norm / perp_norm) if perp_norm > 0 else float("inf")
    positive_evals = np.clip(evals, 0.0, None)
    total_variance = float(positive_evals.sum())
    bulk_variance_explained = (
        float(positive_evals[:k_effective].sum() / total_variance)
        if total_variance > 0 and k_effective > 0
        else 0.0
    )

    return {
        "delta_raw": delta,
        "delta_perp": delta_perp,
        "raw_norm": raw_norm,
        "perp_norm": perp_norm,
        "reduction_ratio": reduction_ratio,
        "retained_fraction": retained_fraction,
        "bulk_variance_explained": bulk_variance_explained,
        "k_bulk_requested": k_bulk,
        "k_bulk_effective": k_effective,
        "numerical_rank": numerical_rank,
        "sample_rank_cap": sample_rank_cap,
        "effective_rank": effective_rank(positive_evals),
        "rank_eps": rank_eps,
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 9C: Semantic Direction Isolation")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--k-bulk", type=int, default=70, help="Number of bulk dimensions to project out")
    parser.add_argument(
        "--min-retained-fraction",
        type=float,
        default=0.10,
        help="Minimum perp/raw norm fraction to preserve after projection.",
    )
    parser.add_argument(
        "--rank-tol",
        type=float,
        default=1e-8,
        help="Relative eigenvalue tolerance for numerical rank estimation.",
    )
    parser.add_argument("--output-dir", type=str, default="logs/phase9/vectors")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(args.data_dir) if f.startswith("layer_") and f.endswith("_stats.npz")]
    manifest = {
        "created_by": "scripts/run_phase9_semantic_dirs.py",
        "vector_dir": Path(args.output_dir).name,
        "vector_key_default": "delta_perp",
        "layers": {},
    }
    
    print(f"Isolating semantic directions for {len(files)} layers (k_bulk={args.k_bulk})...")

    for f in sorted(files):
        layer_idx = int(f.split("_")[1])
        data = np.load(os.path.join(args.data_dir, f))
        
        cov = data["covariance"]
        math_centroid = data["math_centroid"]
        creative_centroid = data["creative_centroid"]
        
        vector_data = isolate_semantic_direction(
            cov=cov,
            math_centroid=math_centroid,
            creative_centroid=creative_centroid,
            k_bulk=args.k_bulk,
            min_retained_fraction=args.min_retained_fraction,
            rank_tol=args.rank_tol,
            n_samples=int(data["n_samples"]) if "n_samples" in data.files else None,
        )
        
        save_path = f"{args.output_dir}/layer_{layer_idx}_vector.npz"
        np.savez(save_path, **vector_data)
        manifest["layers"][str(layer_idx)] = {
            "path": str(Path(Path(args.output_dir).name) / f"layer_{layer_idx}_vector.npz"),
            "available_vector_keys": ["delta_raw", "delta_perp"],
            "raw_norm": float(vector_data["raw_norm"]),
            "perp_norm": float(vector_data["perp_norm"]),
            "retained_fraction": float(vector_data["retained_fraction"]),
            "k_bulk_effective": int(vector_data["k_bulk_effective"]),
        }
        
        print(f"  Layer {layer_idx}:")
        print(f"    Raw Delta Norm: {vector_data['raw_norm']:.3f}")
        print(
            f"    Rank Guardrail: requested={args.k_bulk}, effective={vector_data['k_bulk_effective']}, "
            f"numerical_rank={vector_data['numerical_rank']}, sample_cap={vector_data['sample_rank_cap']}"
        )
        print(
            f"    Perp Delta Norm: {vector_data['perp_norm']:.3f} "
            f"(retained={vector_data['retained_fraction']:.3f}, reduction={vector_data['reduction_ratio']:.1f}x)"
        )
        print(
            f"    Bulk Variance Explained: {vector_data['bulk_variance_explained']:.3f} "
            f"(effective_rank={vector_data['effective_rank']:.2f})"
        )
        print(f"    Vector saved to {save_path}")

    manifest_path = Path(args.output_dir).parent / "semantic_directions.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSemantic direction isolation complete.")
    print(f"Semantic direction manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
