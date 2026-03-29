import os
from pathlib import Path

import numpy as np


def infer_companion_csv(output_csv, suffix):
    root, ext = os.path.splitext(output_csv)
    return f"{root}_{suffix}{ext or '.csv'}"


def infer_companion_png(output_csv, suffix):
    root, _ = os.path.splitext(output_csv)
    return f"{root}_{suffix}.png"


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def layer_type_name(layer, fox_layers=None):
    fox_layers = set(fox_layers or [])
    return "fox" if int(layer) in fox_layers else "gla"


def bootstrap_mean_ci(values, n_boot=4000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(n_boot)):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signflip_test(values, n_perm=20000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "pvalue": 1.0, "n": 0}
    observed = float(np.mean(values))
    ci_low, ci_high = bootstrap_mean_ci(values, seed=seed)
    rng = np.random.default_rng(int(seed))
    perm_means = []
    batch = min(4096, max(256, int(n_perm)))
    remaining = int(n_perm)
    while remaining > 0:
        take = min(batch, remaining)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(take, values.size), replace=True)
        perm_means.append(np.mean(signs * values[None, :], axis=1))
        remaining -= take
    perm_means = np.concatenate(perm_means, axis=0)
    pvalue = float((1.0 + np.sum(np.abs(perm_means) >= abs(observed))) / (perm_means.size + 1.0))
    return {"mean": observed, "ci95_low": ci_low, "ci95_high": ci_high, "pvalue": pvalue, "n": int(values.size)}


def paired_signflip_test(a, b, n_perm=20000, seed=0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired samples must match shape; got {a.shape} vs {b.shape}")
    result = signflip_test(a - b, n_perm=n_perm, seed=seed)
    result["mean_a"] = float(np.mean(a)) if a.size else 0.0
    result["mean_b"] = float(np.mean(b)) if b.size else 0.0
    return result


def compute_representation_metrics(matrix):
    from prism.analysis import compute_spectral_metrics, compute_top_eigenvalues
    import torch
    
    x = np.asarray(matrix, dtype=np.float64)
    n, d = x.shape
    centered = x - x.mean(axis=0, keepdims=True)
    total_variance = float(np.sum(np.var(x, axis=0, ddof=1))) if n > 1 else 0.0
    mean_activation_norm = float(np.mean(np.linalg.norm(x, axis=1)))
    mean_feature_variance = float(np.mean(np.var(x, axis=0, ddof=1))) if n > 1 else 0.0
    
    if n < 2:
        return {
            "participation_ratio": 0.0,
            "spectral_entropy": 0.0,
            "normalized_spectral_entropy": 0.0,
            "pc1_explained_variance_ratio": 0.0,
            "pc5_cumulative_explained_variance_ratio": 0.0,
            "pc10_cumulative_explained_variance_ratio": 0.0,
            "mean_activation_norm": mean_activation_norm,
            "mean_feature_variance": mean_feature_variance,
            "total_variance": total_variance,
        }
        
    spectral_entropy, participation_ratio = compute_spectral_metrics(torch.from_numpy(centered).float())
    
    # normalized_entropy in the old code used len(probs) which is the rank
    # compute_spectral_metrics uses all eigenvalues (some might be tiny)
    # We'll use log(min(n, d)) as the normalization factor to be consistent with standard ID.
    normalized_entropy = float(spectral_entropy / max(np.log(min(n, d)), 1e-12))
    
    # For PC explained variance, use compute_top_eigenvalues
    evals = np.array(compute_top_eigenvalues(torch.from_numpy(centered).float(), k=10))
    # Note: eigenvalues from compute_top_eigenvalues (via eigvalsh on x.T @ x) are NOT scaled by (n-1).
    # The old code's eigvals WERE scaled by (n-1).
    # So explained = (eigvals_old / total_old) = (singular_values**2 / total_singular_values**2)
    # This is equivalent to (evals / sum(evals_all)).
    # sum(evals_all) is trace(x.T @ x) = sum(diag(x.T @ x)) = sum(x**2)
    sum_evals = float(np.sum(centered**2))
    explained = evals / max(sum_evals, 1e-12)
    
    return {
        "participation_ratio": participation_ratio,
        "spectral_entropy": spectral_entropy,
        "normalized_spectral_entropy": normalized_entropy,
        "pc1_explained_variance_ratio": float(np.sum(explained[:1])),
        "pc5_cumulative_explained_variance_ratio": float(np.sum(explained[:5])),
        "pc10_cumulative_explained_variance_ratio": float(np.sum(explained[:10])),
        "mean_activation_norm": mean_activation_norm,
        "mean_feature_variance": mean_feature_variance,
        "total_variance": total_variance,
    }


def mean_pairwise_metrics(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return {
            "mean_pairwise_cosine_distance": 0.0,
            "median_pairwise_cosine_distance": 0.0,
            "mean_pairwise_l2_distance": 0.0,
            "n_pairs": 0,
        }
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.clip(norms, 1e-12, None)
    cosine = x_norm @ x_norm.T
    sq = np.sum(x ** 2, axis=1, keepdims=True)
    l2 = np.sqrt(np.clip(sq + sq.T - 2.0 * (x @ x.T), 0.0, None))
    tri = np.triu_indices(n, k=1)
    cosine_dist = 1.0 - cosine[tri]
    l2_dist = l2[tri]
    return {
        "mean_pairwise_cosine_distance": float(np.mean(cosine_dist)),
        "median_pairwise_cosine_distance": float(np.median(cosine_dist)),
        "mean_pairwise_l2_distance": float(np.mean(l2_dist)),
        "n_pairs": int(cosine_dist.size),
    }


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)