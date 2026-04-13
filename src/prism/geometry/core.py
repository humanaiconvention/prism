import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

VALID_SPECTRAL_MODES = ("full_spectrum", "principal_subspace", "orthogonal_complement")

def compute_cosine(a: torch.Tensor, b: torch.Tensor, eps: float=1e-12) -> float:
    """Computes the cosine similarity between two tensors, flattened to 1D."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = float(torch.norm(a).item() * torch.norm(b).item())
    if denom <= eps:
        return 0.0
    return float(torch.dot(a, b).item() / denom)

def split_vector_by_direction(vector: torch.Tensor, direction: torch.Tensor, eps: float=1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits a vector into components parallel and orthogonal to a given direction."""
    vector = vector.reshape(-1)
    direction = direction.reshape(-1)
    denom = float(torch.dot(direction, direction).item())
    if denom <= eps:
        return torch.zeros_like(vector), vector.clone()
    parallel = (torch.dot(vector, direction) / denom) * direction
    orthogonal = vector - parallel
    return parallel, orthogonal

def parse_spectral_modes(raw_modes: str) -> List[str]:
    """Parses a comma-separated string of spectral modes, validating against known constants."""
    modes = [mode.strip() for mode in str(raw_modes).split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in VALID_SPECTRAL_MODES]
    if unknown:
        raise ValueError(f"Unknown spectral modes: {unknown}. Valid modes: {list(VALID_SPECTRAL_MODES)}")
    if not modes:
        raise ValueError("At least one spectral mode is required.")
    return modes

def project_noise_to_component(noise: torch.Tensor, basis: torch.Tensor, spectral_mode: str) -> torch.Tensor:
    """Projects noise onto a specific spectral subspace component using a given basis."""
    if spectral_mode == "full_spectrum":
        return noise
    if basis is None or basis.numel() == 0:
        raise ValueError(f"basis is required for spectral_mode={spectral_mode}")
        
    flat_noise = noise.reshape(-1, noise.shape[-1])
    principal = (flat_noise @ basis) @ basis.T
    
    if spectral_mode == "principal_subspace":
        return principal.reshape_as(noise)
    if spectral_mode == "orthogonal_complement":
        return (flat_noise - principal).reshape_as(noise)
        
    raise ValueError(f"Unsupported spectral_mode: {spectral_mode}")

def project_orthogonal_noise(noise: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Projects noise completely orthogonally to a given basis."""
    if basis is None or basis.numel() == 0:
        return noise
    flat_noise = noise.reshape(-1, noise.shape[-1])
    proj_flat = (flat_noise @ basis) @ basis.T
    return (flat_noise - proj_flat).reshape_as(noise)

def fit_principal_basis(chunks: List[np.ndarray], top_k: int) -> Dict[str, Any]:
    """Fits PCA to a subset of states to extract their principal subspace using SVD."""
    import numpy as np
    matrix = np.concatenate([np.asarray(chunk, dtype=np.float64) for chunk in chunks], axis=0)
    if matrix.shape[0] < 2:
        raise ValueError("Need at least two activation rows to fit a principal subspace.")
    
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    
    max_rank = min(int(top_k), vh.shape[0], matrix.shape[1], centered.shape[0] - 1)
    if max_rank <= 0:
        raise ValueError(f"Unable to fit a non-empty basis: top_k={top_k}, matrix_shape={matrix.shape}")
        
    variance = singular_values ** 2
    total_variance = float(np.sum(variance))
    explained = variance[:max_rank] / max(total_variance, 1e-12)
    
    return {
        "basis": torch.tensor(vh[:max_rank].T.copy(), dtype=torch.float32),
        "pc1_explained_variance_ratio": float(explained[0]) if explained.size else 0.0,
        "topk_cumulative_explained_variance_ratio": float(np.sum(explained)),
        "orthogonal_complement_variance_ratio": float(max(0.0, 1.0 - np.sum(explained))),
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "top_k_used": int(max_rank)
    }

def unit_vector(vec: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    """Normalizes a tensor into a flattened unit vector."""
    vec = vec.reshape(-1).float()
    return vec / max(float(torch.norm(vec).item()), eps)

def compute_coeff(activation: torch.Tensor, unit_direction: torch.Tensor) -> float:
    """Projects an activation onto a unit vector direction."""
    return float(torch.dot(activation.reshape(-1), unit_direction.reshape(-1)).item())

def orthogonal_residual(state: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Removes the component of state that is parallel to direction."""
    vec = direction.to(device=state.device, dtype=state.dtype)
    coeff = torch.dot(state, vec)
    return state - (coeff * vec)

def fit_pca_bank(states: List[np.ndarray], requested_rank: int) -> Dict[str, Any]:
    """Fits PCA to a collection of states, returning mean and basis."""
    import numpy as np
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Need at least two states to fit a PCA bank")
    
    mean = np.mean(x, axis=0)
    centered = x - mean[None, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    
    max_rank = max(1, min(int(requested_rank), int(vh.shape[0]), int(x.shape[0] - 1)))
    
    energy = singular_values ** 2
    total_energy = float(np.sum(energy))
    per_pc = energy[:max_rank] / max(total_energy, 1e-12)
    cumulative = np.cumsum(per_pc)
    
    return {
        "mean": mean,
        "basis": vh[:max_rank].T,
        "effective_rank": int(max_rank),
        "per_pc_explained_variance_ratio": per_pc.astype(np.float64),
        "cumulative_explained_variance_ratio": cumulative.astype(np.float64),
    }

def make_random_orthogonal_subspace(direction: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Creates a random subspace orthogonal to the given direction."""
    import numpy as np
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    dim = int(direction.shape[0])
    target_rank = max(1, int(rank))
    
    rng = np.random.default_rng(int(seed))
    basis_cols = []
    attempts = 0
    
    while len(basis_cols) < target_rank:
        attempts += 1
        if attempts > 64:
            raise RuntimeError(f"Could not sample a random orthogonal subspace of rank={target_rank}")
            
        candidate = rng.normal(size=(dim, target_rank + 8))
        candidate = candidate - np.outer(direction, direction @ candidate)
        
        for col_idx in range(candidate.shape[1]):
            vec = candidate[:, col_idx]
            for prev in basis_cols:
                vec = vec - (prev @ vec) * prev
                
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                continue
                
            basis_cols.append(vec / norm)
            if len(basis_cols) >= target_rank:
                break
                
    return np.stack(basis_cols[:target_rank], axis=1)

def project_state(state: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> Dict[str, float]:
    """Projects a state onto an affine subspace defined by mean and basis."""
    import numpy as np
    x = np.asarray(state, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    total_energy = float(np.dot(x, x))
    coeff = np.asarray(basis, dtype=np.float64).T @ x
    proj_energy = float(np.dot(coeff, coeff))
    
    return {
        "projection_fraction": float(proj_energy / max(total_energy, 1e-12)) if total_energy > 1e-12 else 0.0,
        "projection_norm": float(np.sqrt(max(proj_energy, 0.0))),
    }

def project_onto_basis(tensor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis is None:
        return torch.zeros_like(tensor)
    flat = tensor.reshape(-1, tensor.shape[-1])
    proj = flat @ basis @ basis.T
    return proj.reshape_as(tensor)

def project_out_basis(tensor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis is None:
        return tensor
    flat = tensor.reshape(-1, tensor.shape[-1])
    proj = flat @ basis @ basis.T
    return (flat - proj).reshape_as(tensor)

def compute_mean_cosine_to_ref(states: List[torch.Tensor], ref_idx: int) -> float:
    import numpy as np
    if len(states) < 2:
        return float("nan")
    ref_idx = int(min(max(ref_idx, 0), len(states) - 1))
    ref = states[ref_idx].reshape(-1)
    ref_norm = torch.norm(ref).clamp_min(1e-8)
    cos_vals = []
    for state in states:
        vec = state.reshape(-1)
        denom = torch.norm(vec).clamp_min(1e-8) * ref_norm
        cos_vals.append(float(torch.dot(vec, ref).item() / denom.item()))
    return float(np.mean(cos_vals))

def outlier_geometry(H_raw: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """Compute massive-activation outlier geometry metrics for a hidden-state matrix.

    Inspired by TurboQuant (Google, ICLR 2026) — the same per-dimension magnitude
    statistics that predict KV-cache quantization error also serve as an early-warning
    signal for representation collapse and activation instability during fine-tuning.

    Accepts either a ``torch.Tensor`` or a ``numpy.ndarray``.  When passed a NumPy
    array the computation is delegated to :func:`outlier_geometry_numpy` so that
    the function is usable without a CUDA device or even a full PyTorch install.

    Args:
        H_raw: Float tensor **or** array of shape ``(seq_len, hidden_dim)``.
               Raw (not mean-centred) hidden states from a single layer and prompt.

    Returns:
        Dict with keys:

        * ``outlier_ratio``          — max dim magnitude / mean dim magnitude.
          >10 indicates a dominant "massive activation".
        * ``activation_kurtosis``    — excess kurtosis of per-dim magnitudes.
          Positive = heavy-tailed distribution.
        * ``cardinal_proximity``     — mean max-abs component of each token unit vector.
          Near 1.0 = vectors are axis-aligned → prone to quantisation snap.
        * ``quantization_hostility`` — composite score in [0, 1].
          >0.7 signals the layer is hostile to low-bit quantisation.
    """
    if isinstance(H_raw, np.ndarray):
        return outlier_geometry_numpy(H_raw)

    import math
    import torch.nn.functional as F

    H = H_raw.float()
    seq, dim = H.shape

    if seq < 1 or dim < 1:
        return {
            "outlier_ratio": 1.0,
            "activation_kurtosis": 0.0,
            "cardinal_proximity": 0.0,
            "quantization_hostility": 0.0,
        }

    dim_mag = H.abs().mean(dim=0)  # (dim,)
    mean_mag = dim_mag.mean()
    max_mag = dim_mag.max()
    outlier_ratio = float((max_mag / (mean_mag + 1e-12)).item())

    mu = dim_mag.mean()
    sigma = dim_mag.std(unbiased=False)
    if sigma < 1e-12:
        activation_kurtosis = 0.0
    else:
        activation_kurtosis = float(
            (((dim_mag - mu) ** 4).mean() / (sigma ** 4 + 1e-12) - 3.0).item()
        )

    h_unit = F.normalize(H, dim=-1)
    cardinal_proximity = float(h_unit.abs().max(dim=-1).values.mean().item())

    or_norm = min(math.log(max(outlier_ratio, 1.0)) / math.log(50.0), 1.0)
    ak_norm = min(max(activation_kurtosis, 0.0) / 20.0, 1.0)
    cp_norm = float(cardinal_proximity)
    quantization_hostility = (or_norm + ak_norm + cp_norm) / 3.0

    return {
        "outlier_ratio": outlier_ratio,
        "activation_kurtosis": activation_kurtosis,
        "cardinal_proximity": cardinal_proximity,
        "quantization_hostility": quantization_hostility,
    }


def outlier_geometry_numpy(H_raw: np.ndarray) -> Dict[str, float]:
    """Pure-NumPy implementation of :func:`outlier_geometry`.

    Identical math; no PyTorch dependency.  Use this when the full PyTorch stack
    is unavailable (e.g. lightweight CI environments, pure-CPU inference servers,
    or when loading hidden states from a pre-computed ``.npy`` file).

    Args:
        H_raw: Float array of shape ``(seq_len, hidden_dim)``.

    Returns:
        Same four-key dict as :func:`outlier_geometry`.
    """
    import math

    H = np.asarray(H_raw, dtype=np.float32)
    if H.ndim != 2 or H.shape[0] < 1 or H.shape[1] < 1:
        return {
            "outlier_ratio": 1.0,
            "activation_kurtosis": 0.0,
            "cardinal_proximity": 0.0,
            "quantization_hostility": 0.0,
        }

    dim_mag = np.abs(H).mean(axis=0)          # (hidden_dim,)
    mean_mag = float(dim_mag.mean())
    max_mag = float(dim_mag.max())
    outlier_ratio = max_mag / (mean_mag + 1e-12)

    mu = float(dim_mag.mean())
    sigma = float(dim_mag.std())
    if sigma < 1e-12:
        activation_kurtosis = 0.0
    else:
        activation_kurtosis = float(
            np.mean(((dim_mag - mu) ** 4)) / (sigma ** 4 + 1e-12) - 3.0
        )

    norms = np.linalg.norm(H, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    h_unit = H / norms
    cardinal_proximity = float(np.abs(h_unit).max(axis=-1).mean())

    or_norm = min(math.log(max(outlier_ratio, 1.0)) / math.log(50.0), 1.0)
    ak_norm = min(max(activation_kurtosis, 0.0) / 20.0, 1.0)
    cp_norm = float(cardinal_proximity)
    quantization_hostility = (or_norm + ak_norm + cp_norm) / 3.0

    return {
        "outlier_ratio": outlier_ratio,
        "activation_kurtosis": activation_kurtosis,
        "cardinal_proximity": cardinal_proximity,
        "quantization_hostility": quantization_hostility,
    }


def apply_givens_rotations(weight: torch.Tensor, rng: Any, rotations: int, angle: float) -> torch.Tensor:
    import numpy as np
    w = weight.detach().clone()
    out_dim, in_dim = w.shape
    if in_dim < 2:
        return w
    for _ in range(int(rotations)):
        i, j = rng.choice(in_dim, size=2, replace=False)
        theta = float(rng.uniform(-angle, angle))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        col_i = w[:, i].clone()
        col_j = w[:, j].clone()
        w[:, i] = c * col_i + s * col_j
        w[:, j] = -s * col_i + c * col_j
    return w

