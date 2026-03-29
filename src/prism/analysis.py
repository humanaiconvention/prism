"""Spectral and structural analysis utilities."""

from typing import Sequence, Tuple, Union, List
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

def compute_eigenvalues_hardware_aware(matrix: torch.Tensor, device_override: str = None) -> torch.Tensor:
    """Computes eigenvalues using a hardware-aware dispatch system."""
    target_device = device_override
    if target_device is None:
        if torch.cuda.is_available():
            try:
                free_mem, total_mem = torch.cuda.mem_get_info()
                needed = matrix.element_size() * matrix.nelement() * 3
                if free_mem > needed:
                    target_device = "cuda"
                else:
                    target_device = "cpu"
            except Exception:
                target_device = "cpu"
        else:
            target_device = "cpu"

    original_device = matrix.device
    
    try:
        compute_matrix = matrix.to(target_device) if matrix.device.type != target_device else matrix
        
        is_square = (compute_matrix.ndim == 2 and compute_matrix.shape[0] == compute_matrix.shape[1]) or \
                    (compute_matrix.ndim == 3 and compute_matrix.shape[1] == compute_matrix.shape[2])
                    
        if is_square:
            vals = torch.linalg.eigvalsh(compute_matrix)
        else:
            vals = torch.linalg.svdvals(compute_matrix)
            
        return vals.to(original_device)

    except Exception as e:
        if target_device == "cuda":
            logger.warning(f"GPU spectral decomposition failed ({e}). Falling back to CPU.")
            return compute_eigenvalues_hardware_aware(matrix, device_override="cpu")
        else:
            logger.error(f"Spectral decomposition failed on CPU: {e}")
            raise e

def compute_spectral_metrics(hidden_window: torch.Tensor, device_override: str = None) -> Tuple[Union[float, List[float]], Union[float, List[float]]]:
    """Compute spectral entropy and effective dimension for a window.

    Args:
        hidden_window: Hidden state window shaped (tokens, hidden_dim) or (batch, tokens, hidden_dim).
        device_override: Optional device string ('cpu', 'cuda').

    Returns:
        Tuple[Union[float, List[float]], Union[float, List[float]]]: 
            If unbatched: (spectral_entropy, effective_dimension).
            If batched: ([spectral_entropies], [effective_dimensions]).
    """
    if hidden_window.numel() == 0:
        return 0.0, 0.0
    
    is_batched = hidden_window.dim() == 3
    if not is_batched:
        hidden_window = hidden_window.unsqueeze(0)
        
    x = hidden_window.float()
    if not torch.isfinite(x).all():
        logger.warning("NaN/Inf detected in hidden states. Skipping spectral metrics.")
        return ([0.0] * x.shape[0], [0.0] * x.shape[0]) if is_batched else (0.0, 0.0)

    # Center before forming the covariance so this metric tracks representational spread,
    # not the mean offset of the activations.
    x = x - x.mean(dim=1, keepdim=True)
    covariance = torch.bmm(x.transpose(1, 2), x)
    try:
        eigenvalues = compute_eigenvalues_hardware_aware(covariance, device_override)
    except Exception as exc:
        logger.error("Eigen decomposition failed: %s", exc)
        return ([0.0] * x.shape[0], [0.0] * x.shape[0]) if is_batched else (0.0, 0.0)
        
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    total = eigenvalues.sum(dim=-1, keepdim=True)
    
    mask = (total.squeeze(-1) > 0)
    probabilities = torch.zeros_like(eigenvalues)
    probabilities[mask] = eigenvalues[mask] / total[mask]
    
    spectral_entropies = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=-1)
    
    denom = (eigenvalues ** 2).sum(dim=-1) + 1e-12
    effective_dims = (total.squeeze(-1) ** 2) / denom
    
    spectral_entropies = spectral_entropies * mask.float()
    effective_dims = effective_dims * mask.float()
    
    if is_batched:
        return spectral_entropies.tolist(), effective_dims.tolist()
    return float(spectral_entropies[0].item()), float(effective_dims[0].item())


def compute_shannon_effective_rank(hidden_window: torch.Tensor) -> float:
    """Compute Shannon effective rank via SVD (Roy & Bhattacharyya 2007).
    
    This is the standard metric used in Pythia/GPT-2 spectral analyses.
    EffRank = exp(H(p)) where p_i = σ_i / Σσ_j and H is Shannon entropy.
    
    Unlike participation ratio (which uses eigenvalues squared), this
    uses singular values directly and is less sensitive to small sample sizes.
    
    Args:
        hidden_window: (tokens, hidden_dim) or (batch, tokens, hidden_dim)
    
    Returns:
        float: Shannon effective rank. Compare to hidden_dim for utilization %.
    """
    if hidden_window.numel() == 0:
        return 0.0
    
    if hidden_window.dim() == 3:
        hidden_window = hidden_window[0]  # Take first batch element
    
    x = hidden_window.float().cpu()
    if not torch.isfinite(x).all():
        return 0.0
    
    # Center
    x = x - x.mean(dim=0, keepdim=True)
    
    # SVD
    try:
        _, S, _ = torch.linalg.svd(x, full_matrices=False)
    except RuntimeError:
        return 0.0
    
    # Normalize singular values to probability distribution
    S = S[S > 1e-12]
    if len(S) == 0:
        return 0.0
    
    p = S / S.sum()
    entropy = -(p * torch.log(p)).sum().item()
    
    return float(np.exp(entropy))


def compute_top_eigenvalues(hidden_window: torch.Tensor, k: int) -> Union[List[float], List[List[float]]]:
    """Compute the top-k eigenvalues from a hidden-state covariance matrix.

    Args:
        hidden_window: Hidden state window shaped (tokens, hidden_dim) or (batch, tokens, hidden_dim).
        k: Number of eigenvalues to return.

    Returns:
        Union[List[float], List[List[float]]]: Top-k eigenvalues in descending order.
    """
    if hidden_window.numel() == 0 or k <= 0:
        return []
        
    is_batched = hidden_window.dim() == 3
    if not is_batched:
        hidden_window = hidden_window.unsqueeze(0)
        
    x = hidden_window.float().cpu()
    if not torch.isfinite(x).all():
        logger.warning("NaN/Inf detected in hidden states. Skipping top eigenvalues.")
        return [[] for _ in range(x.shape[0])] if is_batched else []
        
    covariance = torch.bmm(x.transpose(1, 2), x)
    try:
        eigenvalues = torch.linalg.eigvalsh(covariance)
    except RuntimeError as exc:
        logger.error("Eigen decomposition failed for top eigenvalues: %s", exc)
        return [[] for _ in range(x.shape[0])] if is_batched else []
        
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    if eigenvalues.numel() == 0:
        return [[] for _ in range(x.shape[0])] if is_batched else []
        
    topk = torch.topk(eigenvalues, k=min(k, eigenvalues.shape[-1]), dim=-1)
    
    if is_batched:
        return topk.values.tolist()
    return [float(value) for value in topk.values[0].tolist()]


def compute_top_head_idx(attentions: Sequence[torch.Tensor]) -> str:
    """Identify the attention head with the highest variance.

    Args:
        attentions: Attention tensors per layer.

    Returns:
        str: Top head identifier formatted as "L{layer}_H{head}" or "".
    """
    if not attentions:
        return ""
    best_score = -1.0
    best_layer = -1
    best_head = -1
    for layer_idx, layer_attention in enumerate(attentions):
        if layer_attention is None:
            continue
        attn = layer_attention
        if attn.dim() == 4:
            attn = attn[0]
        if attn.numel() == 0:
            continue
        head_scores = torch.nan_to_num(attn.var(dim=(1, 2)), nan=0.0)
        score, head_idx = torch.max(head_scores, dim=0)
        if float(score.item()) > best_score:
            best_score = float(score.item())
            best_layer = layer_idx
            best_head = int(head_idx.item())
    if best_layer < 0:
        return ""
    return f"L{best_layer}_H{best_head}"
