"""Random subspace principal-angle baseline.

This script samples pairs of random k-dimensional subspaces in R^d, computes their
principal angles, and summarizes how many angles fall below a fixed threshold.

Output:
- Prints summary statistics over all sampled pairs.
- Saves a histogram plot to the requested path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch


def _setup_logging() -> None:
    """Configure basic logging for the script."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _orthonormal_basis(x: torch.Tensor) -> torch.Tensor:
    """Compute an orthonormal basis for the column-span of x via QR.

    Args:
        x: A tensor of shape [d, k].

    Returns:
        A tensor Q of shape [d, k] with orthonormal columns.
    """

    q, _ = torch.linalg.qr(x, mode="reduced")
    return q


def _principal_angles_degrees(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute principal angles (in degrees) between two subspaces.

    Uses the singular values of q1^T q2.

    Args:
        q1: Orthonormal basis [d, k].
        q2: Orthonormal basis [d, k].

    Returns:
        Principal angles as a 1D tensor of length k, in degrees.
    """

    m = q1.T @ q2
    # Singular values are cos(theta) for principal angles theta.
    s = torch.linalg.svdvals(m)
    s = torch.clamp(s, -1.0, 1.0)
    angles_rad = torch.acos(s)
    angles_deg = angles_rad * (180.0 / torch.pi)
    return angles_deg


def run(
    *,
    d: int = 576,
    k: int = 185,
    num_pairs: int = 1000,
    angle_threshold_deg: float = 30.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the random-subspace baseline experiment.

    Args:
        d: Ambient dimension.
        k: Subspace dimension.
        num_pairs: Number of random subspace pairs to sample.
        angle_threshold_deg: Threshold for counting principal angles.
        seed: RNG seed for reproducibility.
        device: Torch device string.

    Returns:
        A tuple (counts, all_angles_deg):
        - counts: shape [num_pairs], number of angles < threshold per pair.
        - all_angles_deg: shape [num_pairs * k], all principal angles in degrees.
    """

    if k > d:
        raise ValueError(f"k must be <= d, got k={k}, d={d}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    counts = np.empty((num_pairs,), dtype=np.int64)
    all_angles = np.empty((num_pairs, k), dtype=np.float64)

    dev = torch.device(device)

    for i in range(num_pairs):
        x1 = torch.randn((d, k), generator=gen, device=dev, dtype=torch.float32)
        x2 = torch.randn((d, k), generator=gen, device=dev, dtype=torch.float32)

        q1 = _orthonormal_basis(x1)
        q2 = _orthonormal_basis(x2)

        angles_deg = _principal_angles_degrees(q1, q2)
        all_angles[i, :] = angles_deg.detach().cpu().double().numpy()

        counts[i] = int((angles_deg < angle_threshold_deg).sum().item())

        if (i + 1) % 100 == 0:
            logging.info("Processed %d/%d pairs", i + 1, num_pairs)

    return counts, all_angles.reshape(-1)


def save_histogram(
    *,
    angles_deg: np.ndarray,
    output_path: Path,
    bins: int = 60,
) -> None:
    """Save a histogram plot of principal angles.

    Args:
        angles_deg: Array of principal angles in degrees.
        output_path: Path to save the PNG histogram.
        bins: Number of histogram bins.

    Returns:
        None.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logging.exception("Failed to import matplotlib; cannot save histogram.")
        raise

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.exception("Failed to create output directory: %s", output_path.parent)
        raise

    plt.figure(figsize=(10, 6))
    plt.hist(angles_deg, bins=bins)
    plt.xlabel("Principal angle (degrees)")
    plt.ylabel("Count")
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=200)
    except Exception:
        logging.exception("Failed to save histogram to: %s", output_path)
        raise
    finally:
        plt.close()


def main() -> None:
    """CLI entrypoint."""

    _setup_logging()

    counts, all_angles_deg = run()

    mean = float(np.mean(counts))
    std = float(np.std(counts, ddof=1))
    p5 = float(np.percentile(counts, 5))
    p95 = float(np.percentile(counts, 95))

    logging.info("Random subspace baseline (num_pairs=%d)", counts.shape[0])
    logging.info("Count of principal angles < 30° per pair (k=185)")
    logging.info("Mean: %.3f", mean)
    logging.info("Std:  %.3f", std)
    logging.info("5th percentile:  %.3f", p5)
    logging.info("95th percentile: %.3f", p95)

    output_path = Path(r"D:\Genesis\logs\random_baseline_principal_angles.png")
    save_histogram(angles_deg=all_angles_deg, output_path=output_path)
    logging.info("Saved histogram: %s", output_path)


if __name__ == "__main__":
    main()
