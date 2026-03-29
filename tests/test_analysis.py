import unittest

import torch

from prism.analysis import compute_spectral_metrics


class TestAnalysis(unittest.TestCase):
    def test_compute_spectral_metrics_unbatched(self) -> None:
        hidden_window = torch.randn(16, 32)
        spectral_entropy, effective_dim = compute_spectral_metrics(hidden_window)

        self.assertIsInstance(spectral_entropy, float)
        self.assertIsInstance(effective_dim, float)
        self.assertTrue(torch.isfinite(torch.tensor(spectral_entropy)).item())
        self.assertTrue(torch.isfinite(torch.tensor(effective_dim)).item())
        self.assertGreaterEqual(effective_dim, 0.0)

    def test_compute_spectral_metrics_batched(self) -> None:
        hidden_window = torch.randn(2, 16, 32)
        spectral_entropies, effective_dims = compute_spectral_metrics(hidden_window)

        self.assertEqual(len(spectral_entropies), 2)
        self.assertEqual(len(effective_dims), 2)
        for value in spectral_entropies + effective_dims:
            self.assertTrue(torch.isfinite(torch.tensor(value)).item())


if __name__ == "__main__":
    unittest.main()

