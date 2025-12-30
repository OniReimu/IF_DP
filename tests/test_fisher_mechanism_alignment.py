import unittest

import torch

from core.fisher_dp_sgd import maha_clip


class TestFisherMechanismAlignment(unittest.TestCase):
    def test_maha_clip_noop_inside_radius(self):
        # Use an orthonormal basis so we can compute the metric norm exactly.
        U = torch.eye(3)
        scaling = torch.tensor([2.0, 1.0, 0.5])  # e.g., sqrt(λ)
        radius = 1.0

        vec = torch.tensor([0.1, 0.2, 0.3])
        metric_norm = ((U.T @ vec) * scaling).norm().item()
        self.assertLessEqual(metric_norm, radius)

        clipped, reported_norm = maha_clip(vec.clone(), U, scaling, radius)
        self.assertTrue(torch.allclose(clipped, vec))
        self.assertAlmostEqual(reported_norm, metric_norm, places=6)

    def test_maha_clip_enforces_scaled_norm(self):
        U = torch.eye(3)
        scaling = torch.tensor([2.0, 1.0, 0.5])  # e.g., sqrt(λ)
        radius = 1.0

        vec = torch.tensor([10.0, 0.0, 0.0])
        clipped, _ = maha_clip(vec.clone(), U, scaling, radius)

        clipped_metric_norm = ((U.T @ clipped) * scaling).norm().item()
        self.assertLessEqual(clipped_metric_norm, radius + 1e-6)
        self.assertAlmostEqual(clipped_metric_norm, radius, places=5)

    def test_fisher_noise_isotropic_in_whitened_coordinates(self):
        # If alpha_noise = (z * 1/sqrt(λ)) * (σ * C_F), then whitening gives:
        #   sqrt(λ) * alpha_noise = z * (σ * C_F)
        lam = torch.tensor([0.5, 2.0, 10.0])
        inv_sqrt_lam = lam.rsqrt()
        sqrt_lam = lam.sqrt()

        sigma = 1.7
        radius = 0.8
        z = torch.randn_like(lam)

        alpha_noise = (z * inv_sqrt_lam) * sigma * radius
        whitened = sqrt_lam * alpha_noise
        expected = z * sigma * radius

        self.assertTrue(torch.allclose(whitened, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()

