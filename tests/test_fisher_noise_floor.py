import unittest

import torch

from core.fisher_dp_sgd import apply_noise_floor, maybe_apply_noise_floor


class TestFisherNoiseFloor(unittest.TestCase):
    def test_apply_noise_floor_clamps(self):
        scaling = torch.tensor([0.5, 1.0, 2.0])
        out = apply_noise_floor(scaling, floor=1.0)
        expected = torch.tensor([1.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(out, expected))

    def test_maybe_apply_noise_floor_disabled(self):
        scaling = torch.tensor([0.5, 1.0, 2.0])
        out = maybe_apply_noise_floor(scaling, full_complement_noise=False, floor=1.0)
        self.assertTrue(torch.allclose(out, scaling))

    def test_maybe_apply_noise_floor_enabled(self):
        scaling = torch.tensor([0.5, 1.0, 2.0])
        out = maybe_apply_noise_floor(scaling, full_complement_noise=True, floor=1.0)
        expected = torch.tensor([1.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
