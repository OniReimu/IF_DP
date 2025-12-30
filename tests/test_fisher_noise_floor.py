import unittest

import torch

from core.fisher_dp_sgd import apply_noise_floor


class TestFisherNoiseFloor(unittest.TestCase):
    def test_apply_noise_floor_clamps(self):
        scaling = torch.tensor([0.5, 1.0, 2.0])
        out = apply_noise_floor(scaling, floor=1.0)
        expected = torch.tensor([1.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
