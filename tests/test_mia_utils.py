import unittest

import torch
from torch.utils.data import Dataset

from core.mia import (
    align_mia_datasets,
    prepare_shadow_splits,
    user_level_loss_attack,
    user_level_shadow_attack,
)
from data.common import SyntheticUserDataset


class DummyTransformDataset(Dataset):
    def __init__(self, n, transform=None):
        self.transform = transform
        self.data = [torch.tensor(float(i)) for i in range(n)]
        self.targets = [0 for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.targets[idx]


class ConstantModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.num_classes, device=x.device)


class TestMiaUtilities(unittest.TestCase):
    def test_prepare_shadow_splits(self):
        train = DummyTransformDataset(100)
        eval_data = DummyTransformDataset(80)
        splits = prepare_shadow_splits(train, eval_data, seed=123)

        self.assertEqual(len(splits["shadow_indices"]), 50)
        self.assertEqual(len(splits["shadow_non_member_indices"]), 50)
        self.assertEqual(splits["non_member_source"], "eval")
        self.assertTrue(max(splits["shadow_indices"]) < 100)
        self.assertTrue(max(splits["shadow_non_member_indices"]) < 80)

    def test_align_mia_datasets(self):
        train = DummyTransformDataset(5, transform=lambda x: x + 1)
        eval_data = DummyTransformDataset(5, transform=lambda x: x + 10)
        priv_ds = SyntheticUserDataset(train, num_users=2)

        train_aligned, priv_aligned, aligned = align_mia_datasets(train, priv_ds, eval_data, 2)
        self.assertTrue(aligned)
        sample = train_aligned[0]
        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        self.assertAlmostEqual(float(x.item()), 10.0)

        priv_sample = priv_aligned[0]
        self.assertEqual(len(priv_sample), 3)
        self.assertAlmostEqual(float(priv_sample[0].item()), 10.0)

    def test_user_level_loss_attack(self):
        dataset = DummyTransformDataset(4, transform=None)
        member_groups = [[0, 1]]
        non_member_groups = [[2, 3]]
        model = ConstantModel()

        result = user_level_loss_attack(
            model,
            member_groups,
            non_member_groups,
            dataset,
            dataset,
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(result["auc"], 0.5, places=6)

    def test_user_level_shadow_attack_empty(self):
        model = ConstantModel()
        result = user_level_shadow_attack(
            model,
            member_groups=[],
            non_member_groups=[],
            priv_ds=None,
            eval_user_ds=None,
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(result["auc"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
