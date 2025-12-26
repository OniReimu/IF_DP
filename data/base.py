"""Dataset builder abstractions used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetConfig:
    dataset_root: str
    allow_download: bool = True
    dataset_size: int = 5000  # number of private samples (simulation setup)
    public_ratio: float = 1.0  # fraction of remaining samples used as public
    calibration_size: int = 5000  # number of public samples reserved for calibration
    batch_size: int = 128
    eval_batch_size: int = 256
    sample_level: bool = True
    num_users: int = 10
    critical_label: Optional[int] = None
    tokenizer_name: str = "bert-base-uncased"
    max_seq_length: int = 512
    # Optional non-IID simulation: exclude specific class labels from the public pretrain split only.
    # Private/calibration/evaluation splits remain unchanged.
    public_pretrain_exclude_classes: Optional[Sequence[int]] = None
    non_iid: bool = False  # If True, enable non-IID split (requires public_pretrain_exclude_classes)
    seed: int = 0


@dataclass
class DatasetLoaders:
    private: DataLoader
    public: DataLoader
    calibration: DataLoader
    evaluation: DataLoader
    critical_eval: DataLoader
    private_base: Dataset
    private_indices: Any


class DatasetBuilder:
    """Base class for datasets supporting a consistent split/build API."""

    task_type: str = "vision"
    num_labels: int = 0

    def get_label_mapping(self) -> Optional[Dict[int, str]]:
        return None

    def build(self, config: DatasetConfig) -> DatasetLoaders:  # pragma: no cover
        raise NotImplementedError


def split_private_public_calibration_indices(
    total_size: int,
    private_size: int,
    calibration_size: int,
    public_ratio: float,
    seed: int,
):
    """Split dataset indices into disjoint private/public/calibration subsets."""

    private_size = int(private_size)
    if private_size <= 0:
        raise ValueError("--dataset-size must be > 0 (number of private samples)")
    if private_size >= total_size:
        raise ValueError(f"--dataset-size ({private_size}) must be < training set size ({total_size})")

    calibration_size = int(calibration_size)
    if calibration_size < 0:
        raise ValueError("--calibration-size must be >= 0")
    if private_size + calibration_size >= total_size:
        raise ValueError(
            f"private ({private_size}) + calibration ({calibration_size}) must be < training set size ({total_size})"
        )

    public_ratio = float(public_ratio)
    if not (0.0 <= public_ratio <= 1.0):
        raise ValueError("--public-ratio must be in [0, 1]")

    # Use a seeded permutation so splits match the randomized strategy.
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(total_size)

    # Match the saber branch slicing order:
    #   pretrain/public first, then calibration, then private last.
    pretrain_size = total_size - private_size - calibration_size
    pub_pool = perm[:pretrain_size]
    calib_idx = perm[pretrain_size : pretrain_size + calibration_size]
    priv_idx = perm[pretrain_size + calibration_size :]

    public_take = int(round(len(pub_pool) * public_ratio))
    pub_idx = pub_pool[:public_take]

    return priv_idx, pub_idx, calib_idx
