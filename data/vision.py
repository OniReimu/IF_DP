"""Vision dataset builders (torchvision)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

from .base import (
    DatasetBuilder,
    DatasetConfig,
    DatasetLoaders,
    split_private_public_calibration_indices,
)
from .common import SyntheticUserDataset, UserBatchSampler
from .registry import register_dataset


def _build_private_loader(private_subset: Dataset, config: DatasetConfig) -> DataLoader:
    if config.sample_level:
        return DataLoader(private_subset, batch_size=config.batch_size, shuffle=True)
    synthetic = SyntheticUserDataset(private_subset, config.num_users)
    sampler = UserBatchSampler(synthetic.uid)
    return DataLoader(synthetic, batch_sampler=sampler)


@dataclass
class _VisionMeta:
    task_type: str
    num_labels: int
    label_mapping: Optional[Dict[int, str]] = None


class _TorchvisionClassificationBuilder(DatasetBuilder):
    dataset_cls = None  # type: ignore[assignment]
    dataset_kwargs = None  # type: ignore[assignment]
    meta: _VisionMeta

    def __init__(self) -> None:
        if self.dataset_kwargs is None:
            self.dataset_kwargs = {}

    @property
    def task_type(self) -> str:
        return self.meta.task_type

    @property
    def num_labels(self) -> int:
        return self.meta.num_labels

    def get_label_mapping(self) -> Optional[Dict[int, str]]:
        return self.meta.label_mapping

    def build(self, config: DatasetConfig) -> DatasetLoaders:
        if self.dataset_cls is None:
            raise ValueError("dataset_cls is not set")

        transform_train = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        trainset = self.dataset_cls(
            root=config.dataset_root,
            train=True,
            download=config.allow_download,
            transform=transform_train,
            **self.dataset_kwargs,
        )
        testset = self.dataset_cls(
            root=config.dataset_root,
            train=False,
            download=config.allow_download,
            transform=transform_test,
            **self.dataset_kwargs,
        )

        priv_idx, pub_idx, calib_idx = split_private_public_calibration_indices(
            len(trainset),
            config.dataset_size,
            config.calibration_size,
            config.public_ratio,
            config.seed,
        )
        private_subset = Subset(trainset, priv_idx.tolist())
        public_subset = Subset(trainset, pub_idx.tolist())
        calibration_subset = Subset(trainset, calib_idx.tolist())

        priv_loader = _build_private_loader(private_subset, config)
        pub_loader = DataLoader(public_subset, batch_size=config.batch_size, shuffle=True)
        calib_loader = DataLoader(calibration_subset, batch_size=config.batch_size, shuffle=True)
        eval_loader = DataLoader(testset, batch_size=config.eval_batch_size, shuffle=False)

        # For now, critical_eval mirrors evaluation; downstream code can create slices.
        crit_loader = eval_loader

        return DatasetLoaders(
            private=priv_loader,
            public=pub_loader,
            calibration=calib_loader,
            evaluation=eval_loader,
            critical_eval=crit_loader,
            private_base=private_subset,
            private_indices=priv_idx,
        )


@register_dataset("cifar10")
class CIFAR10Builder(_TorchvisionClassificationBuilder):
    dataset_cls = torchvision.datasets.CIFAR10
    meta = _VisionMeta(task_type="vision", num_labels=10)

    def build(self, config: DatasetConfig) -> DatasetLoaders:
        loaders = super().build(config)
        # Extract class names from underlying dataset (trainset lives inside Subset)
        base = loaders.public.dataset
        dataset = base.dataset if hasattr(base, "dataset") else base
        classes = getattr(dataset, "classes", None)
        if classes:
            self.meta.label_mapping = {i: str(name) for i, name in enumerate(classes)}
        return loaders


@register_dataset("cifar100")
class CIFAR100Builder(_TorchvisionClassificationBuilder):
    dataset_cls = torchvision.datasets.CIFAR100
    meta = _VisionMeta(task_type="vision", num_labels=100)

    def build(self, config: DatasetConfig) -> DatasetLoaders:
        loaders = super().build(config)
        base = loaders.public.dataset
        dataset = base.dataset if hasattr(base, "dataset") else base
        classes = getattr(dataset, "classes", None)
        if classes:
            self.meta.label_mapping = {i: str(name) for i, name in enumerate(classes)}
        return loaders


@register_dataset("fashion_mnist")
class FashionMNISTBuilder(DatasetBuilder):
    task_type = "vision"
    num_labels = 10

    def get_label_mapping(self) -> Optional[Dict[int, str]]:
        # torchvision uses indices 0..9; names are stable
        return {
            0: "t-shirt/top",
            1: "trouser",
            2: "pullover",
            3: "dress",
            4: "coat",
            5: "sandal",
            6: "shirt",
            7: "sneaker",
            8: "bag",
            9: "ankle boot",
        }

    def build(self, config: DatasetConfig) -> DatasetLoaders:
        # FashionMNIST is 28x28 grayscale; the vision models in this repo expect 3x32x32.
        # We upsample to 32 and repeat channels to keep the training pipeline consistent.
        transform_train = T.Compose(
            [
                T.RandomCrop(28, padding=4),
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1xHxW -> 3xHxW
                T.Resize((32, 32)),
            ]
        )
        transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                T.Resize((32, 32)),
            ]
        )

        trainset = torchvision.datasets.FashionMNIST(
            root=config.dataset_root, train=True, download=config.allow_download, transform=transform_train
        )
        testset = torchvision.datasets.FashionMNIST(
            root=config.dataset_root, train=False, download=config.allow_download, transform=transform_test
        )

        priv_idx, pub_idx, calib_idx = split_private_public_calibration_indices(
            len(trainset),
            config.dataset_size,
            config.calibration_size,
            config.public_ratio,
            config.seed,
        )
        private_subset = Subset(trainset, priv_idx.tolist())
        public_subset = Subset(trainset, pub_idx.tolist())
        calibration_subset = Subset(trainset, calib_idx.tolist())

        priv_loader = _build_private_loader(private_subset, config)
        pub_loader = DataLoader(public_subset, batch_size=config.batch_size, shuffle=True)
        calib_loader = DataLoader(calibration_subset, batch_size=config.batch_size, shuffle=True)
        eval_loader = DataLoader(testset, batch_size=config.eval_batch_size, shuffle=False)

        return DatasetLoaders(
            private=priv_loader,
            public=pub_loader,
            calibration=calib_loader,
            evaluation=eval_loader,
            critical_eval=eval_loader,
            private_base=private_subset,
            private_indices=priv_idx,
        )
