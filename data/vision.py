"""Vision dataset builders (torchvision)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from config import get_logger, REHEARSAL_MAX_EXCLUDED_CLASS_RATIO
from .common import SyntheticUserDataset, UserBatchSampler
from .base import (
    DatasetBuilder,
    DatasetConfig,
    DatasetLoaders,
    split_private_public_calibration_indices,
)
from .registry import register_dataset

logger = get_logger("data")


def _maybe_distributed_sampler(dataset, *, shuffle: bool, drop_last: bool = False):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    return None


def _build_loader(dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    sampler = _maybe_distributed_sampler(dataset, shuffle=shuffle, drop_last=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
    )


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
        
        # Determine split mode
        non_iid = getattr(config, "non_iid", False)
        if non_iid:
            logger.info("Non-IID dataset split mode enabled.")
        else:
            logger.info("IID dataset split mode (default).")
        
        # Optional non-IID simulation:
        # - Exclude some class labels from PUBLIC PRETRAIN only
        # - Move the removed public-pretrain samples into PRIVATE
        # - Calibration and evaluation remain unchanged
        # Only apply non-IID logic if explicitly enabled
        exclude = getattr(config, "public_pretrain_exclude_classes", None)
        if non_iid and exclude:
            exclude_set = {int(x) for x in exclude}
            targets = getattr(trainset, "targets", None)
            if targets is None:
                targets = getattr(trainset, "labels", None)
            if targets is None:
                raise RuntimeError(
                    "public_pretrain_exclude_classes requested, but dataset has no 'targets'/'labels' attribute."
                )
            targets_arr = np.asarray(targets, dtype=int)

            pub_idx_arr = np.asarray(pub_idx, dtype=int)
            priv_idx_arr = np.asarray(priv_idx, dtype=int)
            calib_idx_arr = np.asarray(calib_idx, dtype=int)

            pub_targets = targets_arr[pub_idx_arr]
            remove_mask = np.isin(pub_targets, list(exclude_set))
            removed_from_public = pub_idx_arr[remove_mask]
            pub_idx_filtered = pub_idx_arr[~remove_mask]

            # Move removed public samples into private (private size increases).
            if removed_from_public.size > 0:
                priv_idx_arr = np.concatenate([priv_idx_arr, removed_from_public.astype(int)], axis=0)

            def _count_for(indices: np.ndarray, cls: int) -> int:
                if indices.size == 0:
                    return 0
                return int(np.sum(targets_arr[indices] == int(cls)))

            # ============================================================
            # REHEARSAL BUFFER: Add samples from non-excluded classes to private
            # to prevent catastrophic forgetting (keep excluded classes ≤ 50% of private)
            # ============================================================
            # Identify rehearsal classes (all classes NOT in excluded set)
            all_classes = set(np.unique(targets_arr))
            rehearsal_classes = sorted(all_classes - exclude_set)
            
            # Count excluded-class samples in private
            excluded_in_private = sum(_count_for(priv_idx_arr, cls) for cls in exclude_set)
            total_private = len(priv_idx_arr)
            
            # Calculate how many rehearsal samples we need
            if total_private > 0 and excluded_in_private / total_private > REHEARSAL_MAX_EXCLUDED_CLASS_RATIO:
                # Target: excluded_classes / (total_private + rehearsal_needed) ≤ REHEARSAL_MAX_EXCLUDED_CLASS_RATIO
                # => rehearsal_needed ≥ excluded_in_private / REHEARSAL_MAX_EXCLUDED_CLASS_RATIO - total_private
                rehearsal_needed = int(np.ceil(
                    excluded_in_private / REHEARSAL_MAX_EXCLUDED_CLASS_RATIO - total_private
                ))
                
                if rehearsal_needed > 0 and len(rehearsal_classes) > 0:
                    # Find rehearsal-class samples in public pretrain
                    pub_targets_filtered = targets_arr[pub_idx_filtered]
                    rehearsal_mask = np.isin(pub_targets_filtered, list(rehearsal_classes))
                    rehearsal_candidates = pub_idx_filtered[rehearsal_mask]
                    
                    if len(rehearsal_candidates) >= rehearsal_needed:
                        # Take exactly what we need (randomly if more available)
                        if len(rehearsal_candidates) > rehearsal_needed:
                            rng = np.random.RandomState(int(config.seed))
                            selected_indices = rng.choice(
                                len(rehearsal_candidates), 
                                size=rehearsal_needed, 
                                replace=False
                            )
                            rehearsal_selected = rehearsal_candidates[selected_indices]
                        else:
                            rehearsal_selected = rehearsal_candidates
                        
                        # Move rehearsal samples from public to private
                        priv_idx_arr = np.concatenate([priv_idx_arr, rehearsal_selected.astype(int)], axis=0)
                        pub_idx_filtered = pub_idx_filtered[~np.isin(pub_idx_filtered, rehearsal_selected)]
                        
                        logger.info(
                            "Rehearsal buffer: added %s samples from classes %s",
                            len(rehearsal_selected),
                            rehearsal_classes,
                        )
                        logger.info(
                            "   • Goal: keep excluded classes ≤ %.0f%% of private",
                            REHEARSAL_MAX_EXCLUDED_CLASS_RATIO * 100,
                        )
                        logger.info(
                            "   • Before: excluded=%s/%s (%.1f%%)",
                            excluded_in_private,
                            total_private,
                            excluded_in_private / total_private * 100,
                        )
                        excluded_after = sum(_count_for(priv_idx_arr, cls) for cls in exclude_set)
                        logger.info(
                            "   • After:  excluded=%s/%s (%.1f%%)",
                            excluded_after,
                            len(priv_idx_arr),
                            excluded_after / len(priv_idx_arr) * 100,
                        )
                    else:
                        logger.warn(
                            "Rehearsal buffer: requested %s samples but only %s available in public pretrain",
                            rehearsal_needed,
                            len(rehearsal_candidates),
                        )
                        if len(rehearsal_candidates) > 0:
                            # Take all available
                            priv_idx_arr = np.concatenate([priv_idx_arr, rehearsal_candidates.astype(int)], axis=0)
                            pub_idx_filtered = pub_idx_filtered[~np.isin(pub_idx_filtered, rehearsal_candidates)]
                            logger.info("   • Added %s available samples", len(rehearsal_candidates))

            logger.info("Non-IID simulation: exclude classes from public pretrain and move to private.")
            logger.info("   • Excluded classes: %s", sorted(exclude_set))
            logger.info(
                "   • Public pretrain size: %s -> %s (moved %s)",
                len(pub_idx_arr),
                len(pub_idx_filtered),
                len(removed_from_public),
            )
            logger.info(
                "   • Private size: %s -> %s (calibration unchanged: %s)",
                len(priv_idx),
                len(priv_idx_arr),
                len(calib_idx_arr),
            )
            for cls in sorted(exclude_set):
                logger.info(
                    "   • class %s: public=%s private=%s calib=%s",
                    cls,
                    _count_for(pub_idx_filtered, cls),
                    _count_for(priv_idx_arr, cls),
                    _count_for(calib_idx_arr, cls),
                )

            # Replace indices with our non-IID construction.
            priv_idx = priv_idx_arr
            pub_idx = pub_idx_filtered
        private_subset = Subset(trainset, priv_idx.tolist())
        public_subset = Subset(trainset, pub_idx.tolist())
        calibration_subset = Subset(trainset, calib_idx.tolist())

        priv_loader = _build_private_loader(private_subset, config)
        pub_loader = _build_loader(public_subset, batch_size=config.batch_size, shuffle=True)
        calib_loader = _build_loader(calibration_subset, batch_size=config.batch_size, shuffle=True)
        eval_loader = _build_loader(testset, batch_size=config.eval_batch_size, shuffle=False)

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
        pub_loader = _build_loader(public_subset, batch_size=config.batch_size, shuffle=True)
        calib_loader = _build_loader(calibration_subset, batch_size=config.batch_size, shuffle=True)
        eval_loader = _build_loader(testset, batch_size=config.eval_batch_size, shuffle=False)

        return DatasetLoaders(
            private=priv_loader,
            public=pub_loader,
            calibration=calib_loader,
            evaluation=eval_loader,
            critical_eval=eval_loader,
            private_base=private_subset,
            private_indices=priv_idx,
        )
