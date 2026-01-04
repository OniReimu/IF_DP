"""Text dataset builders powered by HuggingFace datasets/tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from .common import SyntheticUserDataset, UserBatchSampler
from .base import (
    DatasetBuilder,
    DatasetConfig,
    DatasetLoaders,
    split_private_public_calibration_indices,
)
from .registry import register_dataset


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



class HFTextClassificationDataset(Dataset):
    """Lightweight Dataset wrapper that tokenizes samples on the fly."""

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        text_fields: Sequence[str],
        max_length: int,
    ) -> None:
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_fields = tuple(text_fields)
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.dataset)

    def _join_fields(self, record: Dict) -> str:
        pieces = []
        for field in self.text_fields:
            value = record.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                pieces.append(value)
            elif isinstance(value, (list, tuple)):
                pieces.append(" ".join(str(v) for v in value))
            else:
                pieces.append(str(value))
        text = " ".join(piece.strip() for piece in pieces if piece).strip()
        return text or ""

    def __getitem__(self, idx: int):
        record = self.dataset[idx]
        text = self._join_fields(record)
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        features = {k: v.squeeze(0) for k, v in encoded.items()}
        label_key = "labels" if "labels" in record else "label"
        label_value = record[label_key]
        features["labels"] = torch.tensor(int(label_value), dtype=torch.long)
        return features


@dataclass
class _TextMeta:
    task_type: str
    num_labels: int
    text_fields: Sequence[str]
    label_mapping: Optional[Dict[int, str]] = None


class _HFTextDatasetBuilder(DatasetBuilder):
    """Base builder that handles splitting/tokenization for HF corpora."""

    dataset_name: str = ""
    eval_split: str = "test"
    meta: _TextMeta

    def get_label_mapping(self) -> Optional[Dict[int, str]]:
        return self.meta.label_mapping

    @property
    def task_type(self) -> str:
        return self.meta.task_type

    @property
    def num_labels(self) -> int:
        return self.meta.num_labels

    def _load_hf_dataset(self, split: str, cache_dir: str):
        return load_dataset(self.dataset_name, split=split, cache_dir=cache_dir)

    def build(self, config: DatasetConfig) -> DatasetLoaders:
        if not self.dataset_name:
            raise ValueError("dataset_name must be defined for text datasets")

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        train_hf = self._load_hf_dataset("train", cache_dir=config.dataset_root)
        test_hf = self._load_hf_dataset(self.eval_split, cache_dir=config.dataset_root)

        train_dataset = HFTextClassificationDataset(
            train_hf,
            tokenizer,
            self.meta.text_fields,
            config.max_seq_length,
        )
        eval_dataset = HFTextClassificationDataset(
            test_hf,
            tokenizer,
            self.meta.text_fields,
            config.max_seq_length,
        )

        priv_idx, pub_idx, calib_idx = split_private_public_calibration_indices(
            len(train_dataset),
            config.dataset_size,
            config.calibration_size,
            config.public_ratio,
            config.seed,
        )

        private_subset = Subset(train_dataset, priv_idx.tolist())
        public_subset = Subset(train_dataset, pub_idx.tolist())
        calibration_subset = Subset(train_dataset, calib_idx.tolist())

        priv_loader = _build_private_loader(private_subset, config)
        pub_loader = _build_loader(public_subset, batch_size=config.batch_size, shuffle=True)
        calib_loader = _build_loader(calibration_subset, batch_size=config.batch_size, shuffle=True)
        eval_loader = _build_loader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)

        return DatasetLoaders(
            private=priv_loader,
            public=pub_loader,
            calibration=calib_loader,
            evaluation=eval_loader,
            critical_eval=eval_loader,
            private_base=private_subset,
            private_indices=priv_idx,
        )


@register_dataset("ag_news")
class AGNewsBuilder(_HFTextDatasetBuilder):
    dataset_name = "ag_news"
    meta = _TextMeta(
        task_type="text",
        num_labels=4,
        text_fields=("text",),
        label_mapping={
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        },
    )


@register_dataset("dbpedia_ontology")
class DBPediaOntologyBuilder(_HFTextDatasetBuilder):
    dataset_name = "dbpedia_14"
    meta = _TextMeta(
        task_type="text",
        num_labels=14,
        text_fields=("content", "title"),
        label_mapping={
            0: "Company",
            1: "EducationalInstitution",
            2: "Artist",
            3: "Athlete",
            4: "OfficeHolder",
            5: "MeanOfTransportation",
            6: "Building",
            7: "NaturalPlace",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "WrittenWork",
        },
    )
