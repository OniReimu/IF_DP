"""Shared batch utilities for vision + language + user-level DP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, Sampler


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested containers to `device`."""

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def unpack_batch(batch_data: Any) -> Tuple[Any, Any, Optional[Any]]:
    """Accept dict/tuple/list batches and return (x, y, user_id or None)."""

    if isinstance(batch_data, dict):
        labels = batch_data.get("labels")
        if labels is None and "label" in batch_data:
            labels = batch_data["label"]
        user_ids = batch_data.get("user_ids", batch_data.get("user_id"))
        feature_keys = [k for k in batch_data.keys() if k not in {"labels", "label", "user_ids", "user_id"}]
        if not feature_keys:
            raise ValueError("Dictionary batch must contain at least one feature field")
        if len(feature_keys) == 1:
            features = batch_data[feature_keys[0]]
        else:
            features = {k: batch_data[k] for k in feature_keys}
        return features, labels, user_ids

    if isinstance(batch_data, (tuple, list)):
        if len(batch_data) < 2:
            raise ValueError("Batch must contain at least (x, y)")
        # pad to length 3
        x = batch_data[0]
        y = batch_data[1]
        uid = batch_data[2] if len(batch_data) >= 3 else None
        return x, y, uid

    raise ValueError(f"Unsupported batch type: {type(batch_data)}")


def prepare_batch(batch_data: Any, device: torch.device) -> Tuple[Any, torch.Tensor, Optional[torch.Tensor]]:
    """Standardise batch structure across datasets and move to device."""

    features, labels, user_ids = unpack_batch(batch_data)
    features = move_to_device(features, device)

    if labels is None:
        raise ValueError("Batch is missing labels")
    labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
    labels = labels.to(device)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    labels = labels.long()

    if user_ids is None:
        return features, labels, None
    user_ids = user_ids if torch.is_tensor(user_ids) else torch.as_tensor(user_ids)
    user_ids = user_ids.to(device)
    if user_ids.ndim == 0:
        user_ids = user_ids.unsqueeze(0)
    return features, labels, user_ids.long()


class SyntheticUserDataset(Dataset):
    """Wrap a base dataset and assign synthetic user ids in round-robin order."""

    def __init__(self, base: Dataset, num_users: int) -> None:
        self.base = base
        self.num_users = max(1, int(num_users))
        self.uid = torch.arange(len(base), dtype=torch.long) % self.num_users

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        uid = int(self.uid[idx].item())

        if isinstance(sample, dict):
            out = dict(sample)
            # HuggingFace convention
            out["user_ids"] = uid
            # normalise label field name for downstream code
            if "labels" not in out and "label" in out:
                out["labels"] = out["label"]
            return out

        if isinstance(sample, (tuple, list)):
            if len(sample) < 2:
                raise ValueError("Base dataset must return (x, y) or dict")
            x, y = sample[0], sample[1]
            return x, y, uid

        raise ValueError(f"Unsupported sample type: {type(sample)}")


class UserBatchSampler(Sampler[List[int]]):
    """Yield batches containing all samples from one user (one user per batch)."""

    def __init__(self, user_ids: Union[Sequence[int], torch.Tensor], shuffle_users: bool = True) -> None:
        if torch.is_tensor(user_ids):
            user_ids = user_ids.cpu().tolist()
        self.user_ids = list(int(u) for u in user_ids)
        self.shuffle_users = shuffle_users

        user_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, uid in enumerate(self.user_ids):
            user_to_indices[uid].append(idx)
        self._users = sorted(user_to_indices.keys())
        self._user_to_indices = user_to_indices

    def __iter__(self) -> Iterator[List[int]]:
        users = torch.tensor(self._users)
        if self.shuffle_users and len(users) > 1:
            users = users[torch.randperm(len(users))]
        for uid in users.tolist():
            batch = self._user_to_indices.get(int(uid), [])
            if batch:
                yield batch

    def __len__(self) -> int:
        return len(self._users)


