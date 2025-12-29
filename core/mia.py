#!/usr/bin/env python3
# ================================================================
# Membership Inference Attacks (MIA) on CIFAR-10 Models
#    * Yeom et al. confidence-based attack
#    * Evaluation on baseline vs DP models
#    * Supports both user-level and sample-level DP
# ================================================================

import os, argparse, copy
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

from data.common import SyntheticUserDataset, prepare_batch
from models.model import CNN
from config import get_logger

def auc_star(auc: float) -> float:
    """Sign-invariant attack strength: attacker can flip score direction."""
    return float(max(auc, 1.0 - auc))

def auc_advantage(auc: float) -> float:
    """Membership advantage over random guessing (0.0 is best)."""
    return float(abs(auc - 0.5))

def _mia_fallback(reason: str) -> None:
    logger.warn("MIA fallback: %s", reason)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set reproducible random seeds for consistent MIA evaluation
from config import get_dataset_location, set_random_seeds
from .device_utils import resolve_device
set_random_seeds()
dataset_root, allow_download = get_dataset_location(
    dataset_key='cifar10',
    required_subdir='cifar-10-batches-py'
)

NUM_RUNS = 5 
logger = get_logger("mia")


class _TransformOverrideDataset(Dataset):
    """Wrap a dataset but force a specific transform during __getitem__."""

    def __init__(self, base: Dataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        if not hasattr(self.base, "transform"):
            return self.base[idx]
        original = getattr(self.base, "transform")
        setattr(self.base, "transform", self.transform)
        try:
            return self.base[idx]
        finally:
            setattr(self.base, "transform", original)


def _get_dataset_transform(dataset) -> object:
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    return getattr(dataset, "transform", None)


def _apply_transform_override(dataset, transform):
    if transform is None:
        return dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if not hasattr(base, "transform"):
            return dataset
        wrapped = _TransformOverrideDataset(base, transform)
        return Subset(wrapped, dataset.indices)
    if hasattr(dataset, "transform"):
        return _TransformOverrideDataset(dataset, transform)
    return dataset


def align_mia_datasets(train_data, priv_ds, eval_data, num_users):
    """Align member/non-member transforms to eval_data when possible."""
    eval_transform = _get_dataset_transform(eval_data)
    if eval_transform is None:
        return train_data, priv_ds, False

    train_data_aligned = _apply_transform_override(train_data, eval_transform)
    priv_ds_aligned = priv_ds
    if priv_ds is not None and hasattr(priv_ds, "base"):
        base_aligned = _apply_transform_override(priv_ds.base, eval_transform)
        priv_ds_aligned = SyntheticUserDataset(base_aligned, num_users)
    return train_data_aligned, priv_ds_aligned, True


def prepare_shadow_splits(train_data, eval_data, seed=None, shadow_size=None):
    """Prepare a fixed shadow split (members/non-members) for fair comparisons."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    shadow_size = shadow_size or min(len(train_data) // 2, 2000)
    shadow_indices = rng.choice(len(train_data), shadow_size, replace=False)
    if eval_data is not None:
        shadow_non_member_size = min(len(eval_data), shadow_size)
        shadow_non_member_indices = rng.choice(len(eval_data), shadow_non_member_size, replace=False)
        source = "eval"
    else:
        remaining_indices = np.setdiff1d(np.arange(len(train_data)), shadow_indices)
        shadow_non_member_indices = remaining_indices[:shadow_size]
        source = "train_fallback"
    return {
        "shadow_indices": shadow_indices,
        "shadow_non_member_indices": shadow_non_member_indices,
        "shadow_size": shadow_size,
        "non_member_source": source,
    }


def _extract_user_id(sample) -> int:
    if isinstance(sample, dict):
        uid_value = sample.get("user_ids", sample.get("user_id"))
        if isinstance(uid_value, torch.Tensor):
            return int(uid_value.item())
        return int(uid_value) if uid_value is not None else 0
    if isinstance(sample, (list, tuple)) and len(sample) >= 3:
        uid_value = sample[2]
        if isinstance(uid_value, torch.Tensor):
            return int(uid_value.item())
        return int(uid_value)
    return 0


def _extract_label(sample) -> int:
    if isinstance(sample, dict):
        label_value = sample.get("labels", sample.get("label"))
    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
        label_value = sample[1]
    else:
        raise ValueError("Sample is missing label information for MIA matching.")

    if isinstance(label_value, torch.Tensor):
        label_value = label_value.item()
    return int(label_value)


def _collect_label_counts(dataset, indices):
    counts = defaultdict(int)
    for idx in indices:
        label = _extract_label(dataset[idx])
        counts[label] += 1
    return counts


def _build_label_index_map(dataset):
    label_map = defaultdict(list)
    for idx in range(len(dataset)):
        label = _extract_label(dataset[idx])
        label_map[label].append(idx)
    return label_map


def _sample_indices_by_label(dataset, target_counts):
    target_total = sum(target_counts.values()) if target_counts else 0
    if target_total == 0:
        return []

    label_map = _build_label_index_map(dataset)
    all_indices = list(range(len(dataset)))
    selected = []

    for label, count in target_counts.items():
        pool = label_map.get(label, [])
        if not pool:
            logger.warn("MIA non-member matching: no eval samples for label %s; using full eval pool.", label)
            pool = all_indices
        replace = len(pool) < count
        if replace:
            logger.warn(
                "MIA non-member matching: eval label %s has %s < %s; sampling with replacement.",
                label,
                len(pool),
                count,
            )
        chosen = np.random.choice(pool, count, replace=replace).tolist()
        selected.extend(chosen)

    np.random.shuffle(selected)
    return selected


def _collect_user_groups(dataset):
    user_samples = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        uid = _extract_user_id(sample)
        user_samples[uid].append(idx)
    return user_samples


def _excluded_fraction(dataset, indices, excluded_set):
    if not indices:
        return 0.0
    excluded = 0
    for idx in indices:
        label = _extract_label(dataset[idx])
        if label in excluded_set:
            excluded += 1
    return float(excluded) / float(len(indices))


def _split_indices_by_excluded(dataset, excluded_set):
    excluded = []
    rest = []
    for idx in range(len(dataset)):
        label = _extract_label(dataset[idx])
        if label in excluded_set:
            excluded.append(idx)
        else:
            rest.append(idx)
    return excluded, rest


def prepare_user_level_groups(priv_ds, eval_data, num_users, mia_users, *, seed=None, excluded_classes=None):
    """Prepare fixed user groups for a user-level audit (user in / user out).

    Notes:
      - `mia_users` is interpreted as number of USERS to audit (not samples).
      - When `excluded_classes` is provided, we match non-member users by excluded/rest ratio
        to reduce non-IID class-mix artifacts.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    member_groups_map = _collect_user_groups(priv_ds)
    eval_user_ds = SyntheticUserDataset(eval_data, num_users)

    n_users = min(mia_users, len(member_groups_map))
    if n_users <= 0:
        return [], [], eval_user_ds

    member_user_ids = rng.choice(list(member_groups_map.keys()), n_users, replace=False)
    member_groups = [member_groups_map[uid] for uid in member_user_ids]

    group_sizes = [len(group) for group in member_groups]
    total_needed = sum(group_sizes)

    non_member_groups = []
    if excluded_classes:
        excluded_set = set(int(x) for x in excluded_classes)
        eval_excluded, eval_rest = _split_indices_by_excluded(eval_user_ds, excluded_set)
        member_fracs = [
            _excluded_fraction(priv_ds, member_groups_map[uid], excluded_set) for uid in member_user_ids
        ]
        excluded_counts = [int(round(size * frac)) for size, frac in zip(group_sizes, member_fracs)]
        rest_counts = [size - excl for size, excl in zip(group_sizes, excluded_counts)]
        total_excluded_needed = sum(excluded_counts)
        total_rest_needed = sum(rest_counts)

        use_excl_replacement = total_excluded_needed > len(eval_excluded)
        use_rest_replacement = total_rest_needed > len(eval_rest)
        if use_excl_replacement or use_rest_replacement or total_needed > len(eval_user_ds):
            logger.warn(
                "MIA non-member sampling: requested %s samples from eval size %s; sampling with replacement.",
                total_needed,
                len(eval_user_ds),
            )

        if not use_excl_replacement:
            eval_excluded = rng.permutation(eval_excluded).tolist()
        if not use_rest_replacement:
            eval_rest = rng.permutation(eval_rest).tolist()
        excl_cursor = 0
        rest_cursor = 0

        for size, excl_count, rest_count in zip(group_sizes, excluded_counts, rest_counts):
            if excl_count <= 0:
                excl_indices = []
            elif use_excl_replacement:
                excl_indices = rng.choice(eval_excluded, excl_count, replace=True).tolist()
            else:
                excl_indices = eval_excluded[excl_cursor:excl_cursor + excl_count]
                excl_cursor += excl_count

            if rest_count <= 0:
                rest_indices = []
            elif use_rest_replacement:
                rest_indices = rng.choice(eval_rest, rest_count, replace=True).tolist()
            else:
                rest_indices = eval_rest[rest_cursor:rest_cursor + rest_count]
                rest_cursor += rest_count

            group_indices = excl_indices + rest_indices
            rng.shuffle(group_indices)
            non_member_groups.append(group_indices)
        logger.info("   • Non-members: size-matched with excluded/rest ratio.")
    else:
        use_replacement = total_needed > len(eval_user_ds)
        if use_replacement:
            logger.warn(
                "MIA non-member sampling: requested %s samples from eval size %s; sampling with replacement.",
                total_needed,
                len(eval_user_ds),
            )
            for size in group_sizes:
                group_indices = rng.choice(len(eval_user_ds), size, replace=True).tolist()
                non_member_groups.append(group_indices)
        else:
            all_indices = rng.permutation(len(eval_user_ds)).tolist()
            cursor = 0
            for size in group_sizes:
                non_member_groups.append(all_indices[cursor:cursor + size])
                cursor += size
        logger.info("   • Non-members: size-matched from evaluation pool.")

    return member_groups, non_member_groups, eval_user_ds


def _compute_group_scores(model, dataset, groups, device, batch_size=64):
    scores = []
    model.eval()
    with torch.no_grad():
        for indices in groups:
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            total_loss = 0.0
            total_count = 0
            for batch_data in loader:
                features, labels, _ = prepare_batch(batch_data, device)
                output = model(features)
                loss = F.cross_entropy(output, labels, reduction="sum")
                total_loss += float(loss.item())
                total_count += int(labels.size(0))
            if total_count == 0:
                scores.append(0.0)
            else:
                scores.append(-total_loss / total_count)
    return scores


def user_level_loss_attack(model, member_groups, non_member_groups, member_dataset, non_member_dataset, device):
    """User-level audit attack using mean per-user loss (higher score => more likely member)."""
    member_scores = _compute_group_scores(model, member_dataset, member_groups, device)
    non_member_scores = _compute_group_scores(model, non_member_dataset, non_member_groups, device)

    if not member_scores or not non_member_scores:
        _mia_fallback("user-loss: empty member/non-member scores")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base)}

    y_true = np.array([1] * len(member_scores) + [0] * len(non_member_scores))
    scores = np.array(member_scores + non_member_scores)
    if np.std(scores) < 1e-8:
        _mia_fallback("user-loss: zero-variance scores")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base)}

    auc_score = roc_auc_score(y_true, scores)
    return {"auc": auc_score, "auc_star": auc_star(auc_score), "adv": auc_advantage(auc_score)}


def _compute_group_features(model, dataset, groups, device, batch_size=64):
    feature_rows = []
    model.eval()
    with torch.no_grad():
        for indices in groups:
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
            group_features = []
            for batch_data in loader:
                inputs, labels, _ = prepare_batch(batch_data, device)
                output = model(inputs)
                probs = F.softmax(output, dim=1)
                probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
                max_prob = torch.max(probs, dim=1)[0]
                entropy = -torch.sum(probs * torch.log(probs), dim=1)
                entropy = torch.clamp(entropy, min=0.0, max=10.0)
                top3_probs, _ = torch.topk(probs, min(3, probs.size(1)), dim=1)
                if top3_probs.size(1) < 3:
                    padding = torch.zeros(top3_probs.size(0), 3 - top3_probs.size(1), device=device)
                    top3_probs = torch.cat([top3_probs, padding], dim=1)
                batch_features = torch.cat(
                    [max_prob.unsqueeze(1), entropy.unsqueeze(1), top3_probs],
                    dim=1,
                )
                group_features.append(batch_features)
            if not group_features:
                feature_rows.append(np.zeros(5, dtype=np.float32))
            else:
                group_mat = torch.cat(group_features, dim=0)
                feature_rows.append(group_mat.mean(dim=0).cpu().numpy())
    return np.vstack(feature_rows) if feature_rows else np.array([])


def prepare_user_shadow_splits(priv_ds, eval_data, num_users, seed=0, shadow_users=None, eval_user_ds=None):
    rng = np.random.RandomState(seed)
    priv_groups = _collect_user_groups(priv_ds)
    if eval_user_ds is None:
        eval_user_ds = SyntheticUserDataset(eval_data, num_users)
    eval_groups = _collect_user_groups(eval_user_ds)

    max_users = min(len(priv_groups), len(eval_groups))
    if max_users == 0:
        return {
            "shadow_user_ids": [],
            "shadow_non_member_user_ids": [],
            "eval_user_ds": eval_user_ds,
        }

    if shadow_users is None:
        shadow_users = min(max_users // 2, 200)
    shadow_users = max(1, min(shadow_users, max_users))

    shadow_user_ids = rng.choice(list(priv_groups.keys()), shadow_users, replace=False)
    shadow_non_member_ids = rng.choice(list(eval_groups.keys()), shadow_users, replace=False)

    return {
        "shadow_user_ids": list(shadow_user_ids),
        "shadow_non_member_user_ids": list(shadow_non_member_ids),
        "eval_user_ds": eval_user_ds,
    }


def user_level_shadow_attack(
    target_model,
    member_groups,
    non_member_groups,
    priv_ds,
    eval_user_ds,
    device,
    shadow_epochs=3,
    num_shadows=3,
    shadow_splits=None,
):
    """User-level shadow attack using aggregated per-user features."""
    if not member_groups or not non_member_groups:
        _mia_fallback("user-shadow: empty member/non-member groups")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}

    priv_groups = _collect_user_groups(priv_ds)
    if shadow_splits is None:
        eval_groups = _collect_user_groups(eval_user_ds)
        shadow_user_ids = list(priv_groups.keys())
        non_member_user_ids = list(eval_groups.keys())
        max_users = min(len(shadow_user_ids), len(non_member_user_ids))
        if max_users == 0:
            _mia_fallback("user-shadow: no users in shadow pool")
            base = 0.5
            return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}
        shadow_users = min(max_users // 2, 200)
        shadow_users = max(1, shadow_users)
        shadow_user_ids = np.random.choice(shadow_user_ids, shadow_users, replace=False)
        non_member_user_ids = np.random.choice(non_member_user_ids, shadow_users, replace=False)
        eval_groups = _collect_user_groups(eval_user_ds)
    else:
        shadow_user_ids = shadow_splits.get("shadow_user_ids", [])
        non_member_user_ids = shadow_splits.get("shadow_non_member_user_ids", [])
        eval_user_ds = shadow_splits.get("eval_user_ds", eval_user_ds)
        eval_groups = _collect_user_groups(eval_user_ds)

    shadow_member_groups = [priv_groups[uid] for uid in shadow_user_ids if uid in priv_groups]
    shadow_non_member_groups = [eval_groups[uid] for uid in non_member_user_ids if uid in eval_groups]

    if not shadow_member_groups or not shadow_non_member_groups:
        _mia_fallback("user-shadow: empty shadow groups")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}

    shadow_member_indices = [idx for group in shadow_member_groups for idx in group]
    shadow_non_member_indices = [idx for group in shadow_non_member_groups for idx in group]
    shadow_trainset = Subset(priv_ds, shadow_member_indices)
    shadow_non_trainset = Subset(eval_user_ds, shadow_non_member_indices)

    logger.info("   🔧 Training %s shadow models with %s epochs each...", num_shadows, shadow_epochs)
    shadow_models = train_shadow_models(
        shadow_trainset, target_model, num_shadows=num_shadows, epochs=shadow_epochs, device=device
    )

    shadow_features = []
    shadow_labels = []

    for shadow_model in shadow_models:
        member_features = _compute_group_features(shadow_model, priv_ds, shadow_member_groups, device)
        non_member_features = _compute_group_features(shadow_model, eval_user_ds, shadow_non_member_groups, device)

        if member_features.size > 0 and non_member_features.size > 0:
            shadow_features.append(member_features)
            shadow_features.append(non_member_features)
            shadow_labels.extend([1] * len(member_features))
            shadow_labels.extend([0] * len(non_member_features))

    if not shadow_features:
        _mia_fallback("user-shadow: no shadow features")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}

    X_shadow = np.vstack(shadow_features)
    y_shadow = np.array(shadow_labels)
    X_shadow, y_shadow = validate_and_clean_features(X_shadow, y_shadow, "user-level shadow training data")

    if len(X_shadow) < 10 or len(np.unique(y_shadow)) < 2:
        _mia_fallback("user-shadow: insufficient shadow samples")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_train, X_test, y_train, y_test = train_test_split(
        X_shadow, y_shadow, test_size=0.3, random_state=42, stratify=y_shadow
    )

    attack_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
        )),
    ])

    try:
        attack_pipeline.fit(X_train, y_train)
        shadow_predictions = attack_pipeline.predict_proba(X_test)[:, 1]
        shadow_auc = roc_auc_score(y_test, shadow_predictions)
    except Exception:
        _mia_fallback("user-shadow: attack training failed")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}

    target_member_features = _compute_group_features(target_model, priv_ds, member_groups, device)
    target_non_member_features = _compute_group_features(target_model, eval_user_ds, non_member_groups, device)
    if target_member_features.size == 0 or target_non_member_features.size == 0:
        _mia_fallback("user-shadow: empty target features")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}

    X_target = np.vstack([target_member_features, target_non_member_features])
    y_target = np.concatenate([
        np.ones(len(target_member_features)),
        np.zeros(len(target_non_member_features)),
    ])

    try:
        target_predictions_proba = attack_pipeline.predict_proba(X_target)[:, 1]
        target_predictions = attack_pipeline.predict(X_target)
        auc_score = roc_auc_score(y_target, target_predictions_proba)
        attack_accuracy = accuracy_score(y_target, target_predictions)
    except Exception:
        _mia_fallback("user-shadow: attack eval failed")
        base = 0.5
        return {"auc": base, "auc_star": auc_star(base), "adv": auc_advantage(base), "accuracy": 0.5}

    return {
        "auc": auc_score,
        "auc_star": auc_star(auc_score),
        "adv": auc_advantage(auc_score),
        "accuracy": attack_accuracy,
        "shadow_auc": shadow_auc,
        "n_shadow_users": len(shadow_user_ids),
        "n_target_users": len(target_member_features) + len(target_non_member_features),
    }

# ════════════════════════════════════════════════════════════════
# MIA Data Preparation Functions
# ════════════════════════════════════════════════════════════════

def prepare_mia_data_sample_level(train_data, eval_data, private_indices, mia_size):
    """Prepare MIA datasets for sample-level DP"""
    # Use random sampling from actual private training samples as members
    if hasattr(train_data, 'indices'):
        # train_data is a Subset, use random sampling from the subset
        max_members = min(mia_size, len(train_data))
        member_indices = np.random.choice(len(train_data), max_members, replace=False).tolist()
    else:
        max_members = min(mia_size, len(train_data))
        member_indices = np.random.choice(len(train_data), max_members, replace=False).tolist()
    
    member_set = Subset(train_data, member_indices)

    target_counts = _collect_label_counts(train_data, member_indices)
    non_member_indices = _sample_indices_by_label(eval_data, target_counts)
    if len(non_member_indices) < len(member_indices):
        missing = len(member_indices) - len(non_member_indices)
        fallback = np.random.choice(len(eval_data), missing, replace=True).tolist()
        non_member_indices.extend(fallback)

    non_member_set = Subset(eval_data, non_member_indices)

    logger.info("   • Members: %s samples from training data", len(member_set))
    logger.info("   • Non-members: %s label-matched samples from evaluation data", len(non_member_set))
    
    return member_set, non_member_set

def prepare_mia_data_user_level(priv_ds, eval_data, num_users, mia_size):
    """Prepare MIA datasets for user-level DP"""
    # Collect samples per user
    user_samples = {}
    for idx in range(len(priv_ds)):
        sample = priv_ds[idx]
        if isinstance(sample, dict):
            uid_value = sample.get("user_ids", sample.get("user_id"))
            if isinstance(uid_value, torch.Tensor):
                uid = int(uid_value.item())
            else:
                uid = int(uid_value)
        elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
            uid = int(sample[2])
        else:
            uid = 0
        user_samples.setdefault(uid, []).append(idx)
    
    # Select member users (from training) and randomly sample from them
    available_users = list(user_samples.keys())[:num_users]
    all_member_indices = []
    for uid in available_users:
        all_member_indices.extend(user_samples[uid])
    
    # Random sampling from all available member samples
    max_members = min(mia_size, len(all_member_indices))
    member_indices = np.random.choice(all_member_indices, max_members, replace=False).tolist()
    member_set = Subset(priv_ds, member_indices)

    target_counts = _collect_label_counts(priv_ds, member_indices)
    non_member_indices = _sample_indices_by_label(eval_data, target_counts)
    if len(non_member_indices) < len(member_indices):
        missing = len(member_indices) - len(non_member_indices)
        fallback = np.random.choice(len(eval_data), missing, replace=True).tolist()
        non_member_indices.extend(fallback)

    non_member_set = Subset(eval_data, non_member_indices)

    logger.info(
        "   • Members: %s samples from %s training users",
        len(member_set),
        len(available_users),
    )
    logger.info("   • Non-members: %s label-matched samples from evaluation data", len(non_member_set))
    
    return member_set, non_member_set

# ════════════════════════════════════════════════════════════════
# Attack Functions
# ════════════════════════════════════════════════════════════════

def confidence_attack(model, member_loader, non_member_loader, device):
    """Yeom et al. confidence-based membership inference attack"""
    model.eval()
    
    member_confidences = []
    non_member_confidences = []
    member_max_probs = []
    non_member_max_probs = []
    
    # Collect confidences for member samples
    with torch.no_grad():
        for batch_data in tqdm(member_loader, desc="Processing members", leave=False):
            features, labels, _ = prepare_batch(batch_data, device)
            output = model(features)
            probs = F.softmax(output, dim=1)
            
            # Use probability of the correct class (more standard for MIA)
            correct_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            member_confidences.extend(correct_class_probs.cpu().numpy())
            
            # Also track max probabilities for comparison
            max_probs = torch.max(probs, dim=1)[0]
            member_max_probs.extend(max_probs.cpu().numpy())
    
    # Collect confidences for non-member samples
    with torch.no_grad():
        for batch_data in tqdm(non_member_loader, desc="Processing non-members", leave=False):
            features, labels, _ = prepare_batch(batch_data, device)
            output = model(features)
            probs = F.softmax(output, dim=1)
            
            # Use probability of the correct class
            correct_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            non_member_confidences.extend(correct_class_probs.cpu().numpy())
            
            # Also track max probabilities for comparison
            max_probs = torch.max(probs, dim=1)[0]
            non_member_max_probs.extend(max_probs.cpu().numpy())
    
    # Create labels and evaluate
    all_confidences = np.concatenate([member_confidences, non_member_confidences])
    all_labels = np.concatenate([np.ones(len(member_confidences)), np.zeros(len(non_member_confidences))])
    
    # Check if we have valid data
    if len(set(all_labels)) < 2:
        _mia_fallback("confidence attack: single-class labels")
        return {'auc': 0.5, 'accuracy': 0.5, 'member_conf_mean': 0.0, 'non_member_conf_mean': 0.0}
    
    if np.std(all_confidences) < 1e-8:
        _mia_fallback("confidence attack: zero-variance scores")
        return {'auc': 0.5, 'accuracy': 0.5, 'member_conf_mean': np.mean(member_confidences), 'non_member_conf_mean': np.mean(non_member_confidences)}
    
    auc_score = roc_auc_score(all_labels, all_confidences)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_confidences)
    optimal_idx = np.argmax(precision + recall)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    predictions = (all_confidences >= optimal_threshold).astype(int)
    attack_accuracy = accuracy_score(all_labels, predictions)
    
    return {
        'auc': auc_score,
        'accuracy': attack_accuracy,
        'member_conf_mean': np.mean(member_confidences),
        'non_member_conf_mean': np.mean(non_member_confidences)
    }

def _reset_module_parameters(module):
    reset_fn = getattr(module, "reset_parameters", None)
    if callable(reset_fn):
        reset_fn()


def train_shadow_models(shadow_trainset, target_model, num_shadows=3, epochs=5, device='cpu'):
    """Train shadow models for Shokri attack using the target model architecture"""
    shadow_models = []
    
    for i in tqdm(range(num_shadows), desc="Training shadow models"):
        shadow_model = copy.deepcopy(target_model).to(device)
        shadow_model.apply(_reset_module_parameters)
        optimizer = torch.optim.SGD(shadow_model.parameters(), lr=1e-3, momentum=0.9)
        
        # Create data loader for this shadow model
        shadow_loader = DataLoader(shadow_trainset, batch_size=128, shuffle=True)
        
        # Train shadow model
        shadow_model.train()
        for epoch in tqdm(range(epochs), desc=f"  Shadow {i+1}/{num_shadows} epochs", leave=False):
            for batch_data in shadow_loader:
                features, labels, _ = prepare_batch(batch_data, device)
                optimizer.zero_grad()
                output = shadow_model(features)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
        
        shadow_models.append(shadow_model)
    
    return shadow_models

def extract_attack_features(model, data_loader, device):
    """Extract features for shadow model attack"""
    model.eval()
    feature_rows = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            batch_inputs, _, _ = prepare_batch(batch_data, device)
            output = model(batch_inputs)
            probs = F.softmax(output, dim=1)
            
            # Clip probabilities to avoid numerical issues
            probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
            
            # Features: [max_prob, entropy, top-3 probs]
            max_prob = torch.max(probs, dim=1)[0]
            
            # More robust entropy calculation
            entropy = -torch.sum(probs * torch.log(probs), dim=1)
            # Clamp entropy to reasonable range
            entropy = torch.clamp(entropy, min=0.0, max=10.0)
            
            top3_probs, _ = torch.topk(probs, min(3, probs.size(1)), dim=1)
            
            # Pad top3_probs if we have fewer than 3 classes
            if top3_probs.size(1) < 3:
                padding = torch.zeros(top3_probs.size(0), 3 - top3_probs.size(1), device=device)
                top3_probs = torch.cat([top3_probs, padding], dim=1)
            
            batch_features = torch.cat([
                max_prob.unsqueeze(1),
                entropy.unsqueeze(1), 
                top3_probs
            ], dim=1)
            
            # Check for NaN or inf values
            if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                logger.warn("NaN or inf values detected in features, skipping batch.")
                continue
                
            feature_rows.append(batch_features.cpu().numpy())
    
    return np.vstack(feature_rows) if feature_rows else np.array([])

def validate_and_clean_features(features, labels, name="features"):
    """Validate and clean feature arrays for numerical stability"""
    if features.size == 0:
        return features, labels
    
    # Check for NaN or inf values
    nan_mask = np.isnan(features).any(axis=1)
    inf_mask = np.isinf(features).any(axis=1)
    bad_mask = nan_mask | inf_mask
    
    if bad_mask.any():
        features = features[~bad_mask]
        labels = labels[~bad_mask]
    
    # Check for constant features (zero variance)
    if features.size > 0:
        feature_std = np.std(features, axis=0)
        zero_var_mask = feature_std < 1e-8
        if zero_var_mask.any():
            # Add small random noise to zero-variance features
            features[:, zero_var_mask] += np.random.normal(0, 1e-6, 
                                                          (features.shape[0], zero_var_mask.sum()))
    
    return features, labels

def shadow_model_attack(
    target_model,
    member_loader,
    non_member_loader,
    train_data,
    device,
    eval_data=None,
    shadow_epochs=3,
    shadow_splits=None,
):
    """
    Shokri et al. shadow model attack.
    Uses shadow models to learn attack patterns.
    train_data: The actual training data used for the target model (priv_base)
    eval_data: The evaluation data used for target non-members (eval_base)
    shadow_epochs: Number of training epochs for each shadow model (default: 3)
    """
    
    if shadow_splits is None:
        shadow_size = min(len(train_data) // 2, 2000)
        shadow_indices = np.random.choice(len(train_data), shadow_size, replace=False)
        shadow_trainset = Subset(train_data, shadow_indices)

        if eval_data is not None:
            shadow_non_member_size = min(len(eval_data), shadow_size)
            shadow_non_member_indices = np.random.choice(len(eval_data), shadow_non_member_size, replace=False)
            shadow_non_trainset = Subset(eval_data, shadow_non_member_indices)
            logger.info("Shadow attack: Using eval_data for non-members.")
        else:
            remaining_indices = np.setdiff1d(np.arange(len(train_data)), shadow_indices)
            shadow_non_trainset = Subset(train_data, remaining_indices[:shadow_size])
            logger.warn("Shadow attack: Using train_data for non-members.")
    else:
        shadow_indices = shadow_splits["shadow_indices"]
        shadow_trainset = Subset(train_data, shadow_indices)
        shadow_non_member_indices = shadow_splits["shadow_non_member_indices"]
        if eval_data is not None and shadow_splits.get("non_member_source") == "eval":
            shadow_non_trainset = Subset(eval_data, shadow_non_member_indices)
        else:
            shadow_non_trainset = Subset(train_data, shadow_non_member_indices)
        logger.info(
            "Shadow attack: Using shared shadow split (%s members / %s non-members).",
            len(shadow_trainset),
            len(shadow_non_trainset),
        )
    
    # Train shadow models
    logger.info("   🔧 Training %s shadow models with %s epochs each...", 3, shadow_epochs)
    shadow_models = train_shadow_models(shadow_trainset, target_model, num_shadows=3, epochs=shadow_epochs, device=device)
    
    # Generate attack training data using shadow models
    shadow_features = []
    shadow_labels = []
    
    for i, shadow_model in enumerate(shadow_models):
        # For each shadow model:
        # - Members: samples that WERE used to train this shadow model
        # - Non-members: samples that were NOT used to train this shadow model
        
        shadow_member_loader = DataLoader(shadow_trainset, batch_size=128, shuffle=False)
        shadow_non_member_loader = DataLoader(shadow_non_trainset, batch_size=128, shuffle=False)
        
        # Extract features for members (label=1) and non-members (label=0)
        member_features = extract_attack_features(shadow_model, shadow_member_loader, device)
        non_member_features = extract_attack_features(shadow_model, shadow_non_member_loader, device)
        
        if member_features.size > 0 and non_member_features.size > 0:
            shadow_features.append(member_features)
            shadow_features.append(non_member_features)
            shadow_labels.extend([1] * len(member_features))  # Members
            shadow_labels.extend([0] * len(non_member_features))  # Non-members
    
    if not shadow_features:
        _mia_fallback("shadow: no features")
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5}
    
    # Combine all shadow training data
    X_shadow = np.vstack(shadow_features)
    y_shadow = np.array(shadow_labels)
    
    # Validate and clean shadow features
    X_shadow, y_shadow = validate_and_clean_features(X_shadow, y_shadow, "shadow training data")
    
    if len(X_shadow) < 10:  # Need minimum samples for training
        _mia_fallback("shadow: insufficient samples")
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Check class balance
    unique_labels, label_counts = np.unique(y_shadow, return_counts=True)
    
    if len(unique_labels) < 2:
        _mia_fallback("shadow: single-class labels")
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Ensure minimum samples per class
    min_class_size = min(label_counts)
    if min_class_size < 5:
        _mia_fallback("shadow: tiny class size")
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Train attack classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Split shadow data for training the attack classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_shadow, y_shadow, test_size=0.3, random_state=42, stratify=y_shadow
    )
    
    # Create a pipeline with feature scaling and logistic regression
    attack_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,  # Add regularization
            solver='liblinear',  # More stable solver
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Train attack classifier with error handling
    try:
        attack_pipeline.fit(X_train, y_train)
        
        # Test attack classifier on shadow data
        shadow_predictions = attack_pipeline.predict_proba(X_test)[:, 1]
        shadow_auc = roc_auc_score(y_test, shadow_predictions)
        
    except Exception as e:
        _mia_fallback("shadow: attack training failed")
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Extract features from target model for actual attack
    target_member_features = extract_attack_features(target_model, member_loader, device)
    target_non_member_features = extract_attack_features(target_model, non_member_loader, device)
    
    if target_member_features.size == 0 or target_non_member_features.size == 0:
        _mia_fallback("shadow: empty target features")
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': shadow_auc}
    
    # Create target attack dataset
    X_target = np.vstack([target_member_features, target_non_member_features])
    y_target = np.concatenate([
        np.ones(len(target_member_features)),    # Members = 1
        np.zeros(len(target_non_member_features)) # Non-members = 0
    ])
    
    # Check for any remaining numerical issues
    if np.isnan(X_target).any() or np.isinf(X_target).any():
        X_target = np.nan_to_num(X_target, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Perform attack on target model
    try:
        target_predictions_proba = attack_pipeline.predict_proba(X_target)[:, 1]
        target_predictions = attack_pipeline.predict(X_target)
        
        # Evaluate attack performance
        auc_score = roc_auc_score(y_target, target_predictions_proba)
        attack_accuracy = accuracy_score(y_target, target_predictions)
        
    except Exception as e:
        _mia_fallback("shadow: attack eval failed")
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': shadow_auc}
    
    return {
        'auc': auc_score,
        'auc_star': auc_star(auc_score),
        'adv': auc_advantage(auc_score),
        'accuracy': attack_accuracy,
        'shadow_auc': shadow_auc,
        'n_shadow_samples': len(X_shadow),
        'n_target_samples': len(X_target)
    }

# ════════════════════════════════════════════════════════════════
# Main Evaluation Function
# ════════════════════════════════════════════════════════════════

def evaluate_membership_inference(
    baseline_model,
    fisher_dp_model,
    train_data,
    eval_data,
    private_indices,
    priv_ds,
    num_users,
    mia_size,
    sample_level,
    device,
    vanilla_dp_model=None,
    dp_sat_model=None,
    l2_baseline_model=None,
    shadow_epochs=3,
    mia_level="auto",
    mia_attack="shadow",
):
    """Evaluate membership inference attacks on baseline, Fisher DP, and optionally Vanilla DP & DP-SAT models
    
    Args:
        baseline_model: Non-DP baseline model
        fisher_dp_model: Fisher-informed DP model
        train_data: The actual training data used (priv_base from trainset)
        eval_data: The evaluation data for non-members (eval_base from testset)
        private_indices: Original private indices (for compatibility, may not be used)
        priv_ds: User-level dataset object (for user-level DP)
        num_users: Number of users for user-level DP
        mia_size: Number of member/non-member samples for evaluation
        sample_level: Whether using sample-level (True) or user-level (False) DP
        device: Device to run evaluation on
        vanilla_dp_model: Optional vanilla DP model for comparison
        dp_sat_model: Optional DP-SAT model for comparison
        l2_baseline_model: Optional L2 regularized baseline model for comparison
        mia_attack: User-level attack type ("shadow" or "loss")
    """
    
    logger.highlight("Membership Inference Attack Evaluation (audit-only)")
    
    # Build comparison message
    methods = ["Baseline", "Fisher DP"]
    if l2_baseline_model is not None:
        methods.append("L2 Baseline")
    if vanilla_dp_model is not None:
        methods.append("Vanilla DP")
    if dp_sat_model is not None:
        methods.append("DP-SAT")
    logger.info("Comparing: %s", " vs ".join(methods))

    train_data, priv_ds, aligned = align_mia_datasets(train_data, priv_ds, eval_data, num_users)
    if aligned:
        logger.info("MIA transforms: aligned to eval.")

    mia_use_user = False
    mia_use_sample = False
    if mia_level == "auto":
        mia_use_sample = sample_level
        mia_use_user = not sample_level
    elif mia_level == "sample":
        mia_use_sample = True
    elif mia_level == "user":
        mia_use_user = True

    if mia_use_user and priv_ds is None:
        logger.warn("User-level MIA requested but priv_ds is unavailable; falling back to sample-level MIA.")
        mia_use_user = False
        mia_use_sample = True

    if mia_use_sample:
        logger.info("MIA mode: sample-level.")
    else:
        logger.info("MIA mode: user-level.")
        logger.info("User-level MIA attack: %s.", mia_attack)
    
    # Prepare member and non-member datasets
    if mia_use_sample:
        logger.info("Sample-level MIA: using actual private training samples as members.")
        member_set, non_member_set = prepare_mia_data_sample_level(train_data, eval_data, private_indices, mia_size)
    else:
        logger.info("User-level MIA: using actual private users as members.")
        member_set, non_member_set = prepare_mia_data_user_level(priv_ds, eval_data, num_users, mia_size)
    
    member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
    non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
    
    logger.info("   • Members: %s samples", len(member_set))
    logger.info("   • Non-members: %s samples", len(non_member_set))
    
    # Track results across multiple runs for statistical analysis
    num_runs = NUM_RUNS  # Multiple runs for statistical robustness
    all_results = {
        'baseline': {},
        'fisher_dp': {},
    }
    if l2_baseline_model is not None:
        all_results['l2_baseline'] = {}
    if vanilla_dp_model is not None:
        all_results['vanilla_dp'] = {}
    if dp_sat_model is not None:
        all_results['dp_sat'] = {}

    if mia_use_sample:
        for key in all_results.keys():
            all_results[key]['shadow'] = []
    if mia_use_user:
        for key in all_results.keys():
            all_results[key]['user_attack_star'] = []
    
    for run_idx in range(num_runs):
        # Re-sample for each run to get different member/non-member sets
        if mia_use_sample:
            member_set, non_member_set = prepare_mia_data_sample_level(train_data, eval_data, private_indices, mia_size)
        else:
            member_set, non_member_set = prepare_mia_data_user_level(priv_ds, eval_data, num_users, mia_size)
        
        member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
        shadow_splits = prepare_shadow_splits(train_data, eval_data, seed=run_idx)
        user_shadow_splits = None
        if mia_use_user:
            member_groups, non_member_groups, non_member_user_ds = prepare_user_level_groups(
                priv_ds, eval_data, num_users, mia_size
            )
            if mia_attack == "shadow":
                user_shadow_splits = prepare_user_shadow_splits(
                    priv_ds,
                    eval_data,
                    num_users,
                    seed=run_idx,
                    eval_user_ds=non_member_user_ds,
                )
        
        if mia_use_sample:
            logger.highlight(f"Shadow Model Attack (Run {run_idx + 1}/{num_runs})")
        
        if mia_use_sample:
            baseline_shadow = shadow_model_attack(
                baseline_model,
                member_loader,
                non_member_loader,
                train_data,
                device,
                eval_data,
                shadow_epochs=shadow_epochs,
                shadow_splits=shadow_splits,
            )
            fisher_shadow = shadow_model_attack(
                fisher_dp_model,
                member_loader,
                non_member_loader,
                train_data,
                device,
                eval_data,
                shadow_epochs=shadow_epochs,
                shadow_splits=shadow_splits,
            )
            all_results['baseline']['shadow'].append(baseline_shadow['auc_star'])
            all_results['fisher_dp']['shadow'].append(fisher_shadow['auc_star'])
        if mia_use_user:
            if mia_attack == "shadow":
                baseline_user = user_level_shadow_attack(
                    baseline_model,
                    member_groups,
                    non_member_groups,
                    priv_ds,
                    non_member_user_ds,
                    device,
                    shadow_epochs=shadow_epochs,
                    shadow_splits=user_shadow_splits,
                )
                fisher_user = user_level_shadow_attack(
                    fisher_dp_model,
                    member_groups,
                    non_member_groups,
                    priv_ds,
                    non_member_user_ds,
                    device,
                    shadow_epochs=shadow_epochs,
                    shadow_splits=user_shadow_splits,
                )
            else:
                baseline_user = user_level_loss_attack(
                    baseline_model,
                    member_groups,
                    non_member_groups,
                    priv_ds,
                    non_member_user_ds,
                    device,
                )
                fisher_user = user_level_loss_attack(
                    fisher_dp_model,
                    member_groups,
                    non_member_groups,
                    priv_ds,
                    non_member_user_ds,
                    device,
                )
            all_results['baseline']['user_attack_star'].append(baseline_user['auc_star'])
            all_results['fisher_dp']['user_attack_star'].append(fisher_user['auc_star'])
        
        if l2_baseline_model is not None:
            if mia_use_sample:
                l2_baseline_shadow = shadow_model_attack(
                    l2_baseline_model,
                    member_loader,
                    non_member_loader,
                    train_data,
                    device,
                    eval_data,
                    shadow_epochs=shadow_epochs,
                    shadow_splits=shadow_splits,
                )
                all_results['l2_baseline']['shadow'].append(l2_baseline_shadow['auc_star'])
            if mia_use_user:
                if mia_attack == "shadow":
                    l2_user = user_level_shadow_attack(
                        l2_baseline_model,
                        member_groups,
                        non_member_groups,
                        priv_ds,
                        non_member_user_ds,
                        device,
                        shadow_epochs=shadow_epochs,
                        shadow_splits=user_shadow_splits,
                    )
                else:
                    l2_user = user_level_loss_attack(
                        l2_baseline_model,
                        member_groups,
                        non_member_groups,
                        priv_ds,
                        non_member_user_ds,
                        device,
                    )
                all_results['l2_baseline']['user_attack_star'].append(l2_user['auc_star'])
        
        if vanilla_dp_model is not None:
            if mia_use_sample:
                vanilla_shadow = shadow_model_attack(
                    vanilla_dp_model,
                    member_loader,
                    non_member_loader,
                    train_data,
                    device,
                    eval_data,
                    shadow_epochs=shadow_epochs,
                    shadow_splits=shadow_splits,
                )
                all_results['vanilla_dp']['shadow'].append(vanilla_shadow['auc_star'])
            if mia_use_user:
                if mia_attack == "shadow":
                    vanilla_user = user_level_shadow_attack(
                        vanilla_dp_model,
                        member_groups,
                        non_member_groups,
                        priv_ds,
                        non_member_user_ds,
                        device,
                        shadow_epochs=shadow_epochs,
                        shadow_splits=user_shadow_splits,
                    )
                else:
                    vanilla_user = user_level_loss_attack(
                        vanilla_dp_model,
                        member_groups,
                        non_member_groups,
                        priv_ds,
                        non_member_user_ds,
                        device,
                    )
                all_results['vanilla_dp']['user_attack_star'].append(vanilla_user['auc_star'])
        
        if dp_sat_model is not None:
            if mia_use_sample:
                dp_sat_shadow = shadow_model_attack(
                    dp_sat_model,
                    member_loader,
                    non_member_loader,
                    train_data,
                    device,
                    eval_data,
                    shadow_epochs=shadow_epochs,
                    shadow_splits=shadow_splits,
                )
                all_results['dp_sat']['shadow'].append(dp_sat_shadow['auc_star'])
            if mia_use_user:
                if mia_attack == "shadow":
                    dp_sat_user = user_level_shadow_attack(
                        dp_sat_model,
                        member_groups,
                        non_member_groups,
                        priv_ds,
                        non_member_user_ds,
                        device,
                        shadow_epochs=shadow_epochs,
                        shadow_splits=user_shadow_splits,
                    )
                else:
                    dp_sat_user = user_level_loss_attack(
                        dp_sat_model,
                        member_groups,
                        non_member_groups,
                        priv_ds,
                        non_member_user_ds,
                        device,
                    )
                all_results['dp_sat']['user_attack_star'].append(dp_sat_user['auc_star'])
    
    # Statistical analysis of results
    logger.highlight("Final MIA Results (audit-only)")
    
    def print_stats(name, values):
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info("%s: %.4f ± %.4f", name, mean_val, std_val)
        return mean_val, std_val
    
    if mia_use_sample:
        logger.info("Shadow Attack AUC*:")
        baseline_shadow_mean, baseline_shadow_std = print_stats("  Baseline", all_results['baseline']['shadow'])
        fisher_shadow_mean, fisher_shadow_std = print_stats("  Fisher DP", all_results['fisher_dp']['shadow'])
        if l2_baseline_model is not None:
            l2_baseline_shadow_mean, l2_baseline_shadow_std = print_stats("  L2 Baseline", all_results['l2_baseline']['shadow'])
        if vanilla_dp_model is not None:
            vanilla_shadow_mean, vanilla_shadow_std = print_stats("  Vanilla DP", all_results['vanilla_dp']['shadow'])
        if dp_sat_model is not None:
            dp_sat_shadow_mean, dp_sat_shadow_std = print_stats("  DP-SAT", all_results['dp_sat']['shadow'])

    if mia_use_user:
        label = "User-level Shadow AUC*" if mia_attack == "shadow" else "User-level Loss AUC*"
        logger.info("%s:", label)
        print_stats("  Baseline", all_results['baseline']['user_attack_star'])
        print_stats("  Fisher DP", all_results['fisher_dp']['user_attack_star'])
        if l2_baseline_model is not None:
            print_stats("  L2 Baseline", all_results['l2_baseline']['user_attack_star'])
        if vanilla_dp_model is not None:
            print_stats("  Vanilla DP", all_results['vanilla_dp']['user_attack_star'])
        if dp_sat_model is not None:
            print_stats("  DP-SAT", all_results['dp_sat']['user_attack_star'])

    if not mia_use_sample:
        return {
            'statistical_results': all_results,
        }
    
    # Statistical significance tests (t-tests)
    if mia_use_sample:
        from scipy import stats
        
        logger.info("Statistical Significance Tests (p-values):")
        
        # Fisher DP vs Baseline
        _, p_shadow_fisher_base = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['baseline']['shadow'])
        logger.info("  Fisher DP vs Baseline (Shadow): p = %.4f", p_shadow_fisher_base)
        
        # L2 baseline tests (for regularization hypothesis)
        if l2_baseline_model is not None:
            # L2 baseline vs regular baseline
            _, p_shadow_l2_base = stats.ttest_rel(all_results['l2_baseline']['shadow'], all_results['baseline']['shadow'])
            logger.info("  L2 Baseline vs Baseline (Shadow): p = %.4f", p_shadow_l2_base)
            
            # Fisher DP vs L2 baseline (key L2 regularization hypothesis test!)
            _, p_shadow_fisher_l2 = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['l2_baseline']['shadow'])
            logger.info("  Fisher DP vs L2 Baseline (Shadow): p = %.4f", p_shadow_fisher_l2)
        
        if vanilla_dp_model is not None:
            # Vanilla DP vs Baseline
            _, p_shadow_vanilla_base = stats.ttest_rel(all_results['vanilla_dp']['shadow'], all_results['baseline']['shadow'])
            logger.info("  Vanilla DP vs Baseline (Shadow): p = %.4f", p_shadow_vanilla_base)
            
            # Fisher DP vs Vanilla DP (the key comparison!)
            _, p_shadow_fisher_vanilla = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['vanilla_dp']['shadow'])
            logger.info("  Fisher DP vs Vanilla DP (Shadow): p = %.4f", p_shadow_fisher_vanilla)
            
            # L2 baseline vs Vanilla DP (if both available)
        if l2_baseline_model is not None:
            _, p_shadow_l2_vanilla = stats.ttest_rel(all_results['l2_baseline']['shadow'], all_results['vanilla_dp']['shadow'])
            logger.info("  L2 Baseline vs Vanilla DP (Shadow): p = %.4f", p_shadow_l2_vanilla)
    
    if dp_sat_model is not None:
        # DP-SAT vs Baseline
        _, p_shadow_dp_sat_base = stats.ttest_rel(all_results['dp_sat']['shadow'], all_results['baseline']['shadow'])
        logger.info("  DP-SAT vs Baseline (Shadow): p = %.4f", p_shadow_dp_sat_base)
        
        # Fisher DP vs DP-SAT (key comparison!)
        _, p_shadow_fisher_dp_sat = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['dp_sat']['shadow'])
        logger.info("  Fisher DP vs DP-SAT (Shadow): p = %.4f", p_shadow_fisher_dp_sat)
        
        # L2 baseline vs DP-SAT (if both available)
        if l2_baseline_model is not None:
            _, p_shadow_l2_dp_sat = stats.ttest_rel(all_results['l2_baseline']['shadow'], all_results['dp_sat']['shadow'])
            logger.info("  L2 Baseline vs DP-SAT (Shadow): p = %.4f", p_shadow_l2_dp_sat)
        
        # Vanilla DP vs DP-SAT (if both available)
        if vanilla_dp_model is not None:
            _, p_shadow_vanilla_dp_sat = stats.ttest_rel(all_results['vanilla_dp']['shadow'], all_results['dp_sat']['shadow'])
            logger.info("  Vanilla DP vs DP-SAT (Shadow): p = %.4f", p_shadow_vanilla_dp_sat)
    
    # Final assessment based on shadow attack AUC across runs (no need for worst-case since we only have one attack)
    fisher_worst_shadow = max(all_results['fisher_dp']['shadow'])
    
    worst_case_results = {'fisher_dp': fisher_worst_shadow}
    
    if l2_baseline_model is not None:
        l2_baseline_worst_shadow = max(all_results['l2_baseline']['shadow'])
        worst_case_results['l2_baseline'] = l2_baseline_worst_shadow
    
    if vanilla_dp_model is not None:
        vanilla_worst_shadow = max(all_results['vanilla_dp']['shadow'])
        worst_case_results['vanilla_dp'] = vanilla_worst_shadow
    
    if dp_sat_model is not None:
        dp_sat_worst_shadow = max(all_results['dp_sat']['shadow'])
        worst_case_results['dp_sat'] = dp_sat_worst_shadow
    
    logger.highlight("Final Privacy Protection Comparison (audit-only)")
    logger.info("Shadow Attack AUC (worst across runs):")
    logger.info("   • Fisher DP: %.4f", fisher_worst_shadow)
    if l2_baseline_model is not None:
        logger.info("   • L2 Baseline: %.4f", l2_baseline_worst_shadow)
    if vanilla_dp_model is not None:
        logger.info("   • Vanilla DP: %.4f", vanilla_worst_shadow)
    if dp_sat_model is not None:
        logger.info("   • DP-SAT: %.4f", dp_sat_worst_shadow)
    
    # Find the best performing method
    best_method = min(worst_case_results.items(), key=lambda x: x[1])
    best_name, best_auc = best_method
    
    if best_name == 'fisher_dp':
        logger.success("Fisher DP provides the BEST privacy protection.")
        if l2_baseline_model is not None:
            diff_l2 = l2_baseline_worst_shadow - fisher_worst_shadow
            logger.info("   📈 vs L2 Baseline: %.4f AUC reduction", diff_l2)
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_shadow - fisher_worst_shadow
            logger.info("   📈 vs Vanilla DP: %.4f AUC reduction", diff_vanilla)
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_shadow - fisher_worst_shadow
            logger.info("   📈 vs DP-SAT: %.4f AUC reduction", diff_dp_sat)
    elif best_name == 'l2_baseline':
        logger.success("L2 Baseline provides the BEST privacy protection.")
        diff_fisher = fisher_worst_shadow - l2_baseline_worst_shadow
        logger.info("   📈 vs Fisher DP: %.4f AUC reduction", diff_fisher)
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_shadow - l2_baseline_worst_shadow
            logger.info("   📈 vs Vanilla DP: %.4f AUC reduction", diff_vanilla)
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_shadow - l2_baseline_worst_shadow
            logger.info("   📈 vs DP-SAT: %.4f AUC reduction", diff_dp_sat)
    elif best_name == 'vanilla_dp':
        logger.success("Vanilla DP provides the BEST privacy protection.")
        diff_fisher = fisher_worst_shadow - vanilla_worst_shadow
        logger.info("   📈 vs Fisher DP: %.4f AUC reduction", diff_fisher)
        if l2_baseline_model is not None:
            diff_l2 = l2_baseline_worst_shadow - vanilla_worst_shadow
            logger.info("   📈 vs L2 Baseline: %.4f AUC reduction", diff_l2)
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_shadow - vanilla_worst_shadow
            logger.info("   📈 vs DP-SAT: %.4f AUC reduction", diff_dp_sat)
    elif best_name == 'dp_sat':
        logger.success("DP-SAT provides the BEST privacy protection.")
        diff_fisher = fisher_worst_shadow - dp_sat_worst_shadow
        logger.info("   📈 vs Fisher DP: %.4f AUC reduction", diff_fisher)
        if l2_baseline_model is not None:
            diff_l2 = l2_baseline_worst_shadow - dp_sat_worst_shadow
            logger.info("   📈 vs L2 Baseline: %.4f AUC reduction", diff_l2)
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_shadow - dp_sat_worst_shadow
            logger.info("   📈 vs Vanilla DP: %.4f AUC reduction", diff_vanilla)
    
    # Privacy strength assessment for all methods
    privacy_threshold = 0.6  # AUC > 0.6 indicates weak privacy protection
    
    if fisher_worst_shadow < privacy_threshold:
        logger.success("Fisher DP provides STRONG privacy protection (audit).")
    else:
        logger.warn("Fisher DP provides WEAK privacy protection (audit).")
    
    if l2_baseline_model is not None:
        if l2_baseline_worst_shadow < privacy_threshold:
            logger.success("L2 Baseline provides STRONG privacy protection (audit).")
        else:
            logger.warn("L2 Baseline provides WEAK privacy protection (audit).")
        
    if vanilla_dp_model is not None:
        if vanilla_worst_shadow < privacy_threshold:
            logger.success("Vanilla DP provides STRONG privacy protection (audit).")
        else:
            logger.warn("Vanilla DP provides WEAK privacy protection (audit).")
    
    if dp_sat_model is not None:
        if dp_sat_worst_shadow < privacy_threshold:
            logger.success("DP-SAT provides STRONG privacy protection (audit).")
        else:
            logger.warn("DP-SAT provides WEAK privacy protection (audit).")
    
    return {
        'fisher_worst_auc': fisher_worst_shadow,
        'l2_baseline_worst_auc': l2_baseline_worst_shadow if l2_baseline_model else None,
        'vanilla_worst_auc': vanilla_worst_shadow if vanilla_dp_model else None,
        'dp_sat_worst_auc': dp_sat_worst_shadow if dp_sat_model else None,
        'statistical_results': all_results
    }

# ════════════════════════════════════════════════════════════════
# Standalone Evaluation Function
# ════════════════════════════════════════════════════════════════

def get_device(args):
    """Get the appropriate device based on command line arguments"""
    if not hasattr(args, 'cuda_devices'):
        args.cuda_devices = None
    if not hasattr(args, 'multi_gpu'):
        args.multi_gpu = False
    return resolve_device(args)

def prepare_standalone_mia_data(trainset, testset, member_size=5000, non_member_size=5000):
    """Prepare member and non-member datasets for standalone MIA evaluation"""
    
    # Member set: samples that were used for training
    member_indices = np.random.choice(len(trainset), member_size, replace=False)
    member_set = Subset(trainset, member_indices)
    
    # Non-member set: samples from test set (never seen during training)
    non_member_indices = np.random.choice(len(testset), non_member_size, replace=False)
    non_member_set = Subset(testset, non_member_indices)
    
    return member_set, non_member_set, member_indices, non_member_indices

def main():
    """Standalone MIA evaluation script (for backward compatibility)"""
    parser = argparse.ArgumentParser('Membership Inference Attack Evaluation')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--cuda-id', type=int)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda-devices', type=str, default=None,
                       help='Comma-separated CUDA device ids for multi-GPU execution')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Enable torch.nn.DataParallel across the requested CUDA devices')
    parser.add_argument('--member-size', type=int, default=2000,
                       help='Number of member samples for MIA')
    parser.add_argument('--non-member-size', type=int, default=2000,
                       help='Number of non-member samples for MIA')
    
    args = parser.parse_args()
    device = get_device(args)
    
    # Load data
    trans = T.Compose([T.ToTensor(), T.Normalize((.5,.5,.5),(.5,.5,.5))])
    trainset = torchvision.datasets.CIFAR10(dataset_root, train=True, download=allow_download, transform=trans)
    testset = torchvision.datasets.CIFAR10(dataset_root, train=False, download=allow_download, transform=trans)
    
    # Load trained models (you need to train them first using main.py)
    models_dir = './saved_models'
    
    try:
        # Load baseline model and its training metadata
        baseline = CNN().to(device)
        baseline_path = os.path.join(models_dir, '基线模型.pth')
        if os.path.exists(baseline_path):
            checkpoint = torch.load(baseline_path, map_location=device)
            baseline.load_state_dict(checkpoint['model_state_dict'])
            logger.success("Loaded baseline model.")
            
            # Extract training indices if available
            if 'training_indices' in checkpoint:
                training_indices = checkpoint['training_indices']
                sample_level = checkpoint.get('sample_level', True)
                dataset_size = checkpoint.get('dataset_size', len(training_indices))
                num_users = checkpoint.get('num_users', 10)
                logger.success("Found training metadata: %s training samples", len(training_indices))
                mode = "Sample-level" if sample_level else f"User-level ({num_users} users)"
                logger.info("   Mode: %s", mode)
            else:
                logger.warn("No training indices found, using random samples (less accurate).")
                training_indices = None
                sample_level = True
                
        else:
            logger.error("Baseline model not found. Please train models first using main.py.")
            return
        
        # Load DP model
        dp_model = CNN().to(device)
        dp_path = os.path.join(models_dir, 'DP模型.pth')
        if os.path.exists(dp_path):
            checkpoint = torch.load(dp_path, map_location=device)
            dp_model.load_state_dict(checkpoint['model_state_dict'])
            logger.success("Loaded DP model.")
        else:
            logger.error("DP model not found. Please train models first using main.py.")
            return
        
        # Load L2 baseline model (optional)
        l2_baseline = None
        l2_baseline_path = os.path.join(models_dir, 'L2基线模型.pth')
        if os.path.exists(l2_baseline_path):
            l2_baseline = CNN().to(device)
            checkpoint = torch.load(l2_baseline_path, map_location=device)
            l2_baseline.load_state_dict(checkpoint['model_state_dict'])
            weight_decay = checkpoint.get('weight_decay', 0.0)
            logger.success("Loaded L2 baseline model (λ=%s)", weight_decay)
        else:
            logger.info("L2 baseline model not found (optional).")
        
        # Prepare MIA datasets using actual training data
        if training_indices is not None:
            # Use actual training data as members
            member_indices = training_indices[:args.member_size]  # Take subset if needed
            member_set = Subset(trainset, member_indices)
            
            # Use test set as non-members
            non_member_indices = list(range(min(args.non_member_size, len(testset))))
            non_member_set = Subset(testset, non_member_indices)
            
            logger.info("Using ACTUAL training data as members:")
            logger.info("   • Members: %s samples from actual training set", len(member_set))
            logger.info("   • Non-members: %s samples from test set", len(non_member_set))
        else:
            # Fallback to random sampling (legacy behavior)
            member_set, non_member_set, _, _ = prepare_standalone_mia_data(
                trainset, testset, args.member_size, args.non_member_size
            )
            logger.info("Using RANDOM training samples as members (legacy mode):")
            logger.info("   • Members: %s random samples from training set", len(member_set))
            logger.info("   • Non-members: %s samples from test set", len(non_member_set))
        
        # Run MIA evaluation
        member_loader = DataLoader(member_set, batch_size=128, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=128, shuffle=False)
        
        logger.highlight("Standalone Membership Inference Attack Evaluation (audit-only)")
        
        # Run shadow model attack (only attack we use now - more powerful assessment)
        logger.highlight("Shadow Model Attack")
        
        if training_indices is not None:
            # Use actual training data for shadow models
            actual_train_data = Subset(trainset, training_indices)
            logger.info("Using ACTUAL training data for shadow models.")
        else:
            # Fallback to full trainset
            actual_train_data = trainset
            logger.warn("Using full training set for shadow models (less accurate).")
        
        baseline_shadow_results = shadow_model_attack(baseline, member_loader, non_member_loader, actual_train_data, device, eval_data=testset)
        logger.info("Baseline Model:")
        logger.info("   • AUC: %.4f", baseline_shadow_results['auc'])
        logger.info("   • Attack Accuracy: %.4f", baseline_shadow_results['accuracy'])
        
        if l2_baseline is not None:
            l2_baseline_shadow_results = shadow_model_attack(l2_baseline, member_loader, non_member_loader, actual_train_data, device, eval_data=testset)
            logger.info("L2 Baseline Model:")
            logger.info("   • AUC: %.4f", l2_baseline_shadow_results['auc'])
            logger.info("   • Attack Accuracy: %.4f", l2_baseline_shadow_results['accuracy'])
        
        dp_shadow_results = shadow_model_attack(dp_model, member_loader, non_member_loader, actual_train_data, device, eval_data=testset)
        logger.info("DP Model:")
        logger.info("   • AUC: %.4f", dp_shadow_results['auc'])
        logger.info("   • Attack Accuracy: %.4f", dp_shadow_results['accuracy'])
        
        # Overall assessment using shadow attack results only
        dp_auc = dp_shadow_results['auc']
        
        if l2_baseline is not None:
            l2_auc = l2_baseline_shadow_results['auc']
            logger.info("Overall Privacy Protection (Shadow Attack AUC):")
            logger.info("   • DP Model: %.4f", dp_auc)
            logger.info("   • L2 Baseline: %.4f", l2_auc)
            
            if l2_auc < dp_auc:
                diff = dp_auc - l2_auc
                logger.info("   📈 L2 Baseline has %.4f better privacy protection than DP model.", diff)
            elif dp_auc < l2_auc:
                diff = l2_auc - dp_auc
                logger.info("   📈 DP Model has %.4f better privacy protection than L2 baseline.", diff)
            else:
                logger.info("   🔄 Similar privacy protection between DP and L2 baseline.")
        else:
            logger.info("Overall Privacy Protection (Shadow Attack AUC: %.4f)", dp_auc)
            
        if training_indices is not None:
            logger.success("Using actual training data for accurate evaluation.")
        else:
            logger.warn("For accurate evaluation, retrain models or use integrated MIA with --run-mia flag.")
            
        if dp_auc <= 0.55:
            logger.success("DP model provides STRONG privacy protection (audit).")
        elif dp_auc <= 0.65:
            logger.warn("DP model provides MODERATE privacy protection (audit).")
        else:
            logger.error("DP model privacy protection may be INSUFFICIENT (audit).")
        
        logger.info("Using shadow attack only - more realistic privacy assessment (audit).")
            
    except FileNotFoundError as e:
        logger.error("Model files not found: %s", e)
        logger.error("Please first train the models using: uv run training/main.py --mps")

if __name__ == "__main__":
    main() 
