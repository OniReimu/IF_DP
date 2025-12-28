#!/usr/bin/env python3
"""Run MIA audit on cached ablation models without retraining."""

from __future__ import annotations

import argparse
import os
import sys
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_logger, get_dataset_location, get_random_seed, set_random_seeds
from core.device_utils import resolve_device
from core.mia import (
    align_mia_datasets,
    prepare_mia_data_sample_level,
    prepare_mia_data_user_level,
    prepare_shadow_splits,
    prepare_user_level_groups,
    shadow_model_attack,
    user_level_loss_attack,
    prepare_user_shadow_splits,
    user_level_shadow_attack,
)
from data import DATASET_REGISTRY, DatasetConfig, build_dataset_builder
from data.common import prepare_batch
from models import create_model

logger = get_logger("mia_only")
AVAILABLE_DATASETS = tuple(DATASET_REGISTRY.keys())


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data in loader:
            inputs, labels, _ = prepare_batch(batch_data, device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return 100.0 * correct / total if total > 0 else 0.0


def _sanitize_cache_key(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def build_pretrain_cache_path(models_dir: str, dataset_name: str, model_type: str, epochs: int, non_iid: bool) -> str:
    ds = _sanitize_cache_key(dataset_name)
    iid_mode = "noniid" if non_iid else "iid"
    return os.path.join(models_dir, f"Pretrain_{ds}_{model_type}_{int(epochs)}_public_{iid_mode}.pth")


def safe_torch_load(path, map_location=None):
    load_kwargs = {"map_location": map_location}
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MIA audit runner (cached ablation models)")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cuda-id", type=int)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset_name",
        choices=AVAILABLE_DATASETS,
        default="cifar10",
    )
    parser.add_argument("--model-type", type=str, default="cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--non-iid", action="store_true")
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--public-ratio", type=float, default=None)
    parser.add_argument("--public-pretrain-exclude-classes", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-seq-length", type=int, default=512)

    parser.add_argument("--sample-level", action="store_true")
    parser.add_argument("--users", type=int, default=10)

    parser.add_argument("--mia-size", type=int, default=1000)
    parser.add_argument(
        "--mia-level",
        type=str,
        default="auto",
        choices=["auto", "sample", "user"],
        help="MIA mode: auto follows DP mode; sample forces sample-level; user forces user-level.",
    )
    parser.add_argument(
        "--mia-attack",
        type=str,
        default="shadow",
        choices=["shadow", "loss"],
        help="User-level MIA attack: shadow (default) or loss. Ignored for sample-level MIA.",
    )
    parser.add_argument("--shadow-epochs", type=int, default=3)
    parser.add_argument("--models-dir", type=str, default="./saved_models")

    return parser.parse_args()


def _parse_excluded_classes(value: str):
    if not value:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in parts] if parts else None


def _apply_ablation_defaults(args: argparse.Namespace) -> None:
    if args.dataset_size is None:
        args.dataset_size = 30000 if not args.non_iid else 5000

    if args.public_ratio is None:
        if args.non_iid:
            args.public_ratio = 1.0
        else:
            args.public_ratio = 0.667


def load_model(model_type: str, num_labels: int, ckpt_path: str, device) -> torch.nn.Module:
    model = create_model(model_type, num_labels=num_labels)
    checkpoint = safe_torch_load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def main() -> None:
    args = parse_args()
    _apply_ablation_defaults(args)

    set_random_seeds()
    device = resolve_device(args)

    dataset_root, allow_download = get_dataset_location(dataset_key=args.dataset_name)
    dataset_builder = build_dataset_builder(args.dataset_name)
    dataset_num_labels = getattr(dataset_builder, "num_labels", None)
    if dataset_num_labels is None:
        raise RuntimeError("Dataset builder must expose num_labels for MIA-only script.")

    dataset_config = DatasetConfig(
        dataset_root=dataset_root,
        allow_download=allow_download,
        dataset_size=args.dataset_size,
        public_ratio=args.public_ratio,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        sample_level=args.sample_level,
        num_users=args.users,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        public_pretrain_exclude_classes=_parse_excluded_classes(args.public_pretrain_exclude_classes),
        non_iid=args.non_iid,
        seed=get_random_seed(),
    )

    loaders = dataset_builder.build(dataset_config)
    priv_loader = loaders.private
    eval_loader = loaders.evaluation
    priv_base = loaders.private_base
    priv_idx = loaders.private_indices
    priv_ds = None if args.sample_level else getattr(priv_loader, "dataset", None)
    eval_dataset = getattr(eval_loader, "dataset", None)

    if eval_dataset is None:
        raise RuntimeError("Evaluation dataset is required for MIA sampling.")

    models_dir = args.models_dir
    ds_tag = _sanitize_cache_key(args.dataset_name)
    iid_tag = "noniid" if args.non_iid else "iid"

    model_paths = {
        "Baseline (Public Only)": build_pretrain_cache_path(
            models_dir, args.dataset_name, args.model_type, args.epochs, args.non_iid
        ),
        "Vanilla DP-SGD": os.path.join(models_dir, f"Vanilla_DP_{ds_tag}_Ablation_{iid_tag}.pth"),
        "Vanilla DP-SGD + DP-SAT": os.path.join(models_dir, f"Vanilla_DPSAT_{ds_tag}_Ablation_{iid_tag}.pth"),
        "Fisher DP + Normal": os.path.join(models_dir, f"Fisher_Normal_{ds_tag}_Ablation_{iid_tag}.pth"),
        "Fisher DP + DP-SAT": os.path.join(models_dir, f"Fisher_DPSAT_{ds_tag}_Ablation_{iid_tag}.pth"),
        "Fisher DP + Normal + Calib": os.path.join(models_dir, f"Fisher_Normal_{ds_tag}_Calibrated_Ablation_{iid_tag}.pth"),
        "Fisher DP + DP-SAT + Calib": os.path.join(models_dir, f"Fisher_DPSAT_{ds_tag}_Calibrated_Ablation_{iid_tag}.pth"),
    }

    models_to_evaluate = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            logger.warn("Missing model: %s", path)
            continue
        try:
            models_to_evaluate[name] = load_model(args.model_type, dataset_num_labels, path, device)
        except Exception as exc:
            logger.warn("Failed to load %s (%s)", name, exc)

    if not models_to_evaluate:
        raise RuntimeError("No cached models found to evaluate.")

    logger.highlight("Running cached MIA audit")
    eval_source = eval_dataset

    mia_train_data, mia_priv_ds, transform_aligned = align_mia_datasets(
        priv_base, priv_ds, eval_source, args.users
    )
    if transform_aligned:
        logger.info("MIA transforms: aligned to eval.")

    mia_use_user = False
    mia_use_sample = False
    if args.mia_level == "auto":
        mia_use_sample = args.sample_level
        mia_use_user = not args.sample_level
    elif args.mia_level == "sample":
        mia_use_sample = True
    elif args.mia_level == "user":
        mia_use_user = True

    if mia_use_user and mia_priv_ds is None:
        logger.warn("User-level MIA requested but priv_ds is unavailable; falling back to sample-level.")
        mia_use_user = False
        mia_use_sample = True

    logger.info("MIA mode: %s", "user-level" if mia_use_user else "sample-level")
    if mia_use_user:
        logger.info("User-level MIA attack: %s", args.mia_attack)

    member_loader = None
    non_member_loader = None
    shadow_splits = None
    if mia_use_sample:
        member_set, non_member_set = prepare_mia_data_sample_level(
            mia_train_data, eval_source, priv_idx, args.mia_size
        )
        member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
        logger.info("MIA samples: %s members / %s non-members", len(member_set), len(non_member_set))
        shadow_splits = prepare_shadow_splits(mia_train_data, eval_source, seed=get_random_seed())
        logger.info(
            "Shadow split fixed: %s members / %s non-members",
            len(shadow_splits["shadow_indices"]),
            len(shadow_splits["shadow_non_member_indices"]),
        )
    else:
        label = "user-level shadow audit" if args.mia_attack == "shadow" else "user-level loss audit"
        logger.info("MIA samples: %s.", label)

    user_groups = None
    user_shadow_splits = None
    if mia_use_user:
        user_groups = prepare_user_level_groups(mia_priv_ds, eval_source, args.users, args.mia_size)
        logger.info("User-level audit: %s users", len(user_groups[0]))
        if args.mia_attack == "shadow":
            _, _, non_member_user_ds = user_groups
            user_shadow_splits = prepare_user_shadow_splits(
                mia_priv_ds,
                eval_source,
                args.users,
                seed=get_random_seed(),
                eval_user_ds=non_member_user_ds,
            )
            logger.info(
                "User shadow split fixed: %s member users / %s non-members",
                len(user_shadow_splits["shadow_user_ids"]),
                len(user_shadow_splits["shadow_non_member_user_ids"]),
            )

    mia_results = {}
    if mia_use_sample:
        logger.info("Shadow attack results:")
    else:
        label = "User-level shadow attack results:" if args.mia_attack == "shadow" else "User-level loss attack results:"
        logger.info(label)
    for model_name, model in models_to_evaluate.items():
        mia_results[model_name] = {}
        if mia_use_sample:
            shadow_result = shadow_model_attack(
                model,
                member_loader,
                non_member_loader,
                mia_train_data,
                device,
                eval_source,
                shadow_epochs=args.shadow_epochs,
                shadow_splits=shadow_splits,
            )
            mia_results[model_name]["shadow_auc_star"] = shadow_result.get(
                "auc_star",
                max(shadow_result["auc"], 1.0 - shadow_result["auc"]),
            )
            mia_results[model_name]["shadow_adv"] = shadow_result.get(
                "adv",
                abs(shadow_result["auc"] - 0.5),
            )
            logger.info(
                "   • %s: AUC*=%.4f  |AUC-0.5|=%.4f",
                model_name,
                mia_results[model_name]["shadow_auc_star"],
                mia_results[model_name]["shadow_adv"],
            )
        if mia_use_user:
            member_groups, non_member_groups, non_member_user_ds = user_groups
            if args.mia_attack == "shadow":
                user_result = user_level_shadow_attack(
                    model,
                    member_groups,
                    non_member_groups,
                    mia_priv_ds,
                    non_member_user_ds,
                    device,
                    shadow_epochs=args.shadow_epochs,
                    shadow_splits=user_shadow_splits,
                )
            else:
                user_result = user_level_loss_attack(
                    model,
                    member_groups,
                    non_member_groups,
                    mia_priv_ds,
                    non_member_user_ds,
                    device,
                )
            mia_results[model_name]["user_auc_star"] = user_result["auc_star"]
            mia_results[model_name]["user_adv"] = user_result["adv"]
            logger.info(
                "   • %s: User AUC*=%.4f  |AUC-0.5|=%.4f",
                model_name,
                mia_results[model_name]["user_auc_star"],
                mia_results[model_name]["user_adv"],
            )

    logger.highlight("MIA Summary")
    baseline_key = "Baseline (Public Only)"
    baseline_auc_star = None
    if mia_use_sample:
        logger.info("Sample-level audit AUC*:")
        for model_name, res in mia_results.items():
            logger.info("   • %s: %.4f", model_name, res["shadow_auc_star"])
        logger.info("Sample-level audit |AUC-0.5|:")
        for model_name, res in mia_results.items():
            logger.info("   • %s: %.4f", model_name, res["shadow_adv"])
        baseline_auc_star = mia_results.get(baseline_key, {}).get("shadow_auc_star")
    else:
        label = "User-level shadow AUC*:" if args.mia_attack == "shadow" else "User-level loss AUC*:"
        logger.info(label)
        for model_name, res in mia_results.items():
            logger.info("   • %s: %.4f", model_name, res["user_auc_star"])
        label = "User-level shadow |AUC-0.5|:" if args.mia_attack == "shadow" else "User-level loss |AUC-0.5|:"
        logger.info(label)
        for model_name, res in mia_results.items():
            logger.info("   • %s: %.4f", model_name, res["user_adv"])
        baseline_auc_star = mia_results.get(baseline_key, {}).get("user_auc_star")

    if baseline_auc_star is not None:
        logger.info("MIA sanity (baseline AUC*): %.4f (target ~0.5).", baseline_auc_star)
        if abs(baseline_auc_star - 0.5) > 0.05:
            logger.warn("MIA sanity: baseline deviates from 0.5; check member/non-member matching.")

    acc_results = {}
    for model_name, model in models_to_evaluate.items():
        acc_results[model_name] = accuracy(model, eval_loader, device)

    logger.info("Privacy vs Accuracy Tradeoff (%s):", "user-level" if mia_use_user else "sample-level")
    logger.info("   Model                          Accuracy  AttackAUC*  |AUC-0.5|")
    logger.info(f"   {'─'*30} ─────────  ─────────  ─────────")

    model_order = [
        "Baseline (Public Only)",
        "Vanilla DP-SGD",
        "Vanilla DP-SGD + DP-SAT",
        "Fisher DP + Normal",
        "Fisher DP + DP-SAT",
        "Fisher DP + Normal + Calib",
        "Fisher DP + DP-SAT + Calib",
    ]

    for name in model_order:
        if name not in acc_results:
            continue
        acc = acc_results[name]
        if name == "Baseline (Public Only)":
            logger.info(f"   {name:30} {acc:5.1f}%     {'-':>6}    {'-':>7}")
            continue
        if mia_use_sample:
            aucs = mia_results[name]["shadow_auc_star"]
            adv = mia_results[name]["shadow_adv"]
        else:
            aucs = mia_results[name]["user_auc_star"]
            adv = mia_results[name]["user_adv"]
        logger.info(f"   {name:30} {acc:5.1f}%     {aucs:.4f}   {adv:.4f}")


if __name__ == "__main__":
    main()
