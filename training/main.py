#!/usr/bin/env python3
"""Unified IF-DP training entry point with modular datasets/models."""

import argparse
import copy
import glob
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.fisher_dp_sgd import compute_fisher, train_with_dp
from core.dp_sgd import train_with_vanilla_dp
from core.dp_sat import train_with_dp_sat

from data import DATASET_REGISTRY, DatasetConfig, build_dataset_builder
from data.common import prepare_batch
from models import available_models, build_model
from core.mia import evaluate_membership_inference
from core.privacy_accounting import (
    compute_actual_epsilon,
    get_privacy_params_for_target_epsilon,
    print_privacy_summary,
    validate_privacy_comparison,
)
from config import RANDOM_SEED, get_dataset_location, get_logger, set_random_seeds
from core.device_utils import resolve_device, maybe_wrap_model_for_multi_gpu


AVAILABLE_DATASETS = tuple(DATASET_REGISTRY.keys())
AVAILABLE_MODELS = tuple(available_models())
VISION_PRETRAINABLE = {"resnet", "resnet18", "efficientnet", "efficientnet_b0", "vit", "vit_b16"}
HF_SEQUENCE_MODELS = {"bert", "qwen", "llama", "llama3.1-8b"}
logger = get_logger("training")


def get_device(args: argparse.Namespace) -> torch.device:
    return resolve_device(args)


def accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for batch in loader:
            features, labels, _ = prepare_batch(batch, device)
            preds = model(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / max(1, total)


def build_model_from_args(args: argparse.Namespace, num_labels: int) -> torch.nn.Module:
    kwargs = {"num_labels": num_labels}
    model_name = args.model_name.lower()
    if args.use_pretrained_vision and model_name in VISION_PRETRAINABLE:
        kwargs["pretrained"] = True
    if args.model_checkpoint and model_name in HF_SEQUENCE_MODELS:
        kwargs["checkpoint"] = args.model_checkpoint
    model = build_model(model_name, **kwargs)
    return maybe_wrap_model_for_multi_gpu(model, args)


def log_data_split(dataset_name: str, loaders) -> None:
    private_size = len(loaders.private_base)
    public_size = len(loaders.public.dataset) if hasattr(loaders.public, "dataset") else 0
    eval_size = len(loaders.evaluation.dataset) if hasattr(loaders.evaluation, "dataset") else 0
    logger.info("Data split for %s:", dataset_name)
    logger.info("   • Private data : %s samples", private_size)
    logger.info("   • Public data  : %s samples", public_size)
    logger.info("   • Evaluation   : %s samples", eval_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Fisher DP-SGD (modular edition)")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cuda-id", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help='Comma-separated CUDA device ids for multi-GPU execution (e.g., "0,1,2")',
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable torch.nn.DataParallel across the requested CUDA devices",
    )

    parser.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset_name",
        choices=AVAILABLE_DATASETS,
        default="cifar10",
        help="Dataset to train on",
    )
    parser.add_argument(
        "--model",
        "--model-type",
        dest="model_name",
        choices=AVAILABLE_MODELS,
        default="cnn",
        help="Model architecture",
    )
    parser.add_argument("--dataset-size", type=int, default=50000)
    parser.add_argument("--public-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dp-epochs", type=int, default=None,
                       help='Number of DP fine-tuning epochs. If None, uses max(1, ceil(epochs/10)). '
                            'Lower values may help DP-SAT perform better.')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--critical-label", type=int, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--model-checkpoint", type=str, default=None,
                        help="Optional HF checkpoint override for language models")
    parser.add_argument("--use-pretrained-vision", action="store_true",
                        help="Load torchvision backbones with pretrained weights when supported")

    parser.add_argument("--clean", action="store_true", help="Remove saved models before training")

    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip-radius", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--dp-layer", type=str, default="conv1")
    parser.add_argument("--lambda-flatness", type=float, default=0.01)

    parser.add_argument("--sample-level", action="store_true")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument(
        "--positively-correlated-noise",
        action="store_true",
        help="(Unsupported) Mechanism-aligned Fisher DP-SGD uses negatively correlated F^{-1} noise.",
    )
    parser.set_defaults(positively_correlated_noise=False)

    parser.add_argument("--run-mia", action="store_true")
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
    parser.add_argument("--compare-others", action="store_true")

    parser.add_argument("--target-epsilon", type=float, default=None)

    return parser.parse_args()


def validate_privacy_args(args: argparse.Namespace) -> None:
    if args.target_epsilon is None:
        args.target_epsilon = 10.0


def main() -> None:
    args = parse_args()
    validate_privacy_args(args)
    set_random_seeds()

    device = get_device(args)
    models_dir = "./saved_models"
    os.makedirs(models_dir, exist_ok=True)

    if args.clean:
        logger.warn("Cleaning saved models…")
        for path in glob.glob(os.path.join(models_dir, "*.pth")):
            os.remove(path)
            logger.info("  removed %s", path)

    dataset_root, allow_download = get_dataset_location(dataset_key=args.dataset_name)
    dataset_builder = build_dataset_builder(args.dataset_name)
    dataset_config = DatasetConfig(
        dataset_root=dataset_root,
        allow_download=allow_download,
        dataset_size=args.dataset_size,
        public_ratio=args.public_ratio,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        sample_level=args.sample_level,
        num_users=args.users,
        critical_label=args.critical_label,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        seed=RANDOM_SEED,
    )
    loaders = dataset_builder.build(dataset_config)
    priv_loader = loaders.private
    pub_loader = loaders.public
    calib_loader = getattr(loaders, "calibration", None)
    test_loader = loaders.evaluation
    crit_loader = loaders.critical_eval
    priv_base = loaders.private_base
    priv_indices = list(loaders.private_indices)

    log_data_split(args.dataset_name, loaders)

    if args.sample_level:
        logger.info("Using sample-level DP-SGD (traditional).")
        priv_ds_for_mia = None
    else:
        logger.info("Using user-level DP-SGD (%s synthetic users).", args.users)
        priv_ds_for_mia = getattr(priv_loader, "dataset", None)

    baseline = build_model_from_args(args, dataset_builder.num_labels).to(device)
    opt_baseline = torch.optim.SGD(baseline.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0)

    baseline_loader = pub_loader
    if baseline_loader is None:
        raise ValueError("Public loader is required for baseline pretrain in training/main.py.")

    logger.highlight("Training baseline (public-only)")
    for epoch in tqdm(range(args.epochs)):
        baseline.train()
        for batch in baseline_loader:
            features, labels, _ = prepare_batch(batch, device)
            opt_baseline.zero_grad()
            F.cross_entropy(baseline(features), labels).backward()
            opt_baseline.step()

    logger.highlight("Fisher matrix")
    fisher_loader = pub_loader
    if fisher_loader is None:
        raise ValueError("Public loader is required for Fisher estimation but was not available.")
    logger.info("Fisher estimation: using public pretrain data.")
    fisher_matrix, _ = compute_fisher(
        baseline,
        fisher_loader,
        device,
        target_layer=args.dp_layer,
        rho=1e-2,
    )

    logger.highlight("Fisher-informed DP-SGD")
    fisher_dp_model = copy.deepcopy(baseline)

    if args.sample_level:
        sample_rate = float(args.batch_size) / float(len(priv_base))
    else:
        sample_rate = len(priv_loader) / len(priv_base)
    steps_per_epoch = len(priv_loader)
    noise_multiplier, total_steps = get_privacy_params_for_target_epsilon(
        target_epsilon=args.target_epsilon,
        target_delta=args.delta,
        sample_rate=sample_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    sigma = noise_multiplier
    display_epsilon = args.target_epsilon
    logger.highlight("Using Proper Privacy Accounting (Opacus RDP)")
    logger.info("   • Target (ε, δ): (%.4f, %.1e)", args.target_epsilon, args.delta)
    logger.info("   • Sample rate    : %.4f", sample_rate)
    logger.info("   • Noise multiplier: %.4f", noise_multiplier)
    logger.info("   • Sigma (multiplier): %.4f", sigma)
    logger.info("   • Vanilla noise std: σ×Δ₂ = %.4f×%.3f = %.4f", sigma, args.clip_radius, sigma * args.clip_radius)
    logger.info("   • Fisher noise std: σ×C_F where C_F is calibrated from public data inside Fisher DP-SGD")
    logger.info("   • Total steps     : %s", total_steps)

    fisher_dp_model = train_with_dp(
        fisher_dp_model,
        priv_loader,
        fisher_matrix,
        epsilon=display_epsilon,
        delta=args.delta,
        sigma=sigma,
        clip_radius=args.clip_radius,
        k=args.k,
        device=device,
        target_layer=args.dp_layer,
        sample_level=args.sample_level,
        epochs=args.epochs,
        positive_noise_correlation=args.positively_correlated_noise,
        dp_epochs=args.dp_epochs,
    )

    vanilla_dp_model: Optional[torch.nn.Module] = None
    dp_sat_model: Optional[torch.nn.Module] = None

    if args.compare_others:
        logger.highlight("Vanilla DP-SGD (comparison)")
        vanilla_dp_model = copy.deepcopy(baseline)
        vanilla_dp_model = train_with_vanilla_dp(
            vanilla_dp_model,
            priv_loader,
            epsilon=display_epsilon,
            delta=args.delta,
            sigma=sigma,
            clip_radius=args.clip_radius,
            device=device,
            target_layer=args.dp_layer,
            sample_level=args.sample_level,
            epochs=args.epochs,
            dp_epochs=args.dp_epochs,
        )

        logger.highlight("DP-SAT: Sharpness-Aware Training (comparison)")
        dp_sat_model = copy.deepcopy(baseline)
        dp_sat_model = train_with_dp_sat(
            dp_sat_model,
            priv_loader,
            epsilon=display_epsilon,
            delta=args.delta,
            sigma=sigma,
            clip_radius=args.clip_radius,
            device=device,
            target_layer=args.dp_layer,
            sample_level=args.sample_level,
            epochs=args.epochs,
            lambda_flatness=args.lambda_flatness,
        )

        if noise_multiplier is not None:
            actual_eps = compute_actual_epsilon(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                steps=total_steps,
                target_delta=args.delta,
            )
            is_fair = validate_privacy_comparison(actual_eps, actual_eps)
            print_privacy_summary(
                method_name="Fisher DP-SGD",
                target_epsilon=args.target_epsilon,
                actual_epsilon=actual_eps,
                delta=args.delta,
                noise_multiplier=noise_multiplier,
                steps=total_steps,
                sample_rate=sample_rate,
            )
            print_privacy_summary(
                method_name="Vanilla DP-SGD",
                target_epsilon=args.target_epsilon,
                actual_epsilon=actual_eps,
                delta=args.delta,
                noise_multiplier=noise_multiplier,
                steps=total_steps,
                sample_rate=sample_rate,
            )
            print_privacy_summary(
                method_name="DP-SAT",
                target_epsilon=args.target_epsilon,
                actual_epsilon=actual_eps,
                delta=args.delta,
                noise_multiplier=noise_multiplier,
                steps=total_steps,
                sample_rate=sample_rate,
            )
            if is_fair:
                logger.success("Fair privacy comparison: matched ε across methods.")

    def crit_accuracy(model):
        if crit_loader is None:
            return float("nan")
        return accuracy(model, crit_loader, device)

    dp_mode = "Sample-level" if args.sample_level else f"User-level ({args.users} users)"
    baseline_acc = accuracy(baseline, test_loader, device)
    fisher_acc = accuracy(fisher_dp_model, test_loader, device)
    vanilla_acc = accuracy(vanilla_dp_model, test_loader, device) if vanilla_dp_model else float("nan")
    dp_sat_acc = accuracy(dp_sat_model, test_loader, device) if dp_sat_model else float("nan")

    logger.highlight(f"Accuracy summary ({dp_mode})")
    logger.info(" baseline   : %6.2f%% (crit %5.2f%%)", baseline_acc, crit_accuracy(baseline))
    logger.info(" Fisher DP  : %6.2f%% (crit %5.2f%%)", fisher_acc, crit_accuracy(fisher_dp_model))
    if vanilla_dp_model is not None:
        logger.info(" Vanilla DP : %6.2f%% (crit %5.2f%%)", vanilla_acc, crit_accuracy(vanilla_dp_model))
    if dp_sat_model is not None:
        logger.info(" DP-SAT     : %6.2f%% (crit %5.2f%%)", dp_sat_acc, crit_accuracy(dp_sat_model))

    if vanilla_dp_model is not None:
        logger.info(" Fisher vs Vanilla: %+5.2f%%", fisher_acc - vanilla_acc)
    if dp_sat_model is not None:
        logger.info(" Fisher vs DP-SAT : %+5.2f%%", fisher_acc - dp_sat_acc)

    logger.highlight("Saving models for MIA evaluation")
    baseline_path = os.path.join(models_dir, "baseline_model.pth")
    torch.save(
        {
            "model_state_dict": baseline.state_dict(),
            "model_name": args.model_name,
            "dataset": args.dataset_name,
            "accuracy": baseline_acc,
            "critical_accuracy": crit_accuracy(baseline),
            "dataset_size": args.dataset_size,
            "sample_level": args.sample_level,
            "num_users": args.users if not args.sample_level else None,
        },
        baseline_path,
    )
    logger.success("Saved baseline model to %s", baseline_path)

    fisher_path = os.path.join(models_dir, "fisher_dp_model.pth")
    torch.save(
        {
            "model_state_dict": fisher_dp_model.state_dict(),
            "model_name": args.model_name,
            "dataset": args.dataset_name,
            "accuracy": fisher_acc,
            "critical_accuracy": crit_accuracy(fisher_dp_model),
            "epsilon": display_epsilon,
            "clip_radius": args.clip_radius,
            "dataset_size": args.dataset_size,
            "sample_level": args.sample_level,
            "num_users": args.users if not args.sample_level else None,
        },
        fisher_path,
    )
    logger.success("Saved Fisher DP model to %s", fisher_path)

    if vanilla_dp_model is not None:
        vanilla_path = os.path.join(models_dir, "vanilla_dp_model.pth")
        torch.save(
            {
                "model_state_dict": vanilla_dp_model.state_dict(),
                "model_name": args.model_name,
                "dataset": args.dataset_name,
                "accuracy": vanilla_acc,
                "critical_accuracy": crit_accuracy(vanilla_dp_model),
                "epsilon": display_epsilon,
                "clip_radius": args.clip_radius,
                "dataset_size": args.dataset_size,
                "sample_level": args.sample_level,
                "num_users": args.users if not args.sample_level else None,
            },
            vanilla_path,
        )
        logger.success("Saved Vanilla DP model to %s", vanilla_path)

    if dp_sat_model is not None:
        dp_sat_path = os.path.join(models_dir, "dp_sat_model.pth")
        torch.save(
            {
                "model_state_dict": dp_sat_model.state_dict(),
                "model_name": args.model_name,
                "dataset": args.dataset_name,
                "accuracy": dp_sat_acc,
                "critical_accuracy": crit_accuracy(dp_sat_model),
                "epsilon": display_epsilon,
                "clip_radius": args.clip_radius,
                "lambda_flatness": args.lambda_flatness,
                "dataset_size": args.dataset_size,
                "sample_level": args.sample_level,
                "num_users": args.users if not args.sample_level else None,
            },
            dp_sat_path,
        )
        logger.success("Saved DP-SAT model to %s", dp_sat_path)

    logger.info("To evaluate privacy protection, run with --run-mia or invoke mia.py directly.")

    if args.run_mia:
        eval_dataset = getattr(test_loader, "dataset", None)
        evaluate_membership_inference(
            baseline,
            fisher_dp_model,
            priv_base,
            eval_dataset,
            priv_indices,
            priv_ds_for_mia,
            args.users,
            args.mia_size,
            args.sample_level,
            device,
            vanilla_dp_model,
            dp_sat_model,
            None,
            mia_level=args.mia_level,
            mia_attack=args.mia_attack,
        )


if __name__ == "__main__":
    main()
