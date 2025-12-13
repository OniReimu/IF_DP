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
from core.config import RANDOM_SEED, get_dataset_location, set_random_seeds


AVAILABLE_DATASETS = tuple(DATASET_REGISTRY.keys())
AVAILABLE_MODELS = tuple(available_models())
VISION_PRETRAINABLE = {"resnet", "resnet18", "efficientnet", "efficientnet_b0", "vit", "vit_b16"}
HF_SEQUENCE_MODELS = {"bert", "qwen", "llama", "llama3.1-8b"}


def get_device(args: argparse.Namespace) -> torch.device:
    if args.cpu:
        return torch.device("cpu")
    if args.mps and torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        idx = 0 if args.cuda_id is None else args.cuda_id
        print(f"Using CUDA:{idx}")
        return torch.device(f"cuda:{idx}")
    print("Using CPU")
    return torch.device("cpu")


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
    return build_model(model_name, **kwargs)


def log_data_split(dataset_name: str, loaders) -> None:
    private_size = len(loaders.private_base)
    public_size = len(loaders.public.dataset) if hasattr(loaders.public, "dataset") else 0
    eval_size = len(loaders.evaluation.dataset) if hasattr(loaders.evaluation, "dataset") else 0
    print(f"\n📊 Data split for {dataset_name}:")
    print(f"   • Private data : {private_size} samples")
    print(f"   • Public data  : {public_size} samples")
    print(f"   • Evaluation   : {eval_size} samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Fisher DP-SGD (modular edition)")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--cuda-id", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

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

    parser.add_argument("--adaptive-clip", action="store_true")
    parser.add_argument("--quantile", type=float, default=0.95)

    parser.add_argument("--sample-level", action="store_true")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--full-complement-noise", action="store_true")
    parser.add_argument("--positively-correlated-noise", action="store_true",
                        help="Use positively correlated Fisher noise (default: negatively correlated)")
    parser.set_defaults(positively_correlated_noise=False)

    parser.add_argument("--run-mia", action="store_true")
    parser.add_argument("--mia-size", type=int, default=1000)
    parser.add_argument("--compare-others", action="store_true")

    privacy_group = parser.add_mutually_exclusive_group()
    privacy_group.add_argument("--use-legacy-accounting", action="store_true")

    epsilon_group = parser.add_mutually_exclusive_group()
    epsilon_group.add_argument("--epsilon", type=float, default=None)
    epsilon_group.add_argument("--target-epsilon", type=float, default=None)

    return parser.parse_args()


def validate_privacy_args(args: argparse.Namespace) -> None:
    if args.use_legacy_accounting:
        if args.epsilon is None:
            raise SystemExit("❌ --use-legacy-accounting requires --epsilon")
        if args.target_epsilon is not None:
            raise SystemExit("❌ Cannot set --target-epsilon with legacy accounting")
    else:
        if args.epsilon is not None:
            raise SystemExit("❌ --epsilon is only valid with --use-legacy-accounting")
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
        print("Cleaning saved models…")
        for path in glob.glob(os.path.join(models_dir, "*.pth")):
            os.remove(path)
            print(f"  removed {path}")

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
    test_loader = loaders.evaluation
    crit_loader = loaders.critical_eval
    priv_base = loaders.private_base
    priv_indices = list(loaders.private_indices)

    log_data_split(args.dataset_name, loaders)

    if args.sample_level:
        print("📊 Using SAMPLE-level DP-SGD (traditional)")
        priv_ds_for_mia = None
    else:
        print(f"👥 Using USER-level DP-SGD ({args.users} synthetic users)")
        priv_ds_for_mia = getattr(priv_loader, "dataset", None)

    baseline = build_model_from_args(args, dataset_builder.num_labels).to(device)
    opt_baseline = torch.optim.SGD(baseline.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0)

    print("\n⚙️  Training baseline…")
    for epoch in tqdm(range(args.epochs)):
        baseline.train()
        for batch in priv_loader:
            features, labels, _ = prepare_batch(batch, device)
            opt_baseline.zero_grad()
            F.cross_entropy(baseline(features), labels).backward()
            opt_baseline.step()

    print("\n🔍  Fisher matrix…")
    fisher_matrix, _ = compute_fisher(
        baseline,
        priv_loader,
        device,
        target_layer=args.dp_layer,
        rho=1e-2,
    )

    print("\n🚀 Fisher-informed DP-SGD…")
    fisher_dp_model = copy.deepcopy(baseline)

    if args.use_legacy_accounting:
        sigma = math.sqrt(2 * math.log(1.25 / args.delta)) / args.epsilon
        display_epsilon = args.epsilon
        noise_multiplier = None
        total_steps = args.epochs * len(priv_loader)
        sample_rate = len(priv_loader) / len(priv_base)
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
        sigma = noise_multiplier * args.clip_radius
        display_epsilon = args.target_epsilon
        print("\n🔒 Using Proper Privacy Accounting (Opacus RDP)")
        print(f"   • Target (ε, δ): ({args.target_epsilon}, {args.delta})")
        print(f"   • Sample rate    : {sample_rate:.4f}")
        print(f"   • Noise multiplier: {noise_multiplier:.4f}")
        print(f"   • Sigma           : {sigma:.4f}")
        print(f"   • Total steps     : {total_steps}")

    fisher_dp_model = train_with_dp(
        fisher_dp_model,
        priv_loader,
        fisher_matrix,
        epsilon=display_epsilon,
        delta=args.delta,
        sigma=sigma,
        full_complement_noise=args.full_complement_noise,
        clip_radius=args.clip_radius,
        k=args.k,
        device=device,
        target_layer=args.dp_layer,
        adaptive_clip=args.adaptive_clip,
        quantile=args.quantile,
        sample_level=args.sample_level,
        epochs=args.epochs,
        positive_noise_correlation=args.positively_correlated_noise,
    )

    vanilla_dp_model: Optional[torch.nn.Module] = None
    dp_sat_model: Optional[torch.nn.Module] = None

    if args.compare_others:
        print("\n📐 Vanilla DP-SGD (comparison)…")
        vanilla_dp_model = copy.deepcopy(baseline)
        vanilla_dp_model = train_with_vanilla_dp(
            vanilla_dp_model,
            priv_loader,
            epsilon=display_epsilon,
            delta=args.delta,
            sigma=None if args.use_legacy_accounting else sigma,
            clip_radius=args.clip_radius,
            device=device,
            target_layer=args.dp_layer,
            adaptive_clip=args.adaptive_clip,
            quantile=args.quantile,
            sample_level=args.sample_level,
            epochs=args.epochs,
        )

        print("\n🔺 DP-SAT: Sharpness-Aware Training (comparison)…")
        dp_sat_model = copy.deepcopy(baseline)
        dp_sat_model = train_with_dp_sat(
            dp_sat_model,
            priv_loader,
            epsilon=display_epsilon,
            delta=args.delta,
            sigma=None if args.use_legacy_accounting else sigma,
            clip_radius=args.clip_radius,
            device=device,
            target_layer=args.dp_layer,
            adaptive_clip=args.adaptive_clip,
            quantile=args.quantile,
            sample_level=args.sample_level,
            epochs=args.epochs,
            lambda_flatness=args.lambda_flatness,
        )

        if not args.use_legacy_accounting and noise_multiplier is not None:
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
                print("\n✅ Fair privacy comparison: matched ε across methods")

    def crit_accuracy(model):
        if crit_loader is None:
            return float("nan")
        return accuracy(model, crit_loader, device)

    dp_mode = "Sample-level" if args.sample_level else f"User-level ({args.users} users)"
    baseline_acc = accuracy(baseline, test_loader, device)
    fisher_acc = accuracy(fisher_dp_model, test_loader, device)
    vanilla_acc = accuracy(vanilla_dp_model, test_loader, device) if vanilla_dp_model else float("nan")
    dp_sat_acc = accuracy(dp_sat_model, test_loader, device) if dp_sat_model else float("nan")

    print(f"\n📊 Accuracy summary ({dp_mode})")
    print(f" baseline   : {baseline_acc:6.2f}% (crit {crit_accuracy(baseline):5.2f}%)")
    print(f" Fisher DP  : {fisher_acc:6.2f}% (crit {crit_accuracy(fisher_dp_model):5.2f}%)")
    if vanilla_dp_model is not None:
        print(f" Vanilla DP : {vanilla_acc:6.2f}% (crit {crit_accuracy(vanilla_dp_model):5.2f}%)")
    if dp_sat_model is not None:
        print(f" DP-SAT     : {dp_sat_acc:6.2f}% (crit {crit_accuracy(dp_sat_model):5.2f}%)")

    if vanilla_dp_model is not None:
        print(f" Fisher vs Vanilla: {fisher_acc - vanilla_acc:+5.2f}%")
    if dp_sat_model is not None:
        print(f" Fisher vs DP-SAT : {fisher_acc - dp_sat_acc:+5.2f}%")

    print("\n💾 Saving models for MIA evaluation…")
    baseline_path = os.path.join(models_dir, "baseline_model.pth")
    torch.save(
        {
            "model_state_dict": baseline.state_dict(),
            "model_name": args.model_name,
            "dataset": args.dataset_name,
            "accuracy": baseline_acc,
            "critical_accuracy": crit_accuracy(baseline),
            "training_indices": priv_indices,
            "dataset_size": args.dataset_size,
            "sample_level": args.sample_level,
            "num_users": args.users if not args.sample_level else None,
        },
        baseline_path,
    )
    print(f"✅ Saved baseline model to {baseline_path}")

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
            "adaptive_clip": args.adaptive_clip,
            "quantile": args.quantile if args.adaptive_clip else None,
            "training_indices": priv_indices,
            "dataset_size": args.dataset_size,
            "sample_level": args.sample_level,
            "num_users": args.users if not args.sample_level else None,
        },
        fisher_path,
    )
    print(f"✅ Saved Fisher DP model to {fisher_path}")

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
                "adaptive_clip": args.adaptive_clip,
                "quantile": args.quantile if args.adaptive_clip else None,
                "training_indices": priv_indices,
                "dataset_size": args.dataset_size,
                "sample_level": args.sample_level,
                "num_users": args.users if not args.sample_level else None,
            },
            vanilla_path,
        )
        print(f"✅ Saved Vanilla DP model to {vanilla_path}")

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
                "adaptive_clip": args.adaptive_clip,
                "quantile": args.quantile if args.adaptive_clip else None,
                "lambda_flatness": args.lambda_flatness,
                "training_indices": priv_indices,
                "dataset_size": args.dataset_size,
                "sample_level": args.sample_level,
                "num_users": args.users if not args.sample_level else None,
            },
            dp_sat_path,
        )
        print(f"✅ Saved DP-SAT model to {dp_sat_path}")

    print("\n🛡️  To evaluate privacy protection, run with --run-mia or invoke mia.py directly.")

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
        )


if __name__ == "__main__":
    main()
