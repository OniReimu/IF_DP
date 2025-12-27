#!/usr/bin/env python3
"""Plot ε(δ) privacy profile using ablation-style arguments."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install with `uv pip install matplotlib`."
    ) from exc

from opacus.accountants import RDPAccountant

# Ensure project root on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_logger, get_dataset_location, get_random_seed, set_random_seeds
from core.privacy_accounting import get_privacy_params_for_target_epsilon
from data import DATASET_REGISTRY, DatasetConfig, build_dataset_builder

logger = get_logger("privacy_profile")
AVAILABLE_DATASETS = tuple(DATASET_REGISTRY.keys())


def compute_epsilons(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    deltas: Iterable[float],
) -> List[float]:
    """Compute ε for each δ using a fixed RDP accountant history."""
    accountant = RDPAccountant()
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    return [accountant.get_epsilon(delta=float(d)) for d in deltas]


def _parse_excluded_classes(value: str) -> Optional[Sequence[int]]:
    if not value:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if not parts:
        return None
    try:
        return [int(v) for v in parts]
    except ValueError as exc:
        raise SystemExit(f"Invalid --public-pretrain-exclude-classes: '{value}'") from exc


def _apply_ablation_defaults(args: argparse.Namespace) -> None:
    if args.dataset_size is None:
        args.dataset_size = 30000 if not args.non_iid else 5000

    if args.public_ratio is None:
        if args.non_iid:
            args.public_ratio = 1.0
        else:
            # IID mode default approximates CIFAR-10 public split size.
            args.public_ratio = 0.667


def _resolve_dp_epochs(args: argparse.Namespace) -> int:
    if args.dp_epochs is not None:
        return int(args.dp_epochs)
    return max(1, int(math.ceil(args.epochs / 10)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ε(δ) privacy profile using ablation-style arguments."
    )

    # Dataset arguments (match ablation defaults)
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset_name",
        choices=AVAILABLE_DATASETS,
        default="cifar10",
        help="Dataset identifier registered in the data package",
    )
    parser.add_argument("--non-iid", action="store_true")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="Number of private samples to draw from the dataset",
    )
    parser.add_argument(
        "--public-ratio",
        type=float,
        default=None,
        help="Fraction of remaining samples reserved for public pretrain",
    )
    parser.add_argument(
        "--public-pretrain-exclude-classes",
        type=str,
        default="",
        help="Comma-separated class indices to exclude from public pretrain",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-seq-length", type=int, default=512)

    # DP arguments (match ablation defaults)
    parser.add_argument("--target-epsilon", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip-radius", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dp-epochs", type=int, default=None)
    parser.add_argument("--sample-level", action="store_true")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument(
        "--accounting-mode",
        type=str,
        default="repo_q_eff",
        choices=["repo_q_eff", "repo", "user_poisson", "both"],
        help=(
            "Which sampling-rate model to use for user-level DP. "
            "'repo_q_eff' (alias 'repo') matches current ablation scripts (q_eff=len(loader)/len(private)); "
            "'user_poisson' uses q_user=1/users (Poisson subsampling approximation); "
            "'both' overlays both curves."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_ablation_defaults(args)

    if args.non_iid and not args.public_pretrain_exclude_classes.strip():
        raise SystemExit(
            "--non-iid requires --public-pretrain-exclude-classes (e.g., '0,1')."
        )

    set_random_seeds()
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
        critical_label=None,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        public_pretrain_exclude_classes=_parse_excluded_classes(
            args.public_pretrain_exclude_classes
        ),
        non_iid=args.non_iid,
        seed=get_random_seed(),
    )

    loaders = dataset_builder.build(dataset_config)
    priv_loader = loaders.private
    priv_base = loaders.private_base

    private_size = len(priv_base)
    steps_per_epoch = len(priv_loader)
    dp_epochs = _resolve_dp_epochs(args)

    deltas = np.logspace(math.log10(1e-8), math.log10(1e-2), 60)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    def _solve_and_plot(label: str, q: float, steps_per_epoch_local: int, color: str) -> tuple[float, int, float]:
        sigma, total_steps_local = get_privacy_params_for_target_epsilon(
            target_epsilon=args.target_epsilon,
            target_delta=args.delta,
            sample_rate=q,
            epochs=dp_epochs,
            steps_per_epoch=steps_per_epoch_local,
        )
        accountant = RDPAccountant()
        accountant.history = [(sigma, q, total_steps_local)]
        actual_eps = float(accountant.get_epsilon(delta=args.delta))

        eps_curve = compute_epsilons(sigma, q, total_steps_local, deltas)
        ax.plot(deltas, eps_curve, linewidth=1.8, label=label, color=color)
        ax.scatter([args.delta], [actual_eps], color=color, zorder=3, s=26)
        return sigma, total_steps_local, actual_eps

    dp_mode = "sample-level" if args.sample_level else "user-level"
    logger.highlight("DP Guarantee (Accountant)")
    logger.info("Accountant: RDP (Opacus)")
    logger.info("Target (ε, δ): (%.4f, %.1e)", args.target_epsilon, args.delta)
    logger.info("DP mode: %s", dp_mode)
    if not args.sample_level:
        logger.info("User definition: %s synthetic users", args.users)
    logger.info("Private samples: %s", private_size)
    logger.info("Steps/epoch (training loop): %s", steps_per_epoch)

    if args.sample_level:
        q_sample = steps_per_epoch / private_size
        logger.info("Sampling model: sample-level DP (Poisson subsampling approximation)")
        sigma, total_steps, actual_epsilon = _solve_and_plot(
            label=f"sample-level (q={q_sample:.4f})",
            q=q_sample,
            steps_per_epoch_local=steps_per_epoch,
            color="#1f77b4",
        )
        logger.info("Sampling rate q: %.6f", q_sample)
        logger.info("Steps: %s (dp_epochs=%s, steps/epoch=%s)", total_steps, dp_epochs, steps_per_epoch)
        logger.info("Noise multiplier σ: %.4f", sigma)
        logger.info("Computed ε at target δ: %.4f", actual_epsilon)
        logger.info("Noise std: %.4f (σ×C)", sigma * args.clip_radius)
    else:
        q_eff = steps_per_epoch / private_size
        q_user = 1.0 / max(1, int(args.users))

        logger.info("Sampling model: user-level DP (one user per step)")
        logger.info("   • Standard Poisson user subsampling: q_user = 1/users = %.6f", q_user)
        logger.info("   • Repo effective rate: q_eff = len(priv_loader)/len(priv_base) = %.6f", q_eff)
        if q_eff > q_user * 1.0001:
            logger.warn(
                "q_eff (%.6f) > q_user (%.6f): solving for σ with q_eff is typically more conservative (larger σ) at fixed ε.",
                q_eff,
                q_user,
            )

        mode = str(args.accounting_mode)
        if mode in {"repo_q_eff", "repo"}:
            sigma, total_steps, actual_epsilon = _solve_and_plot(
                label=f"repo q_eff (q={q_eff:.4f})",
                q=q_eff,
                steps_per_epoch_local=steps_per_epoch,
                color="#1f77b4",
            )
            logger.info("Using accounting-mode=repo_q_eff (matches current ablation scripts).")
            logger.info("Sampling rate q: %.6f", q_eff)
            logger.info("Steps: %s (dp_epochs=%s, steps/epoch=%s)", total_steps, dp_epochs, steps_per_epoch)
            logger.info("Noise multiplier σ: %.4f", sigma)
            logger.info("Computed ε at target δ: %.4f", actual_epsilon)
            logger.info("Noise std: %.4f (σ×C)", sigma * args.clip_radius)
        elif mode == "user_poisson":
            sigma, total_steps, actual_epsilon = _solve_and_plot(
                label=f"user Poisson q_user (q={q_user:.4f})",
                q=q_user,
                steps_per_epoch_local=steps_per_epoch,
                color="#ff7f0e",
            )
            logger.info("Using accounting-mode=user_poisson (standard reporting for papers).")
            logger.info("Sampling rate q: %.6f", q_user)
            logger.info("Steps: %s (dp_epochs=%s, steps/epoch=%s)", total_steps, dp_epochs, steps_per_epoch)
            logger.info("Noise multiplier σ: %.4f", sigma)
            logger.info("Computed ε at target δ: %.4f", actual_epsilon)
            logger.info("Noise std: %.4f (σ×C)", sigma * args.clip_radius)
            logger.info("Note: This uses Poisson subsampling as an approximation to per-epoch user iteration.")
        else:  # both
            sigma_eff, total_steps_eff, eps_eff = _solve_and_plot(
                label=f"repo q_eff (q={q_eff:.4f})",
                q=q_eff,
                steps_per_epoch_local=steps_per_epoch,
                color="#1f77b4",
            )
            sigma_user, total_steps_user, eps_user = _solve_and_plot(
                label=f"user Poisson q_user (q={q_user:.4f})",
                q=q_user,
                steps_per_epoch_local=steps_per_epoch,
                color="#ff7f0e",
            )
            logger.info("Using accounting-mode=both (overlay).")
            logger.info("Repo:  q=%.6f, steps=%s, σ=%.4f, ε(δ)=%.4f", q_eff, total_steps_eff, sigma_eff, eps_eff)
            logger.info("User:  q=%.6f, steps=%s, σ=%.4f, ε(δ)=%.4f", q_user, total_steps_user, sigma_user, eps_user)
            logger.info("Note: 'user Poisson' uses Poisson subsampling as an approximation to per-epoch user iteration.")

    ax.set_xscale("log")
    ax.set_xlabel("δ (log scale)")
    ax.set_ylabel("ε")
    ax.set_title("Privacy Profile: ε(δ)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    ax.axvline(args.delta, color="#555555", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
