#!/usr/bin/env python3
"""
Proper Privacy Accounting for DP-SGD using Opacus
==================================================

This module provides accurate privacy accounting for both Fisher DP-SGD and Vanilla DP-SGD
using Opacus' PrivacyEngine with the Rényi Differential Privacy (RDP) accountant.

Key Features:
- Accurate composition across multiple epochs and steps
- Support for both sample-level and user-level DP
- Fair comparison between different DP methods at the same actual privacy level
"""

import math
import warnings
from typing import Tuple, Optional

import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from config import get_logger

logger = get_logger("privacy")


def get_privacy_params_for_target_epsilon(
    target_epsilon: float,
    target_delta: float, 
    sample_rate: float,
    epochs: int,
    steps_per_epoch: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate the noise multiplier needed to achieve target (ε, δ)-DP
    over the specified number of training steps.
    
    Args:
        target_epsilon: Target epsilon for (ε, δ)-DP
        target_delta: Target delta for (ε, δ)-DP  
        sample_rate: Sampling rate (batch_size / dataset_size)
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch (if None, computed from sample_rate)
    
    Returns:
        Tuple of (noise_multiplier, actual_steps)
    """
    
    if steps_per_epoch is None:
        steps_per_epoch = math.ceil(1.0 / sample_rate)
    
    total_steps = epochs * steps_per_epoch
    
    # Use Opacus to find the required noise multiplier
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp"
    )
    
    return noise_multiplier, total_steps


def compute_actual_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float
) -> float:
    """
    Compute the actual epsilon achieved given training parameters.
    
    Args:
        noise_multiplier: Noise multiplier used in training
        sample_rate: Sampling rate (batch_size / dataset_size)
        steps: Total number of training steps
        target_delta: Target delta for (ε, δ)-DP
    
    Returns:
        Actual epsilon achieved
    """
    
    accountant = RDPAccountant()
    
    # Account for all steps
    accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    
    # For multiple steps, we need to call step() multiple times
    # But this is inefficient, so we use the batch method
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    
    # Compute the actual epsilon
    actual_epsilon = accountant.get_epsilon(delta=target_delta)
    
    return actual_epsilon


class PrivacyTracker:
    """
    Track privacy expenditure during training using RDP accounting.
    """
    
    def __init__(self, 
                 target_delta: float,
                 sample_rate: float,
                 noise_multiplier: float):
        """
        Initialize privacy tracker.
        
        Args:
            target_delta: Target delta for (ε, δ)-DP
            sample_rate: Sampling rate (batch_size / dataset_size)
            noise_multiplier: Noise multiplier used in training
        """
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier
        self.accountant = RDPAccountant()
        self.steps = 0
    
    def step(self) -> None:
        """Record one training step."""
        self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate
        )
        self.steps += 1
    
    def get_epsilon(self) -> float:
        """Get current epsilon."""
        if self.steps == 0:
            return 0.0
        return self.accountant.get_epsilon(delta=self.target_delta)
    
    def get_privacy_budget(self) -> Tuple[float, float, int]:
        """Get current privacy budget: (epsilon, delta, steps)."""
        return self.get_epsilon(), self.target_delta, self.steps


def validate_privacy_comparison(
    fisher_epsilon: float,
    vanilla_epsilon: float,
    tolerance: float = 0.1
) -> bool:
    """
    Validate that two methods are being compared at similar privacy levels.
    
    Args:
        fisher_epsilon: Actual epsilon for Fisher DP method
        vanilla_epsilon: Actual epsilon for Vanilla DP method  
        tolerance: Relative tolerance for epsilon difference
    
    Returns:
        True if epsilons are within tolerance, False otherwise
    """
    if fisher_epsilon <= 0 or vanilla_epsilon <= 0:
        return False
    
    relative_diff = abs(fisher_epsilon - vanilla_epsilon) / max(fisher_epsilon, vanilla_epsilon)
    
    if relative_diff > tolerance:
        warnings.warn(
            f"Privacy levels differ significantly: Fisher ε={fisher_epsilon:.3f}, "
            f"Vanilla ε={vanilla_epsilon:.3f} (relative diff: {relative_diff:.1%}). "
            f"Comparison may not be fair."
        )
        return False
    
    return True


def calculate_sigma_from_noise_multiplier(
    noise_multiplier: float,
    clip_radius: float
) -> float:
    """
    Convert Opacus noise multiplier to noise standard deviation.
    
    In Opacus: noise_std = noise_multiplier * clip_radius
    
    Args:
        noise_multiplier: Opacus noise multiplier
        clip_radius: Gradient clipping radius
    
    Returns:
        Noise standard deviation (sigma)
    """
    return noise_multiplier * clip_radius


def print_privacy_summary(
    method_name: str,
    target_epsilon: float,
    actual_epsilon: float,
    delta: float,
    noise_multiplier: float,
    steps: int,
    sample_rate: float
) -> None:
    """Print a summary of privacy parameters."""
    
    logger.highlight(f"{method_name} Privacy Summary")
    logger.info("   • Target (ε, δ): (%.4f, %.1e)", target_epsilon, delta)
    logger.info("   • Actual ε: %.4f", actual_epsilon)
    logger.info("   • Noise multiplier: %.4f", noise_multiplier)
    logger.info("   • Total steps: %s", steps)
    logger.info("   • Sample rate: %.4f", sample_rate)
    
    if abs(actual_epsilon - target_epsilon) > 0.1:
        logger.warn("Privacy level differs from target.")


# Example usage for fair comparison
def setup_fair_comparison(
    target_epsilon: float = 10.0,
    target_delta: float = 1e-6,
    dataset_size: int = 10000,
    batch_size: int = 128,
    epochs: int = 50
) -> Tuple[float, float, int]:
    """
    Set up parameters for fair comparison between DP methods.
    
    Returns:
        Tuple of (noise_multiplier, sample_rate, total_steps)
    """
    
    sample_rate = batch_size / dataset_size
    noise_multiplier, total_steps = get_privacy_params_for_target_epsilon(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=epochs
    )
    
    logger.highlight("Fair Comparison Setup")
    logger.info("   • Target: (ε=%s, δ=%s)", target_epsilon, target_delta)
    logger.info("   • Dataset size: %s, Batch size: %s", dataset_size, batch_size)
    logger.info("   • Sample rate: %.4f", sample_rate)
    logger.info("   • Epochs: %s, Total steps: %s", epochs, total_steps)
    logger.info("   • Required noise multiplier: %.4f", noise_multiplier)
    
    return noise_multiplier, sample_rate, total_steps


if __name__ == "__main__":
    # Example: Calculate privacy parameters for CIFAR-10 experiment
    logger.highlight("Example: CIFAR-10 Privacy Accounting")
    
    # Typical CIFAR-10 private training setup
    dataset_size = 10000  # Private training samples
    batch_size = 128
    epochs = 50
    target_epsilon = 10.0
    target_delta = 1e-6
    
    noise_mult, sample_rate, total_steps = setup_fair_comparison(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        dataset_size=dataset_size,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Verify the calculation
    actual_epsilon = compute_actual_epsilon(
        noise_multiplier=noise_mult,
        sample_rate=sample_rate,
        steps=total_steps,
        target_delta=target_delta
    )
    
    logger.success("Verification:")
    logger.info("   • Computed noise multiplier: %.4f", noise_mult)
    logger.info("   • Actual epsilon achieved: %.4f", actual_epsilon)
    logger.info("   • Error: %.4f", abs(actual_epsilon - target_epsilon))
