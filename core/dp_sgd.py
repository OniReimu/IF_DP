# dp-sgd.py  – Vanilla DP-SGD implementation for comparison
# ------------------------------------------------------------
#  * Standard Euclidean gradient clipping
#  * Isotropic Gaussian noise
#  * Per-sample gradient computation
#  * Comparison baseline for Fisher-informed DP-SGD

import math, numpy as np, torch, torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm

from data.common import prepare_batch
from models.utils import compute_loss
from core.param_selection import select_parameters_by_budget
from config import get_logger

logger = get_logger("dp_sgd")

def train_with_vanilla_dp(model, train_loader, epsilon=8.0, delta=1e-6,
                         clip_radius=10.0, device="cuda", target_layer="conv1",
                         sample_level=None, epochs=1, sigma=None, dp_param_count=None, dp_epochs=None,
                         lr=1e-3, public_loader=None, rehearsal_lambda=1.0):
    """
    Vanilla DP-SGD training with Euclidean clipping and isotropic noise.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data (private)
        epsilon: Privacy budget
        delta: Privacy parameter
        clip_radius: Clipping radius for gradients (L2 norm)
        device: Device to run on
        target_layer: Which layer(s) to apply DP to (same as Fisher version)
        sample_level: If True, use sample-level DP (clip per sample).
                     If False, use user-level DP (clip per user).
                     If None, auto-detect from batch structure.
        epochs: Number of training epochs (for privacy accounting)
        sigma: If provided, use this sigma directly (for proper privacy accounting).
               If None, compute sigma from epsilon using legacy method.
        public_loader: Optional DataLoader for public rehearsal (non-DP gradient).
                      If provided, combines DP private gradient with non-DP public gradient:
                      g_total = g_priv_DP + λ * g_public
        rehearsal_lambda: Mixing weight for public rehearsal gradient (default: 1.0).
                         Only used if public_loader is provided.
    
    Returns:
        Trained model with vanilla DP-SGD
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Privacy accounting
    if sigma is not None:
        # Use provided sigma (proper privacy accounting)
        logger.info("   • Using provided sigma: %.4f", sigma)
    else:
        # Legacy privacy accounting for multi-epoch training
        # Use simple composition bound: σ_total = σ_single / √T for T epochs
        sigma_single_epoch = math.sqrt(2*math.log(1.25/delta)) / epsilon
        # defer adjustment until dp_epochs is defined

    # Use shared parameter selection utility for consistency across all DP methods
    names, params, stats = select_parameters_by_budget(
        model, dp_param_count, target_layer, verbose=True
    )
    dp_mask = None  # No masking needed - all complete parameters
    
    # Strict DP: Freeze all other layers
    frozen_count = 0
    for name, p in model.named_parameters():
        if name not in names:
            p.requires_grad = False
            frozen_count += 1
        else:
            p.requires_grad = True
    
    if frozen_count > 0:
        logger.info("   🔒 Strict DP: Frozen %s parameter groups (trained on public data)", frozen_count)
    
    # Auto-detect DP mode if not specified
    if sample_level is None:
        first_batch = next(iter(train_loader))
        _, _, first_user_ids = prepare_batch(first_batch, device)
        if first_user_ids is None:
            sample_level = True
        else:
            unique_users = torch.unique(first_user_ids)
            sample_level = unique_users.numel() > 1
    
    mode_str = "Sample-level" if sample_level else "User-level"
    selection_desc = f"dp-param-count={dp_param_count}" if dp_param_count is not None else f"layers={target_layer}"
    logger.highlight(f"Vanilla DP-SGD config: {mode_str} DP  {selection_desc}  ε={epsilon}")
    if sigma is not None:
        logger.info("   • Proper privacy accounting: σ=%.4f", sigma)
    else:
        logger.info("   • Multi-epoch privacy: T=%s, σ_single=%.3f, σ_adjusted=%.3f", epochs, sigma_single_epoch, sigma)
    logger.info("   • Euclidean clipping with radius %s", clip_radius)
    logger.info("   • Isotropic Gaussian noise")
    
    if not sample_level:
        logger.info("   • User-level mode: Clipping aggregated user gradients")
    else:
        logger.info("   • Sample-level mode: Clipping individual sample gradients")
    
    # Public rehearsal setup (uses public pretrain dataset)
    if public_loader is not None and rehearsal_lambda > 0:
        logger.info("   • Public rehearsal enabled: λ=%s (using public pretrain dataset)", rehearsal_lambda)
        public_iter = iter(public_loader)
    else:
        public_iter = None
        if public_loader is not None and rehearsal_lambda == 0:
            logger.info("   • Public rehearsal disabled (λ=0)")
    logger.info(" ")
    
    noise_l2, grad_norm = [], []
    actual_radius = clip_radius  # Start with provided radius

    # Determine DP fine-tuning epochs
    if dp_epochs is None:
        dp_epochs = max(1, int(math.ceil(epochs / 10)))
    if sigma is None:
        sigma = sigma_single_epoch / math.sqrt(dp_epochs)
        logger.info("   • Legacy accounting: T=%s, σ_single=%.3f, σ_adjusted=%.3f", dp_epochs, sigma_single_epoch, sigma)
    logger.info("   • DP finetuning epochs: %s (requested %s)", dp_epochs, epochs)
    
    for epoch in range(dp_epochs):
        # Reset public loader iterator each epoch
        if public_loader is not None:
            public_iter = iter(public_loader)
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Vanilla DP-SGD ({mode_str}) epoch {epoch+1}/{dp_epochs}", leave=False)):
            features, labels, user_ids = prepare_batch(batch_data, device)
            batch_size = labels.size(0)
            model.zero_grad()
            logits = model(features)
            losses = F.cross_entropy(logits, labels, reduction="none")
            
            if sample_level:
                # SAMPLE-LEVEL DP: Compute per-sample gradients
                per_g = []
                for i in range(batch_size):
                    gi = grad(losses[i], params, retain_graph=True)
                    per_g.append(torch.cat([g.view(-1) for g in gi]).detach())
                per_g = torch.stack(per_g)
                
                # Vanilla Euclidean clipping for each sample
                for i in range(per_g.size(0)):
                    norm = per_g[i].norm()
                    if norm > actual_radius:
                        per_g[i].mul_(actual_radius / norm)
                    grad_norm.append(norm.item())
                
                # Average clipped gradients
                g_bar = per_g.mean(0)
                
            else:
                # USER-LEVEL DP: Compute gradient per user (more robust approach)
                user_gradients = []
                unique_users = torch.unique(user_ids) if user_ids is not None else [0]
                
                for uid in unique_users:
                    if user_ids is not None:
                        mask = (user_ids == uid)
                        user_losses = losses[mask]
                    else:
                        # If no user_ids (shouldn't happen in user-level mode), treat all as one user
                        user_losses = losses
                        mask = torch.ones_like(losses, dtype=torch.bool)
                    
                    # Compute gradient of user's total loss: ∇_θ ∑_{i ∈ user} ℓ(x_i, y_i, θ)
                    user_total_loss = user_losses.sum()
                    user_grad = grad(user_total_loss, params, retain_graph=True)
                    user_grad_flat = torch.cat([g.view(-1) for g in user_grad]).detach()
                    user_gradients.append(user_grad_flat)
                    
                # In user-level DP with UserBatchSampler, we should have exactly one user per batch
                if len(user_gradients) != 1:
                    logger.warn("Expected 1 user per batch, got %s users", len(user_gradients))
                    logger.warn("   Unique users in batch: %s", unique_users.tolist())
                
                # Clip each user's gradient (Euclidean clipping)
                clipped_user_grads = []
                for user_grad_flat in user_gradients:
                    user_norm = user_grad_flat.norm()
                    if user_norm > actual_radius:
                        user_grad_flat.mul_(actual_radius / user_norm)
                    grad_norm.append(user_norm.item())
                    clipped_user_grads.append(user_grad_flat)
                
                # Average across users in batch (should be just one user for UserBatchSampler)
                g_bar = torch.stack(clipped_user_grads).mean(0)
            
            # Add isotropic Gaussian noise
            iso_noise = torch.randn_like(g_bar) * sigma * actual_radius
            g_priv = g_bar + iso_noise
            noise_l2.append(iso_noise.norm().item())
            
            # ============================================================
            # Public Rehearsal: Add non-DP gradient from public data
            # ============================================================
            if public_iter is not None:
                try:
                    public_batch = next(public_iter)
                except StopIteration:
                    # Reset iterator if exhausted
                    public_iter = iter(public_loader)
                    public_batch = next(public_iter)
                
                # Compute non-DP gradient on public batch
                pub_features, pub_labels, _ = prepare_batch(public_batch, device)
                model.zero_grad()
                pub_logits = model(pub_features)
                pub_loss = F.cross_entropy(pub_logits, pub_labels, reduction="mean")
                pub_grad = grad(pub_loss, params, retain_graph=False)
                g_public = torch.cat([g.view(-1) for g in pub_grad]).detach()

                # Diagnostic: if public rehearsal is weak relative to DP noisy update, λ=1 won't help much.
                # (We only log DP-safe quantities here: g_priv already includes noise.)
                if batch_idx == 0:
                    g_priv_norm = float(g_priv.norm().item())
                    g_pub_norm = float(g_public.norm().item())
                    ratio = g_pub_norm / (g_priv_norm + 1e-12)
                    logger.info(
                        "   📌 Rehearsal strength (batch0): ‖g_priv‖=%.2f, ‖g_pub‖=%.2f, ‖g_pub‖/‖g_priv‖=%.4f",
                        g_priv_norm,
                        g_pub_norm,
                        ratio,
                    )
                
                # Combine: g_total = g_priv_DP + λ * g_public
                g_total = g_priv + rehearsal_lambda * g_public
            else:
                g_total = g_priv
            
            # scatter back to model parameters
            idx = 0
            for p in params:
                n = p.numel()
                p.grad = g_total[idx:idx+n].view_as(p)
                idx += n
            opt.step()
    
    grad_type = "‖g_user‖₂" if not sample_level else "‖g‖₂"
    logger.info("Vanilla DP-SGD final stats:")
    logger.info("   • Median %s = %.2f", grad_type, np.median(grad_norm))
    logger.info("   • Isotropic noise ℓ₂ ∈ [%.1f,%.1f]", min(noise_l2), max(noise_l2))
    logger.info("   • Privacy: (ε=%s, δ=%s) over %s DP epochs", epsilon, delta, dp_epochs)
    
    return model
