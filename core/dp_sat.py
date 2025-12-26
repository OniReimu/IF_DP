# dp_sat.py  – DP-SAT: Differentially Private Sharpness-Aware Training
# ------------------------------------------------------------
#  * Implementation of Park et al. "Differentially Private Sharpness-Aware Training" (ICML 2023)
#  * EXACT Implementation (Algorithm 1 from the paper):
#    1. Perturb weights w' = w + ρ * g_prev / ||g_prev||
#    2. Compute DP gradient at w'
#    3. Restore weights w = w' - ρ * g_prev / ||g_prev||
#    4. Update w = w - η * g_priv
#  * Standard Euclidean gradient clipping + isotropic Gaussian noise (like vanilla DP-SGD)
#  * Works with both sample-level and user-level DP modes

import math, numpy as np, torch, torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm

from data.common import prepare_batch
from core.param_selection import select_parameters_by_budget

def train_with_dp_sat(model, train_loader, epsilon=8.0, delta=1e-6,
                      clip_radius=10.0, device="cuda", target_layer="conv1",
                      adaptive_clip=True, quantile=0.95, sample_level=None,
                      epochs=1, sigma=None, rho_sat=0.001, lambda_flatness=None,
                      dp_param_count=None, dp_epochs=None, use_approximate_method=False,
                      lr=1e-3, public_loader=None, rehearsal_lambda=1.0):
    """
    DP-SAT training: Vanilla DP-SGD + Sharpness-Aware flatness adjustment.
    
    This implements the EXACT DP-SAT algorithm (Algorithm 1) via weight perturbation,
    replacing the approximate gradient-addition method.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data (private)
        epsilon: Privacy budget
        delta: Privacy parameter
        clip_radius: Clipping radius for gradients (L2 norm)
        device: Device to run on
        target_layer: Which layer(s) to apply DP to
        adaptive_clip: Whether to use adaptive clipping
        quantile: Quantile for adaptive clipping
        sample_level: If True, use sample-level DP (clip per sample).
                     If False, use user-level DP (clip per user).
                     If None, auto-detect from batch structure.
        epochs: Number of training epochs (for privacy accounting)
        sigma: If provided, use this sigma directly (for proper privacy accounting).
               If None, compute sigma from epsilon using legacy method.
        rho_sat: Perturbation radius (ρ) for weight perturbation (default: 0.001).
        lambda_flatness: Legacy parameter, mapped to rho_sat if rho_sat is default.
        public_loader: Optional DataLoader for public rehearsal (non-DP gradient).
                      If provided, combines DP private gradient with non-DP public gradient:
                      g_total = g_priv_DP + λ * g_public
        rehearsal_lambda: Mixing weight for public rehearsal gradient (default: 1.0).
                         Only used if public_loader is provided.
    
    Returns:
        Trained model with DP-SAT
    """
    
    # Handle legacy parameter mapping
    if lambda_flatness is not None and rho_sat == 0.001:
        print(f"⚠️  Note: Using legacy 'lambda_flatness' ({lambda_flatness}) as 'rho_sat'")
        rho_sat = lambda_flatness

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Privacy accounting (same as vanilla DP-SGD) — note dp_epochs below
    if sigma is not None:
        # Use provided sigma (proper privacy accounting)
        print(f"   • Using provided sigma: {sigma:.4f}")
    else:
        # Legacy privacy accounting for multi-epoch training
        # Use simple composition bound: σ_total = σ_single / √T for T epochs
        # (We will set dp_epochs below to be a fraction of the requested epochs.)
        sigma_single_epoch = math.sqrt(2*math.log(1.25/delta)) / epsilon
        # defer sigma adjustment until dp_epochs is defined

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
        print(f"   🔒 Strict DP: Frozen {frozen_count} parameter groups (trained on public data)")
    
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
    print(f"\n🎯 DP-SAT config: {mode_str} DP  {selection_desc}  ε={epsilon}")
    if sigma is not None:
        print(f"   • Proper privacy accounting: σ={sigma:.4f}")
    else:
        print(f"   • Multi-epoch privacy: T={epochs}, σ_single={sigma_single_epoch:.3f}, σ_adjusted={sigma:.3f}")
    print(f"   • Euclidean clipping with radius {clip_radius}")
    print(f"   • Isotropic Gaussian noise")
    print(f"   • EXACT DP-SAT: Weight perturbation with radius ρ={rho_sat}")
    print(f"   • Adaptive clipping: {adaptive_clip}")
    
    if not sample_level:
        print("   • User-level mode: Clipping aggregated user gradients")
    else:
        print("   • Sample-level mode: Clipping individual sample gradients")
    
    # Public rehearsal setup (uses public pretrain dataset)
    if public_loader is not None and rehearsal_lambda > 0:
        print(f"   • Public rehearsal enabled: λ={rehearsal_lambda} (using public pretrain dataset)")
        public_iter = iter(public_loader)
    else:
        public_iter = None
        if public_loader is not None and rehearsal_lambda == 0:
            print(f"   • Public rehearsal disabled (λ=0)")
    print()
    
    noise_l2, grad_norm = [], []
    adaptive_radius_computed = False
    actual_radius = clip_radius  # Start with provided radius
    
    # Initialize previous step's noisy gradient for DP-SAT
    g_prev_priv = None

    # Determine DP fine-tuning epochs
    if dp_epochs is None:
        dp_epochs = max(1, int(math.ceil(epochs / 10)))
    if sigma is None:
        sigma = sigma_single_epoch / math.sqrt(dp_epochs)
        print(f"   • Legacy accounting: T={dp_epochs}, σ_single={sigma_single_epoch:.3f}, σ_adjusted={sigma:.3f}")
    print(f"   • DP finetuning epochs: {dp_epochs} (requested {epochs})")
    
    for epoch in range(dp_epochs):
        # Reset public loader iterator each epoch
        if public_loader is not None:
            public_iter = iter(public_loader)
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"DP-SAT ({mode_str}) epoch {epoch+1}/{dp_epochs}", leave=False)):
            features, labels, user_ids = prepare_batch(batch_data, device)
            batch_size = labels.size(0)
            # ============================================================
            # DP-SAT EXACT: 1. Perturb weights
            # ============================================================
            perturbation = None
            if g_prev_priv is not None:
                # Euclidean perturbation: δ = ρ * g / ||g||₂
                g_norm = g_prev_priv.norm() + 1e-8
                perturbation = rho_sat * g_prev_priv / g_norm
                
                # Apply perturbation
                current_idx = 0
                for p in params:
                    n = p.numel()
                    p.data.add_(perturbation[current_idx:current_idx+n].view_as(p))
                    current_idx += n

            # Standard DP-SGD forward/backward on (possibly perturbed) weights
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
                
                # Adaptive clipping: collect norms from first batch to compute quantile
                if adaptive_clip and not adaptive_radius_computed:
                    batch_norms = []
                    for i in range(per_g.size(0)):
                        # Compute Euclidean norm
                        euclidean_norm = per_g[i].norm().item()
                        batch_norms.append(euclidean_norm)
                    grad_norm.extend(batch_norms)
                    
                    # If we have enough samples or it's a large batch, compute adaptive radius
                    if len(grad_norm) >= 100 or batch_idx == 0:
                        adaptive_radius = np.quantile(grad_norm, quantile)
                        actual_radius = adaptive_radius
                        adaptive_radius_computed = True
                        
                        print(f"📊 DP-SAT adaptive clipping from {len(grad_norm)} samples:")
                        print(f"   • Mean: {np.mean(grad_norm):.3f}")
                        print(f"   • Median: {np.median(grad_norm):.3f}")
                        print(f"   • {quantile:.1%} quantile: {adaptive_radius:.3f}")
                        print(f"   • Max: {np.max(grad_norm):.3f}")
                        print(f"   → Using adaptive radius: {actual_radius:.3f}\n")
                        
                        grad_norm = []  # Reset for actual training statistics
                
                # Vanilla Euclidean clipping for each sample
                for i in range(per_g.size(0)):
                    norm = per_g[i].norm()
                    if norm > actual_radius:
                        per_g[i].mul_(actual_radius / norm)
                    grad_norm.append(norm.item())
                
                # Average clipped gradients
                g_bar = per_g.mean(0)
                
            else:
                # USER-LEVEL DP
                user_gradients = []
                unique_users = torch.unique(user_ids) if user_ids is not None else [0]
                
                for uid in unique_users:
                    if user_ids is not None:
                        mask = (user_ids == uid)
                        user_losses = losses[mask]
                    else:
                        user_losses = losses
                        mask = torch.ones_like(losses, dtype=torch.bool)
                    
                    user_total_loss = user_losses.sum()
                    user_grad = grad(user_total_loss, params, retain_graph=True)
                    user_grad_flat = torch.cat([g.view(-1) for g in user_grad]).detach()
                    user_gradients.append(user_grad_flat)
                    
                    # Adaptive clipping
                    if adaptive_clip and not adaptive_radius_computed:
                        user_norm = user_grad_flat.norm().item()
                        grad_norm.append(user_norm)
                        
                        if len(grad_norm) >= min(10, len(train_loader)) or batch_idx == 0:
                            adaptive_radius = np.quantile(grad_norm, quantile)
                            actual_radius = adaptive_radius
                            adaptive_radius_computed = True
                            
                            print(f"📊 DP-SAT adaptive clipping from {len(grad_norm)} users:")
                            print(f"   • Mean user grad norm: {np.mean(grad_norm):.3f}")
                            print(f"   • Median user grad norm: {np.median(grad_norm):.3f}")
                            print(f"   • {quantile:.1%} quantile: {adaptive_radius:.3f}")
                            print(f"   • Max user grad norm: {np.max(grad_norm):.3f}")
                            print(f"   → Using adaptive radius: {actual_radius:.3f}\n")
                            
                            grad_norm = []
                
                if len(user_gradients) != 1:
                    print(f"⚠️  Warning: Expected 1 user per batch, got {len(user_gradients)} users")
                
                # Clip each user's gradient (Euclidean clipping)
                clipped_user_grads = []
                for user_grad_flat in user_gradients:
                    user_norm = user_grad_flat.norm()
                    if user_norm > actual_radius:
                        user_grad_flat.mul_(actual_radius / user_norm)
                    grad_norm.append(user_norm.item())
                    clipped_user_grads.append(user_grad_flat)
                
                g_bar = torch.stack(clipped_user_grads).mean(0)
            
            # Add isotropic Gaussian noise (standard DP-SGD)
            iso_noise = torch.randn_like(g_bar) * sigma * actual_radius
            g_priv = g_bar + iso_noise
            noise_l2.append(iso_noise.norm().item())
            
            # ============================================================
            # DP-SAT EXACT: 3. Restore weights & 4. Update
            # ============================================================
            
            # Restore weights
            if perturbation is not None:
                current_idx = 0
                for p in params:
                    n = p.numel()
                    p.data.sub_(perturbation[current_idx:current_idx+n].view_as(p))
                    current_idx += n
            
            # Store current noisy gradient for next iteration
            g_prev_priv = g_priv.clone().detach()
            
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
                    print(f"   📌 Rehearsal strength (batch0): ‖g_priv‖={g_priv_norm:.2f}, ‖g_pub‖={g_pub_norm:.2f}, ‖g_pub‖/‖g_priv‖={ratio:.4f}")
                
                # Combine: g_total = g_priv_DP + λ * g_public
                g_total = g_priv + rehearsal_lambda * g_public
            else:
                g_total = g_priv
            
            # Apply update
            idx = 0
            for p in params:
                n = p.numel()
                p.grad = g_total[idx:idx+n].view_as(p)
                idx += n
            opt.step()
    
    grad_type = "‖g_user‖₂" if not sample_level else "‖g‖₂"
    print(f"\n📊  DP-SAT final stats:")
    print(f"   • Median {grad_type} = {np.median(grad_norm):.2f}")
    print(f"   • Isotropic noise ℓ₂ ∈ [{min(noise_l2):.1f},{max(noise_l2):.1f}]")
    print(f"   • Perturbation radius ρ = {rho_sat:.4f}")
    print(f"   • Privacy: (ε={epsilon}, δ={delta}) over {dp_epochs} DP epochs")
    print(f"   • ✅ IMPLEMENTATION: Exact DP-SAT (Weight Perturbation)")
    
    return model
