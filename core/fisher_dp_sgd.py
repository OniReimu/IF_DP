# fisher_dp.py  – curvature-aware Fisher + DP-SGD
# ------------------------------------------------------------
#  * Supports single layer, multiple layers, or "all" layers
#  * Works on GPU, CPU, *and* Apple-Silicon MPS (CPU fallback for eigvals)
#  * Fisher-metric clipping (aligned with F^{-1} noise), low-rank anisotropic Gaussian noise
#  * Accepts batches of (x, y)  or  (x, y, user_id)

import math, numpy as np, torch, torch.nn.functional as F
from torch.autograd import grad
from scipy.linalg import eigh
from tqdm import tqdm

from data.common import prepare_batch
from models.utils import compute_loss
from core.param_selection import select_parameters_by_budget
from config import get_logger
from core.device_utils import freeze_batchnorm_stats
from core.privacy_accounting import compute_actual_epsilon

logger = get_logger("fisher_dp")

# ============================================================
# 0.  Utilities
# ============================================================
# ============================================================
# 1.  Fisher estimation (damped) on chosen layer set
# ============================================================
def compute_fisher(model, dataloader, device,
                   target_layer="conv1",
                   rho: float = 1e-2,
                   dp_param_count=None):
    """
    Return F  =  (1/N) GᵀG + ρI   for the parameters in target_layer or dp_param_count

    target_layer:
        "conv1"            – single layer substring
        "conv1,conv2"      – comma-separated list
        "all"              – every parameter
    dp_param_count:
        If specified, select first N parameters in model order (overrides target_layer)
    """
    model.eval()

    # ------------ choose parameters ------------
    # Use shared parameter selection utility for consistency across all DP methods
    # Note: compute_fisher only needs names, not the actual parameter objects
    tgt_names, _, stats = select_parameters_by_budget(
        model, dp_param_count, target_layer, verbose=True
    )
    # Adjust print prefix for Fisher computation context
    if dp_param_count is not None:
        logger.info("Computing Fisher for up to %s parameters (head-first)", dp_param_count)
        logger.info("   ✅ Selected %s complete parameters", len(tgt_names))
        logger.info(
            "      Budget: %s | Used: %s | Unused: %s (%.1f%% efficiency)",
            dp_param_count,
            stats['total_selected'],
            stats['unused'],
            stats['efficiency'],
        )
        if stats['head_total_params'] > 0:
            logger.info(
                "      Head params: selected %s tensors (head scalars available: %s)",
                stats['head_selected'],
                stats['head_total_params'],
            )
            if stats['head_selected'] == 0 and stats['head_min_tensor'] > 0 and dp_param_count < stats['head_min_tensor']:
                logger.warn(
                    "Budget too small to include the smallest head tensor (min head tensor size=%s).",
                    stats['head_min_tensor'],
                )
                logger.warn("Increase --dp-param-count or use --dp-layer to target the classifier/head directly.")
    else:
        if target_layer == "all":
            logger.info("Computing Fisher for ALL layers")
        elif "," in str(target_layer):
            layers = [s.strip() for s in str(target_layer).split(",")]
            logger.info("Computing Fisher for layers %s", layers)
        else:
            logger.info("Computing Fisher for layer '%s'", target_layer)

    if not tgt_names:
        raise ValueError(f"No parameters match selection (layer='{target_layer}', dp_param_count={dp_param_count})")

    P = sum(dict(model.named_parameters())[n].numel() for n in tgt_names)
    logger.info("Target parameters: %s", tgt_names)
    logger.info("Total parameters: %s", P)
    logger.info("Damping coefficient ρ = %s", rho)

    mem_gb = P**2 * 4 / 1e9
    logger.info("Fisher matrix size: %s×%s ≈ %.2f GB", P, P, mem_gb)
    if mem_gb > 2:
        logger.warn("Fisher matrix is large; eigendecomp may be slow.")

    # ------------ accumulate per-sample grads ------------
    grads = []
    desc_label = f"dp-param-count={dp_param_count}" if dp_param_count is not None else target_layer
    for batch_data in tqdm(dataloader, desc=f"Fisher pass ({desc_label})"):
        features, labels, _ = prepare_batch(batch_data, device)

        model.zero_grad()
        logits = model(features)
        loss = compute_loss(model, logits, labels)
        g = grad(loss,
                 [dict(model.named_parameters())[n] for n in tgt_names],
                 retain_graph=False, create_graph=False)
        grads.append(torch.cat([v.flatten() for v in g]).detach())

    G = torch.vstack(grads)                     # [N_rows, P]
    logger.info("Gradient matrix shape: %s", tuple(G.shape))

    fisher = (G.T @ G) / len(G) + rho * torch.eye(P, device=device)
    logger.info("Fisher matrix shape: %s", tuple(fisher.shape))

    # ------------ condition-number (CPU fallback on MPS) ------------
    try:
        eigvals = torch.linalg.eigvals(fisher.cpu() if fisher.device.type=="mps" else fisher).real
        cond    = (eigvals.max() / eigvals.min()).item()
        rank_est= (eigvals > eigvals.max()*1e-10).sum().item()
        logger.info("Condition number ≈ %.2e   Estimated rank %s/%s", cond, rank_est, P)
    except Exception as e:
        logger.warn("Condition-number computation failed (ignored): %s", e)

    return fisher, tgt_names

# ============================================================
# 2.  Top-k eigendecomposition with numeric floor
# ============================================================
def topk_eigh_with_floor(mat: torch.Tensor,
                         k: int = 128,
                         lam_floor: float = 5e-1):
    # Ensure k doesn't exceed matrix size
    actual_k = min(k, mat.shape[0])
    logger.info("Compute top-%s eigenpairs (requested %s), λ_floor = %s", actual_k, k, lam_floor)
    
    mat_cpu = mat.cpu().numpy()
    try:
        start   = max(0, mat.shape[0]-actual_k)
        lam, U  = eigh(mat_cpu, subset_by_index=[start, mat.shape[0]-1])
        lam, U  = lam[::-1].copy(), U[:, ::-1].copy()    # descending order
        lam     = np.maximum(lam, lam_floor)
    except Exception as e:
        logger.warn("eigh failed (%s) – using diagonal fallback", e)
        diag  = np.diag(mat_cpu)
        idx   = np.argpartition(-diag, actual_k)[:actual_k]
        lam   = np.maximum(diag[idx], lam_floor)
        U     = np.zeros((mat.shape[0], actual_k))
        for i,j in enumerate(idx): U[j,i]=1.0
    
    # Ensure we have the right number of eigenvalues and eigenvectors
    actual_returned_k = len(lam)
    if actual_returned_k != actual_k:
        logger.warn("Returned %s eigenpairs instead of requested %s", actual_returned_k, actual_k)
    
    logger.info("Eigenvalue range: [%.3f, %.3f] (got %s eigenpairs)", lam[-1], lam[0], len(lam))
    return torch.from_numpy(lam).float(), torch.from_numpy(U).float()

# ============================================================
# 3.  Fisher-metric clipping helpers
# ============================================================
def maha_clip(vec, U, inv_sqrt_lam, radius):
    """
    Clip a vector using a quadratic metric defined by (U, scaling).

    If `scaling = sqrt_lam`, this computes the Fisher (F) norm in the subspace:
        ||g||_F = || sqrt(Λ) U^T g ||_2
    If `scaling = inv_sqrt_lam`, this computes the F^{-1} norm in the subspace:
        ||g||_{F^{-1}} = || (1/sqrt(Λ)) U^T g ||_2

    Note: this scales the *full* vector `vec` by a single factor, which is fine when
    the update will be projected into span(U) afterwards (subspace-only updates).
    """
    proj = (U.T @ vec) * inv_sqrt_lam
    norm = proj.norm() + 1e-10
    if norm > radius: vec.mul_(radius / norm)
    return vec, norm.item()


def l2_clip(vec, radius):
    norm = vec.norm() + 1e-10
    if norm > radius:
        vec.mul_(radius / norm)
    return vec, norm.item()


def calibrate_mahalanobis_radius(
    model,
    loader,
    params,
    U,
    clip_scaling,
    euclidean_target,
    sample_level,
    device,
    max_batches=5,
):
    """
    Calibrate a Fisher-metric clipping radius using public data only.

    Option 1 semantics (paper default):
      - `euclidean_target` is the *vanilla DP-SGD* Euclidean clip radius Δ₂.
      - We pick a Fisher-metric radius so the Fisher clip rate matches the Euclidean
        clip rate measured on public gradients.

    The Fisher metric here is determined by the caller-provided `clip_scaling`:
      - for aligned Fisher DP-SGD, pass `sqrt_lam` so the clipped norm is ||g||_F.
    """
    if loader is None:
        logger.warn("Public loader unavailable for norm calibration; using Δ₂ as the Fisher radius (fallback).")
        return euclidean_target, False

    was_training = model.training
    model.eval()

    euclidean_norms = []
    mahalanobis_norms = []
    min_samples = 50 if sample_level else 5

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= max_batches:
            break
        features, labels, _ = prepare_batch(batch_data, device)
        logits = model(features)
        losses = F.cross_entropy(logits, labels, reduction="none")

        if sample_level:
            for i in range(losses.size(0)):
                gi = grad(losses[i], params, retain_graph=True)
                g_flat = torch.cat([g.view(-1) for g in gi]).detach()
                euclidean_norms.append(g_flat.norm().item())
                proj = (U.T @ g_flat) * clip_scaling
                mahalanobis_norms.append(proj.norm().item())
        else:
            user_total_loss = losses.sum()
            gi = grad(user_total_loss, params, retain_graph=True)
            g_flat = torch.cat([g.view(-1) for g in gi]).detach()
            euclidean_norms.append(g_flat.norm().item())
            proj = (U.T @ g_flat) * clip_scaling
            mahalanobis_norms.append(proj.norm().item())

        if len(euclidean_norms) >= min_samples:
            break

    if was_training:
        model.train()

    if not euclidean_norms or not mahalanobis_norms:
        logger.warn("Public norm calibration had no usable gradients; using fixed Δ₂.")
        return euclidean_target, False

    euclidean_norms = np.array(euclidean_norms)
    mahalanobis_norms = np.array(mahalanobis_norms)

    if sample_level:
        euclidean_clip_rate = np.mean(euclidean_norms > euclidean_target)
        maha_low, maha_high = mahalanobis_norms.min(), mahalanobis_norms.max()
        for _ in range(10):
            maha_mid = (maha_low + maha_high) / 2
            maha_clip_rate = np.mean(mahalanobis_norms > maha_mid)
            if maha_clip_rate > euclidean_clip_rate:
                maha_low = maha_mid
            else:
                maha_high = maha_mid
        actual_radius = (maha_low + maha_high) / 2
        logger.info("Public norm calibration (sample-level):")
        logger.info("   • Target Euclidean clip radius (vanilla Δ₂): %.3f", euclidean_target)
        logger.info("   • Calibrated Fisher radius: %.3f", actual_radius)
        logger.info("   • Euclidean clip rate: %.1f%%", euclidean_clip_rate * 100.0)
        logger.info("   • Fisher clip rate: %.1f%%", np.mean(mahalanobis_norms > actual_radius) * 100.0)
    else:
        ratios = euclidean_norms / (mahalanobis_norms + 1e-8)
        ratio = float(np.median(ratios))
        actual_radius = euclidean_target / (ratio + 1e-8)
        logger.info("Public norm calibration (user-level):")
        logger.info("   • Target Euclidean clip radius (vanilla Δ₂): %.3f", euclidean_target)
        logger.info("   • Calibrated Fisher radius: %.3f", actual_radius)
        logger.info("   • Median ratio: ||g||₂ / ||g||_F ≈ %.3f", ratio)

    return actual_radius, True

# ============================================================
# 4.  DP-SGD with curvature-aware low-rank noise
# ============================================================
def train_with_dp(model, train_loader, fisher,
                  epsilon=8.0, delta=1e-6,
                  clip_radius=10.0, k=32, lam_floor=5e-1,
                  device="cuda", target_layer="conv1",
                  sample_level=None, epochs=1, sigma=None,
                  positive_noise_correlation=False,
                  dp_sat_mode="none", rho_sat=0.001, dp_epochs=None,
                  lr=1e-3, public_loader=None, rehearsal_lambda=1.0):
    """
    Train with Fisher-informed DP-SGD
    
    Args:
        clip_radius: Target Euclidean clip radius Δ₂ (same as vanilla DP-SGD for fair comparison).
                    This is used only to calibrate a Fisher radius (public-only), which is then used for Fisher clipping.
        sample_level: If True, use sample-level DP (clip per sample).
                     If False, use user-level DP (clip per user).
                     If None, auto-detect from batch structure.
        epochs: Number of training epochs (for privacy accounting)
        sigma: If provided, use this sigma directly (for proper privacy accounting).
               If None, compute sigma from epsilon using legacy method.
        positive_noise_correlation: If False (default), use negatively correlated noise (noise ∝ 1/√λ).
                                   If True, use positively correlated noise (noise ∝ √λ).
        dp_sat_mode: "none" (default), "euclidean" (Euclidean DP-SAT), "fisher" (Fisher DP-SAT).
        rho_sat: Perturbation radius for Exact DP-SAT.
        public_loader: Optional DataLoader for public rehearsal (non-DP gradient).
                      If provided, combines DP private gradient with non-DP public gradient:
                      g_total = g_priv_DP + λ * g_public
        rehearsal_lambda: Mixing weight for public rehearsal gradient (default: 1.0).
                         Only used if public_loader is provided.
    """
    model.train()
    bn_frozen = freeze_batchnorm_stats(model)
    if bn_frozen:
        logger.info("   • BatchNorm stats frozen during DP fine-tuning (%s modules)", bn_frozen)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    lam, U       = topk_eigh_with_floor(fisher, k=k, lam_floor=lam_floor)
    lam, U       = lam.to(device), U.to(device)
    
    # Aligned Fisher DP-SGD (paper default):
    #   - Clip in Fisher norm: ||g||_F = || sqrt(Λ) U^T g ||_2
    #   - Add noise with covariance ∝ F^{-1}: in eigen-coordinates, scale by 1/sqrt(λ)
    inv_sqrt_lam = lam.rsqrt()
    sqrt_lam = lam.sqrt()

    if positive_noise_correlation:
        raise ValueError(
            "positive_noise_correlation is not supported in mechanism-aligned Fisher DP-SGD. "
            "Keep it disabled for standard accountant claims."
        )

    # For calibration and clipping we pass sqrt_lam so the clipped norm is ||g||_F.
    clip_scaling = sqrt_lam
    
    # Privacy accounting
    if sigma is not None:
        # Use provided sigma (proper privacy accounting)
        logger.info("   • Using provided sigma: %.4f", sigma)
    else:
        # Legacy privacy accounting for multi-epoch training
        # Use simple composition bound: σ_total = σ_single / √T for T epochs
        sigma_single_epoch = math.sqrt(2*math.log(1.25/delta)) / epsilon
        sigma = sigma_single_epoch / math.sqrt(epochs)  # Adjust for T epochs
        logger.info("   • Legacy accounting: σ_single=%.3f, σ_adjusted=%.3f", sigma_single_epoch, sigma)
    
    # Use the actual k returned by eigendecomposition
    actual_k = len(lam)
    if actual_k != k:
        logger.warn("Using k=%s eigenpairs (requested %s) due to matrix rank constraints", actual_k, k)

    # gather parameter objects
    if target_layer == "all":
        names = [n for n,_ in model.named_parameters()]
    elif "," in target_layer:
        layers = [s.strip() for s in target_layer.split(",")]
        names  = [n for n,_ in model.named_parameters()
                  if any(l in n for l in layers)]
    else:
        names = [n for n,_ in model.named_parameters()
                 if target_layer in n]
    params = [dict(model.named_parameters())[n] for n in names]
    param_dim = sum(p.numel() for p in params)

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
    logger.highlight(f"Fisher DP-SGD config (subspace-only): {mode_str} DP  layers={target_layer}  ε={epsilon}")
    if sigma is not None:
        logger.info("   • Proper privacy accounting: σ=%.4f", sigma)
    else:
        logger.info("   • Multi-epoch privacy: T=%s, σ_single=%.3f, σ_adjusted=%.3f", epochs, sigma_single_epoch, sigma)
    logger.info("   • Fisher subspace: k=%s, DP param dim=%s", actual_k, param_dim)
    logger.info("   • Target Euclidean clip radius (vanilla Δ₂): %.3f (used only to calibrate Fisher radius)", clip_radius)
    logger.info("   • Private updates: projected into span(U) (no complement updates)")
    logger.info("   • Clipping norm: Fisher (F) norm, ||sqrt(Λ) U^T g||₂")
    logger.info("   • Noise: covariance ∝ F^{-1} (eigen scaling 1/sqrt(λ)), multiplier σ")
    
    if dp_sat_mode != "none":
        logger.info("   • DP-SAT enabled: mode=%s, ρ=%s", dp_sat_mode, rho_sat)
        if dp_sat_mode == "fisher":
            logger.info("     ✨ Using Fisher-whitened weight perturbation (Fisher DP-SAT)")
        elif dp_sat_mode == "euclidean":
            logger.warn("     Using Euclidean weight perturbation (Euclidean DP-SAT)")
    
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
    euclidean_target = clip_radius  # Store the target Euclidean sensitivity
    actual_radius, _ = calibrate_mahalanobis_radius(
        model,
        public_loader,
        params,
        U,
        clip_scaling,
        euclidean_target,
        sample_level,
        device,
    )
    calibration_computed = True  # Avoid any private-data calibration in the training loop

    # Initialize previous step's noisy gradient for DP-SAT
    g_prev_priv = None

    # Determine DP fine-tuning epochs
    if dp_epochs is None:
        dp_epochs = max(1, int(math.ceil(epochs / 10)))
    logger.info("   • DP finetuning epochs: %s (requested %s)", dp_epochs, epochs)

    # Reporting-only ε from standard accountant using (σ, q, T).
    # The linear transform argument applies because (U, Λ) are computed from public data.
    if sample_level:
        batch_size = getattr(train_loader, "batch_size", None)
        q = float(batch_size) / float(len(train_loader.dataset)) if batch_size else 0.0
    else:
        num_users = getattr(getattr(train_loader, "dataset", None), "num_users", None)
        q = 1.0 / float(num_users) if num_users else float(len(train_loader)) / float(len(train_loader.dataset))
    total_steps = int(dp_epochs) * int(len(train_loader))
    eps_report = compute_actual_epsilon(
        noise_multiplier=float(sigma),
        sample_rate=float(q),
        steps=int(total_steps),
        target_delta=float(delta),
    )
    logger.info(
        "   • Accounting (reporting): σ=%.4f, q=%.6f, steps=%s → ε≈%.4f at δ=%.1e",
        float(sigma),
        float(q),
        int(total_steps),
        float(eps_report),
        float(delta),
    )

    for epoch in range(dp_epochs):
        # Reset public loader iterator each epoch
        if public_loader is not None:
            public_iter = iter(public_loader)
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Fisher DP-SGD ({mode_str}) epoch {epoch+1}/{dp_epochs}", leave=False)):
            features, labels, user_ids = prepare_batch(batch_data, device)
            batch_size = labels.size(0)

            # ============================================================
            # DP-SAT Exact Mode: Weight Perturbation (Start of Step)
            # ============================================================
            perturbation = None
            if dp_sat_mode != "none" and g_prev_priv is not None:
                if dp_sat_mode == "euclidean":
                    # Euclidean perturbation: δ = ρ * g / ||g||₂
                    g_norm = g_prev_priv.norm() + 1e-8
                    perturbation = rho_sat * g_prev_priv / g_norm
                elif dp_sat_mode == "fisher":
                    # Fisher perturbation: δ = ρ * F⁻¹g / ||F⁻¹g||_F
                    # Derived as: ρ * U(α_white / ||α_white|| * 1/√λ)
                    alpha = U.T @ g_prev_priv
                    alpha_white = alpha * inv_sqrt_lam
                    alpha_white_norm = alpha_white.norm() + 1e-8
                    
                    # Direction in whitened space scaled back to parameter space
                    perturbation = rho_sat * (U @ (alpha_white / alpha_white_norm * inv_sqrt_lam))

                # Apply perturbation
                if perturbation is not None:
                    current_idx = 0
                    for p in params:
                        n = p.numel()
                        p.data.add_(perturbation[current_idx:current_idx+n].view_as(p))
                        current_idx += n

            model.zero_grad()
            logits = model(features)
            losses = F.cross_entropy(logits, labels, reduction="none")

            if sample_level:
                # SAMPLE-LEVEL DP (subspace-only): project, clip in Fisher norm, then average.
                alpha_list = []
                for i in range(batch_size):
                    gi = grad(losses[i], params, retain_graph=True)
                    g_flat = torch.cat([g.view(-1) for g in gi]).detach()
                    alpha = U.T @ g_flat
                    f_norm = (sqrt_lam * alpha).norm() + 1e-10
                    if f_norm > actual_radius:
                        alpha = alpha * (actual_radius / f_norm)
                    grad_norm.append(float(f_norm.item()))
                    alpha_list.append(alpha)
                alpha_bar = torch.stack(alpha_list).mean(0)

            else:
                # USER-LEVEL DP (subspace-only): aggregate per user, then project+clip in Fisher norm.
                alpha_users = []
                unique_users = torch.unique(user_ids) if user_ids is not None else [0]

                for uid in unique_users:
                    if user_ids is not None:
                        mask = (user_ids == uid)
                        user_losses = losses[mask]
                    else:
                        user_losses = losses

                    user_total_loss = user_losses.sum()
                    user_grad = grad(user_total_loss, params, retain_graph=True)
                    g_flat = torch.cat([g.view(-1) for g in user_grad]).detach()
                    alpha = U.T @ g_flat
                    f_norm = (sqrt_lam * alpha).norm() + 1e-10
                    if f_norm > actual_radius:
                        alpha = alpha * (actual_radius / f_norm)
                    grad_norm.append(float(f_norm.item()))
                    alpha_users.append(alpha)

                if len(alpha_users) != 1:
                    logger.warn("Expected 1 user per batch, got %s users", len(alpha_users))
                    logger.warn("   Unique users in batch: %s", unique_users.tolist())

                alpha_bar = torch.stack(alpha_users).mean(0)

            # ============================================================
            # Subspace-only Fisher DP noise (mechanism-aligned):
            #   - Clip in Fisher norm (actual_radius)
            #   - Add noise with covariance ∝ F^{-1}
            #   - Update is projected into span(U)
            # ============================================================
            z = torch.randn(actual_k, device=device)
            alpha_noise = (z * inv_sqrt_lam) * float(sigma) * float(actual_radius)
            alpha_priv = alpha_bar + alpha_noise
            g_priv = U @ alpha_priv
            
            # ============================================================
            # DP-SAT: Sharpness-Aware Optimization
            # ============================================================
            
            # 1. Restore weights if using Exact Mode
            if perturbation is not None:
                current_idx = 0
                for p in params:
                    n = p.numel()
                    p.data.sub_(perturbation[current_idx:current_idx+n].view_as(p))
                    current_idx += n
            
            # Store current noisy gradient for next iteration
            g_prev_priv = g_priv.clone().detach()
            
            total_noise_norm = float((U @ alpha_noise).norm().item())
            noise_l2.append(total_noise_norm)

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

            # scatter back
            idx = 0
            for p in params:
                n = p.numel()
                p.grad = g_total[idx:idx+n].view_as(p)
                idx += n
            opt.step()

    grad_type = "‖g_user‖_F" if not sample_level else "‖g‖_F"
    logger.info("Fisher DP-SGD final stats:")
    logger.info("   • Target Euclidean clip radius (vanilla Δ₂): %.3f", euclidean_target)
    logger.info("   • Calibrated Fisher radius (||g||_F bound): %.3f", actual_radius)
    logger.info("   • Median %s = %.2f", grad_type, np.median(grad_norm))
    logger.info("   • Total noise ℓ₂ ∈ [%.1f,%.1f]", min(noise_l2), max(noise_l2))
    logger.info("   • Privacy: (ε=%s, δ=%s) over %s DP epochs", epsilon, delta, dp_epochs)
    logger.info("   • Noise multiplier: σ=%.4f; Fisher radius C_F=%.3f; noise std in transformed space: σ×C_F=%.3f", float(sigma), float(actual_radius), float(sigma) * float(actual_radius))

    return model
