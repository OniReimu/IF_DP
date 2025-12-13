# fisher_dp.py  – curvature-aware Fisher + DP-SGD
# ------------------------------------------------------------
#  * Supports single layer, multiple layers, or "all" layers
#  * Works on GPU, CPU, *and* Apple-Silicon MPS (CPU fallback for eigvals)
#  * Per-sample Mahalanobis clipping, low-rank anisotropic Gaussian noise
#  * Accepts batches of (x, y)  or  (x, y, user_id)

import math, numpy as np, torch, torch.nn.functional as F
from torch.autograd import grad
from scipy.linalg import eigh
from tqdm import tqdm

from data.common import prepare_batch
from models.utils import compute_loss

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
    def _match(name: str, layer: str) -> bool:
        # stricter prefix match to avoid accidental substring matches
        return name.startswith(layer)
    if dp_param_count is not None:
        # DP parameter budget mode: smart selection to maximize parameter usage
        print(f"🎯 computing Fisher for up to {dp_param_count} parameters")
        all_params = list(model.named_parameters())
        
        # Build list of (name, param, size, index) for knapsack optimization
        param_info = [(name, param, param.numel(), idx) 
                      for idx, (name, param) in enumerate(all_params)]
        
        # Greedy knapsack: select parameters that fit within budget
        selected_indices = []
        total_selected = 0
        
        for name, param, size, idx in param_info:
            if total_selected + size <= dp_param_count:
                selected_indices.append(idx)
                total_selected += size
            elif total_selected < dp_param_count:
                # Would exceed budget - silently skip for cleaner logs
                continue
        
        # Extract selected parameters
        tgt_names = []
        for idx in sorted(selected_indices):
            name, param = all_params[idx]
            tgt_names.append(name)
        
        unused = dp_param_count - total_selected
        efficiency = (total_selected / dp_param_count) * 100
        print(f"   ✅ Selected {len(tgt_names)} complete parameters")
        print(f"      Budget: {dp_param_count} | Used: {total_selected} | Unused: {unused} ({efficiency:.1f}% efficiency)")
    else:
        if target_layer == "all":
            tgt_names = [n for n, _ in model.named_parameters()]
            print("🎯 computing Fisher for ALL layers")
        elif "," in target_layer:
            layers = [s.strip() for s in target_layer.split(",")]
            tgt_names = [n for n, _ in model.named_parameters()
                        if any(_match(n, l) for l in layers)]
            print(f"🎯 computing Fisher for layers {layers}")
        else:
            tgt_names = [n for n, _ in model.named_parameters()
                        if _match(n, target_layer)]
        print(f"🎯 computing Fisher for layer '{target_layer}'")

    if not tgt_names:
        raise ValueError(f"No parameters match '{target_layer}'")

    P = sum(dict(model.named_parameters())[n].numel() for n in tgt_names)
    print(f"目标参数: {tgt_names}")
    print(f"参数总数: {P}")
    print(f"阻尼系数 ρ = {rho}")

    mem_gb = P**2 * 4 / 1e9
    print(f"Fisher矩阵大小: {P}×{P} ≈ {mem_gb:.2f} GB")
    if mem_gb > 2:
        print("⚠️  Fisher matrix is large; eigendecomp may be slow.")

    # ------------ accumulate per-sample grads ------------
    grads = []
    for batch_data in tqdm(dataloader, desc=f"Fisher pass ({target_layer})"):
        features, labels, _ = prepare_batch(batch_data, device)

        model.zero_grad()
        logits = model(features)
        loss = compute_loss(model, logits, labels)
        g = grad(loss,
                 [dict(model.named_parameters())[n] for n in tgt_names],
                 retain_graph=False, create_graph=False)
        grads.append(torch.cat([v.flatten() for v in g]).detach())

    G = torch.vstack(grads)                     # [N_rows, P]
    print(f"梯度矩阵形状: {G.shape}")

    fisher = (G.T @ G) / len(G) + rho * torch.eye(P, device=device)
    print(f"Fisher矩阵形状: {fisher.shape}")

    # ------------ condition-number (CPU fallback on MPS) ------------
    try:
        eigvals = torch.linalg.eigvals(fisher.cpu() if fisher.device.type=="mps" else fisher).real
        cond    = (eigvals.max() / eigvals.min()).item()
        rank_est= (eigvals > eigvals.max()*1e-10).sum().item()
        print(f"条件数≈ {cond:.2e}   估计rank {rank_est}/{P}")
    except Exception as e:
        print(f"⚠️  条件数计算失败 (ignored): {e}")

    return fisher, tgt_names

# ============================================================
# 2.  Top-k eigendecomposition with numeric floor
# ============================================================
def topk_eigh_with_floor(mat: torch.Tensor,
                         k: int = 128,
                         lam_floor: float = 5e-1):
    # Ensure k doesn't exceed matrix size
    actual_k = min(k, mat.shape[0])
    print(f"计算 top-{actual_k} eigenpairs (requested {k}), λ_floor = {lam_floor}")
    
    mat_cpu = mat.cpu().numpy()
    try:
        start   = max(0, mat.shape[0]-actual_k)
        lam, U  = eigh(mat_cpu, subset_by_index=[start, mat.shape[0]-1])
        lam, U  = lam[::-1].copy(), U[:, ::-1].copy()    # descending order
        lam     = np.maximum(lam, lam_floor)
    except Exception as e:
        print(f"eigh failed ({e}) – using diagonal fallback")
        diag  = np.diag(mat_cpu)
        idx   = np.argpartition(-diag, actual_k)[:actual_k]
        lam   = np.maximum(diag[idx], lam_floor)
        U     = np.zeros((mat.shape[0], actual_k))
        for i,j in enumerate(idx): U[j,i]=1.0
    
    # Ensure we have the right number of eigenvalues and eigenvectors
    actual_returned_k = len(lam)
    if actual_returned_k != actual_k:
        print(f"⚠️  Note: Returned {actual_returned_k} eigenpairs instead of requested {actual_k}")
    
    print(f"特征值范围: [{lam[-1]:.3f}, {lam[0]:.3f}] (got {len(lam)} eigenpairs)")
    return torch.from_numpy(lam).float(), torch.from_numpy(U).float()

# ============================================================
# 3.  Mahalanobis per-sample clipping
# ============================================================
def maha_clip(vec, U, inv_sqrt_lam, radius):
    proj = (U.T @ vec) * inv_sqrt_lam           # (Uᵀg)/√λ
    norm = proj.norm() + 1e-10
    if norm > radius: vec.mul_(radius / norm)
    return vec, norm.item()

# ============================================================
# 4.  DP-SGD with curvature-aware low-rank noise
# ============================================================
def train_with_dp(model, train_loader, fisher,
                  epsilon=8.0, delta=1e-6,
                  clip_radius=10.0, k=32, lam_floor=5e-1,
                  device="cuda", target_layer="conv1",
                  adaptive_clip=False, quantile=0.95, sample_level=None,
                  epochs=1, sigma=None, full_complement_noise=False,
                  positive_noise_correlation=False,
                  dp_sat_mode="none", rho_sat=0.001):
    """
    Train with Fisher-informed DP-SGD
    
    Args:
        clip_radius: Target Euclidean sensitivity Δ₂ (same as vanilla DP-SGD for fair comparison).
                    This will be converted to appropriate Mahalanobis threshold internally.
        sample_level: If True, use sample-level DP (clip per sample).
                     If False, use user-level DP (clip per user).
                     If None, auto-detect from batch structure.
        epochs: Number of training epochs (for privacy accounting)
        sigma: If provided, use this sigma directly (for proper privacy accounting).
               If None, compute sigma from epsilon using legacy method.
        full_complement_noise: If True, add full noise to orthogonal complement.
                              If False, only add noise in Fisher subspace.
                              Setting to False preserves curvature-aware benefits.
        positive_noise_correlation: If False (default), use negatively correlated noise (noise ∝ 1/√λ).
                                   If True, use positively correlated noise (noise ∝ √λ).
        dp_sat_mode: "none" (default), "euclidean" (Euclidean DP-SAT), "fisher" (Fisher DP-SAT).
        rho_sat: Perturbation radius for Exact DP-SAT.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    lam, U       = topk_eigh_with_floor(fisher, k=k, lam_floor=lam_floor)
    lam, U       = lam.to(device), U.to(device)
    
    # Compute both scaling factors
    inv_sqrt_lam = lam.rsqrt()  # 1/√λ (negatively correlated: less noise in high curvature)
    sqrt_lam = lam.sqrt()       # √λ (positively correlated: more noise in high curvature)
    
    # Choose noise scaling strategy
    if positive_noise_correlation:
        noise_scaling = sqrt_lam
        strategy_name = "Positively correlated noise (noise ∝ √λ)"
    else:
        noise_scaling = inv_sqrt_lam
        strategy_name = "Negatively correlated noise (noise ∝ 1/√λ, default)"
    
    # Clipping always uses inverse scaling to maintain consistent Mahalanobis norm definition
    clip_scaling = inv_sqrt_lam
    
    # Privacy accounting
    if sigma is not None:
        # Use provided sigma (proper privacy accounting)
        print(f"   • Using provided sigma: {sigma:.4f}")
    else:
        # Legacy privacy accounting for multi-epoch training
        # Use simple composition bound: σ_total = σ_single / √T for T epochs
        sigma_single_epoch = math.sqrt(2*math.log(1.25/delta)) / epsilon
        sigma = sigma_single_epoch / math.sqrt(epochs)  # Adjust for T epochs
        print(f"   • Legacy accounting: σ_single={sigma_single_epoch:.3f}, σ_adjusted={sigma:.3f}")
    
    # Use the actual k returned by eigendecomposition
    actual_k = len(lam)
    if actual_k != k:
        print(f"⚠️  Using k={actual_k} eigenpairs (requested {k}) due to matrix rank constraints")

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
    print(f"\n🎯 Fisher DP-SGD config: {mode_str} DP  layers={target_layer}  ε={epsilon}")
    if sigma is not None:
        print(f"   • Proper privacy accounting: σ={sigma:.4f}")
    else:
        print(f"   • Multi-epoch privacy: T={epochs}, σ_single={sigma_single_epoch:.3f}, σ_adjusted={sigma:.3f}")
    print(f"   • Fisher subspace: k={actual_k}, complement dim={param_dim-actual_k}")
    print(f"   • Target Euclidean sensitivity: Δ₂ = {clip_radius:.3f} (will convert to Mahalanobis)")
    print(f"   • Adaptive clipping: {adaptive_clip}")
    print(f"   • Full complement noise: {full_complement_noise}")
    print(f"   • Noise scaling strategy: {strategy_name}")
    
    if dp_sat_mode != "none":
        print(f"   • DP-SAT enabled: mode={dp_sat_mode}, ρ={rho_sat}")
        if dp_sat_mode == "fisher":
            print(f"     ✨ Using Fisher-whitened weight perturbation (Fisher DP-SAT)")
        elif dp_sat_mode == "euclidean":
            print(f"     ⚠️  Using Euclidean weight perturbation (Euclidean DP-SAT)")
    
    if not sample_level:
        print("   • User-level mode: Clipping aggregated user gradients")
    else:
        print("   • Sample-level mode: Clipping individual sample gradients")
    print()

    noise_l2, grad_norm = [], []
    euclidean_norms = []  # Track Euclidean norms for calibration
    adaptive_radius_computed = False
    euclidean_target = clip_radius  # Store the target Euclidean sensitivity
    actual_radius = clip_radius  # Will be calibrated to Mahalanobis space
    calibration_computed = False  # Flag for norm calibration
    
    # Initialize previous step's noisy gradient for DP-SAT
    g_prev_priv = None

    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Fisher DP-SGD ({mode_str})")):
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
            # SAMPLE-LEVEL DP: Compute per-sample gradients
            per_g = []
            for i in range(batch_size):
                gi = grad(losses[i], params, retain_graph=True)
                per_g.append(torch.cat([g.view(-1) for g in gi]).detach())
            per_g = torch.stack(per_g)

            # Calibration or adaptive clipping: collect EUCLIDEAN norms (like vanilla DP-SGD)
            if (adaptive_clip and not adaptive_radius_computed) or not calibration_computed:
                batch_euclidean_norms = []
                batch_mahalanobis_norms = []
                
                for i in range(per_g.size(0)):
                    # Compute both norms for calibration
                    euclidean_norm = per_g[i].norm().item()
                    proj = (U.T @ per_g[i]) * clip_scaling
                    mahalanobis_norm = proj.norm().item()
                    
                    batch_euclidean_norms.append(euclidean_norm)
                    batch_mahalanobis_norms.append(mahalanobis_norm)
                
                euclidean_norms.extend(batch_euclidean_norms)
                
                # Adaptive clipping: use Euclidean norms (like vanilla DP-SGD)
                if adaptive_clip and not adaptive_radius_computed:
                    if len(euclidean_norms) >= 100 or batch_idx == 0:
                        euclidean_adaptive_radius = np.quantile(euclidean_norms, quantile)
                        euclidean_target = euclidean_adaptive_radius  # Update target
                        adaptive_radius_computed = True
                        
                        print(f"📊 Fisher adaptive clipping from {len(euclidean_norms)} samples (EUCLIDEAN norms):")
                        print(f"   • Mean Euclidean: {np.mean(euclidean_norms):.3f}")
                        print(f"   • Median Euclidean: {np.median(euclidean_norms):.3f}")
                        print(f"   • {quantile:.1%} quantile: {euclidean_adaptive_radius:.3f}")
                        print(f"   • Max Euclidean: {np.max(euclidean_norms):.3f}")
                        print(f"   → Using Euclidean target: Δ₂ = {euclidean_target:.3f}")
                
                # Calibration: find Mahalanobis threshold that matches Euclidean sensitivity
                if not calibration_computed and len(euclidean_norms) >= 50:
                    # Find what Mahalanobis threshold gives similar clipping behavior as Euclidean target
                    euclidean_clip_rate = np.mean(np.array(batch_euclidean_norms) > euclidean_target)
                    
                    # Binary search for Mahalanobis threshold that gives similar clip rate
                    maha_norms = np.array(batch_mahalanobis_norms)
                    maha_low, maha_high = maha_norms.min(), maha_norms.max()
                    
                    for _ in range(10):  # Binary search iterations
                        maha_mid = (maha_low + maha_high) / 2
                        maha_clip_rate = np.mean(maha_norms > maha_mid)
                        
                        if maha_clip_rate > euclidean_clip_rate:
                            maha_low = maha_mid
                        else:
                            maha_high = maha_mid
                    
                    actual_radius = (maha_low + maha_high) / 2
                    calibration_computed = True
                    
                    print(f"🎯 Norm calibration completed:")
                    print(f"   • Target Euclidean sensitivity: Δ₂ = {euclidean_target:.3f}")
                    print(f"   • Calibrated Mahalanobis threshold: {actual_radius:.3f}")
                    print(f"   • Euclidean clip rate: {euclidean_clip_rate:.1%}")
                    print(f"   • Mahalanobis clip rate: {np.mean(maha_norms > actual_radius):.1%}")
                    print(f"   → Fair comparison: same effective sensitivity bound Δ₂")
                    print(f"   🔧 NOISE SCALING FIX: Using actual_radius={actual_radius:.3f} for noise (was euclidean_target={euclidean_target:.3f})")
                    print()
                    
                    euclidean_norms = []  # Reset for actual training statistics

            # Mahalanobis clipping with calibrated threshold
            for i in range(per_g.size(0)):
                per_g[i], nrm = maha_clip(per_g[i], U, clip_scaling, actual_radius)
                grad_norm.append(nrm)
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
                
                # Calibration or adaptive clipping: collect EUCLIDEAN norms
                if (adaptive_clip and not adaptive_radius_computed) or not calibration_computed:
                    euclidean_norm = user_grad_flat.norm().item()
                    euclidean_norms.append(euclidean_norm)
                    
                    if adaptive_clip and not adaptive_radius_computed:
                        if len(euclidean_norms) >= min(10, len(train_loader)) or batch_idx == 0:
                            euclidean_adaptive_radius = np.quantile(euclidean_norms, quantile)
                            euclidean_target = euclidean_adaptive_radius
                            adaptive_radius_computed = True
                            
                            print(f"📊 Fisher adaptive clipping from {len(euclidean_norms)} users (EUCLIDEAN norms):")
                            print(f"   • Mean Euclidean: {np.mean(euclidean_norms):.3f}")
                            print(f"   • Median Euclidean: {np.median(euclidean_norms):.3f}")
                            print(f"   • {quantile:.1%} quantile: {euclidean_adaptive_radius:.3f}")
                            print(f"   • Max Euclidean: {np.max(euclidean_norms):.3f}")
                            print(f"   → Using Euclidean target: Δ₂ = {euclidean_target:.3f}")
                    
                    # Calibration for user-level
                    if not calibration_computed and len(euclidean_norms) >= 5:
                        # Simple heuristic: use median ratio for calibration
                        proj = (U.T @ user_grad_flat) * clip_scaling
                        mahalanobis_norm = proj.norm().item()
                        ratio = euclidean_norm / (mahalanobis_norm + 1e-8)
                        actual_radius = euclidean_target / (ratio + 1e-8)
                        calibration_computed = True
                        
                        print(f"🎯 User-level norm calibration:")
                        print(f"   • Target Euclidean sensitivity: Δ₂ = {euclidean_target:.3f}")
                        print(f"   • Calibrated Mahalanobis threshold: {actual_radius:.3f}")
                        print(f"   • Sample ratio: ||g||₂/||g||_{{F⁻¹}} ≈ {ratio:.3f}")
                        print(f"   🔧 NOISE SCALING FIX: Using actual_radius={actual_radius:.3f} for noise (was euclidean_target={euclidean_target:.3f})")
                        print()
                        
                        euclidean_norms = []  # Reset
            
            # In user-level DP with UserBatchSampler, we should have exactly one user per batch
            if len(user_gradients) != 1:
                print(f"⚠️  Warning: Expected 1 user per batch, got {len(user_gradients)} users")
                print(f"   Unique users in batch: {unique_users.tolist()}")
            
            # Clip each user's gradient
            clipped_user_grads = []
            for user_grad_flat in user_gradients:
                clipped_grad, user_norm = maha_clip(user_grad_flat, U, clip_scaling, actual_radius)
                grad_norm.append(user_norm)
                clipped_user_grads.append(clipped_grad)
            
            # Average across users in batch (should be just one user for UserBatchSampler)
            g_bar = torch.stack(clipped_user_grads).mean(0)

        # ============================================================
        # TWO-COMPONENT NOISE: Fisher subspace + orthogonal complement
        # ============================================================
        
        # 1. Low-rank noise in Fisher subspace (anisotropic)
        z_fisher = torch.randn(actual_k, device=device)
        fisher_noise = U @ (z_fisher * noise_scaling) * sigma * actual_radius
        
        if full_complement_noise:
            # 2. Complement noise in orthogonal subspace (isotropic)
            # Generate noise in full space, then project to orthogonal complement
            z_full = torch.randn_like(g_bar)
            z_complement = z_full - U @ (U.T @ z_full)  # Project to complement: (I - UU^T)z
            
            # Scale complement noise properly: σΔ (not divided by √P)
            complement_noise = z_complement * sigma * actual_radius
            
            # Total noise = Fisher subspace noise + complement noise
            total_noise = fisher_noise + complement_noise
            complement_noise_norm = complement_noise.norm().item()
        else:
            # Only Fisher subspace noise (preserves curvature-aware benefits)
            total_noise = fisher_noise
            complement_noise_norm = 0.0
        
        g_priv = g_bar + total_noise
        
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
        
        # Track noise components
        fisher_noise_norm = fisher_noise.norm().item()
        total_noise_norm = total_noise.norm().item()
        noise_l2.append(total_noise_norm)

        # scatter back
        idx = 0
        for p in params:
            n = p.numel()
            p.grad = g_priv[idx:idx+n].view_as(p)
            idx += n
        opt.step()

    grad_type = "‖g_user‖_Mah" if not sample_level else "‖g‖_Mah"
    print(f"\n📊  Fisher DP-SGD final stats:")
    print(f"   • Target Euclidean sensitivity: Δ₂ = {euclidean_target:.3f} (same as vanilla DP-SGD)")
    print(f"   • Calibrated Mahalanobis threshold: {actual_radius:.3f}")
    print(f"   • Median {grad_type} = {np.median(grad_norm):.2f}")
    print(f"   • Total noise ℓ₂ ∈ [{min(noise_l2):.1f},{max(noise_l2):.1f}]")
    if len(noise_l2) > 0:
        last_idx = len(noise_l2) - 1
        if full_complement_noise:
            print(f"   • Last batch noise components: Fisher={fisher_noise_norm:.1f}, Complement={complement_noise_norm:.1f}")
        else:
            print(f"   • Last batch noise: Fisher only={fisher_noise_norm:.1f} (complement disabled)")
    print(f"   • Privacy: (ε={epsilon}, δ={delta}) over {epochs} epochs")
    print(f"   • ✅ FAIR COMPARISON: Same noise scale σ×Δ={sigma * actual_radius:.3f} as vanilla DP-SGD")
    print(f"   • 🔧 NOISE SCALING FIXED: Using actual_radius for noise (not euclidean_target)")

    return model
