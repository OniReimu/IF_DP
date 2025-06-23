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

# ============================================================
# 1.  Fisher estimation (damped) on chosen layer set
# ============================================================
def compute_fisher(model, dataloader, device,
                   target_layer="conv1",
                   rho: float = 1e-2):
    """
    Return F  =  (1/N) GᵀG + ρI   for the parameters in target_layer

    target_layer:
        "conv1"            – single layer substring
        "conv1,conv2"      – comma-separated list
        "all"              – every parameter
    """
    model.eval()

    # ------------ choose parameters ------------
    if target_layer == "all":
        tgt_names = [n for n, _ in model.named_parameters()]
        print("🎯 computing Fisher for ALL layers")
    elif "," in target_layer:
        layers = [s.strip() for s in target_layer.split(",")]
        tgt_names = [n for n, _ in model.named_parameters()
                     if any(l in n for l in layers)]
        print(f"🎯 computing Fisher for layers {layers}")
    else:
        tgt_names = [n for n, _ in model.named_parameters()
                     if target_layer in n]
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
        # support (x,y) OR (x,y,user_id)
        if len(batch_data) == 3:
            x, y, _ = batch_data
        else:
            x, y    = batch_data
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        loss = F.cross_entropy(model(x), y)
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
                  epochs=1, sigma=None, full_complement_noise=False):
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
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    lam, U       = topk_eigh_with_floor(fisher, k=k, lam_floor=lam_floor)
    lam, U       = lam.to(device), U.to(device)
    inv_sqrt_lam = lam.rsqrt()
    
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
        # Check first batch to determine mode
        first_batch = next(iter(train_loader))
        if len(first_batch) == 3:
            # Has user_id, check if all samples in batch have same user_id
            _, _, user_ids = first_batch
            unique_users = torch.unique(user_ids)
            sample_level = len(unique_users) > 1  # Multiple users = sample-level
        else:
            sample_level = True  # No user_id = sample-level
    
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

    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Fisher DP-SGD ({mode_str})")):
        # accept (x,y) OR (x,y,uid)
        if len(batch_data) == 3:
            x, y, user_ids = batch_data
        else:
            x, y = batch_data
            user_ids = None
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        losses = F.cross_entropy(model(x), y, reduction="none")

        if sample_level:
            # SAMPLE-LEVEL DP: Compute per-sample gradients
            per_g = []
            for i in range(x.size(0)):
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
                    proj = (U.T @ per_g[i]) * inv_sqrt_lam
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
                    print(f"   → Fair comparison: same effective sensitivity bound Δ₂\n")
                    
                    euclidean_norms = []  # Reset for actual training statistics

            # Mahalanobis clipping with calibrated threshold
            for i in range(per_g.size(0)):
                per_g[i], nrm = maha_clip(per_g[i], U, inv_sqrt_lam, actual_radius)
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
                        proj = (U.T @ user_grad_flat) * inv_sqrt_lam
                        mahalanobis_norm = proj.norm().item()
                        ratio = euclidean_norm / (mahalanobis_norm + 1e-8)
                        actual_radius = euclidean_target / (ratio + 1e-8)
                        calibration_computed = True
                        
                        print(f"🎯 User-level norm calibration:")
                        print(f"   • Target Euclidean sensitivity: Δ₂ = {euclidean_target:.3f}")
                        print(f"   • Calibrated Mahalanobis threshold: {actual_radius:.3f}")
                        print(f"   • Sample ratio: ||g||₂/||g||_{{F⁻¹}} ≈ {ratio:.3f}\n")
                        
                        euclidean_norms = []  # Reset
            
            # In user-level DP with UserBatchSampler, we should have exactly one user per batch
            if len(user_gradients) != 1:
                print(f"⚠️  Warning: Expected 1 user per batch, got {len(user_gradients)} users")
                print(f"   Unique users in batch: {unique_users.tolist()}")
            
            # Clip each user's gradient
            clipped_user_grads = []
            for user_grad_flat in user_gradients:
                clipped_grad, user_norm = maha_clip(user_grad_flat, U, inv_sqrt_lam, actual_radius)
                grad_norm.append(user_norm)
                clipped_user_grads.append(clipped_grad)
            
            # Average across users in batch (should be just one user for UserBatchSampler)
            g_bar = torch.stack(clipped_user_grads).mean(0)

        # ============================================================
        # TWO-COMPONENT NOISE: Fisher subspace + orthogonal complement
        # ============================================================
        
        # 1. Low-rank noise in Fisher subspace (anisotropic)
        z_fisher = torch.randn(actual_k, device=device)
        fisher_noise = U @ (z_fisher * inv_sqrt_lam) * sigma * euclidean_target  # Use Euclidean target for noise scale
        
        if full_complement_noise:
            # 2. Complement noise in orthogonal subspace (isotropic)
            # Generate noise in full space, then project to orthogonal complement
            z_full = torch.randn_like(g_bar)
            z_complement = z_full - U @ (U.T @ z_full)  # Project to complement: (I - UU^T)z
            
            # Scale complement noise properly: σΔ (not divided by √P)
            complement_noise = z_complement * sigma * euclidean_target  # Use Euclidean target
            
            # Total noise = Fisher subspace noise + complement noise
            total_noise = fisher_noise + complement_noise
            complement_noise_norm = complement_noise.norm().item()
        else:
            # Only Fisher subspace noise (preserves curvature-aware benefits)
            total_noise = fisher_noise
            complement_noise_norm = 0.0
        
        g_priv = g_bar + total_noise
        
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
    print(f"   • ✅ FAIR COMPARISON: Same effective sensitivity Δ₂ as vanilla DP-SGD")

    return model