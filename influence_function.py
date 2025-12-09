# influence_function.py
# ==============================================================
# Curvature-aware DP -- Influence-function calibration utilities
# ==============================================================

import torch, numpy as np, cvxpy as cp, copy
import torch.nn.functional as F
from tqdm import tqdm

# --------------------------------------------------------------
# 0.  Small helpers
# --------------------------------------------------------------
def unpack_batch(batch_data):
    """Accept (x,y) or (x,y,user_id) batches and return x,y,(user_id|None)."""
    return (batch_data + (None,))[:3]

# --------------------------------------------------------------
# 1.  Critical slice extraction + gradients
# --------------------------------------------------------------
def get_evaluation_slice(eval_loader, target_class="all", max_samples_per_class=200, device="cpu"):
    """
    Extract evaluation slice for calibration.
    
    Paper mapping — Step a (Critical slice):
    S_crit = {all classes in the public test split} when target_class == "all".
    
    Args:
        target_class: 
            - "all": Use all classes (recommended for general utility improvement)
            - int: Use specific class only (for targeted fairness applications)
        max_samples_per_class: Maximum samples per class to avoid memory issues
    """
    if target_class == "all":
        print(f"🎯 Using ALL CLASSES for calibration (general utility improvement)")
        
        # Collect samples by class for balanced sampling
        class_samples = {}
        
        for batch_data in eval_loader:
            # Handle both (x, y) and (x, y, user_id) formats
            if len(batch_data) == 3:
                x, y, _ = batch_data  # x, y, user_id
            else:
                x, y = batch_data     # x, y only
            
            for i in range(len(x)):
                class_id = y[i].item()
                if class_id not in class_samples:
                    class_samples[class_id] = {'x': [], 'y': []}
                
                # Add sample if we haven't reached the limit for this class
                if len(class_samples[class_id]['x']) < max_samples_per_class:
                    class_samples[class_id]['x'].append(x[i:i+1])
                    class_samples[class_id]['y'].append(y[i:i+1])
        
        # Combine all classes
        all_x, all_y = [], []
        total_samples = 0
        
        for class_id in sorted(class_samples.keys()):
            class_x = torch.cat(class_samples[class_id]['x'])
            class_y = torch.cat(class_samples[class_id]['y'])
            all_x.append(class_x)
            all_y.append(class_y)
            total_samples += len(class_x)
            print(f"   • Class {class_id}: {len(class_x)} samples")
        
        if not all_x:
            print(f"⚠️  No samples found in evaluation data")
            return torch.empty(0, 3, 32, 32, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)
        
        eval_x = torch.cat(all_x).to(device)
        eval_y = torch.cat(all_y).to(device)
        
        print(f"✅ Evaluation slice: {total_samples} samples across {len(class_samples)} classes")
        
    else:
        # Single class mode (legacy behavior)
        print(f"🎯 Using SINGLE CLASS {target_class} for calibration (targeted improvement)")
        
        eval_x, eval_y = [], []
        for batch_data in eval_loader:
            # Handle both (x, y) and (x, y, user_id) formats
            if len(batch_data) == 3:
                x, y, _ = batch_data  # x, y, user_id
            else:
                x, y = batch_data     # x, y only
                
            m = y == target_class
            if m.any():
                eval_x.append(x[m])
                eval_y.append(y[m])
        
        if not eval_x:
            print(f"⚠️  No samples of class {target_class}")
            return torch.empty(0, 3, 32, 32, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)
        
        eval_x = torch.cat(eval_x).to(device)
        eval_y = torch.cat(eval_y).to(device)
        print(f"🎯 Evaluation slice: {len(eval_x)} samples of class {target_class}")
    
    return eval_x, eval_y


# Keep old function name for backward compatibility but mark as deprecated
def get_critical_slice(eval_loader, target_class: int = 3, device="cpu"):
    """
    ⚠️  DEPRECATED: This function is deprecated and causes overfitting to single class.
    Use get_evaluation_slice with target_class="all" for general utility improvement.
    """
    print(f"⚠️  WARNING: get_critical_slice is deprecated and causes single-class overfitting!")
    print(f"⚠️  Recommendation: Use get_evaluation_slice(target_class='all') instead")
    return get_evaluation_slice(eval_loader, target_class=target_class, device=device)


@torch.no_grad()
def _frob_norm(tensor_dict):
    return torch.sqrt(
        sum(v.pow(2).sum() for v in tensor_dict.values())
    )


def compute_slice_gradient(model, crit_x, crit_y, device):
    """J = ∇_θ 1/m Σ ℓ(s,θ)  on critical slice."""
    if not len(crit_x):
        return {n: torch.zeros_like(p.data) for n, p in model.named_parameters()}
    model.zero_grad()
    F.cross_entropy(model(crit_x), crit_y, reduction='mean').backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in model.named_parameters()}

# --------------------------------------------------------------
# 2.  Influence-vector bank  H⁻¹ ∇ℓ(z)
# --------------------------------------------------------------
def compute_influence_vectors(model, public_loader, train_loader,
                              device, method="linear",
                              reg=0.1,  # ✨ PRIVACY-PRESERVING: Reduced regularization
                              strict=True):
    """
    🔒 PRIVACY-PRESERVING: Compute influence vectors using ONLY public data and DP model.
    
    IMPORTANT: For post-processing theorem compliance, this function must NOT use
    any statistics computed on the private training data (train_loader).
    
    Only 'linear' method is privacy-preserving. Other methods violate DP guarantees.
    """
    # flatten public data to list[(x,y)]
    public_samples = []
    for batch_data in public_loader:
        # Handle both (x, y) and (x, y, user_id) formats
        if len(batch_data) == 3:
            x, y, _ = batch_data  # x, y, user_id
        else:
            x, y = batch_data     # x, y only
            
        for i in range(x.size(0)):
            public_samples.append((x[i:i+1].to(device),
                                   y[i:i+1].to(device)))

    infl_vecs = []

    if method == "linear":                                 # -----------------
        print(f"🔒 Using privacy-preserving 'linear' method (reg={reg})")
        for x, y in tqdm(public_samples, desc="Privacy-preserving linear-IF"):
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            vec = {}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    # ✨ MORE CONSERVATIVE: Scale by (1 + ||g||) instead of ||g|| alone for stability
                    g_norm = g.norm() + 1e-8
                    scaling_factor = 1.0 + g_norm  # More conservative scaling
                    vec[n] = (g / (scaling_factor * reg)).cpu()  # store on CPU to avoid MPS/GPU OOM
                else:
                    vec[n] = torch.zeros_like(p, device="cpu")
            infl_vecs.append(vec)

    elif method == "public-fisher":                       # -----------------
        print(f"🔒 Using privacy-preserving 'public-fisher' method (reg={reg})")
        print(f"   Computing Fisher information from PUBLIC data only")
        
        # Use a smaller subset for Fisher computation to avoid memory issues
        max_fisher_samples = min(100, len(public_samples))  # Much smaller for memory efficiency
        fisher_samples = public_samples[:max_fisher_samples]
        
        print(f"   Using {max_fisher_samples} public samples for Fisher matrix computation")
        
        # Compute Fisher matrix from public data only (privacy-preserving)
        public_grads = []
        for x, y in tqdm(fisher_samples, desc="Computing public Fisher"):
            try:
                model.zero_grad()
                F.cross_entropy(model(x), y).backward()
                grad_vec = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                public_grads.append(grad_vec.detach().cpu())  # Move to CPU to save GPU memory
            except RuntimeError as e:
                print(f"⚠️  Error computing gradient: {e}")
                continue
        
        if len(public_grads) >= 10:  # Need minimum samples for stable Fisher
            try:
                G_public = torch.stack(public_grads)
                dim = G_public.shape[1]
                
                # Add strong regularization for stability
                fisher_public = (G_public.T @ G_public) / len(G_public) + reg * torch.eye(dim)
                
                print(f"   Public Fisher shape: {fisher_public.shape}")
                print(f"   Successfully computed Fisher from {len(public_grads)} public samples")
                
                # Compute influence vectors using public Fisher
                for x, y in tqdm(public_samples[:200], desc="Public-Fisher influence vectors"):  # Limit total computations
                    try:
                        model.zero_grad()
                        F.cross_entropy(model(x), y).backward()
                        grad_vec = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).cpu()
                        
                        # Solve: fisher_public * v = grad_vec
                        try:
                            influence_vec_flat = torch.linalg.solve(fisher_public, grad_vec.unsqueeze(1)).squeeze(1)
                        except:
                            # Fallback to pseudo-inverse if singular
                            print("   Using pseudo-inverse for stability")
                            influence_vec_flat = torch.pinverse(fisher_public) @ grad_vec
                        
                        # Reconstruct parameter-wise influence vector
                        vec = {}
                        idx = 0
                        for n, p in model.named_parameters():
                            if p.grad is not None:
                                n_params = p.numel()
                                vec[n] = influence_vec_flat[idx:idx+n_params].view_as(p).to(device)
                                idx += n_params
                            else:
                                vec[n] = torch.zeros_like(p)
                        infl_vecs.append(vec)
                    except Exception as e:
                        print(f"⚠️  Error in influence computation: {e}")
                        # Fallback: create zero influence vector
                        vec = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
                        infl_vecs.append(vec)
                        
            except Exception as e:
                print(f"⚠️  Error computing public Fisher matrix: {e}")
                print(f"   Falling back to linear method")
                return compute_influence_vectors(model, public_loader, train_loader, device, method="linear", reg=reg, strict=strict)
        else:
            print(f"⚠️  Insufficient public gradients ({len(public_grads)} < 10), falling back to linear method")
            return compute_influence_vectors(model, public_loader, train_loader, device, method="linear", reg=reg, strict=strict)

    elif method == "batch":                                # -----------------
        raise ValueError(
            f"🚨 PRIVACY VIOLATION: 'batch' method uses private training data statistics. "
            f"This violates the post-processing theorem. Use method='linear' instead."
        )

    elif method == "original":                             # -----------------
        raise ValueError(
            f"🚨 PRIVACY VIOLATION: 'original' method uses private training data statistics. "
            f"This violates the post-processing theorem. Use method='linear' instead."
        )

    else:                                                  # -----------------
        raise ValueError(f"unknown method '{method}'. Choose 'linear' for privacy-preserving influence functions.")

    print(f"✅ Computed {len(infl_vecs)} privacy-preserving influence vectors using only public data")
    return infl_vecs, public_samples

# --------------------------------------------------------------
# 3.  Trust-region scaling (shared util)
# --------------------------------------------------------------
def apply_trust_region(calib_dict, ref_norm, tau=0.05):
    """
    Scale Δθ so that ‖Δθ‖₂ ≤ τ·ref_norm.
    (ref_norm = ‖θ̂_DP‖₂  or any baseline scale)
    """
    corr_norm = _frob_norm(calib_dict)
    if corr_norm < 1e-12:
        return calib_dict
    scale = min(1.0, tau * ref_norm / corr_norm)
    if scale < 1.0:
        for n in calib_dict:
            calib_dict[n].mul_(scale)
    return calib_dict

# --------------------------------------------------------------
# 4.  Research-protocol calibration  (lightly modified)
# --------------------------------------------------------------
def calibrate_model_research_protocol(model,
                                      public_loader,
                                      train_loader,
                                      crit_x, crit_y,
                                      device,
                                      method="linear",
                                      eta=200,
                                      target_improve=0.1,
                                      trust_tau=0.01,  # ✨ MUCH MORE CONSERVATIVE: 1% instead of 10%
                                      strict=True,
                                      clean_model=None,
                                      reg=10.0):  # ✨ MUCH MORE CONSERVATIVE: Higher regularization
    """
    🔒 PRIVACY-PRESERVING influence calibration following post-processing theorem.
    
    This function only uses:
    1. DP model parameters (output of DP algorithm)  
    2. Public dataset (dataset B)
    3. No statistics from private training data (dataset A)
    
    Args:
        clean_model: Non-DP baseline model for measuring ΔL_DP (utility drop due to DP)
        trust_tau: Trust region parameter (default 0.1 = 10% of model norm)
        reg: Regularization for influence vector computation (smaller = less regularization)
    """

    model.eval()

    # ---- 0) MEASURE ΔL_DP: Utility drop due to DP noise
    # Paper mapping — Step b (Measure utility drop):
    #   ΔL_DP = (1/|S_crit|) Σ_{s∈S_crit} [ℓ(s, θ̂_DP) − ℓ(s, θ̂_clean)]
    if clean_model is not None and len(crit_x) > 0:
        print(f"\n📏 MEASURING ΔL_DP: Utility drop due to DP noise on critical slice")
        
        clean_model.eval()
        model.eval()
        
        with torch.no_grad():
            # Loss on clean model
            clean_loss = F.cross_entropy(clean_model(crit_x), crit_y, reduction='mean')
            # Loss on DP model  
            dp_loss = F.cross_entropy(model(crit_x), crit_y, reduction='mean')
            
            delta_L_DP = dp_loss.item() - clean_loss.item()
            
        print(f"   • Clean model loss: {clean_loss.item():.4f}")
        print(f"   • DP model loss:    {dp_loss.item():.4f}")
        print(f"   • ΔL_DP = L_DP - L_clean = {delta_L_DP:+.4f}")
        
        if delta_L_DP > 0:
            print(f"   📈 DP noise caused utility degradation (higher loss)")
            print(f"   🎯 Goal: Use calibration to recover utility loss")
        else:
            print(f"   📉 DP model performs better than clean (unusual)")
    else:
        if clean_model is None:
            print(f"\n⚠️  SKIPPING ΔL_DP measurement: No clean_model provided")
        else:
            print(f"\n⚠️  SKIPPING ΔL_DP measurement: No critical slice samples")

    # ---- a) slice gradient J
    # Paper mapping — Step c(a): J = ∇_θ (1/m Σ_{s∈S_crit} ℓ(s, θ̂)) at the current θ̂
    J = compute_slice_gradient(model, crit_x, crit_y, device)
    J_flat = torch.cat([v.flatten() for v in J.values()])

    # ---- b) influence vectors (PRIVACY-PRESERVING: only uses public data + DP model)
    # Paper mapping — Step c(b): v(z) ≈ H_{θ̂}^{-1} ∇_θ ℓ(z, θ̂) for z in the public pool
    infl_vecs, public_samples = compute_influence_vectors(model, public_loader,
                                             train_loader, device,
                                             method=method, reg=reg, strict=strict)

    # ---- c) influence scores: how much would adding each public sample help the evaluation slice?
    # Paper mapping — Step c(c): α(z) = - J^T v(z).
    # If J points to increasing loss and v(z) points to decreasing loss, then α(z) > 0 is helpful.
    # Note: α(z) depends on current θ̂, so it must be recomputed after each update in the refinement loop.
    scores = np.array([
        -torch.dot(
            J_flat,
            torch.cat([v[n].flatten() for n in J.keys()]).to(J_flat.device)
        ).item()
        for v in infl_vecs
    ])

    # ---- d) choose η most helpful samples 
    # Paper mapping — Step d(i): Initial selection via sparse re-weighting
    # ✨ Implementation: indicator weights by selecting the η largest (most helpful) α(z)
    idx = np.argsort(scores)[-eta:]  # Select largest (most positive) scores
    w = np.zeros_like(scores)
    w[idx] = 1.0

    print(f"\n📊 Influence score statistics:")
    print(f"   • Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   • Selected {eta} samples with HIGHEST scores: [{scores[idx].min():.4f}, {scores[idx].max():.4f}]")
    print(f"   • Mean selected score: {scores[idx].mean():.4f}")
    print(f"   • Score std: {scores.std():.4f}")
    print(f"   • Median score: {np.median(scores):.4f}")
    print(f"   🎯 Goal: Select samples that help reduce evaluation slice loss (positive scores)")
    
    # Additional diagnostic: check if we have any helpful samples
    positive_scores = scores[scores > 0]
    negative_scores = scores[scores < 0]
    print(f"   • Helpful samples (score > 0): {len(positive_scores)} out of {len(scores)}")
    print(f"   • Harmful samples (score < 0): {len(negative_scores)} out of {len(scores)}")
    
    if len(positive_scores) == 0:
        print(f"   ⚠️  WARNING: No helpful public samples found! All would increase evaluation loss.")
        print(f"   💡 This suggests domain mismatch between public and evaluation data.")
    elif len(positive_scores) < eta:
        print(f"   ⚠️  WARNING: Only {len(positive_scores)} helpful samples, but requesting {eta}")
        print(f"   💡 Reducing eta to {len(positive_scores)} to avoid harmful samples")
        # Only use actually helpful samples
        helpful_idx = np.where(scores > 0)[0]
        w = np.zeros_like(scores)
        w[helpful_idx] = 1.0
        eta = len(helpful_idx)

    # 🔍 DIAGNOSTIC: Check influence vector statistics
    infl_norms = [torch.sqrt(sum(v[n].pow(2).sum() for n in v.keys())).item() for v in infl_vecs]
    print(f"\n🔍 Influence vector diagnostics:")
    print(f"   • Influence vector norms range: [{min(infl_norms):.4f}, {max(infl_norms):.4f}]")
    print(f"   • Mean influence vector norm: {np.mean(infl_norms):.4f}")
    print(f"   • Influence vectors used (nonzero weights): {np.sum(w > 0)}")

    # ---- e) Bias computation
    # Paper mapping — Step d(ii): Δθ = - (1/n) H_{θ̂}^{-1} Σ_z w_z ∇ℓ(z, θ̂)
    # Here we average v(z) ≈ H^{-1}∇ℓ(z) over selected z, implementing the same update up to scaling.
    # Move in direction that helpful public samples suggest
    n_selected = np.sum(w > 0)  # Number of selected samples
    if n_selected == 0:
        print(f"⚠️  WARNING: No helpful samples found for calibration - skipping update")
        print(f"   💡 This suggests the public data can't help improve the evaluation slice")
        return model
        
    delta = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    for i, weight in enumerate(w):
        if weight > 0:  # Only sum over helpful samples
            for n in delta:
                delta[n] += infl_vecs[i][n].to(delta[n].device) * weight / n_selected  # Average of helpful influence vectors

    # 🔍 DIAGNOSTIC: Check delta before trust region
    delta_norm_before = _frob_norm(delta)
    print(f"\n🔍 Parameter update diagnostics:")
    print(f"   • Raw Δθ norm before trust region: {delta_norm_before:.4f}")
    print(f"   • Using CORRECTED formula: Δθ = (1/η) Σ_{{helpful}} H⁻¹ ∇ℓ(public_sample)")
    print(f"   • Helpful samples used η: {n_selected}")
    print(f"   • Direction: Move model toward what helpful public samples suggest")
    print(f"   • Privacy-preserving: Uses only public data + DP model output")

    # ---- f) trust-region clip (IMPROVED: more generous)
    # Paper mapping — Step d(iii): Back-tracking line search over α ∈ {1, 1/2, 1/4, ...}
    # Note: We enforce a conservative trust region here. The explicit back-tracking
    #       line search that picks α by minimizing L_crit(θ̂ + αΔθ) is executed in
    #       ablation_optimized.calibrate_with_line_search(), which wraps this update.
    ref_norm = torch.sqrt(sum(p.pow(2).sum()
                              for p in model.parameters())).item()
    print(f"   • Model parameter norm ‖θ‖: {ref_norm:.4f}")
    print(f"   • Trust region limit (τ × ‖θ‖): {trust_tau * ref_norm:.4f}")
    
    delta = apply_trust_region(delta, ref_norm, tau=trust_tau)
    delta_norm_after = _frob_norm(delta)
    print(f"   • Final Δθ norm after trust region: {delta_norm_after:.4f}")
    print(f"   • Trust region scaling factor: {delta_norm_after / (delta_norm_before + 1e-12):.4f}")
    
    # 🔍 DIAGNOSTIC: Check relative parameter change
    relative_change = delta_norm_after / ref_norm
    print(f"   • Relative parameter change: {relative_change:.4f} ({relative_change*100:.2f}%)")
    
    if relative_change > 0.2:
        print(f"   ⚠️  WARNING: Large parameter change (>{20:.0f}%) may cause instability")
    elif relative_change < 0.005:
        print(f"   ⚠️  WARNING: Very small parameter change (<{0.5:.1f}%) may be ineffective")
        print(f"   💡 SUGGESTION: Try decreasing reg parameter or increasing trust_tau")

    # ---- g) apply calibration: θ̂* = θ̂_DP + Δθ
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(delta[n])

    print(f"✅ Privacy-preserving calibration applied (‖Δθ‖₂ ≈ {_frob_norm(delta):.4f}, "
          f"trust τ={trust_tau})")
    print(f"🔒 Post-processing theorem: Privacy guarantees preserved")
    return model


def calibrate_model(model, public_loader, train_loader, fisher,
                    top_k=200, device=None, target_class="all", **kw):
    """
    🔄 UPDATED: Now uses ALL CLASSES by default for general utility improvement.
    
    Args:
        target_class: "all" for general utility (default), or int for specific class
    """
    eval_x, eval_y = get_evaluation_slice(public_loader, target_class, device=device)
    return calibrate_model_research_protocol(
        model, public_loader, train_loader,
        eval_x, eval_y, device,
        method="linear", eta=top_k, trust_tau=0.01, reg=10.0, strict=True, **kw)


def calibrate_model_efficient(model, public_loader, train_loader, fisher,
                             top_k=200, device=None, method="linear",
                             target_class="all", **kw):
    """
    🔄 UPDATED: Now uses ALL CLASSES by default for general utility improvement.
    
    Args:
        target_class: "all" for general utility (default), or int for specific class
    """
    eval_x, eval_y = get_evaluation_slice(public_loader, target_class, device=device)
    return calibrate_model_research_protocol(
        model, public_loader, train_loader,
        eval_x, eval_y, device,
        method=method, eta=top_k, trust_tau=0.01, reg=10.0, strict=True, **kw)