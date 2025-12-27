# influence_function.py
# ==============================================================
# Curvature-aware DP -- Influence-function calibration utilities
# ==============================================================

import torch, numpy as np, cvxpy as cp, copy
import torch.nn.functional as F
from tqdm import tqdm
from data.common import move_to_device
from config import get_logger

logger = get_logger("calibration")

# --------------------------------------------------------------
# 0.  Small helpers
# --------------------------------------------------------------
def unpack_batch(batch_data):
    """
    Accept dict/tuple/list batches and return (x, y, user_id or None).
    Mirrors data.common.unpack_batch so calibration works with text loaders.
    """
    if isinstance(batch_data, dict):
        labels = batch_data.get("labels")
        if labels is None and "label" in batch_data:
            labels = batch_data["label"]
        user_ids = batch_data.get("user_ids")
        feature_keys = [k for k in batch_data.keys() if k not in {"labels", "label", "user_ids"}]
        if not feature_keys:
            raise ValueError("Dictionary batch must contain feature fields")
        if len(feature_keys) == 1:
            features = batch_data[feature_keys[0]]
        else:
            features = {k: batch_data[k] for k in feature_keys}
        return features, labels, user_ids

    if isinstance(batch_data, (tuple, list)):
        if len(batch_data) < 2:
            raise ValueError("Batch must contain at least (x, y)")
        batch_tuple = tuple(batch_data) + (None,)
        return batch_tuple[0], batch_tuple[1], batch_tuple[2]

    raise ValueError(f"Unsupported batch type: {type(batch_data)}")


def _infer_batch_size(features):
    """Best-effort batch size detection for tensor/dict/list feature containers."""
    if torch.is_tensor(features):
        return features.size(0)
    if isinstance(features, dict):
        for value in features.values():
            return _infer_batch_size(value)
        raise ValueError("Dictionary batch must contain at least one feature tensor")
    if isinstance(features, (list, tuple)):
        if not features:
            raise ValueError("Empty feature container")
        first = features[0]
        if torch.is_tensor(first) or isinstance(first, (dict, list, tuple)):
            return _infer_batch_size(first)
        return len(features)
    raise ValueError(f"Cannot infer batch size from type {type(features)}")


def _slice_feature(features, idx):
    """Return idx-th sample from tensor/dict/list feature container."""
    if torch.is_tensor(features):
        return features[idx:idx+1]
    if isinstance(features, dict):
        return {k: _slice_feature(v, idx) for k, v in features.items()}
    if isinstance(features, list):
        if not features:
            raise ValueError("Empty feature list")
        first = features[0]
        if torch.is_tensor(first) or isinstance(first, (dict, list, tuple)):
            return [_slice_feature(v, idx) for v in features]
        return features[idx]
    if isinstance(features, tuple):
        if not features:
            raise ValueError("Empty feature tuple")
        first = features[0]
        if torch.is_tensor(first) or isinstance(first, (dict, list, tuple)):
            return tuple(_slice_feature(v, idx) for v in features)
        return features[idx]
    raise ValueError(f"Unsupported feature type for slicing: {type(features)}")


def _ensure_tensor(values):
    if torch.is_tensor(values):
        return values
    return torch.as_tensor(values)

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
        logger.info("Using ALL classes for calibration (general utility improvement).")
        
        # Collect samples by class for balanced sampling
        class_samples = {}
        
        for batch_data in eval_loader:
            x, y, _ = unpack_batch(batch_data)
            
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
            logger.info("   • Class %s: %s samples", class_id, len(class_x))
        
        if not all_x:
            logger.warn("No samples found in evaluation data.")
            return torch.empty(0, 3, 32, 32, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)
        
        eval_x = torch.cat(all_x).to(device)
        eval_y = torch.cat(all_y).to(device)
        
        logger.success("Evaluation slice: %s samples across %s classes", total_samples, len(class_samples))
        
    else:
        # Single class mode (legacy behavior)
        logger.info("Using SINGLE CLASS %s for calibration (targeted improvement).", target_class)
        
        eval_x, eval_y = [], []
        for batch_data in eval_loader:
            x, y, _ = unpack_batch(batch_data)
                
            m = y == target_class
            if m.any():
                eval_x.append(x[m])
                eval_y.append(y[m])
        
        if not eval_x:
            logger.warn("No samples of class %s", target_class)
            return torch.empty(0, 3, 32, 32, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)
        
        eval_x = torch.cat(eval_x).to(device)
        eval_y = torch.cat(eval_y).to(device)
        logger.success("Evaluation slice: %s samples of class %s", len(eval_x), target_class)
    
    return eval_x, eval_y


# Keep old function name for backward compatibility but mark as deprecated
def get_critical_slice(eval_loader, target_class: int = 3, device="cpu"):
    """
    ⚠️  DEPRECATED: This function is deprecated and causes overfitting to single class.
    Use get_evaluation_slice with target_class="all" for general utility improvement.
    """
    logger.warn("get_critical_slice is deprecated and causes single-class overfitting.")
    logger.warn("Recommendation: use get_evaluation_slice(target_class='all') instead.")
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
        x, y, _ = unpack_batch(batch_data)
        batch_size = _infer_batch_size(x)
        labels = _ensure_tensor(y)

        for i in range(batch_size):
            sample_x = _slice_feature(x, i)
            sample_y = labels[i:i+1]
            public_samples.append((
                move_to_device(sample_x, device),
                move_to_device(sample_y, device),
            ))

    infl_vecs = []

    if method == "linear":                                 # -----------------
        logger.info("Using privacy-preserving 'linear' method (reg=%s)", reg)
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
        logger.info("Using privacy-preserving 'public-fisher' method (reg=%s)", reg)
        logger.info("   Computing Fisher information from PUBLIC data only")
        
        # Use a smaller subset for Fisher computation to avoid memory issues
        max_fisher_samples = min(100, len(public_samples))  # Much smaller for memory efficiency
        fisher_samples = public_samples[:max_fisher_samples]
        
        logger.info("   Using %s public samples for Fisher matrix computation", max_fisher_samples)
        
        # Compute Fisher matrix from public data only (privacy-preserving)
        public_grads = []
        for x, y in tqdm(fisher_samples, desc="Computing public Fisher"):
            try:
                model.zero_grad()
                F.cross_entropy(model(x), y).backward()
                grad_vec = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                public_grads.append(grad_vec.detach().cpu())  # Move to CPU to save GPU memory
            except RuntimeError as e:
                logger.warn("Error computing gradient: %s", e)
                continue
        
        if len(public_grads) >= 10:  # Need minimum samples for stable Fisher
            try:
                G_public = torch.stack(public_grads)
                dim = G_public.shape[1]
                
                # Add strong regularization for stability
                fisher_public = (G_public.T @ G_public) / len(G_public) + reg * torch.eye(dim)
                
                logger.info("   Public Fisher shape: %s", tuple(fisher_public.shape))
                logger.success("Computed Fisher from %s public samples", len(public_grads))
                
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
                            logger.warn("Using pseudo-inverse for stability")
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
                        logger.warn("Error in influence computation: %s", e)
                        # Fallback: create zero influence vector
                        vec = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
                        infl_vecs.append(vec)
                        
            except Exception as e:
                logger.warn("Error computing public Fisher matrix: %s", e)
                logger.warn("Falling back to linear method")
                return compute_influence_vectors(model, public_loader, train_loader, device, method="linear", reg=reg, strict=strict)
        else:
            logger.warn("Insufficient public gradients (%s < 10), falling back to linear method", len(public_grads))
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

    logger.success("Computed %s privacy-preserving influence vectors using only public data", len(infl_vecs))
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
        logger.info("MEASURING ΔL_DP: Utility drop due to DP noise on critical slice")
        
        clean_model.eval()
        model.eval()
        
        with torch.no_grad():
            # Loss on clean model
            clean_loss = F.cross_entropy(clean_model(crit_x), crit_y, reduction='mean')
            # Loss on DP model  
            dp_loss = F.cross_entropy(model(crit_x), crit_y, reduction='mean')
            
            delta_L_DP = dp_loss.item() - clean_loss.item()
            
        logger.info("   • Clean model loss: %.4f", clean_loss.item())
        logger.info("   • DP model loss:    %.4f", dp_loss.item())
        logger.info("   • ΔL_DP = L_DP - L_clean = %+0.4f", delta_L_DP)
        
        if delta_L_DP > 0:
            logger.info("   📈 DP noise caused utility degradation (higher loss).")
            logger.info("   🎯 Goal: use calibration to recover utility loss.")
        else:
            logger.warn("   📉 DP model performs better than clean (unusual).")
    else:
        if clean_model is None:
            logger.warn("Skipping ΔL_DP measurement: no clean_model provided.")
        else:
            logger.warn("Skipping ΔL_DP measurement: no critical slice samples.")

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

    logger.info("Influence score statistics:")
    logger.info("   • Score range: [%.4f, %.4f]", scores.min(), scores.max())
    logger.info(
        "   • Selected %s samples with HIGHEST scores: [%.4f, %.4f]",
        eta,
        scores[idx].min(),
        scores[idx].max(),
    )
    logger.info("   • Mean selected score: %.4f", scores[idx].mean())
    logger.info("   • Score std: %.4f", scores.std())
    logger.info("   • Median score: %.4f", np.median(scores))
    logger.info("   🎯 Goal: Select samples that help reduce evaluation slice loss (positive scores)")
    
    # Additional diagnostic: check if we have any helpful samples
    positive_scores = scores[scores > 0]
    negative_scores = scores[scores < 0]
    logger.info("   • Helpful samples (score > 0): %s out of %s", len(positive_scores), len(scores))
    logger.info("   • Harmful samples (score < 0): %s out of %s", len(negative_scores), len(scores))
    
    if len(positive_scores) == 0:
        logger.warn("No helpful public samples found. All would increase evaluation loss.")
        logger.warn("This suggests domain mismatch between public and evaluation data.")
    elif len(positive_scores) < eta:
        logger.warn("Only %s helpful samples, but requesting %s", len(positive_scores), eta)
        logger.warn("Reducing eta to %s to avoid harmful samples", len(positive_scores))
        # Only use actually helpful samples
        helpful_idx = np.where(scores > 0)[0]
        w = np.zeros_like(scores)
        w[helpful_idx] = 1.0
        eta = len(helpful_idx)

    # 🔍 DIAGNOSTIC: Check influence vector statistics
    infl_norms = [torch.sqrt(sum(v[n].pow(2).sum() for n in v.keys())).item() for v in infl_vecs]
    logger.info("Influence vector diagnostics:")
    logger.info("   • Influence vector norms range: [%.4f, %.4f]", min(infl_norms), max(infl_norms))
    logger.info("   • Mean influence vector norm: %.4f", np.mean(infl_norms))
    logger.info("   • Influence vectors used (nonzero weights): %s", np.sum(w > 0))

    # ---- e) Bias computation
    # Paper mapping — Step d(ii): Δθ = - (1/n) H_{θ̂}^{-1} Σ_z w_z ∇ℓ(z, θ̂)
    # Here we average v(z) ≈ H^{-1}∇ℓ(z) over selected z, implementing the same update up to scaling.
    # Move in direction that helpful public samples suggest
    n_selected = np.sum(w > 0)  # Number of selected samples
    if n_selected == 0:
        logger.warn("No helpful samples found for calibration - skipping update")
        logger.warn("This suggests the public data can't help improve the evaluation slice")
        return model
        
    delta = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    for i, weight in enumerate(w):
        if weight > 0:  # Only sum over helpful samples
            for n in delta:
                delta[n] += infl_vecs[i][n].to(delta[n].device) * weight / n_selected  # Average of helpful influence vectors

    # 🔍 DIAGNOSTIC: Check delta before trust region
    delta_norm_before = _frob_norm(delta)
    logger.info("Parameter update diagnostics:")
    logger.info("   • Raw Δθ norm before trust region: %.4f", delta_norm_before)
    logger.info("   • Using CORRECTED formula: Δθ = (1/η) Σ_{helpful} H⁻¹ ∇ℓ(public_sample)")
    logger.info("   • Helpful samples used η: %s", n_selected)
    logger.info("   • Direction: Move model toward what helpful public samples suggest")
    logger.info("   • Privacy-preserving: Uses only public data + DP model output")

    # ---- f) trust-region clip (IMPROVED: more generous)
    # Paper mapping — Step d(iii): Back-tracking line search over α ∈ {1, 1/2, 1/4, ...}
    # Note: We enforce a conservative trust region here. The explicit back-tracking
    #       line search that picks α by minimizing L_crit(θ̂ + αΔθ) is executed in
    #       ablation_optimized.calibrate_with_line_search(), which wraps this update.
    ref_norm = torch.sqrt(sum(p.pow(2).sum()
                              for p in model.parameters())).item()
    logger.info("   • Model parameter norm ‖θ‖: %.4f", ref_norm)
    logger.info("   • Trust region limit (τ × ‖θ‖): %.4f", trust_tau * ref_norm)
    
    delta = apply_trust_region(delta, ref_norm, tau=trust_tau)
    delta_norm_after = _frob_norm(delta)
    logger.info("   • Final Δθ norm after trust region: %.4f", delta_norm_after)
    logger.info(
        "   • Trust region scaling factor: %.4f",
        delta_norm_after / (delta_norm_before + 1e-12),
    )
    
    # 🔍 DIAGNOSTIC: Check relative parameter change
    relative_change = delta_norm_after / ref_norm
    logger.info("   • Relative parameter change: %.4f (%.2f%%)", relative_change, relative_change * 100)
    
    if relative_change > 0.2:
        logger.warn("Large parameter change (>20%%) may cause instability")
    elif relative_change < 0.005:
        logger.warn("Very small parameter change (<0.5%%) may be ineffective")
        logger.warn("Suggestion: try decreasing reg parameter or increasing trust_tau")

    # ---- g) apply calibration: θ̂* = θ̂_DP + Δθ
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(delta[n])

    logger.success(
        "Privacy-preserving calibration applied (‖Δθ‖₂ ≈ %.4f, trust τ=%s)",
        _frob_norm(delta),
        trust_tau,
    )
    logger.success("Post-processing theorem: privacy guarantees preserved.")
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
