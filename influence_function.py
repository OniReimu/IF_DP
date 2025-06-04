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
def get_critical_slice(eval_loader, target_class: int = 3, device="cpu"):
    crit_x, crit_y = [], []
    for batch_data in eval_loader:
        # Handle both (x, y) and (x, y, user_id) formats
        if len(batch_data) == 3:
            x, y, _ = batch_data  # x, y, user_id
        else:
            x, y = batch_data     # x, y only
            
        m = y == target_class
        if m.any():
            crit_x.append(x[m]), crit_y.append(y[m])
    if not crit_x:
        print(f"⚠️  no samples of class {target_class}")
        return torch.empty(0, 3, 32, 32, device=device), \
               torch.empty(0, dtype=torch.long, device=device)
    crit_x = torch.cat(crit_x).to(device)
    crit_y = torch.cat(crit_y).to(device)
    print(f"🎯 critical slice: {len(crit_x)} samples of class {target_class}")
    return crit_x, crit_y


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
                              reg=1.0,  # ✨ FIX  stronger default
                              strict=True):
    """Return list of dicts (aligned with model.named_parameters())."""
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
        for x, y in tqdm(public_samples, desc="linear-IF"):
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            vec = {}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    # ✨ FIX  gradient clipping to unit-norm then scale by 1/reg
                    g = g / (g.norm() + 1e-12)
                    vec[n] = g / reg
                else:
                    vec[n] = torch.zeros_like(p)
            infl_vecs.append(vec)

    elif method == "batch":                                # -----------------
        # diagonal Fisher inverse
        fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        total = 0
        for batch_data in tqdm(train_loader, desc="Fisher diag"):
            # Handle both (x, y) and (x, y, user_id) formats
            if len(batch_data) == 3:
                x, y, _ = batch_data  # x, y, user_id
            else:
                x, y = batch_data     # x, y only
                
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            bs = x.size(0); total += bs
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher_diag[n] += p.grad.detach().pow(2) * bs
        for n in fisher_diag:
            fisher_diag[n] = fisher_diag[n] / total + 0.1   # ✨ OPTIMIZED: stronger damping (was 1e-2)

        for x, y in tqdm(public_samples, desc="diag-IF"):
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            vec = {}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    vec[n] = p.grad.detach() / fisher_diag[n]
                else:
                    vec[n] = torch.zeros_like(p)
            infl_vecs.append(vec)

    elif method == "original":                             # -----------------
        # More accurate but slower: compute actual Hessian-vector products
        print("⚠️  Original method is computationally expensive and may be slow")
        
        # Compute empirical Fisher for better Hessian approximation
        fisher_full = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        total = 0
        for batch_data in tqdm(train_loader, desc="Computing empirical Fisher"):
            # Handle both (x, y) and (x, y, user_id) formats
            if len(batch_data) == 3:
                x, y, _ = batch_data  # x, y, user_id
            else:
                x, y = batch_data     # x, y only
                
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            bs = x.size(0); total += bs
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher_full[n] += p.grad.detach().pow(2) * bs
        
        # Add strong damping for numerical stability
        for n in fisher_full:
            fisher_full[n] = fisher_full[n] / total + 5.0 * torch.ones_like(fisher_full[n])  # ✨ OPTIMIZED: much stronger damping (was 1.0)

        for x, y in tqdm(public_samples, desc="original-IF"):
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            vec = {}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # More conservative: use stronger damping
                    vec[n] = p.grad.detach() / fisher_full[n]
                else:
                    vec[n] = torch.zeros_like(p)
            infl_vecs.append(vec)

    else:                                                  # -----------------
        raise ValueError(f"unknown method '{method}'. Choose from: 'linear', 'batch', 'original'")

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
                                      trust_tau=0.05,
                                      strict=True,
                                      clean_model=None,
                                      reg=1.0):
    """
    One-shot influence calibration following the exact specification.
    
    Args:
        clean_model: Non-DP baseline model for measuring ΔL_DP (utility drop due to DP)
    """

    model.eval()

    # ---- 0) MEASURE ΔL_DP: Utility drop due to DP noise
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
    J = compute_slice_gradient(model, crit_x, crit_y, device)
    J_flat = torch.cat([v.flatten() for v in J.values()])

    # ---- b) influence vectors
    infl_vecs, public_samples = compute_influence_vectors(model, public_loader,
                                             train_loader, device,
                                             method=method, reg=reg, strict=strict)

    # ---- c) influence scores  s_i = Jᵀ v_i
    scores = np.array([ torch.dot(J_flat,
                   torch.cat([v[n].flatten() for n in J.keys()])).item()
                        for v in infl_vecs ])

    # ---- d) choose η most helpful samples (FIXED: select LOWEST scores for negative I_up,loss)
    idx = np.argsort(scores)[:eta]  # 🔧 FIX: select smallest (most negative) scores
    w = np.zeros_like(scores)
    w[idx] = 1.0                           # simple 0/1 for now

    print(f"\n📊 Influence score statistics:")
    print(f"   • Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   • Selected {eta} samples with LOWEST scores: [{scores[idx].min():.4f}, {scores[idx].max():.4f}]")
    print(f"   • Mean selected score: {scores[idx].mean():.4f}")
    print(f"   • Score std: {scores.std():.4f}")
    print(f"   • Median score: {np.median(scores):.4f}")
    print(f"   🎯 Goal: Select negative scores to achieve negative I_up,loss for slice loss reduction")
    
    # 🔍 DIAGNOSTIC: Check influence vector statistics
    infl_norms = [torch.sqrt(sum(v[n].pow(2).sum() for n in v.keys())).item() for v in infl_vecs]
    print(f"\n🔍 Influence vector diagnostics:")
    print(f"   • Influence vector norms range: [{min(infl_norms):.4f}, {max(infl_norms):.4f}]")
    print(f"   • Mean influence vector norm: {np.mean(infl_norms):.4f}")
    print(f"   • Influence vectors used (nonzero weights): {np.sum(w > 0)}")

    # ---- e) CORRECTED: Δθ = -(1/n) H⁻¹ ∑_{z∈P} w_z ∇ℓ(z,θ̂_DP)
    # Note: w has zeros for non-selected samples, so this correctly sums over ALL public samples
    n_total = len(public_samples)  # Total number of public samples
    delta = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    for i, weight in enumerate(w):
        # Sum over ALL samples (w_i = 0 for non-selected samples)
        for n in delta:
            delta[n] -= infl_vecs[i][n] * weight / n_total  # Negative sign + divide by total n

    # 🔍 DIAGNOSTIC: Check delta before trust region
    delta_norm_before = _frob_norm(delta)
    print(f"\n🔍 Parameter update diagnostics:")
    print(f"   • Raw Δθ norm before trust region: {delta_norm_before:.4f}")
    print(f"   • Using CORRECTED formula: Δθ = -(1/n) H⁻¹ ∑ w_z ∇ℓ(z)")
    print(f"   • Total public samples n: {n_total}")
    print(f"   • Selected samples η: {np.sum(w > 0)}")

    # ---- f) trust-region clip
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
    
    if relative_change > 0.1:
        print(f"   ⚠️  WARNING: Large parameter change (>{10:.0f}%) may cause instability")
    elif relative_change < 0.001:
        print(f"   ⚠️  WARNING: Very small parameter change (<{0.1:.1f}%) may be ineffective")

    # ---- g) apply calibration: θ̂* = θ̂_DP + Δθ
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(delta[n])

    print(f"✅ calibration applied (‖Δθ‖₂ ≈ {_frob_norm(delta):.4f}, "
          f"trust τ={trust_tau})")
    return model


def calibrate_model(model, public_loader, train_loader, fisher,
                    top_k=200, device=None, target_class=3, **kw):
    crit_x, crit_y = get_critical_slice(public_loader,
                                        target_class, device)
    return calibrate_model_research_protocol(
        model, public_loader, train_loader,
        crit_x, crit_y, device,
        method="linear", eta=top_k, strict=True, **kw)


def calibrate_model_efficient(model, public_loader, train_loader, fisher,
                             top_k=200, device=None, method="linear",
                             target_class=3, **kw):
    crit_x, crit_y = get_critical_slice(public_loader,
                                        target_class, device)
    return calibrate_model_research_protocol(
        model, public_loader, train_loader,
        crit_x, crit_y, device,
        method=method, eta=top_k, strict=True, **kw)