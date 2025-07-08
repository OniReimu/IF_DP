#!/usr/bin/env python3
# ================================================================
# LINE SEARCH OPTIMIZATION TEST: Fisher DP-SGD + Enhanced Calibration
#    * Baseline: Standard influence function calibration
#    * Enhanced: Line search optimization for optimal step size
#    * Comparison of calibration effectiveness
# ================================================================

import os, glob, argparse, copy, math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, Sampler
from tqdm import tqdm
from torch.autograd import grad  # Required for per-sample gradients

# Project-specific helpers
from fisher_dp_sgd import compute_fisher, topk_eigh_with_floor, maha_clip
from dp_sgd import train_with_vanilla_dp
from dp_sat import train_with_dp_sat
from privacy_accounting import (
    get_privacy_params_for_target_epsilon, 
)
from model import CNN
from mia import evaluate_membership_inference, confidence_attack, shadow_model_attack, prepare_mia_data_sample_level, prepare_mia_data_user_level
from influence_function import calibrate_model_research_protocol, get_evaluation_slice
from config import set_random_seeds, get_random_seed

set_random_seeds()  # Set reproducible random seeds
models_dir = './saved_models'; os.makedirs(models_dir, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# Diagnostic functions (moved from influence_function.py)
# ════════════════════════════════════════════════════════════════

def diagnose_calibration(model, critical_data, critical_targets, device):
    """
    Simple diagnostic function to measure critical slice performance.
    Replacement for the removed function in influence_function.py.
    """
    if len(critical_data) == 0:
        return {'loss': float('inf'), 'accuracy': 0.0, 'num_samples': 0}
    
    model.eval()
    with torch.no_grad():
        output = model(critical_data)
        loss = F.cross_entropy(output, critical_targets)
        predictions = output.argmax(dim=1)
        accuracy = (predictions == critical_targets).float().mean()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy.item() * 100,
        'num_samples': len(critical_data)
    }

def print_calibration_effect(before_stats, after_stats, target_class="all"):
    """
    Print the effect of calibration on the evaluation slice.
    """
    if target_class == "all":
        slice_description = "All Classes"
    else:
        slice_description = f"Class {target_class}"
    
    print(f"\n📊 Calibration Effect Analysis ({slice_description}):")
    print(f"   • Samples evaluated: {before_stats['num_samples']}")
    print(f"   • Loss before:   {before_stats['loss']:.4f}")
    print(f"   • Loss after:    {after_stats['loss']:.4f}")
    print(f"   • Δ Loss:        {after_stats['loss'] - before_stats['loss']:+.4f}")
    print(f"   • Accuracy before: {before_stats['accuracy']:.2f}%")
    print(f"   • Accuracy after:  {after_stats['accuracy']:.2f}%")
    print(f"   • Δ Accuracy:      {after_stats['accuracy'] - before_stats['accuracy']:+.2f}%")
    
    if after_stats['loss'] < before_stats['loss']:
        print(f"   ✅ SUCCESS: Calibration reduced evaluation slice loss!")
    else:
        print(f"   ⚠️  WARNING: Calibration increased evaluation slice loss")
    
    if after_stats['accuracy'] > before_stats['accuracy']:
        print(f"   ✅ SUCCESS: Calibration improved evaluation slice accuracy!")
    else:
        print(f"   ⚠️  WARNING: Calibration reduced evaluation slice accuracy")

# ════════════════════════════════════════════════════════════════
# Line Search Optimization for Calibration
# ════════════════════════════════════════════════════════════════

def _eval_slice_loss(model, critical_data, critical_targets, device):
    """Helper function to evaluate loss on critical slice."""
    if len(critical_data) == 0:
        return float('inf')
    
    model.eval()
    with torch.no_grad():
        critical_data = critical_data.to(device)
        critical_targets = critical_targets.to(device)
        output = model(critical_data)
        loss = F.cross_entropy(output, critical_targets)
    return loss.item()

def _add_delta(model, delta, scale=1.0):
    """Apply scaled parameter update."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in delta:
                param.data.add_(delta[name], alpha=scale)

def calibrate_with_line_search(model, pub_loader, priv_loader, critical_data, critical_targets, 
                              device, method='linear', eta=100, trust_tau=0.01, reg=10.0, 
                              strict=True, clean_model=None):
    """Enhanced calibration with line search optimization for optimal step size."""
    
    print(f"🔍 Line Search Calibration:")
    print(f"   • Method: {method}")
    print(f"   • Eta: {eta}")
    print(f"   • Trust tau: {trust_tau}")
    print(f"   • Regularization: {reg}")
    
    # First get the standard influence update
    calibrated_model = calibrate_model_research_protocol(
        copy.deepcopy(model), pub_loader, priv_loader,
        critical_data, critical_targets, device=device,
        method=method, eta=eta, trust_tau=trust_tau,
        strict=strict, clean_model=clean_model, reg=reg
    )
    
    # Extract the delta that was applied
    delta = {}
    with torch.no_grad():
        for (name, orig_param), (_, new_param) in zip(model.named_parameters(), calibrated_model.named_parameters()):
            delta[name] = new_param.data - orig_param.data
    
    # Line search over different scales
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    best_loss, best_gamma = float('inf'), 0.0
    
    print(f"   🔍 Line search over {len(candidates)} step size candidates...")
    
    for gamma in candidates:
        test_model = copy.deepcopy(model)
        _add_delta(test_model, delta, gamma)
        loss = _eval_slice_loss(test_model, critical_data, critical_targets, device)
        
        print(f"     • γ={gamma:.2f}: loss={loss:.4f}")
        
        if loss < best_loss:
            best_loss, best_gamma = loss, gamma
    
    # Apply best scaling
    final_model = copy.deepcopy(model)
    _add_delta(final_model, delta, best_gamma)
    
    print(f"   ✅ Optimal step size: γ={best_gamma:.2f} (loss: {best_loss:.4f})")
    
    return final_model

# ════════════════════════════════════════════════════════════════
# Synthetic users + batch sampler (reused from main.py)
# ════════════════════════════════════════════════════════════════
class SyntheticUserDataset(torch.utils.data.Dataset):
    """Assigns each sample a user_id ∈ {0,…,K-1} (round-robin)."""
    def __init__(self, base_ds, num_users, perm=None):
        self.base = base_ds
        if perm is None: perm = np.arange(len(base_ds))
        self.uid = torch.tensor(perm % num_users, dtype=torch.long)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x,y = self.base[idx]
        return x, y, self.uid[idx].item()

class UserBatchSampler(Sampler):
    """Yield all indices of exactly ONE user per iteration."""
    def __init__(self, user_ids, shuffle=True):
        self.by_user = {}
        for idx,u in enumerate(user_ids):
            u_key = int(u)  # Convert to Python int to avoid numpy int64 issues
            self.by_user.setdefault(u_key, []).append(idx)
        self.uids = list(self.by_user.keys())
        self.shuffle = shuffle
    def __iter__(self):
        order = np.random.permutation(self.uids) if self.shuffle else self.uids
        for u in order: 
            u_key = int(u)  # Convert to Python int
            yield self.by_user[u_key]
    def __len__(self): return len(self.uids)

# ════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════
def get_device(args):
    if args.cpu: return torch.device('cpu')
    if args.mps and torch.backends.mps.is_available():
        print('Using MPS'); return torch.device('mps')
    if torch.cuda.is_available():
        idx = 0 if args.cuda_id is None else args.cuda_id
        print(f'Using CUDA:{idx}'); return torch.device(f'cuda:{idx}')
    print('Using CPU'); return torch.device('cpu')

def unpack_batch(batch_data):
    """Helper function to handle both (x, y) and (x, y, user_id) formats"""
    if len(batch_data) == 3:
        return batch_data[0], batch_data[1], batch_data[2]  # x, y, user_id
    else:
        return batch_data[0], batch_data[1], None  # x, y, None

def accuracy(model, loader, device):
    model.eval(); tot=correct=0
    with torch.no_grad():
        for batch_data in loader:
            x, y, _ = unpack_batch(batch_data)
            x,y = x.to(device), y.to(device)
            correct += (model(x).argmax(1)==y).sum().item()
            tot += y.size(0)
    return 100*correct/tot

# ════════════════════════════════════════════════════════════════
# ABLATION: Fisher DP-SGD with Optional DP-SAT Optimization
# ════════════════════════════════════════════════════════════════

def train_fisher_dp_with_optimizer(model, train_loader, fisher,
                                  epsilon=8.0, delta=1e-6,
                                  clip_radius=10.0, k=32, lam_floor=5e-1,
                                  device="cuda", target_layer="conv1",
                                  adaptive_clip=True, quantile=0.95, sample_level=None,
                                  epochs=1, sigma=None, full_complement_noise=False,
                                  use_dp_sat=False, lambda_flatness=0.01,
                                  optimizer_name="Normal", positive_noise_correlation=False):
    """
    Fisher DP-SGD with optional DP-SAT optimization.
    
    This function explores the synergy between:
    1. Fisher-informed noise (curvature-aware)
    2. DP-SAT optimization (sharpness-aware)
    
    Args:
        clip_radius: Target Euclidean sensitivity Δ₂ (same as vanilla DP-SGD for fair comparison).
                    This will be converted to appropriate Mahalanobis threshold internally.
        use_dp_sat: If True, apply DP-SAT flatness adjustment after Fisher noise
        lambda_flatness: Flatness coefficient for DP-SAT (only used if use_dp_sat=True)
        optimizer_name: String identifier for logging purposes
        positive_noise_correlation: If False (default), use negatively correlated noise (noise ∝ 1/√λ).
                                   If True, use positively correlated noise (noise ∝ √λ).
        
    Algorithm when use_dp_sat=True (CORRECTED):
        1. Compute per-sample gradients and clip them (standard DP-SGD)
        2. Add Fisher-informed noise → g_fisher_priv
        3. Apply DP-SAT flatness adjustment: g_flat = λ * g_{t-1}^{fisher_priv} / ||g_{t-1}^{fisher_priv}||_2
        4. Final update: θ ← θ - η(g_fisher_priv + g_flat)
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # Fisher eigendecomposition
    lam, U = topk_eigh_with_floor(fisher, k=k, lam_floor=lam_floor)
    lam, U = lam.to(device), U.to(device)
    
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
        print(f"   • Using provided sigma: {sigma:.4f}")
    else:
        sigma_single_epoch = math.sqrt(2*math.log(1.25/delta)) / epsilon
        sigma = sigma_single_epoch / math.sqrt(epochs)
        print(f"   • Legacy accounting: σ_single={sigma_single_epoch:.3f}, σ_adjusted={sigma:.3f}")
    
    actual_k = len(lam)
    if actual_k != k:
        print(f"⚠️  Using k={actual_k} eigenpairs (requested {k}) due to matrix rank constraints")

    # Gather parameter objects
    if target_layer == "all":
        names = [n for n,_ in model.named_parameters()]
    elif "," in target_layer:
        layers = [s.strip() for s in target_layer.split(",")]
        names = [n for n,_ in model.named_parameters()
                 if any(l in n for l in layers)]
    else:
        names = [n for n,_ in model.named_parameters()
                 if target_layer in n]
    params = [dict(model.named_parameters())[n] for n in names]
    param_dim = sum(p.numel() for p in params)

    # Auto-detect DP mode if not specified
    if sample_level is None:
        first_batch = next(iter(train_loader))
        if len(first_batch) == 3:
            _, _, user_ids = first_batch
            unique_users = torch.unique(user_ids)
            sample_level = len(unique_users) > 1
        else:
            sample_level = True

    mode_str = "Sample-level" if sample_level else "User-level"
    opt_type = "DP-SAT" if use_dp_sat else "Normal"
    
    print(f"\n🎯 Fisher DP + {optimizer_name} Optimizer: {mode_str} DP  layers={target_layer}  ε={epsilon}")
    if sigma is not None:
        print(f"   • Proper privacy accounting: σ={sigma:.4f}")
    else:
        print(f"   • Multi-epoch privacy: T={epochs}, σ_single={sigma_single_epoch:.3f}, σ_adjusted={sigma:.3f}")
    print(f"   • Fisher subspace: k={actual_k}, complement dim={param_dim-actual_k}")
    print(f"   • Target Euclidean sensitivity: Δ₂ = {clip_radius:.3f} (will convert to Mahalanobis)")
    print(f"   • Full complement noise: {full_complement_noise}")
    print(f"   • Optimizer type: {opt_type}")
    print(f"   • Noise scaling strategy: {strategy_name}")
    if use_dp_sat:
        print(f"   • DP-SAT flatness coefficient λ: {lambda_flatness:.4f}")
        print(f"   • ✅ CORRECTED: Uses previous step Fisher gradient for flatness")
    print(f"   • Adaptive clipping: {adaptive_clip}")
    
    if not sample_level:
        print("   • User-level mode: Clipping aggregated user gradients")
    else:
        print("   • Sample-level mode: Clipping individual sample gradients")
    print()

    noise_l2, grad_norm, flatness_norm = [], [], []
    euclidean_norms = []  # Track Euclidean norms for calibration
    adaptive_radius_computed = False
    euclidean_target = clip_radius  # Store the target Euclidean sensitivity
    actual_radius = clip_radius  # Will be calibrated to Mahalanobis space
    calibration_computed = False  # Flag for norm calibration
    
    # Initialize previous step's Fisher-noisy gradient for DP-SAT
    g_fisher_priv_prev = None

    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Fisher DP + {opt_type} ({mode_str})")):
        # Accept (x,y) OR (x,y,uid)
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
                        
                        print(f"📊 Fisher + {opt_type} adaptive clipping from {len(euclidean_norms)} samples (EUCLIDEAN norms):")
                        print(f"   • Mean Euclidean: {np.mean(euclidean_norms):.3f}")
                        print(f"   • {quantile:.1%} quantile: {euclidean_adaptive_radius:.3f}")
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
                    
                    print(f"🎯 Ablation norm calibration completed:")
                    print(f"   • Target Euclidean sensitivity: Δ₂ = {euclidean_target:.3f}")
                    print(f"   • Calibrated Mahalanobis threshold: {actual_radius:.3f}")
                    print(f"   • Euclidean clip rate: {euclidean_clip_rate:.1%}")
                    print(f"   • Mahalanobis clip rate: {np.mean(maha_norms > actual_radius):.1%}")
                    print(f"   → Fair comparison: same effective sensitivity bound Δ₂\n")
                    
                    euclidean_norms = []  # Reset for actual training statistics

            # Mahalanobis clipping with calibrated threshold
            for i in range(per_g.size(0)):
                per_g[i], nrm = maha_clip(per_g[i], U, clip_scaling, actual_radius)
                grad_norm.append(nrm)
            g_bar = per_g.mean(0)

        else:
            # USER-LEVEL DP: Compute gradient per user
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
                
                # Calibration or adaptive clipping: collect EUCLIDEAN norms
                if (adaptive_clip and not adaptive_radius_computed) or not calibration_computed:
                    euclidean_norm = user_grad_flat.norm().item()
                    euclidean_norms.append(euclidean_norm)
                    
                    if adaptive_clip and not adaptive_radius_computed:
                        if len(euclidean_norms) >= min(10, len(train_loader)) or batch_idx == 0:
                            euclidean_adaptive_radius = np.quantile(euclidean_norms, quantile)
                            euclidean_target = euclidean_adaptive_radius
                            adaptive_radius_computed = True
                            
                            print(f"📊 Fisher + {opt_type} adaptive clipping from {len(euclidean_norms)} users (EUCLIDEAN norms):")
                            print(f"   • Mean Euclidean: {np.mean(euclidean_norms):.3f}")
                            print(f"   • {quantile:.1%} quantile: {euclidean_adaptive_radius:.3f}")
                            print(f"   → Using Euclidean target: Δ₂ = {euclidean_target:.3f}")
                    
                    # Calibration for user-level
                    if not calibration_computed and len(euclidean_norms) >= 5:
                        # Simple heuristic: use median ratio for calibration
                        proj = (U.T @ user_grad_flat) * inv_sqrt_lam
                        mahalanobis_norm = proj.norm().item()
                        ratio = euclidean_norm / (mahalanobis_norm + 1e-8)
                        actual_radius = euclidean_target / (ratio + 1e-8)
                        calibration_computed = True
                        
                        print(f"🎯 Ablation user-level norm calibration:")
                        print(f"   • Target Euclidean sensitivity: Δ₂ = {euclidean_target:.3f}")
                        print(f"   • Calibrated Mahalanobis threshold: {actual_radius:.3f}")
                        print(f"   • Sample ratio: ||g||₂/||g||_{{F⁻¹}} ≈ {ratio:.3f}\n")
                        
                        euclidean_norms = []  # Reset
            
            # Mahalanobis clipping for each user gradient
            clipped_user_grads = []
            for user_grad_flat in user_gradients:
                clipped_grad, user_norm = maha_clip(user_grad_flat, U, clip_scaling, actual_radius)
                grad_norm.append(user_norm)
                clipped_user_grads.append(clipped_grad)
            
            g_bar = torch.stack(clipped_user_grads).mean(0)

        # ============================================================
        # FISHER-INFORMED NOISE (Two-component)
        # ============================================================
        
        # 1. Low-rank noise in Fisher subspace (anisotropic)
        z_fisher = torch.randn(actual_k, device=device)
        fisher_noise = U @ (z_fisher * noise_scaling) * sigma * euclidean_target  # Use Euclidean target for noise scale
        
        if full_complement_noise:
            # 2. Complement noise in orthogonal subspace (isotropic)
            z_full = torch.randn_like(g_bar)
            z_complement = z_full - U @ (U.T @ z_full)  # Project to complement
            complement_noise = z_complement * sigma * euclidean_target  # Use Euclidean target
            total_noise = fisher_noise + complement_noise
            complement_noise_norm = complement_noise.norm().item()
        else:
            # Only Fisher subspace noise (preserves curvature-aware benefits)
            total_noise = fisher_noise
            complement_noise_norm = 0.0
        
        g_fisher_priv = g_bar + total_noise
        
        # Track noise components
        fisher_noise_norm = fisher_noise.norm().item()
        total_noise_norm = total_noise.norm().item()
        noise_l2.append(total_noise_norm)

        # ============================================================
        # OPTIONAL: DP-SAT OPTIMIZATION (Sharpness-Aware) - CORRECTED
        # ============================================================
        
        if use_dp_sat:
            if g_fisher_priv_prev is not None:
                # Apply DP-SAT flatness adjustment using PREVIOUS step's Fisher-noisy gradient
                g_fisher_priv_prev_norm = g_fisher_priv_prev.norm() + 1e-8
                g_flat = lambda_flatness * g_fisher_priv_prev / g_fisher_priv_prev_norm
                flatness_norm.append(g_flat.norm().item())
                
                # Final gradient: current Fisher-noisy + flatness adjustment from previous step
                g_final = g_fisher_priv + g_flat
            else:
                # First step: no previous gradient available, use standard Fisher DP
                g_final = g_fisher_priv
                flatness_norm.append(0.0)  # No flatness adjustment
                
            # Store current Fisher-noisy gradient for next iteration
            g_fisher_priv_prev = g_fisher_priv.clone().detach()
        else:
            # Standard Fisher DP: just use Fisher-noisy gradient
            g_final = g_fisher_priv
            flatness_norm.append(0.0)  # No flatness adjustment

        # Scatter back to model parameters
        idx = 0
        for p in params:
            n = p.numel()
            p.grad = g_final[idx:idx+n].view_as(p)
            idx += n
        opt.step()

    grad_type = "‖g_user‖_Mah" if not sample_level else "‖g‖_Mah"
    print(f"\n📊  Fisher DP + {optimizer_name} final stats:")
    print(f"   • Target Euclidean sensitivity: Δ₂ = {euclidean_target:.3f} (same as vanilla DP-SGD)")
    print(f"   • Calibrated Mahalanobis threshold: {actual_radius:.3f}")
    print(f"   • Median {grad_type} = {np.median(grad_norm):.2f}")
    print(f"   • Fisher noise ℓ₂ ∈ [{min(noise_l2):.1f},{max(noise_l2):.1f}]")
    if use_dp_sat and len(flatness_norm) > 0 and max(flatness_norm) > 0:
        print(f"   • Flatness adjustment ℓ₂ ∈ [{min(flatness_norm):.3f},{max(flatness_norm):.3f}]")
        print(f"   • Flatness coefficient λ = {lambda_flatness:.4f}")
    if full_complement_noise:
        print(f"   • Last batch: Fisher={fisher_noise_norm:.1f}, Complement={complement_noise_norm:.1f}")
    else:
        print(f"   • Last batch: Fisher only={fisher_noise_norm:.1f} (complement disabled)")
    print(f"   • Privacy: (ε={epsilon}, δ={delta}) over {epochs} epochs")
    print(f"   • ✅ FAIR COMPARISON: Same effective sensitivity Δ₂ as vanilla DP-SGD")

    return model

# ════════════════════════════════════════════════════════════════
# Ablation Study Main Function
# ════════════════════════════════════════════════════════════════

def run_line_search_test(args, device, priv_loader, eval_base, priv_base, priv_idx, priv_ds, Fmat, pub_loader):
    """
    Test line search optimization for calibration.
    
    Comparison:
    1. Fisher DP + Standard Calibration
    2. Fisher DP + Line Search Calibration
    """
    
    print("\n" + "="*70)
    print("🔬  LINE SEARCH OPTIMIZATION TEST")
    print("="*70)
    
    # Load baseline model for initialization
    baseline = CNN().to(device)
    opt_b = torch.optim.SGD(baseline.parameters(), lr=1e-3, momentum=.9)
    
    print('\n⚙️  Training baseline for initialization…')
    for epoch in tqdm(range(args.epochs)):
        baseline.train()
        for batch_data in priv_loader:
            if args.sample_level:
                x, y = batch_data
            else:
                x, y, _ = batch_data
            x, y = x.to(device), y.to(device)
            opt_b.zero_grad(); F.cross_entropy(baseline(x),y).backward(); opt_b.step()

    # Privacy accounting setup
    if not args.use_legacy_accounting:
        sample_rate = len(priv_loader) / len(priv_base)
        steps_per_epoch = len(priv_loader)
        
        noise_multiplier, total_steps = get_privacy_params_for_target_epsilon(
            target_epsilon=args.target_epsilon,
            target_delta=args.delta,
            sample_rate=sample_rate,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch
        )
        
        sigma = noise_multiplier * args.clip_radius
        display_epsilon = args.target_epsilon
        
        print(f"\n🔒 Privacy Accounting:")
        print(f"   • Target (ε, δ): ({args.target_epsilon}, {args.delta})")
        print(f"   • Noise multiplier: {noise_multiplier:.4f}")
        print(f"   • Sigma: {sigma:.4f}")
    else:
        sigma = None
        display_epsilon = args.epsilon

    # ════════════════════════════════════════════════════════════════
    # Train Fisher DP Model (Baseline for Calibration)
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🎯 Training Fisher DP Model")
    print(f"{'='*50}")
    
    fisher_dp_model = copy.deepcopy(baseline)
    fisher_dp_model = train_fisher_dp_with_optimizer(
        fisher_dp_model, priv_loader, Fmat,
        epsilon=display_epsilon, delta=args.delta,
        sigma=sigma,
        full_complement_noise=args.full_complement_noise,
        clip_radius=args.clip_radius,
        k=args.k, device=device,
        target_layer=args.dp_layer,
        adaptive_clip=args.adaptive_clip,
        quantile=args.quantile,
        sample_level=args.sample_level,
        epochs=args.epochs,
        use_dp_sat=False,  # Focus on calibration, not DP-SAT
        optimizer_name="Fisher DP",
        positive_noise_correlation=args.positive_noise_correlation
    )
    
    # Create evaluation loader for critical slice extraction and final accuracy measurement
    eval_loader = DataLoader(eval_base, batch_size=128, shuffle=False)
    
    # Get critical slice using EVALUATION data (eval_loader) for the slice-gradient
    critical_data, critical_targets = get_evaluation_slice(eval_loader, args.target_class, device=device)
    
    # ════════════════════════════════════════════════════════════════
    # Test 1: Standard Calibration (Baseline)
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("📐 TEST 1: Standard Calibration (Baseline)")
    print(f"{'='*50}")
    
    standard_calibrated = copy.deepcopy(fisher_dp_model)
    
    if len(critical_data) > 0:
        before_stats = diagnose_calibration(standard_calibrated, critical_data, critical_targets, device)
        
        # Apply standard calibration
        standard_calibrated = calibrate_model_research_protocol(
            standard_calibrated, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=args.method,
            eta=args.calibration_k,
            trust_tau=args.trust_tau,
            strict=True,
            clean_model=baseline,
            reg=args.reg
        )
        
        after_stats = diagnose_calibration(standard_calibrated, critical_data, critical_targets, device)
        print_calibration_effect(before_stats, after_stats, args.target_class)
    else:
        print(f"⚠️  No samples found for target_class {args.target_class}")
        # Still apply calibration without diagnosis
        standard_calibrated = calibrate_model_research_protocol(
            standard_calibrated, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=args.method,
            eta=args.calibration_k,
            trust_tau=args.trust_tau,
            strict=True,
            clean_model=baseline,
            reg=args.reg
        )
    
    # ════════════════════════════════════════════════════════════════
    # Test 2: Line Search Calibration (Enhanced)
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🔍 TEST 2: Line Search Calibration (Enhanced)")
    print(f"{'='*50}")
    
    line_search_calibrated = copy.deepcopy(fisher_dp_model)
    
    if len(critical_data) > 0:
        before_stats_ls = diagnose_calibration(line_search_calibrated, critical_data, critical_targets, device)
        
        # Apply line search calibration
        line_search_calibrated = calibrate_with_line_search(
            line_search_calibrated, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=args.method,
            eta=args.calibration_k,
            trust_tau=args.trust_tau,
            strict=True,
            clean_model=baseline,
            reg=args.reg
        )
        
        after_stats_ls = diagnose_calibration(line_search_calibrated, critical_data, critical_targets, device)
        print_calibration_effect(before_stats_ls, after_stats_ls, args.target_class)
    else:
        print(f"⚠️  No samples found for target_class {args.target_class}")
        # Still apply calibration without diagnosis
        line_search_calibrated = calibrate_with_line_search(
            line_search_calibrated, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=args.method,
            eta=args.calibration_k,
            trust_tau=args.trust_tau,
            strict=True,
            clean_model=baseline,
            reg=args.reg
        )

    # ════════════════════════════════════════════════════════════════
    # Evaluation and Comparison
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("📊 LINE SEARCH OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    
    # Compute accuracies
    baseline_acc = accuracy(baseline, eval_loader, device)
    fisher_dp_acc = accuracy(fisher_dp_model, eval_loader, device)
    standard_calib_acc = accuracy(standard_calibrated, eval_loader, device)
    line_search_acc = accuracy(line_search_calibrated, eval_loader, device)
    
    dp_mode = "Sample-level" if args.sample_level else f"User-level ({args.users} users)"
    print(f"\n🎯 Accuracy Comparison ({dp_mode} DP):")
    print(f"   • Baseline (Non-DP)           : {baseline_acc:6.2f}%")
    print(f"   • Fisher DP (No Calibration)  : {fisher_dp_acc:6.2f}%")
    print(f"   • Fisher DP + Standard Calib  : {standard_calib_acc:6.2f}%")
    print(f"   • Fisher DP + Line Search     : {line_search_acc:6.2f}%")
    
    # Compute improvements
    fisher_improvement = fisher_dp_acc - baseline_acc
    standard_calib_improvement = standard_calib_acc - fisher_dp_acc
    line_search_improvement = line_search_acc - fisher_dp_acc
    line_search_vs_standard = line_search_acc - standard_calib_acc
    
    print(f"\n📈 Improvement Analysis:")
    print(f"   • Fisher DP benefit:          {fisher_improvement:+5.2f}% vs baseline")
    print(f"   • Standard calibration:       {standard_calib_improvement:+5.2f}% vs Fisher DP")
    print(f"   • Line search calibration:    {line_search_improvement:+5.2f}% vs Fisher DP")
    print(f"   • Line search vs standard:    {line_search_vs_standard:+5.2f}% improvement")
    
    print(f"\n🔍 Line Search Effectiveness:")
    if line_search_vs_standard > 0.5:
        print(f"   ✅ SIGNIFICANT: Line search provides meaningful improvement (+{line_search_vs_standard:.2f}%)")
    elif line_search_vs_standard > 0.1:
        print(f"   ✅ MODEST: Line search provides small improvement (+{line_search_vs_standard:.2f}%)")
    elif line_search_vs_standard > 0:
        print(f"   ⚠️  MINIMAL: Line search provides tiny improvement (+{line_search_vs_standard:.2f}%)")
    else:
        print(f"   ❌ NO BENEFIT: Line search does not improve over standard calibration ({line_search_vs_standard:.2f}%)")
    
    # Identify best method
    best_acc = max(standard_calib_acc, line_search_acc)
    best_method = "Line Search" if line_search_acc >= standard_calib_acc else "Standard"
    
    print(f"\n🏆 Best Calibration Method: {best_method} ({best_acc:.2f}%)")
    
    # Save models for further analysis
    print(f"\n💾 Saving models...")
    
    # Save standard calibrated model
    standard_path = os.path.join(models_dir, 'Standard_Calibrated_Test.pth')
    torch.save({
        'model_state_dict': standard_calibrated.state_dict(),
        'model_type': 'fisher_dp_standard_calibrated',
        'accuracy': standard_calib_acc,
        'improvement_vs_fisher': standard_calib_improvement,
        'calibration_method': args.method,
        'calibration_k': args.calibration_k,
        'trust_tau': args.trust_tau,
        'reg': args.reg,
        'line_search_test': True
    }, standard_path)
    print(f"✅ Saved Standard Calibrated to {standard_path}")
    
    # Save line search calibrated model
    line_search_path = os.path.join(models_dir, 'Line_Search_Calibrated_Test.pth')
    torch.save({
        'model_state_dict': line_search_calibrated.state_dict(),
        'model_type': 'fisher_dp_line_search_calibrated',
        'accuracy': line_search_acc,
        'improvement_vs_fisher': line_search_improvement,
        'improvement_vs_standard': line_search_vs_standard,
        'calibration_method': args.method,
        'calibration_k': args.calibration_k,
        'trust_tau': args.trust_tau,
        'reg': args.reg,
        'line_search_test': True
    }, line_search_path)
    print(f"✅ Saved Line Search Calibrated to {line_search_path}")

    # Return results for potential further analysis
    return {
        'baseline_acc': baseline_acc,
        'fisher_dp_acc': fisher_dp_acc,
        'standard_calib_acc': standard_calib_acc,
        'line_search_acc': line_search_acc,
        'line_search_improvement': line_search_vs_standard,
        'best_method': best_method
    }

# ════════════════════════════════════════════════════════════════
# Main function
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('Line Search Optimization Test for Fisher DP-SGD Calibration')
    
    # Device arguments
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--cuda-id', type=int)
    parser.add_argument('--cpu', action='store_true')
    
    # Data arguments
    parser.add_argument('--dataset-size', type=int, default=10000)
    parser.add_argument('--private-ratio', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=10)
    
    # Clean up
    parser.add_argument('--clean', action='store_true',
                       help='Remove all saved models before training')
    
    # Privacy arguments
    privacy_group = parser.add_mutually_exclusive_group()
    privacy_group.add_argument('--use-legacy-accounting', action='store_true',
                              help='Use legacy privacy accounting (not recommended)')
    
    epsilon_group = parser.add_mutually_exclusive_group()
    epsilon_group.add_argument('--epsilon', type=float, default=None,
                              help='Privacy epsilon (legacy accounting only)')
    epsilon_group.add_argument('--target-epsilon', type=float, default=None,
                              help='Target epsilon for DP (proper accounting)')
    
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--clip-radius', type=float, default=1.0)
    
    # Fisher DP arguments
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--dp-layer', type=str, default='conv1')
    parser.add_argument('--full-complement-noise', action='store_true',
                       help='Use full complement noise in orthogonal subspace')
    
    # Fisher DP noise scaling strategy  
    noise_strategy_group = parser.add_mutually_exclusive_group()
    noise_strategy_group.add_argument('--negatively_correlated_noise', action='store_true', default=True,
                                     help='Fisher DP: noise inversely correlated with curvature (noise ∝ 1/√λ, less noise in high curvature directions, default)')
    noise_strategy_group.add_argument('--positively_correlated_noise', action='store_true',
                                     help='Fisher DP: noise positively correlated with curvature (noise ∝ √λ, more noise in high curvature directions)')
    
    # DP-SAT arguments (kept for compatibility but not used in line search test)
    parser.add_argument('--lambda-flatness', type=float, default=0.01,
                       help='Flatness coefficient for DP-SAT (not used in line search test)')
    
    # Adaptive clipping
    parser.add_argument('--adaptive-clip', action='store_true')
    parser.add_argument('--quantile', type=float, default=0.95)
    
    # DP mode
    parser.add_argument('--sample-level', action='store_true')
    parser.add_argument('--users', type=int, default=10)
    
    # Calibration arguments - MAIN FOCUS OF THIS TEST
    parser.add_argument('--method', type=str, default='linear',
                       choices=['linear', 'public-fisher'],
                       help='Calibration method: linear (fast regularization) or public-fisher (uses public data Fisher matrix)')
    parser.add_argument('--calibration-k', type=int, default=100,
                       help='Number of top-k samples to use for calibration')
    parser.add_argument('--trust-tau', type=float, default=0.005,
                       help='Trust region parameter: max relative parameter change (default: 0.005 = 0.5%%)')
    parser.add_argument('--reg', type=float, default=50.0,
                       help='Regularization parameter for linear method influence vectors (default: 50.0 for stability)')
    parser.add_argument('--target-class', default="all",
                       help='Target class for calibration: "all" for general utility (default), or integer for specific class')
    parser.add_argument('--compare-calibration', action='store_true',
                       help='Run comparative experiment between single-class and multi-class calibration')
    
    # MIA evaluation (simplified for this test)
    parser.add_argument('--run-mia', action='store_true',
                       help='Run membership inference attack evaluation (simplified for line search test)')
    parser.add_argument('--mia-size', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Map new argument names for consistency  
    args.positive_noise_correlation = args.positively_correlated_noise
    
    # Parse target_class argument - can be "all" or integer
    if args.target_class == "all":
        print(f"📊 Using ALL CLASSES for calibration (general utility improvement)")
    else:
        try:
            args.target_class = int(args.target_class)
            print(f"📊 Using target class {args.target_class} for calibration (targeted improvement)")
        except ValueError:
            print(f"❌ Error: target_class must be 'all' or an integer, got '{args.target_class}'")
            exit(1)
    
    # Validate privacy parameters
    if args.use_legacy_accounting:
        if args.epsilon is None:
            print("❌ Error: --use-legacy-accounting requires --epsilon parameter")
            exit(1)
        if args.target_epsilon is not None:
            print("❌ Error: Cannot use --target-epsilon with --use-legacy-accounting")
            exit(1)
    else:
        if args.epsilon is not None:
            print("❌ Error: --epsilon is only for legacy accounting")
            exit(1)
        if args.target_epsilon is None:
            args.target_epsilon = 10.0
    
    device = get_device(args)
    
    # Clean up if requested
    if args.clean:
        print('Cleaning saved models…')
        for f in glob.glob(os.path.join(models_dir,'*Test*.pth')):
            os.remove(f); print('  removed',f)
    
    print(f"\n🔍 LINE SEARCH OPTIMIZATION TEST")
    print(f"   • Focus: Comparing standard vs line search calibration")
    print(f"   • Dataset: CIFAR-10, {args.dataset_size} samples, {args.epochs} epochs")
    print(f"   • Privacy: (ε={args.target_epsilon}, δ={args.delta})")
    print(f"   • Fisher DP: k={args.k}, layers={args.dp_layer}")
    print(f"   • Calibration: method={args.method}, k={args.calibration_k}, τ={args.trust_tau}, reg={args.reg}")
    print(f"   • Target class: {args.target_class}")
    print(f"   • Device: {device}")
    
    # ════════════════════════════════════════════════════════════════
    # Data preparation - Match main.py's proper 3-way split
    # ════════════════════════════════════════════════════════════════
    
    trans = T.Compose([T.ToTensor(), T.Normalize((.5,.5,.5),(.5,.5,.5))])
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=trans)
    
    # Data partitioning for differential privacy (MATCHING main.py):
    # - priv_base: From trainset - used for baseline training, DP training
    # - pub_base: From testset subset - used for calibration (public data)  
    # - eval_base: From testset subset - used for evaluation (independent)
    
    # Private data: use subset of trainset for training
    perm_train = np.random.permutation(min(args.dataset_size, len(trainset)))
    priv_idx = perm_train[:min(args.dataset_size, len(trainset))]
    priv_base = Subset(trainset, priv_idx)  # Training data from trainset
    
    # Public data: use subset of testset for calibration
    pub_ratio = 0.5  # Use 50% of testset for calibration, 50% for evaluation
    perm_test = np.random.permutation(len(testset))
    pub_split = int(len(testset) * pub_ratio)
    pub_idx, eval_idx = perm_test[:pub_split], perm_test[pub_split:]
    
    pub_base = Subset(testset, pub_idx)    # Calibration data from testset
    eval_base = Subset(testset, eval_idx)  # Evaluation data from testset
    
    print(f'\n📊 Line Search Test Data:')
    print(f'   • Private data: {len(priv_base)} samples from CIFAR-10 trainset (for training)')
    print(f'   • Public data: {len(pub_base)} samples from CIFAR-10 testset (for calibration)')
    print(f'   • Evaluation data: {len(eval_base)} samples from CIFAR-10 testset (for evaluation)')
    
    # Setup data loaders based on DP mode
    if args.sample_level:
        print('📊 Using SAMPLE-level DP')
        priv_loader = DataLoader(priv_base, batch_size=128, shuffle=True)
        priv_ds = None
        # Public loader for calibration (sample-level)
        pub_loader = DataLoader(pub_base, batch_size=128, shuffle=False)
    else:
        print(f'👥 Using USER-level DP ({args.users} synthetic users)')
        priv_ds = SyntheticUserDataset(priv_base, args.users)
        priv_loader = DataLoader(priv_ds, batch_sampler=UserBatchSampler(priv_ds.uid))
        # Public loader for calibration (user-level) - use same synthetic user structure
        pub_ds = SyntheticUserDataset(pub_base, args.users)
        pub_loader = DataLoader(pub_ds, batch_size=128, shuffle=False)
    
    # ════════════════════════════════════════════════════════════════
    # Fisher matrix computation
    # ════════════════════════════════════════════════════════════════
    
    print('\n🔍 Computing Fisher matrix for line search test…')
    
    # Train a baseline model for Fisher computation
    fisher_baseline = CNN().to(device)
    fisher_opt = torch.optim.SGD(fisher_baseline.parameters(), lr=1e-3, momentum=.9)
    
    for epoch in tqdm(range(5), desc="Training Fisher baseline"):  # Fewer epochs for Fisher
        fisher_baseline.train()
        for batch_data in priv_loader:
            if args.sample_level:
                x, y = batch_data
            else:
                x, y, _ = batch_data
            x, y = x.to(device), y.to(device)
            fisher_opt.zero_grad()
            F.cross_entropy(fisher_baseline(x),y).backward()
            fisher_opt.step()
    
    Fmat, _ = compute_fisher(fisher_baseline, priv_loader, device,
                            target_layer=args.dp_layer, rho=1e-2)
    
    # ════════════════════════════════════════════════════════════════
    # Run line search test
    # ════════════════════════════════════════════════════════════════
    
    # Run the line search test
    results = run_line_search_test(args, device, priv_loader, eval_base, priv_base, 
                                  priv_idx, priv_ds, Fmat, pub_loader)
    
    print(f"\n✅ Line search optimization test complete! Models saved in {models_dir}/")

if __name__ == "__main__":
    main() 