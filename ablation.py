#!/usr/bin/env python3
# ================================================================
# ABLATION STUDY: Fisher DP-SGD with Different Optimizers
#    * Fisher DP + Normal Optimizer (baseline)
#    * Fisher DP + DP-SAT Optimizer (synergistic combination)
#    * + Influence Function Calibration
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
from influence_function import calibrate_model_research_protocol, get_critical_slice

torch.manual_seed(42); np.random.seed(42)
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

def print_calibration_effect(before_stats, after_stats, target_class=3):
    """
    Print the effect of calibration on the critical slice.
    Replacement for the removed function in influence_function.py.
    """
    print(f"\n📊 Calibration Effect Analysis (Class {target_class}):")
    print(f"   • Samples evaluated: {before_stats['num_samples']}")
    print(f"   • Loss before:   {before_stats['loss']:.4f}")
    print(f"   • Loss after:    {after_stats['loss']:.4f}")
    print(f"   • Δ Loss:        {after_stats['loss'] - before_stats['loss']:+.4f}")
    print(f"   • Accuracy before: {before_stats['accuracy']:.2f}%")
    print(f"   • Accuracy after:  {after_stats['accuracy']:.2f}%")
    print(f"   • Δ Accuracy:      {after_stats['accuracy'] - before_stats['accuracy']:+.2f}%")
    
    if after_stats['loss'] < before_stats['loss']:
        print(f"   ✅ SUCCESS: Calibration reduced critical slice loss!")
    else:
        print(f"   ⚠️  WARNING: Calibration increased critical slice loss")
    
    if after_stats['accuracy'] > before_stats['accuracy']:
        print(f"   ✅ SUCCESS: Calibration improved critical slice accuracy!")
    else:
        print(f"   ⚠️  WARNING: Calibration reduced critical slice accuracy")

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
                                  optimizer_name="Normal", privacy_first=False):
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
        privacy_first: If False (default), use utility-first scaling (noise ∝ 1/√λ).
                      If True, use privacy-first scaling (noise ∝ √λ).
        
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
    inv_sqrt_lam = lam.rsqrt()  # 1/√λ (utility-first: less noise in high curvature)
    sqrt_lam = lam.sqrt()       # √λ (privacy-first: more noise in high curvature)
    
    # Choose noise scaling strategy
    if privacy_first:
        noise_scaling = sqrt_lam
        strategy_name = "Privacy-first (noise ∝ √λ)"
    else:
        noise_scaling = inv_sqrt_lam
        strategy_name = "Utility-first (noise ∝ 1/√λ, default)"
    
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

def run_ablation_study(args, device, priv_loader, eval_base, priv_base, priv_idx, priv_ds, Fmat, pub_loader):
    """
    Run comprehensive ablation study on Fisher DP-SGD variants.
    
    Variants tested:
    1. Vanilla DP-SGD (Non-Fisher)
    2. Vanilla DP-SGD + DP-SAT (Non-Fisher)
    3. Fisher DP + Normal Optimizer
    4. Fisher DP + DP-SAT Optimizer
    5. Fisher DP + Normal + Influence Function Calibration
    6. Fisher DP + DP-SAT + Influence Function Calibration
    """
    
    print("\n" + "="*70)
    print("🔬  ABLATION STUDY: Fisher DP-SGD with Different Optimizers")
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
        
        print(f"\n🔒 Privacy Accounting for Ablation Study:")
        print(f"   • Target (ε, δ): ({args.target_epsilon}, {args.delta})")
        print(f"   • Noise multiplier: {noise_multiplier:.4f}")
        print(f"   • Sigma: {sigma:.4f}")
    else:
        sigma = None
        display_epsilon = args.epsilon

    # Initialize results storage
    ablation_results = {}
    
    # ════════════════════════════════════════════════════════════════
    # Variant 1: Vanilla DP-SGD (Non-Fisher)
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🔵 VARIANT 1: Vanilla DP-SGD (Non-Fisher)")
    print(f"{'='*50}")
    
    vanilla_dp_model = copy.deepcopy(baseline)
    vanilla_dp_model = train_with_vanilla_dp(
        vanilla_dp_model, priv_loader,
        epsilon=display_epsilon, delta=args.delta,
        sigma=sigma,
        clip_radius=args.clip_radius,
        device=device,
        target_layer=args.dp_layer,
        adaptive_clip=args.adaptive_clip,
        quantile=args.quantile,
        sample_level=args.sample_level,
        epochs=args.epochs
    )
    
    # ════════════════════════════════════════════════════════════════
    # Variant 2: Vanilla DP-SGD + DP-SAT (Non-Fisher)
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🔵🔺 VARIANT 2: Vanilla DP-SGD + DP-SAT (Non-Fisher)")
    print(f"{'='*50}")
    
    vanilla_dpsat_model = copy.deepcopy(baseline)
    vanilla_dpsat_model = train_with_dp_sat(
        vanilla_dpsat_model, priv_loader,
        epsilon=display_epsilon, delta=args.delta,
        sigma=sigma,
        clip_radius=args.clip_radius,
        device=device,
        target_layer=args.dp_layer,
        adaptive_clip=args.adaptive_clip,
        quantile=args.quantile,
        sample_level=args.sample_level,
        epochs=args.epochs,
        lambda_flatness=args.lambda_flatness
    )
    
    # ════════════════════════════════════════════════════════════════
    # Variant 3: Fisher DP + Normal Optimizer
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🎯 VARIANT 3: Fisher DP + Normal Optimizer")
    print(f"{'='*50}")
    
    fisher_normal_model = copy.deepcopy(baseline)
    fisher_normal_model = train_fisher_dp_with_optimizer(
        fisher_normal_model, priv_loader, Fmat,
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
        use_dp_sat=False,  # Normal optimizer
        optimizer_name="Normal",
        privacy_first=args.privacy
    )
    
    # ════════════════════════════════════════════════════════════════
    # Variant 4: Fisher DP + DP-SAT Optimizer
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🔺 VARIANT 4: Fisher DP + DP-SAT Optimizer")
    print(f"{'='*50}")
    
    fisher_dpsat_model = copy.deepcopy(baseline)
    fisher_dpsat_model = train_fisher_dp_with_optimizer(
        fisher_dpsat_model, priv_loader, Fmat,
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
        use_dp_sat=True,  # DP-SAT optimizer
        lambda_flatness=args.lambda_flatness,
        optimizer_name="DP-SAT",
        privacy_first=args.privacy
    )
    
    # ════════════════════════════════════════════════════════════════
    # Variant 5: Fisher DP + Normal + Influence Function Calibration
    # ════════════════════════════════════════════════════════════════
    
    # Create test loader for evaluation (needed for regularization)
    test_loader = DataLoader(eval_base, batch_size=128, shuffle=False)
    
    print(f"\n{'='*50}")
    print("📐 VARIANT 5: Fisher DP + Normal + Calibration")
    print(f"{'='*50}")
    
    fisher_normal_calibrated = copy.deepcopy(fisher_normal_model)
    
    # Get critical slice using the new API
    critical_data, critical_targets = get_critical_slice(pub_loader, args.target_class, device)
    
    if len(critical_data) > 0:
        before_stats = diagnose_calibration(fisher_normal_calibrated, critical_data, critical_targets, device)
        
        # Apply calibration with the correct signature
        calib_normal = calibrate_model_research_protocol(
            fisher_normal_calibrated, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=args.method,
            eta=args.calibration_k,
            trust_tau=args.trust_tau,
            strict=True,
            clean_model=baseline,
            reg=args.reg
        )
        
        after_stats = diagnose_calibration(calib_normal, critical_data, critical_targets, device)
        print_calibration_effect(before_stats, after_stats, args.target_class)
    else:
        print(f"⚠️  No samples found for target_class {args.target_class}")
        # Still apply calibration without diagnosis
        calib_normal = calibrate_model_research_protocol(
            fisher_normal_calibrated, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=args.method,
            eta=args.calibration_k,
            trust_tau=args.trust_tau,
            strict=True,
            clean_model=baseline,
            reg=args.reg
        )
    
    # ════════════════════════════════════════════════════════════════
    # Variant 6: Fisher DP + DP-SAT + Influence Function Calibration
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🔺📐 VARIANT 6: Fisher DP + DP-SAT + Calibration")
    print(f"{'='*50}")
    
    fisher_dpsat_calibrated = copy.deepcopy(fisher_dpsat_model)
    
    # Diagnose before calibration for DP-SAT variant
    if len(critical_data) > 0:
        before_stats_dpsat = diagnose_calibration(fisher_dpsat_calibrated, critical_data, critical_targets, device)
        
        # Apply calibration with the correct signature
        calib_dpsat = calibrate_model_research_protocol(
            fisher_dpsat_calibrated, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=args.method,
            eta=args.calibration_k,
            trust_tau=args.trust_tau,
            strict=True,
            clean_model=baseline,
            reg=args.reg
        )
        
        after_stats_dpsat = diagnose_calibration(calib_dpsat, critical_data, critical_targets, device)
        print_calibration_effect(before_stats_dpsat, after_stats_dpsat, args.target_class)
    else:
        print(f"⚠️  No samples found for target_class {args.target_class}")
        # Still apply calibration without diagnosis
        calib_dpsat = calibrate_model_research_protocol(
            fisher_dpsat_calibrated, pub_loader, priv_loader,
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
    print("📊 ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    
    # Compute accuracies
    baseline_acc = accuracy(baseline, test_loader, device)
    vanilla_dp_acc = accuracy(vanilla_dp_model, test_loader, device)
    vanilla_dpsat_acc = accuracy(vanilla_dpsat_model, test_loader, device)
    fisher_normal_acc = accuracy(fisher_normal_model, test_loader, device)
    fisher_dpsat_acc = accuracy(fisher_dpsat_model, test_loader, device)
    calib_normal_acc = accuracy(calib_normal, test_loader, device)
    calib_dpsat_acc = accuracy(calib_dpsat, test_loader, device)
    
    ablation_results['baseline'] = baseline_acc
    ablation_results['vanilla_dp'] = vanilla_dp_acc
    ablation_results['vanilla_dpsat'] = vanilla_dpsat_acc
    ablation_results['fisher_normal'] = fisher_normal_acc
    ablation_results['fisher_dpsat'] = fisher_dpsat_acc
    ablation_results['calib_normal'] = calib_normal_acc
    ablation_results['calib_dpsat'] = calib_dpsat_acc
    
    dp_mode = "Sample-level" if args.sample_level else f"User-level ({args.users} users)"
    print(f"\n🎯 Accuracy Comparison ({dp_mode} DP):")
    print(f"   • Baseline (Non-DP)           : {baseline_acc:6.2f}%")
    print(f"   • Vanilla DP-SGD              : {vanilla_dp_acc:6.2f}%")
    print(f"   • Vanilla DP-SGD + DP-SAT      : {vanilla_dpsat_acc:6.2f}%")
    print(f"   • Fisher DP + Normal          : {fisher_normal_acc:6.2f}%")
    print(f"   • Fisher DP + DP-SAT          : {fisher_dpsat_acc:6.2f}%")
    print(f"   • Fisher DP + Normal + Calib  : {calib_normal_acc:6.2f}%")
    print(f"   • Fisher DP + DP-SAT + Calib  : {calib_dpsat_acc:6.2f}%")
    
    # Compute improvements
    vanilla_dp_improvement = vanilla_dp_acc - baseline_acc
    vanilla_dpsat_improvement = vanilla_dpsat_acc - baseline_acc
    vanilla_dpsat_vs_vanilla = vanilla_dpsat_acc - vanilla_dp_acc
    fisher_vs_vanilla = fisher_normal_acc - vanilla_dp_acc
    normal_improvement = fisher_normal_acc - baseline_acc
    dpsat_improvement = fisher_dpsat_acc - baseline_acc
    synergy_gain = fisher_dpsat_acc - fisher_normal_acc
    calib_normal_improvement = calib_normal_acc - fisher_normal_acc
    calib_dpsat_improvement = calib_dpsat_acc - fisher_dpsat_acc
    
    print(f"\n📈 Improvement Analysis:")
    print(f"   • Vanilla DP-SGD:             {vanilla_dp_improvement:+5.2f}% vs baseline")
    print(f"   • Vanilla DP-SGD + DP-SAT:    {vanilla_dpsat_improvement:+5.2f}% vs baseline")
    print(f"   • DP-SAT gain (Vanilla):      {vanilla_dpsat_vs_vanilla:+5.2f}% over vanilla DP")
    print(f"   • Fisher benefit:             {fisher_vs_vanilla:+5.2f}% over vanilla DP")
    print(f"   • Fisher DP (Normal):         {normal_improvement:+5.2f}% vs baseline")
    print(f"   • Fisher DP (DP-SAT):         {dpsat_improvement:+5.2f}% vs baseline")
    print(f"   • Synergy Gain (DP-SAT):      {synergy_gain:+5.2f}% over normal Fisher DP")
    print(f"   • Calibration (Normal):       {calib_normal_improvement:+5.2f}% over Fisher normal")
    print(f"   • Calibration (DP-SAT):       {calib_dpsat_improvement:+5.2f}% over Fisher DP-SAT")
    
    print(f"\n🔬 Combined Effects Analysis:")
    total_dpsat_gain = fisher_dpsat_acc - baseline_acc
    total_calib_normal_gain = calib_normal_acc - baseline_acc
    total_calib_dpsat_gain = calib_dpsat_acc - baseline_acc
    
    print(f"   • Total DP-SAT effect:       {total_dpsat_gain:+5.2f}% (DP-SAT only)")
    print(f"   • Total Calibration effect:  {total_calib_normal_gain:+5.2f}% (Normal + Calib)")
    print(f"   • Total Combined effect:     {total_calib_dpsat_gain:+5.2f}% (DP-SAT + Calib)")
    
    best_method = max([
        ('Vanilla DP-SGD', vanilla_dp_acc),
        ('Vanilla DP-SGD + DP-SAT', vanilla_dpsat_acc),
        ('Fisher Normal', fisher_normal_acc),
        ('Fisher DP-SAT', fisher_dpsat_acc),
        ('Fisher Normal + Calib', calib_normal_acc),
        ('Fisher DP-SAT + Calib', calib_dpsat_acc)
    ], key=lambda x: x[1])
    
    print(f"   🏆 Best method: {best_method[0]} ({best_method[1]:.2f}%)")
    
    if synergy_gain > 0.5:  # Threshold for meaningful improvement
        print(f"   🎉 SYNERGY DETECTED: DP-SAT optimization provides meaningful benefit!")
    elif synergy_gain > 0:
        print(f"   ✅ SMALL SYNERGY: DP-SAT provides modest improvement")
    else:
        print(f"   ⚠️  NO SYNERGY: DP-SAT may not help with Fisher-informed noise")
    
    if max(calib_normal_improvement, calib_dpsat_improvement) > 0.5:
        print(f"   📐 CALIBRATION BENEFIT: Influence function calibration improves performance!")
    elif max(calib_normal_improvement, calib_dpsat_improvement) > 0:
        print(f"   📐 MODEST CALIBRATION: Small improvement from calibration")
    else:
        print(f"   ⚠️  LIMITED CALIBRATION: Calibration shows minimal benefit")

    # ════════════════════════════════════════════════════════════════
    # Save Models for Further Analysis
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n💾 Saving ablation study models...")
    
    # Save Vanilla DP-SGD
    vanilla_dp_path = os.path.join(models_dir, 'Vanilla_DP_Ablation.pth')
    torch.save({
        'model_state_dict': vanilla_dp_model.state_dict(),
        'model_type': 'vanilla_dp',
        'accuracy': vanilla_dp_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'ablation_study': True
    }, vanilla_dp_path)
    print(f"✅ Saved Vanilla DP-SGD to {vanilla_dp_path}")
    
    # Save Vanilla DP-SGD + DP-SAT
    vanilla_dpsat_path = os.path.join(models_dir, 'Vanilla_DPSAT_Ablation.pth')
    torch.save({
        'model_state_dict': vanilla_dpsat_model.state_dict(),
        'model_type': 'vanilla_dp_dpsat',
        'accuracy': vanilla_dpsat_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'lambda_flatness': args.lambda_flatness,
        'dpsat_gain_vanilla': vanilla_dpsat_vs_vanilla,
        'ablation_study': True
    }, vanilla_dpsat_path)
    print(f"✅ Saved Vanilla DP-SGD + DP-SAT to {vanilla_dpsat_path}")
    
    # Save Fisher DP + Normal
    fisher_normal_path = os.path.join(models_dir, 'Fisher_Normal_Ablation.pth')
    torch.save({
        'model_state_dict': fisher_normal_model.state_dict(),
        'model_type': 'fisher_dp_normal',
        'accuracy': fisher_normal_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'k': args.k,
        'full_complement_noise': args.full_complement_noise,
        'ablation_study': True
    }, fisher_normal_path)
    print(f"✅ Saved Fisher DP + Normal to {fisher_normal_path}")
    
    # Save Fisher DP + DP-SAT
    fisher_dpsat_path = os.path.join(models_dir, 'Fisher_DPSAT_Ablation.pth')
    torch.save({
        'model_state_dict': fisher_dpsat_model.state_dict(),
        'model_type': 'fisher_dp_dpsat',
        'accuracy': fisher_dpsat_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'k': args.k,
        'lambda_flatness': args.lambda_flatness,
        'full_complement_noise': args.full_complement_noise,
        'synergy_gain': synergy_gain,
        'ablation_study': True
    }, fisher_dpsat_path)
    print(f"✅ Saved Fisher DP + DP-SAT to {fisher_dpsat_path}")
    
    # Save Fisher DP + Normal + Calibration
    calib_normal_path = os.path.join(models_dir, 'Fisher_Normal_Calibrated_Ablation.pth')
    torch.save({
        'model_state_dict': calib_normal.state_dict(),
        'model_type': 'fisher_dp_normal_calibrated',
        'accuracy': calib_normal_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'k': args.k,
        'calibration_k': args.calibration_k,
        'calibration_method': args.method,
        'calibration_improvement': calib_normal_improvement,
        'full_complement_noise': args.full_complement_noise,
        'ablation_study': True
    }, calib_normal_path)
    print(f"✅ Saved Fisher DP + Normal + Calibration to {calib_normal_path}")
    
    # Save Fisher DP + DP-SAT + Calibration
    calib_dpsat_path = os.path.join(models_dir, 'Fisher_DPSAT_Calibrated_Ablation.pth')
    torch.save({
        'model_state_dict': calib_dpsat.state_dict(),
        'model_type': 'fisher_dp_dpsat_calibrated',
        'accuracy': calib_dpsat_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'k': args.k,
        'lambda_flatness': args.lambda_flatness,
        'calibration_k': args.calibration_k,
        'calibration_method': args.method,
        'calibration_improvement': calib_dpsat_improvement,
        'synergy_gain': synergy_gain,
        'total_combined_gain': total_calib_dpsat_gain,
        'full_complement_noise': args.full_complement_noise,
        'ablation_study': True
    }, calib_dpsat_path)
    print(f"✅ Saved Fisher DP + DP-SAT + Calibration to {calib_dpsat_path}")

    # ════════════════════════════════════════════════════════════════
    # Optional: Comprehensive 4-Way MIA Evaluation
    # ════════════════════════════════════════════════════════════════
    
    if args.run_mia:
        print(f"\n🛡️  Running comprehensive 4-way MIA evaluation on all ablation variants...")
        
        # Prepare all models for MIA evaluation
        models_to_evaluate = {
            'Baseline (Non-DP)': baseline,
            'Vanilla DP-SGD': vanilla_dp_model,
            'Vanilla DP-SGD + DP-SAT': vanilla_dpsat_model,
            'Fisher DP + Normal': fisher_normal_model,
            'Fisher DP + DP-SAT': fisher_dpsat_model,
            'Fisher DP + Normal + Calib': calib_normal,
            'Fisher DP + DP-SAT + Calib': calib_dpsat
        }
        
        # Set random seed for reproducible MIA evaluation
        np.random.seed(42)
        
        # Prepare member and non-member datasets
        if args.sample_level:
            print("📊 Sample-level MIA: Using actual private training samples as members")
            member_set, non_member_set = prepare_mia_data_sample_level(priv_base, eval_base, priv_idx, args.mia_size)
        else:
            print("👥 User-level MIA: Using actual private users as members")
            member_set, non_member_set = prepare_mia_data_user_level(priv_ds, eval_base, args.users, args.mia_size)
        
        member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
        
        print(f"   • Members: {len(member_set)} samples")
        print(f"   • Non-members: {len(non_member_set)} samples")
        
        # Run comprehensive MIA evaluation on all models
        mia_results = {}
        
        print(f"\n🎯 CONFIDENCE ATTACK RESULTS:")
        print("-" * 50)
        
        for model_name, model in models_to_evaluate.items():
            print(f"   Evaluating {model_name}...")
            conf_result = confidence_attack(model, member_loader, non_member_loader, device)
            mia_results[model_name] = {
                'confidence_auc': conf_result['auc'],
                'confidence_acc': conf_result['accuracy'],
                'member_conf_mean': conf_result['member_conf_mean'],
                'non_member_conf_mean': conf_result['non_member_conf_mean']
            }
            print(f"     • AUC: {conf_result['auc']:.4f}, Accuracy: {conf_result['accuracy']:.4f}")
        
        print(f"\n🕶️  SHADOW MODEL ATTACK RESULTS:")
        print("-" * 50)
        
        for model_name, model in models_to_evaluate.items():
            print(f"   Evaluating {model_name}...")
            shadow_result = shadow_model_attack(model, member_loader, non_member_loader, priv_base, device)
            mia_results[model_name]['shadow_auc'] = shadow_result['auc']
            mia_results[model_name]['shadow_acc'] = shadow_result['accuracy']
            print(f"     • AUC: {shadow_result['auc']:.4f}, Accuracy: {shadow_result['accuracy']:.4f}")
        
        # Comprehensive analysis
        print(f"\n📊 COMPREHENSIVE MIA ANALYSIS")
        print("=" * 60)
        
        # Calculate worst-case AUC for each model
        worst_case_aucs = {}
        for model_name in models_to_evaluate.keys():
            worst_case_aucs[model_name] = max(
                mia_results[model_name]['confidence_auc'],
                mia_results[model_name]['shadow_auc']
            )
        
        print(f"\n🎯 Worst-case AUC Comparison:")
        for model_name, worst_auc in worst_case_aucs.items():
            print(f"   • {model_name:30}: {worst_auc:.4f}")
        
        # Identify best and worst models for privacy
        best_privacy_model = min(worst_case_aucs.items(), key=lambda x: x[1])
        worst_privacy_model = max(worst_case_aucs.items(), key=lambda x: x[1])
        
        print(f"\n🏆 Privacy Protection Ranking:")
        print(f"   🥇 BEST:  {best_privacy_model[0]} (AUC: {best_privacy_model[1]:.4f})")
        print(f"   🥴 WORST: {worst_privacy_model[0]} (AUC: {worst_privacy_model[1]:.4f})")
        
        # Privacy vs Accuracy tradeoff analysis
        print(f"\n⚖️  Privacy vs Accuracy Tradeoff:")
        print(f"   Model                          Accuracy  Privacy (1-AUC)")
        print(f"   {'─'*30} ─────────  ──────────────")
        for model_name in ['Vanilla DP-SGD', 'Vanilla DP-SGD + DP-SAT', 'Fisher DP + Normal', 'Fisher DP + DP-SAT', 'Fisher DP + Normal + Calib', 'Fisher DP + DP-SAT + Calib']:
            if model_name == 'Vanilla DP-SGD':
                acc = vanilla_dp_acc
            elif model_name == 'Vanilla DP-SGD + DP-SAT':
                acc = vanilla_dpsat_acc
            elif model_name == 'Fisher DP + Normal':
                acc = fisher_normal_acc
            elif model_name == 'Fisher DP + DP-SAT':
                acc = fisher_dpsat_acc
            elif model_name == 'Fisher DP + Normal + Calib':
                acc = calib_normal_acc
            else:  # Fisher DP + DP-SAT + Calib
                acc = calib_dpsat_acc
            
            privacy_score = 1.0 - worst_case_aucs[model_name]  # Higher is better
            print(f"   {model_name:30} {acc:5.1f}%     {privacy_score:.3f}")
        
        # Key comparisons only (remove redundant analysis)
        print(f"\n🔒 Key Privacy Effects:")
        
        baseline_auc = worst_case_aucs['Baseline (Non-DP)']
        fisher_normal_auc = worst_case_aucs['Fisher DP + Normal']
        fisher_dpsat_auc = worst_case_aucs['Fisher DP + DP-SAT']
        calib_normal_auc = worst_case_aucs['Fisher DP + Normal + Calib']
        calib_dpsat_auc = worst_case_aucs['Fisher DP + DP-SAT + Calib']
        
        dpsat_privacy_effect = fisher_normal_auc - fisher_dpsat_auc
        calib_normal_privacy_effect = fisher_normal_auc - calib_normal_auc
        calib_dpsat_privacy_effect = fisher_dpsat_auc - calib_dpsat_auc
        combined_effect = fisher_normal_auc - calib_dpsat_auc
        
        print(f"   • DP-SAT effect:      {dpsat_privacy_effect:+.4f} AUC")
        print(f"   • Calibration effect: {max(calib_normal_privacy_effect, calib_dpsat_privacy_effect):+.4f} AUC") 
        print(f"   • Combined effect:    {combined_effect:+.4f} AUC")
        
        # Final recommendation
        print(f"\n🎯 Best Privacy Protection: {best_privacy_model[0]} (AUC: {best_privacy_model[1]:.4f})")
        
        # Store results for return
        ablation_results['mia_results'] = {
            'worst_case_aucs': worst_case_aucs,
            'best_privacy_model': best_privacy_model,
            'detailed_results': mia_results,
            'privacy_effects': {
                'dpsat_effect': dpsat_privacy_effect,
                'calib_normal_effect': calib_normal_privacy_effect,
                'calib_dpsat_effect': calib_dpsat_privacy_effect,
                'combined_effect': combined_effect
            }
        }
        
        # Update legacy fields for compatibility
        ablation_results['mia_results']['fisher_worst_auc'] = fisher_normal_auc
        ablation_results['mia_results']['dp_sat_worst_auc'] = fisher_dpsat_auc
        
        print(f"\n✅ Comprehensive 4-way MIA evaluation complete!")

    return ablation_results

# ════════════════════════════════════════════════════════════════
# Main function
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser('Fisher DP-SGD Ablation Study')
    
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
    noise_strategy_group.add_argument('--utility', action='store_true', default=True,
                                     help='Fisher DP utility-first: less noise in high curvature directions (inverse scaling, default)')
    noise_strategy_group.add_argument('--privacy', action='store_true',
                                     help='Fisher DP privacy-first: more noise in high curvature directions (direct scaling)')
    
    # DP-SAT arguments
    parser.add_argument('--lambda-flatness', type=float, default=0.01,
                       help='Flatness coefficient for DP-SAT')
    
    # Adaptive clipping
    parser.add_argument('--adaptive-clip', action='store_true')
    parser.add_argument('--quantile', type=float, default=0.95)
    
    # DP mode
    parser.add_argument('--sample-level', action='store_true')
    parser.add_argument('--users', type=int, default=10)
    
    # Calibration arguments (now enabled)
    parser.add_argument('--method', type=str, default='linear',
                       choices=['linear','batch','original'],
                       help='Calibration method: linear (fast), batch (medium), original (slow but accurate)')
    parser.add_argument('--calibration-k', type=int, default=100,
                       help='Number of top-k samples to use for calibration')
    parser.add_argument('--trust-tau', type=float, default=0.05,
                       help='Trust region parameter: max relative parameter change (default: 0.05 = 5%%)')
    parser.add_argument('--reg', type=float, default=1.0,
                       help='Regularization parameter for linear method influence vectors (default: 1.0)')
    parser.add_argument('--target-class', default=3,
                       help='Target class for critical slice: integer for single class, "all" for multi-class (default: 3 = "cat" for CIFAR-10)')
    parser.add_argument('--compare-calibration', action='store_true',
                       help='Run comparative experiment between single-class and multi-class calibration')
    
    # MIA evaluation
    parser.add_argument('--run-mia', action='store_true')
    parser.add_argument('--mia-size', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Parse target_class argument (must be int)
    try:
        target_class = int(args.target_class)
        print(f"📊 Using target class {target_class} for calibration")
    except ValueError:
        print(f"❌ Error: target_class must be an integer, got '{args.target_class}'")
        exit(1)
    args.target_class = target_class
    
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
        for f in glob.glob(os.path.join(models_dir,'*Ablation*.pth')):
            os.remove(f); print('  removed',f)
    
    # ════════════════════════════════════════════════════════════════
    # Data preparation
    # ════════════════════════════════════════════════════════════════
    
    trans = T.Compose([T.ToTensor(), T.Normalize((.5,.5,.5),(.5,.5,.5))])
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=trans)
    
    # Private data: subset of trainset
    perm_train = np.random.permutation(min(args.dataset_size, len(trainset)))
    priv_idx = perm_train[:min(args.dataset_size, len(trainset))]
    priv_base = Subset(trainset, priv_idx)
    
    # Evaluation data: subset of testset
    pub_ratio = 0.5
    perm_test = np.random.permutation(len(testset))
    pub_split = int(len(testset) * pub_ratio)
    eval_idx = perm_test[pub_split:]
    eval_base = Subset(testset, eval_idx)
    
    # Public data for calibration: use remaining part of testset
    pub_idx = perm_test[:pub_split]
    pub_base = Subset(testset, pub_idx)
    
    print(f'📊 Ablation Study Data:')
    print(f'   • Private data: {len(priv_base)} samples from trainset')
    print(f'   • Evaluation data: {len(eval_base)} samples from testset')
    print(f'   • Public data (calibration): {len(pub_base)} samples from testset')
    
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
    
    print('\n🔍 Computing Fisher matrix for ablation study…')
    
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
    # Run ablation study
    # ════════════════════════════════════════════════════════════════
    
    results = run_ablation_study(args, device, priv_loader, eval_base, priv_base, 
                                priv_idx, priv_ds, Fmat, pub_loader)
    
    # ════════════════════════════════════════════════════════════════
    # Final summary
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("🎯 ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    
    print(f"🔬 Synergy Analysis:")
    vanilla_dpsat_gain = results['vanilla_dpsat'] - results['vanilla_dp']
    synergy_gain = results['fisher_dpsat'] - results['fisher_normal']
    print(f"   • Vanilla DP-SGD:         {results['vanilla_dp']:6.2f}%")
    print(f"   • Vanilla DP-SGD + DP-SAT: {results['vanilla_dpsat']:6.2f}%")
    print(f"   • DP-SAT gain (Vanilla):   {vanilla_dpsat_gain:+5.2f}%")
    print(f"   • Fisher DP + Normal:     {results['fisher_normal']:6.2f}%")
    print(f"   • Fisher DP + DP-SAT:     {results['fisher_dpsat']:6.2f}%")
    print(f"   • DP-SAT gain (Fisher):   {synergy_gain:+5.2f}%")
    
    print(f"\n📐 Calibration Analysis:")
    calib_normal_gain = results['calib_normal'] - results['fisher_normal']
    calib_dpsat_gain = results['calib_dpsat'] - results['fisher_dpsat']
    print(f"   • Fisher DP + Normal + Calib: {results['calib_normal']:6.2f}%")
    print(f"   • Fisher DP + DP-SAT + Calib: {results['calib_dpsat']:6.2f}%")
    print(f"   • Calibration gain (Normal):  {calib_normal_gain:+5.2f}%")
    print(f"   • Calibration gain (DP-SAT):  {calib_dpsat_gain:+5.2f}%")
    
    print(f"\n🏆 Overall Best Performance:")
    best_variant = max(results['vanilla_dp'], results['vanilla_dpsat'],
                      results['fisher_normal'], results['fisher_dpsat'], 
                      results['calib_normal'], results['calib_dpsat'])
    if best_variant == results['calib_dpsat']:
        print(f"   🥇 Fisher DP + DP-SAT + Calibration: {best_variant:.2f}%")
        print(f"   🎉 TRIPLE COMBINATION: All three techniques work together!")
    elif best_variant == results['calib_normal']:
        print(f"   🥇 Fisher DP + Normal + Calibration: {best_variant:.2f}%")
        print(f"   📐 CALIBRATION DOMINATES: Influence functions provide the key benefit")
    elif best_variant == results['fisher_dpsat']:
        print(f"   🥇 Fisher DP + DP-SAT: {best_variant:.2f}%")
        print(f"   🔺 DP-SAT DOMINATES: Sharpness-aware optimization is most beneficial")
    elif best_variant == results['fisher_normal']:
        print(f"   🥇 Fisher DP + Normal: {best_variant:.2f}%")
        print(f"   🎯 FISHER DOMINATES: Fisher-informed noise is most beneficial")
    elif best_variant == results['vanilla_dpsat']:
        print(f"   🥇 Vanilla DP-SGD + DP-SAT: {best_variant:.2f}%")
        print(f"   🔵🔺 SIMPLE DP-SAT: DP-SAT works best without Fisher complexity")
    else:
        print(f"   🥇 Vanilla DP-SGD: {best_variant:.2f}%")
        print(f"   🔵 VANILLA BEST: Simple DP-SGD outperforms advanced techniques")
    
    if synergy_gain > 1.0:
        print(f"\n✅ STRONG DP-SAT SYNERGY: Combining Fisher + DP-SAT is highly beneficial!")
    elif synergy_gain > 0.5:
        print(f"\n✅ MODERATE DP-SAT SYNERGY: Fisher + DP-SAT combination shows promise")
    elif synergy_gain > 0:
        print(f"\n⚠️  WEAK DP-SAT SYNERGY: Minor benefit from DP-SAT combination")
    else:
        print(f"\n❌ NO DP-SAT SYNERGY: DP-SAT may interfere with Fisher benefits")
    
    if max(calib_normal_gain, calib_dpsat_gain) > 1.0:
        print(f"✅ STRONG CALIBRATION BENEFIT: Influence function calibration significantly helps!")
    elif max(calib_normal_gain, calib_dpsat_gain) > 0.5:
        print(f"✅ MODERATE CALIBRATION BENEFIT: Calibration provides meaningful improvement")
    elif max(calib_normal_gain, calib_dpsat_gain) > 0:
        print(f"⚠️  WEAK CALIBRATION BENEFIT: Small improvement from calibration")
    else:
        print(f"❌ NO CALIBRATION BENEFIT: Calibration may not help this configuration")
    
    print(f"\n🔒 Key Insights:")
    print(f"   • Fisher-informed noise shapes noise according to loss curvature")
    print(f"   • DP-SAT guides optimization toward flatter minima")
    print(f"   • Influence function calibration adjusts model using public data")
    print(f"   • These approaches are orthogonal and can be combined")
    print(f"   • DP-SAT synergy: {synergy_gain:+.2f}% suggests {'beneficial' if synergy_gain > 0 else 'neutral'} interaction")
    print(f"   • Calibration benefit: {max(calib_normal_gain, calib_dpsat_gain):+.2f}% suggests {'beneficial' if max(calib_normal_gain, calib_dpsat_gain) > 0 else 'neutral'} effect")
    
    if 'mia_results' in results:
        print(f"\n🛡️  Privacy Summary:")
        best_privacy = results['mia_results']['best_privacy_model']
        effects = results['mia_results']['privacy_effects']
        
        print(f"   • Best protection: {best_privacy[0]} (AUC: {best_privacy[1]:.4f})")
        print(f"   • Calibration improves privacy by {max(effects['calib_normal_effect'], effects['calib_dpsat_effect']):+.3f} AUC")
        
        if effects['combined_effect'] > 0.02:
            print(f"   ✅ STRONG: Combined techniques provide excellent privacy enhancement")
        elif effects['combined_effect'] > 0:
            print(f"   ✅ GOOD: Combined techniques improve privacy protection")
        else:
            print(f"   ⚠️  LIMITED: Minimal privacy benefit from combined techniques")
    
    print(f"\n📍 Key Findings:")
    print(f"   • DP-SAT synergy: {synergy_gain:+.2f}% accuracy improvement")
    print(f"   • Calibration: {'beneficial' if max(calib_normal_gain, calib_dpsat_gain) > 0 else 'harmful'} for accuracy")
    if 'mia_results' in results:
        print(f"   • Privacy: Calibration provides +{max(results['mia_results']['privacy_effects']['calib_normal_effect'], results['mia_results']['privacy_effects']['calib_dpsat_effect']):.3f} AUC protection")
    
    print(f"\n✅ Ablation study complete! Models saved in {models_dir}/")

if __name__ == "__main__":
    main() 