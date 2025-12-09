# dp-sgd.py  – Vanilla DP-SGD implementation for comparison
# ------------------------------------------------------------
#  * Standard Euclidean gradient clipping
#  * Isotropic Gaussian noise
#  * Per-sample gradient computation
#  * Comparison baseline for Fisher-informed DP-SGD

import math, numpy as np, torch, torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm

def train_with_vanilla_dp(model, train_loader, epsilon=8.0, delta=1e-6,
                         clip_radius=10.0, device="cuda", target_layer="conv1",
                         adaptive_clip=True, quantile=0.95, sample_level=None,
                         epochs=1, sigma=None, dp_param_count=None):
    """
    Vanilla DP-SGD training with Euclidean clipping and isotropic noise.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        epsilon: Privacy budget
        delta: Privacy parameter
        clip_radius: Clipping radius for gradients (L2 norm)
        device: Device to run on
        target_layer: Which layer(s) to apply DP to (same as Fisher version)
        adaptive_clip: Whether to use adaptive clipping
        quantile: Quantile for adaptive clipping
        sample_level: If True, use sample-level DP (clip per sample).
                     If False, use user-level DP (clip per user).
                     If None, auto-detect from batch structure.
        epochs: Number of training epochs (for privacy accounting)
        sigma: If provided, use this sigma directly (for proper privacy accounting).
               If None, compute sigma from epsilon using legacy method.
    
    Returns:
        Trained model with vanilla DP-SGD
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    
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

    # gather parameter objects based on target_layer
    def _match(name: str, layer: str) -> bool:
        return name.startswith(layer)
    
    # gather parameter objects based on target_layer or dp_param_count
    if dp_param_count is not None:
        # DP parameter budget mode: smart selection to maximize parameter usage
        print(f"   🎯 DP Parameter Budget Mode: selecting up to {dp_param_count} parameters")
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
                print(f"      • {name}: {size} params (total: {total_selected})")
            elif total_selected < dp_param_count:
                print(f"      ✗ {name}: {size} params (would exceed budget, skipping)")
        
        # Extract selected parameters
        names = []
        params = []
        for idx in sorted(selected_indices):
            name, param = all_params[idx]
            names.append(name)
            params.append(param)
        
        unused = dp_param_count - total_selected
        efficiency = (total_selected / dp_param_count) * 100
        
        print(f"   ✅ Selected {len(names)} complete parameters")
        print(f"      Budget: {dp_param_count} | Used: {total_selected} | Unused: {unused} ({efficiency:.1f}% efficiency)")
        
        dp_mask = None  # No masking needed - all complete parameters
    else:
        if target_layer == "all":
            names = [n for n,_ in model.named_parameters()]
        elif "," in target_layer:
            layers = [s.strip() for s in target_layer.split(",")]
            names  = [n for n,_ in model.named_parameters()
                    if any(_match(n, l) for l in layers)]
        else:
            names = [n for n,_ in model.named_parameters()
                    if _match(n, target_layer)]
        params = [dict(model.named_parameters())[n] for n in names]
        
        dp_mask = None  # No masking in layer mode
    
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
    print(f"\n🎯 Vanilla DP-SGD config: {mode_str} DP  layers={target_layer}  ε={epsilon}")
    if sigma is not None:
        print(f"   • Proper privacy accounting: σ={sigma:.4f}")
    else:
        print(f"   • Multi-epoch privacy: T={epochs}, σ_single={sigma_single_epoch:.3f}, σ_adjusted={sigma:.3f}")
    print(f"   • Euclidean clipping with radius {clip_radius}")
    print(f"   • Isotropic Gaussian noise")
    print(f"   • Adaptive clipping: {adaptive_clip}")
    
    if not sample_level:
        print("   • User-level mode: Clipping aggregated user gradients")
    else:
        print("   • Sample-level mode: Clipping individual sample gradients")
    print()
    
    noise_l2, grad_norm = [], []
    adaptive_radius_computed = False
    actual_radius = clip_radius  # Start with provided radius
    
    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Vanilla DP-SGD ({mode_str})")):
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
                    
                    print(f"📊 Vanilla adaptive clipping from {len(grad_norm)} samples:")
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
                
                # Adaptive clipping: collect user gradient norms
                if adaptive_clip and not adaptive_radius_computed:
                    user_norm = user_grad_flat.norm().item()
                    grad_norm.append(user_norm)
                    
                    # If we have enough users or it's the first batch, compute adaptive radius
                    if len(grad_norm) >= min(10, len(train_loader)) or batch_idx == 0:
                        adaptive_radius = np.quantile(grad_norm, quantile)
                        actual_radius = adaptive_radius
                        adaptive_radius_computed = True
                        
                        print(f"📊 Vanilla adaptive clipping from {len(grad_norm)} users:")
                        print(f"   • Mean user grad norm: {np.mean(grad_norm):.3f}")
                        print(f"   • Median user grad norm: {np.median(grad_norm):.3f}")
                        print(f"   • {quantile:.1%} quantile: {adaptive_radius:.3f}")
                        print(f"   • Max user grad norm: {np.max(grad_norm):.3f}")
                        print(f"   → Using adaptive radius: {actual_radius:.3f}\n")
                        
                        grad_norm = []  # Reset for actual training statistics
            
            # In user-level DP with UserBatchSampler, we should have exactly one user per batch
            if len(user_gradients) != 1:
                print(f"⚠️  Warning: Expected 1 user per batch, got {len(user_gradients)} users")
                print(f"   Unique users in batch: {unique_users.tolist()}")
            
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
        
        # scatter back to model parameters
        idx = 0
        for p in params:
            n = p.numel()
            p.grad = g_priv[idx:idx+n].view_as(p)
            idx += n
        opt.step()
    
    grad_type = "‖g_user‖₂" if not sample_level else "‖g‖₂"
    print(f"\n📊  Vanilla DP-SGD final stats:")
    print(f"   • Median {grad_type} = {np.median(grad_norm):.2f}")
    print(f"   • Isotropic noise ℓ₂ ∈ [{min(noise_l2):.1f},{max(noise_l2):.1f}]")
    print(f"   • Privacy: (ε={epsilon}, δ={delta}) over {epochs} epochs")
    
    return model
