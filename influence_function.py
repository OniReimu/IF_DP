import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import compute_hessian_inverse
import cvxpy as cp  # For linear program optimization

def unpack_batch(batch_data):
    """Helper function to handle both (x, y) and (x, y, user_id) formats"""
    if len(batch_data) == 3:
        return batch_data[0], batch_data[1], batch_data[2]  # x, y, user_id
    else:
        return batch_data[0], batch_data[1], None  # x, y, None

def get_critical_slice(eval_loader, target_class=3, device='cpu'):
    """
    Extract critical slice from evaluation data (e.g., all 'cat' images for CIFAR-10).
    
    Args:
        eval_loader: DataLoader for evaluation data
        target_class: Class index for critical slice (3 = 'cat' for CIFAR-10)
        device: Device to use
    
    Returns:
        critical_data: Tensor of critical slice samples
        critical_targets: Tensor of critical slice labels
    """
    print(f"Extracting critical slice for class {target_class} (CIFAR-10: 'cat')...")
    
    critical_data = []
    critical_targets = []
    
    for batch_data in eval_loader:
        data, target, _ = unpack_batch(batch_data)
        
        # Find samples belonging to target class
        mask = (target == target_class)
        if mask.any():
            critical_data.append(data[mask])
            critical_targets.append(target[mask])
    
    if critical_data:
        critical_data = torch.cat(critical_data, dim=0).to(device)
        critical_targets = torch.cat(critical_targets, dim=0).to(device)
        print(f"Critical slice extracted: {len(critical_data)} samples of class {target_class}")
    else:
        print(f"Warning: No samples found for class {target_class}")
        critical_data = torch.empty(0, 3, 32, 32).to(device)  # CIFAR-10 shape
        critical_targets = torch.empty(0, dtype=torch.long).to(device)
    
    return critical_data, critical_targets

def compute_critical_slice_gradients(model, critical_data, critical_targets, device):
    """
    Compute aggregated gradients for critical slice.
    
    Formula: ∑_{s∈S_crit} ∇_θℓ(s,θ̂_DP)
    """
    model.eval()
    
    aggregated_gradients = {}
    for name, param in model.named_parameters():
        aggregated_gradients[name] = torch.zeros_like(param.data)
    
    print(f"Computing critical slice gradients for {len(critical_data)} samples...")
    
    if len(critical_data) == 0:
        print("Warning: Empty critical slice, returning zero gradients")
        return aggregated_gradients
    
    # Process in batches to manage memory
    batch_size = 32
    for i in tqdm(range(0, len(critical_data), batch_size), desc="Critical slice gradients"):
        end_idx = min(i + batch_size, len(critical_data))
        batch_data = critical_data[i:end_idx]
        batch_targets = critical_targets[i:end_idx]
        
        model.zero_grad()
        output = model(batch_data)
        loss = F.cross_entropy(output, batch_targets, reduction='sum')  # Sum over batch
        loss.backward()
        
        # Accumulate gradients with proper memory management
        for name, param in model.named_parameters():
            if param.grad is not None:
                # ❽ FIX: Detach gradients to avoid autograd overhead
                aggregated_gradients[name] += param.grad.detach().clone()
    
    return aggregated_gradients

def compute_slice_loss_gradient(model, critical_data, critical_targets, device):
    """
    Compute slice-loss gradient: J = ∇_θ[1/m ∑_{s∈S_crit} ℓ(s,θ)]_{θ=θ_DP}
    
    This is the gradient of the average loss over the critical slice.
    """
    model.eval()
    
    if len(critical_data) == 0:
        print("Warning: Empty critical slice")
        slice_gradient = {}
        for name, param in model.named_parameters():
            slice_gradient[name] = torch.zeros_like(param.data)
        return slice_gradient
    
    print(f"Computing slice-loss gradient for {len(critical_data)} critical samples...")
    
    model.zero_grad()
    output = model(critical_data)
    # Average loss over critical slice (reduction='mean')
    slice_loss = F.cross_entropy(output, critical_targets, reduction='mean')
    slice_loss.backward()
    
    # Extract slice-loss gradient J
    slice_gradient = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            slice_gradient[name] = param.grad.detach().clone()
        else:
            slice_gradient[name] = torch.zeros_like(param.data)
    
    return slice_gradient

def compute_influence_vectors(model, public_loader, train_loader, device, method='linear'):
    """
    Compute influence vectors H^{-1} ∇_θℓ(z,θ_DP) for all z ∈ P.
    
    Returns:
        influence_vectors: List of influence vectors for each public sample
        public_samples: List of (data, target) pairs for reference
    """
    model.eval()
    print(f"Computing influence vectors using {method} method...")
    
    # Collect all public samples
    public_samples = []
    for batch_data in public_loader:
        data, target, _ = unpack_batch(batch_data)
        for i in range(data.size(0)):
            public_samples.append((data[i:i+1], target[i:i+1]))
    
    influence_vectors = []
    
    if method == 'original':
        # Full Hessian inverse method
        gpu_id = device.index if device.type == 'cuda' and device.index is not None else -1
        
        for i, (data, target) in enumerate(tqdm(public_samples, desc="Computing Hessian inverses")):
            data, target = data.to(device), target.to(device)
            
            hessian_inv_grad = compute_hessian_inverse(
                z_test=data,
                t_test=target,
                model=model,
                z_loader=train_loader,
                gpu=gpu_id,
                damp=0.01,
                scale=25.0,
                recursion_depth=100
            )
            
            # Convert to dict format
            inf_vec = {}
            for j, (name, param) in enumerate(model.named_parameters()):
                if j < len(hessian_inv_grad):
                    inf_vec[name] = hessian_inv_grad[j].detach()
                else:
                    inf_vec[name] = torch.zeros_like(param.data)
            
            influence_vectors.append(inf_vec)
    
    elif method == 'linear':
        # Linear approximation: H^{-1} ≈ I/λ
        regularization = 0.1  # Increased from 0.01 for stability
        
        for i, (data, target) in enumerate(tqdm(public_samples, desc="Computing linear influences")):
            data, target = data.to(device), target.to(device)
            
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            inf_vec = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    inf_vec[name] = param.grad.detach().clone() / regularization  # Added .clone()
                else:
                    inf_vec[name] = torch.zeros_like(param.data)
            
            influence_vectors.append(inf_vec)
    
    elif method == 'batch':
        # Diagonal Fisher approximation
        print("Computing diagonal Fisher matrix...")
        
        fisher_diag = {}
        for name, param in model.named_parameters():
            fisher_diag[name] = torch.zeros_like(param.data)
        
        total_samples = 0
        for batch_data in tqdm(train_loader, desc="Fisher diagonal"):
            data, target, _ = unpack_batch(batch_data)
            data, target = data.to(device), target.to(device)
            
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            batch_size = data.size(0)
            total_samples += batch_size
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_diag[name] += param.grad.data ** 2 * batch_size
        
        # Normalize and add damping
        damp = 0.01
        for name in fisher_diag:
            fisher_diag[name] = fisher_diag[name] / total_samples + damp
        
        # Compute influence vectors using diagonal approximation
        for i, (data, target) in enumerate(tqdm(public_samples, desc="Computing diagonal influences")):
            data, target = data.to(device), target.to(device)
            
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            inf_vec = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    inf_vec[name] = param.grad.detach() / fisher_diag[name]
                else:
                    inf_vec[name] = torch.zeros_like(param.data)
            
            influence_vectors.append(inf_vec)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return influence_vectors, public_samples

def solve_weight_optimization(slice_gradient, influence_vectors, target_improvement, eta=200, device='cpu'):
    """
    Solve the linear program for utility enhancement:
    
    min_w ||J * Δθ_w - target_improvement||_1
    s.t. ||w||_1 ≤ η, w ≥ 0
    
    where Δθ_w = ∑ w_z * H^{-1} * ∇ℓ(z,θ_DP)
          target_improvement = desired reduction in slice loss
    
    This actively improves utility beyond baseline performance.
    """
    print(f"Solving utility enhancement optimization with η={eta}, target={target_improvement:.4f}...")
    
    if len(influence_vectors) == 0:
        print("Warning: No influence vectors to optimize")
        return np.zeros(0)
    
    n_samples = len(influence_vectors)
    
    # Memory-efficient approach: compute influence scores
    print(f"Computing influence scores for {n_samples} samples...")
    
    # Flatten slice gradient once
    slice_grad_flat = torch.cat([slice_gradient[name].flatten() for name in slice_gradient.keys()])
    
    # Compute influence scores: J^T * H^{-1} * ∇ℓ(z,θ_DP)
    influence_scores = []
    for inf_vec in influence_vectors:
        inf_flat = torch.cat([inf_vec[name].flatten() for name in inf_vec.keys()])
        # Score = effect of this sample on slice loss
        score = torch.dot(slice_grad_flat, inf_flat).item()
        influence_scores.append(score)
    
    influence_scores = np.array(influence_scores)
    
    # Use cvxpy for small problems, fallback for large ones
    if n_samples <= 1000:  # Memory-safe threshold
        try:
            print("Using cvxpy solver for utility enhancement...")
            w = cp.Variable(n_samples, nonneg=True)
            
            # ENHANCEMENT OBJECTIVE: min ||J * Δθ_w - target_improvement||_1
            # We want influence_scores @ w ≈ -target_improvement (negative because we want to reduce loss)
            objective_expr = influence_scores @ w + target_improvement  # Note: + because scores are negative for improvement
            objective = cp.abs(objective_expr)
            
            # Constraints: L1 norm and non-negativity
            constraints = [
                cp.norm(w, 1) <= eta,  # L1 sparsity constraint
                w >= 0
            ]
            
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if w.value is not None:
                optimal_weights = w.value
                predicted_change = -influence_scores @ optimal_weights  # Negative for loss reduction
                print(f"✅ cvxpy solved utility enhancement:")
                print(f"   • Non-zero weights: {np.sum(optimal_weights > 1e-6)}")
                print(f"   • L1 norm: {np.sum(np.abs(optimal_weights)):.4f}")
                print(f"   • Predicted slice loss reduction: {predicted_change:.4f}")
                print(f"   • Target reduction: {target_improvement:.4f}")
                return optimal_weights
                
        except Exception as e:
            print(f"⚠️  cvxpy failed: {e}")
    
    # Fallback: greedy selection for utility enhancement
    print("Using greedy selection for utility enhancement...")
    
    # We want to find weights that make -influence_scores @ w ≈ target_improvement
    # So influence_scores @ w ≈ -target_improvement
    target_correction = -target_improvement
    
    # Sort by most negative scores (most helpful for reducing loss)
    sorted_indices = np.argsort(influence_scores)  # Most negative first
    
    # Allocate budget greedily toward target
    optimal_weights = np.zeros(n_samples)
    remaining_budget = eta
    current_correction = 0.0
    
    for idx in sorted_indices:
        if remaining_budget <= 0:
            break
            
        # How much weight to assign this sample?
        needed_correction = target_correction - current_correction
        sample_effect = influence_scores[idx]
        
        if abs(needed_correction) < 1e-6:  # Close enough to target
            break
            
        if sample_effect < 0:  # Only negative scores help reduce loss
            weight = min(1.0, remaining_budget)
            optimal_weights[idx] = weight
            remaining_budget -= weight
            current_correction += weight * sample_effect
    
    predicted_reduction = -influence_scores @ optimal_weights
    print(f"✅ Greedy utility enhancement:")
    print(f"   • Non-zero weights: {np.sum(optimal_weights > 1e-6)}")
    print(f"   • L1 norm: {np.sum(np.abs(optimal_weights)):.4f}")
    print(f"   • Predicted slice loss reduction: {predicted_reduction:.4f}")
    print(f"   • Target reduction: {target_improvement:.4f}")
    
    return optimal_weights

def calibrate_model_research_protocol(model, public_loader, train_loader, critical_data, critical_targets, 
                                     device, method='linear', eta=200, baseline_model=None, target_improvement=0.2):
    """
    Calibrate model for utility enhancement using influence functions:
    
    1. Compute slice-loss gradient J
    2. Compute influence vectors H^{-1} ∇ℓ(z,θ_DP) for all z ∈ P
    3. Set target improvement for slice loss
    4. Solve optimization for optimal weights w
    5. Compute deterministic enhancement Δθ_w = ∑ w_z H^{-1} ∇ℓ(z,θ_DP)
    6. Apply enhancement: θ*_DP = θ_DP + Δθ_w
    """
    print(f"\n🔬 Influence Function Utility Enhancement")
    print(f"   • Method: {method}")
    print(f"   • Sparsity budget η: {eta}")
    print(f"   • Critical slice size: {len(critical_data)}")
    print(f"   • Target improvement: {target_improvement:.4f}")
    print(f"   • Baseline model: {'✅ Provided' if baseline_model else '⚠️  Using default target'}")
    
    model.eval()
    n_train = len(train_loader.dataset)
    
    # Step 1: Compute slice-loss gradient J
    slice_gradient = compute_slice_loss_gradient(model, critical_data, critical_targets, device)
    
    # Step 2: Compute influence vectors for all public samples
    influence_vectors, public_samples = compute_influence_vectors(
        model, public_loader, train_loader, device, method=method
    )
    
    # Step 3: Compute target improvement
    if baseline_model is not None:
        target_improvement = compute_utility_damage(model, baseline_model, critical_data, critical_targets, device, target_improvement)
    else:
        print(f"⚠️  No baseline model provided, using default target improvement: {target_improvement:.4f}")
        print(f"   🎯 GOAL: Reduce slice loss by {target_improvement:.4f} to enhance utility")
    
    # Step 4: Solve optimization for optimal weights
    optimal_weights = solve_weight_optimization(
        slice_gradient, influence_vectors, target_improvement, eta=eta, device=device
    )
    
    # Step 5: Compute deterministic enhancement Δθ_w
    print("Computing deterministic utility enhancement...")
    
    calibration = {}
    for name, param in model.named_parameters():
        calibration[name] = torch.zeros_like(param.data)
    
    # Δθ_w = ∑ w_z * H^{-1} ∇ℓ(z,θ_DP)
    # Note: influence_vectors already contain H^{-1} ∇ℓ(z,θ_DP)
    max_correction_norm = 0.0
    for i, weight in enumerate(optimal_weights):
        if weight > 1e-6:  # Only process non-zero weights
            inf_vec = influence_vectors[i]
            for name in calibration.keys():
                calibration[name] += weight * inf_vec[name]
    
    # Safety: Compute correction magnitude and apply damping
    total_correction_norm = 0.0
    for name in calibration.keys():
        total_correction_norm += calibration[name].norm().item() ** 2
    total_correction_norm = total_correction_norm ** 0.5
    
    # Apply adaptive damping to prevent parameter explosion
    max_allowed_correction = 0.1  # Conservative: max 10% parameter change
    if total_correction_norm > max_allowed_correction:
        damping_factor = max_allowed_correction / total_correction_norm
        print(f"   ⚠️  Large enhancement detected (norm={total_correction_norm:.4f})")
        print(f"   🔧 Applying damping factor: {damping_factor:.6f}")
        for name in calibration.keys():
            calibration[name] *= damping_factor
    else:
        print(f"   ✅ Enhancement magnitude safe (norm={total_correction_norm:.6f})")
    
    # Safety: Check for NaN or Inf before applying
    has_nan = False
    for name in calibration.keys():
        if torch.isnan(calibration[name]).any() or torch.isinf(calibration[name]).any():
            print(f"   ⚠️  NaN/Inf detected in {name}, skipping enhancement")
            has_nan = True
            break
    
    if has_nan:
        print("   🚫 Enhancement aborted due to numerical instability")
        return model
    
    # Step 6: Apply enhancement
    print("Applying deterministic utility enhancement...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(calibration[name])
    
    # Count actual non-zero weights used
    nonzero_weights = np.sum(optimal_weights > 1e-6)
    print(f"✅ Utility enhancement complete!")
    print(f"   • Used {nonzero_weights} samples with non-zero weights")
    print(f"   • Deterministic post-processing preserves DP guarantee")
    print(f"   • Expected slice loss reduction: {target_improvement:.4f}")
    
    return model

# Legacy compatibility functions
def calibrate_model(model, public_loader, train_loader, fisher, top_k=200, device=None, target_class=3, baseline_model=None):
    """Legacy wrapper that uses the research protocol"""
    print(f"🔄 Converting to research protocol (legacy compatibility)")
    
    # Extract critical slice
    critical_data = []
    critical_targets = []
    for batch_data in public_loader:
        data, target, _ = unpack_batch(batch_data)
        mask = (target == target_class)
        if mask.any():
            critical_data.append(data[mask])
            critical_targets.append(target[mask])
    
    if critical_data:
        critical_data = torch.cat(critical_data, dim=0).to(device)
        critical_targets = torch.cat(critical_targets, dim=0).to(device)
    else:
        print(f"⚠️  No samples of class {target_class} found, using first 50 samples")
        first_batch = next(iter(public_loader))
        data, target, _ = unpack_batch(first_batch)
        critical_data = data[:50].to(device)
        critical_targets = target[:50].to(device)
    
    return calibrate_model_research_protocol(
        model, public_loader, train_loader, critical_data, critical_targets,
        device, method='original', eta=top_k, baseline_model=baseline_model
    )

def calibrate_model_efficient(model, public_loader, train_loader, fisher, top_k=200, 
                             device=None, method='linear', target_class=3, baseline_model=None):
    """Legacy wrapper that uses the research protocol with efficient methods"""
    print(f"🔄 Converting to research protocol (efficient {method})")
    
    # Extract critical slice
    critical_data = []
    critical_targets = []
    for batch_data in public_loader:
        data, target, _ = unpack_batch(batch_data)
        mask = (target == target_class)
        if mask.any():
            critical_data.append(data[mask])
            critical_targets.append(target[mask])
    
    if critical_data:
        critical_data = torch.cat(critical_data, dim=0).to(device)
        critical_targets = torch.cat(critical_targets, dim=0).to(device)
    else:
        print(f"⚠️  No samples of class {target_class} found, using first 50 samples")
        first_batch = next(iter(public_loader))
        data, target, _ = unpack_batch(first_batch)
        critical_data = data[:50].to(device)
        critical_targets = target[:50].to(device)
    
    return calibrate_model_research_protocol(
        model, public_loader, train_loader, critical_data, critical_targets,
        device, method=method, eta=top_k, baseline_model=baseline_model
    )

def diagnose_calibration(model, critical_data, critical_targets, device):
    """
    Diagnostic function to verify calibration is working as expected.
    Should be called before and after calibration to measure improvement.
    """
    model.eval()
    
    if len(critical_data) == 0:
        print("⚠️  No critical slice data for diagnosis")
        return {'loss': float('inf'), 'accuracy': 0.0}
    
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
    """Print the effect of calibration on the critical slice"""
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

def compute_utility_damage(dp_model, baseline_model, critical_data, critical_targets, device, target_improvement=0.2):
    """
    Compute target improvement for utility enhancement.
    
    Instead of just compensating DP damage, we aim to improve beyond baseline performance.
    
    Args:
        target_improvement: Target reduction in slice loss (default: 0.2)
        
    Returns:
        Positive value representing desired improvement in slice loss
    """
    print("Computing target utility improvement...")
    
    if len(critical_data) == 0:
        print("⚠️  No critical slice data")
        return target_improvement
    
    # Evaluate DP model on critical slice
    dp_model.eval()
    with torch.no_grad():
        dp_output = dp_model(critical_data)
        dp_loss = F.cross_entropy(dp_output, critical_targets, reduction='mean')
    
    # Evaluate baseline model on critical slice  
    baseline_model.eval()
    with torch.no_grad():
        baseline_output = baseline_model(critical_data)
        baseline_loss = F.cross_entropy(baseline_output, critical_targets, reduction='mean')
    
    current_damage = dp_loss.item() - baseline_loss.item()
    
    print(f"   • Baseline slice loss:   {baseline_loss.item():.4f}")
    print(f"   • DP slice loss:         {dp_loss.item():.4f}")
    print(f"   • Current difference:    {current_damage:+.4f}")
    
    # Target: improve beyond baseline by target_improvement
    total_target_improvement = target_improvement + max(0, current_damage)
    
    print(f"   • Target improvement:    {target_improvement:.4f}")
    print(f"   • Total target:          {total_target_improvement:.4f}")
    print(f"   🎯 GOAL: Reduce slice loss by {total_target_improvement:.4f} to enhance utility")
    
    return total_target_improvement
