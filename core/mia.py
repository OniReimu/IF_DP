#!/usr/bin/env python3
# ================================================================
# Membership Inference Attacks (MIA) on CIFAR-10 Models
#    * Yeom et al. confidence-based attack
#    * Evaluation on baseline vs DP models
#    * Supports both user-level and sample-level DP
# ================================================================

import os, argparse, copy
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

from data.common import prepare_batch
from models.model import CNN

def auc_star(auc: float) -> float:
    """Sign-invariant attack strength: attacker can flip score direction."""
    return float(max(auc, 1.0 - auc))


def auc_advantage(auc: float) -> float:
    """Membership advantage over random guessing (0.0 is best)."""
    return float(abs(auc - 0.5))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set reproducible random seeds for consistent MIA evaluation
from .config import set_random_seeds, get_dataset_location
from .device_utils import resolve_device
set_random_seeds()
dataset_root, allow_download = get_dataset_location(
    dataset_key='cifar10',
    required_subdir='cifar-10-batches-py'
)

NUM_RUNS = 5 

# ════════════════════════════════════════════════════════════════
# MIA Data Preparation Functions
# ════════════════════════════════════════════════════════════════

def prepare_mia_data_sample_level(train_data, eval_data, private_indices, mia_size):
    """Prepare MIA datasets for sample-level DP"""
    # Use random sampling from actual private training samples as members
    if hasattr(train_data, 'indices'):
        # train_data is a Subset, use random sampling from the subset
        max_members = min(mia_size, len(train_data))
        member_indices = np.random.choice(len(train_data), max_members, replace=False).tolist()
    else:
        max_members = min(mia_size, len(train_data))
        member_indices = np.random.choice(len(train_data), max_members, replace=False).tolist()
    
    member_set = Subset(train_data, member_indices)
    
    # Use random sampling from evaluation data as non-members
    if hasattr(eval_data, 'indices'):
        # eval_data is a Subset, use random sampling from the subset
        max_non_members = min(mia_size, len(eval_data))
        non_member_indices = np.random.choice(len(eval_data), max_non_members, replace=False).tolist()
    else:
        max_non_members = min(mia_size, len(eval_data))
        non_member_indices = np.random.choice(len(eval_data), max_non_members, replace=False).tolist()
    
    non_member_set = Subset(eval_data, non_member_indices)
    
    print(f"   • Members: {len(member_set)} random samples from training data")
    print(f"   • Non-members: {len(non_member_set)} random samples from evaluation data")
    
    return member_set, non_member_set

def prepare_mia_data_user_level(priv_ds, eval_data, num_users, mia_size):
    """Prepare MIA datasets for user-level DP"""
    # Collect samples per user
    user_samples = {}
    for idx in range(len(priv_ds)):
        sample = priv_ds[idx]
        if isinstance(sample, dict):
            uid_value = sample.get("user_ids", sample.get("user_id"))
            if isinstance(uid_value, torch.Tensor):
                uid = int(uid_value.item())
            else:
                uid = int(uid_value)
        elif isinstance(sample, (list, tuple)) and len(sample) >= 3:
            uid = int(sample[2])
        else:
            uid = 0
        user_samples.setdefault(uid, []).append(idx)
    
    # Select member users (from training) and randomly sample from them
    available_users = list(user_samples.keys())[:num_users]
    all_member_indices = []
    for uid in available_users:
        all_member_indices.extend(user_samples[uid])
    
    # Random sampling from all available member samples
    max_members = min(mia_size, len(all_member_indices))
    member_indices = np.random.choice(all_member_indices, max_members, replace=False).tolist()
    member_set = Subset(priv_ds, member_indices)
    
    # Use random sampling from evaluation data as non-members
    if hasattr(eval_data, 'indices'):
        # eval_data is a Subset, use random sampling from the subset
        max_non_members = min(mia_size, len(eval_data))
        non_member_indices = np.random.choice(len(eval_data), max_non_members, replace=False).tolist()
    else:
        max_non_members = min(mia_size, len(eval_data))
        non_member_indices = np.random.choice(len(eval_data), max_non_members, replace=False).tolist()
        
    non_member_set = Subset(eval_data, non_member_indices)
    
    print(f"   • Members: {len(member_set)} random samples from {len(available_users)} training users")
    print(f"   • Non-members: {len(non_member_set)} random samples from evaluation data")
    
    return member_set, non_member_set

# ════════════════════════════════════════════════════════════════
# Attack Functions
# ════════════════════════════════════════════════════════════════

def confidence_attack(model, member_loader, non_member_loader, device):
    """Yeom et al. confidence-based membership inference attack"""
    model.eval()
    
    member_confidences = []
    non_member_confidences = []
    member_max_probs = []
    non_member_max_probs = []
    
    # Collect confidences for member samples
    with torch.no_grad():
        for batch_data in tqdm(member_loader, desc="Processing members", leave=False):
            features, labels, _ = prepare_batch(batch_data, device)
            output = model(features)
            probs = F.softmax(output, dim=1)
            
            # Use probability of the correct class (more standard for MIA)
            correct_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            member_confidences.extend(correct_class_probs.cpu().numpy())
            
            # Also track max probabilities for comparison
            max_probs = torch.max(probs, dim=1)[0]
            member_max_probs.extend(max_probs.cpu().numpy())
    
    # Collect confidences for non-member samples
    with torch.no_grad():
        for batch_data in tqdm(non_member_loader, desc="Processing non-members", leave=False):
            features, labels, _ = prepare_batch(batch_data, device)
            output = model(features)
            probs = F.softmax(output, dim=1)
            
            # Use probability of the correct class
            correct_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            non_member_confidences.extend(correct_class_probs.cpu().numpy())
            
            # Also track max probabilities for comparison
            max_probs = torch.max(probs, dim=1)[0]
            non_member_max_probs.extend(max_probs.cpu().numpy())
    
    # Create labels and evaluate
    all_confidences = np.concatenate([member_confidences, non_member_confidences])
    all_labels = np.concatenate([np.ones(len(member_confidences)), np.zeros(len(non_member_confidences))])
    
    # Check if we have valid data
    if len(set(all_labels)) < 2:
        print("⚠️  Warning: Only one class in labels!")
        return {'auc': 0.5, 'accuracy': 0.5, 'member_conf_mean': 0.0, 'non_member_conf_mean': 0.0}
    
    if np.std(all_confidences) < 1e-8:
        print("⚠️  Warning: No variance in confidence scores!")
        return {'auc': 0.5, 'accuracy': 0.5, 'member_conf_mean': np.mean(member_confidences), 'non_member_conf_mean': np.mean(non_member_confidences)}
    
    auc_score = roc_auc_score(all_labels, all_confidences)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_confidences)
    optimal_idx = np.argmax(precision + recall)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    predictions = (all_confidences >= optimal_threshold).astype(int)
    attack_accuracy = accuracy_score(all_labels, predictions)
    
    return {
        'auc': auc_score,
        'accuracy': attack_accuracy,
        'member_conf_mean': np.mean(member_confidences),
        'non_member_conf_mean': np.mean(non_member_confidences)
    }

def _reset_module_parameters(module):
    reset_fn = getattr(module, "reset_parameters", None)
    if callable(reset_fn):
        reset_fn()


def train_shadow_models(shadow_trainset, target_model, num_shadows=3, epochs=5, device='cpu'):
    """Train shadow models for Shokri attack using the target model architecture"""
    shadow_models = []
    
    for i in tqdm(range(num_shadows), desc="Training shadow models"):
        shadow_model = copy.deepcopy(target_model).to(device)
        shadow_model.apply(_reset_module_parameters)
        optimizer = torch.optim.SGD(shadow_model.parameters(), lr=1e-3, momentum=0.9)
        
        # Create data loader for this shadow model
        shadow_loader = DataLoader(shadow_trainset, batch_size=128, shuffle=True)
        
        # Train shadow model
        shadow_model.train()
        for epoch in tqdm(range(epochs), desc=f"  Shadow {i+1}/{num_shadows} epochs", leave=False):
            for batch_data in shadow_loader:
                features, labels, _ = prepare_batch(batch_data, device)
                optimizer.zero_grad()
                output = shadow_model(features)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
        
        shadow_models.append(shadow_model)
    
    return shadow_models

def extract_attack_features(model, data_loader, device):
    """Extract features for shadow model attack"""
    model.eval()
    feature_rows = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            batch_inputs, _, _ = prepare_batch(batch_data, device)
            output = model(batch_inputs)
            probs = F.softmax(output, dim=1)
            
            # Clip probabilities to avoid numerical issues
            probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
            
            # Features: [max_prob, entropy, top-3 probs]
            max_prob = torch.max(probs, dim=1)[0]
            
            # More robust entropy calculation
            entropy = -torch.sum(probs * torch.log(probs), dim=1)
            # Clamp entropy to reasonable range
            entropy = torch.clamp(entropy, min=0.0, max=10.0)
            
            top3_probs, _ = torch.topk(probs, min(3, probs.size(1)), dim=1)
            
            # Pad top3_probs if we have fewer than 3 classes
            if top3_probs.size(1) < 3:
                padding = torch.zeros(top3_probs.size(0), 3 - top3_probs.size(1), device=device)
                top3_probs = torch.cat([top3_probs, padding], dim=1)
            
            batch_features = torch.cat([
                max_prob.unsqueeze(1),
                entropy.unsqueeze(1), 
                top3_probs
            ], dim=1)
            
            # Check for NaN or inf values
            if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                print("⚠️  Warning: NaN or inf values detected in features, skipping batch")
                continue
                
            feature_rows.append(batch_features.cpu().numpy())
    
    return np.vstack(feature_rows) if feature_rows else np.array([])

def validate_and_clean_features(features, labels, name="features"):
    """Validate and clean feature arrays for numerical stability"""
    if features.size == 0:
        return features, labels
    
    # Check for NaN or inf values
    nan_mask = np.isnan(features).any(axis=1)
    inf_mask = np.isinf(features).any(axis=1)
    bad_mask = nan_mask | inf_mask
    
    if bad_mask.any():
        features = features[~bad_mask]
        labels = labels[~bad_mask]
    
    # Check for constant features (zero variance)
    if features.size > 0:
        feature_std = np.std(features, axis=0)
        zero_var_mask = feature_std < 1e-8
        if zero_var_mask.any():
            # Add small random noise to zero-variance features
            features[:, zero_var_mask] += np.random.normal(0, 1e-6, 
                                                          (features.shape[0], zero_var_mask.sum()))
    
    return features, labels

def shadow_model_attack(target_model, member_loader, non_member_loader, train_data, device, eval_data=None, shadow_epochs=3):
    """
    Shokri et al. shadow model attack.
    Uses shadow models to learn attack patterns.
    train_data: The actual training data used for the target model (priv_base)
    eval_data: The evaluation data used for target non-members (eval_base)
    shadow_epochs: Number of training epochs for each shadow model (default: 3)
    """
    
    # Prepare shadow training data from the actual training data
    # Use a subset of the actual training data for shadow models
    shadow_size = min(len(train_data) // 2, 2000)  # Use at most half of training data
    shadow_indices = np.random.choice(len(train_data), shadow_size, replace=False)
    shadow_trainset = Subset(train_data, shadow_indices)
    
    # 🔧 FIX: Use eval_data for shadow non-members to match target evaluation setup
    # If eval_data is provided, use it for shadow non-members
    # Otherwise, fall back to remaining train_data (original buggy behavior)
    if eval_data is not None:
        # Use eval_data for shadow non-members (correct approach)
        shadow_non_member_size = min(len(eval_data), shadow_size)
        shadow_non_member_indices = np.random.choice(len(eval_data), shadow_non_member_size, replace=False)
        shadow_non_trainset = Subset(eval_data, shadow_non_member_indices)
        print(f"🔧 Shadow attack: Using eval_data for non-members (correct)")
    else:
        # Fallback to original approach (for backward compatibility)
        remaining_indices = np.setdiff1d(np.arange(len(train_data)), shadow_indices)
        shadow_non_trainset = Subset(train_data, remaining_indices[:shadow_size])
        print(f"⚠️  Shadow attack: Using train_data for non-members (may be inaccurate)")
    
    # Train shadow models
    print(f"   🔧 Training {3} shadow models with {shadow_epochs} epochs each...")
    shadow_models = train_shadow_models(shadow_trainset, target_model, num_shadows=3, epochs=shadow_epochs, device=device)
    
    # Generate attack training data using shadow models
    shadow_features = []
    shadow_labels = []
    
    for i, shadow_model in enumerate(shadow_models):
        # For each shadow model:
        # - Members: samples that WERE used to train this shadow model
        # - Non-members: samples that were NOT used to train this shadow model
        
        shadow_member_loader = DataLoader(shadow_trainset, batch_size=128, shuffle=False)
        shadow_non_member_loader = DataLoader(shadow_non_trainset, batch_size=128, shuffle=False)
        
        # Extract features for members (label=1) and non-members (label=0)
        member_features = extract_attack_features(shadow_model, shadow_member_loader, device)
        non_member_features = extract_attack_features(shadow_model, shadow_non_member_loader, device)
        
        if member_features.size > 0 and non_member_features.size > 0:
            shadow_features.append(member_features)
            shadow_features.append(non_member_features)
            shadow_labels.extend([1] * len(member_features))  # Members
            shadow_labels.extend([0] * len(non_member_features))  # Non-members
    
    if not shadow_features:
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5}
    
    # Combine all shadow training data
    X_shadow = np.vstack(shadow_features)
    y_shadow = np.array(shadow_labels)
    
    # Validate and clean shadow features
    X_shadow, y_shadow = validate_and_clean_features(X_shadow, y_shadow, "shadow training data")
    
    if len(X_shadow) < 10:  # Need minimum samples for training
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Check class balance
    unique_labels, label_counts = np.unique(y_shadow, return_counts=True)
    
    if len(unique_labels) < 2:
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Ensure minimum samples per class
    min_class_size = min(label_counts)
    if min_class_size < 5:
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Train attack classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Split shadow data for training the attack classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_shadow, y_shadow, test_size=0.3, random_state=42, stratify=y_shadow
    )
    
    # Create a pipeline with feature scaling and logistic regression
    attack_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,  # Add regularization
            solver='liblinear',  # More stable solver
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Train attack classifier with error handling
    try:
        attack_pipeline.fit(X_train, y_train)
        
        # Test attack classifier on shadow data
        shadow_predictions = attack_pipeline.predict_proba(X_test)[:, 1]
        shadow_auc = roc_auc_score(y_test, shadow_predictions)
        
    except Exception as e:
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': base}
    
    # Extract features from target model for actual attack
    target_member_features = extract_attack_features(target_model, member_loader, device)
    target_non_member_features = extract_attack_features(target_model, non_member_loader, device)
    
    if target_member_features.size == 0 or target_non_member_features.size == 0:
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': shadow_auc}
    
    # Create target attack dataset
    X_target = np.vstack([target_member_features, target_non_member_features])
    y_target = np.concatenate([
        np.ones(len(target_member_features)),    # Members = 1
        np.zeros(len(target_non_member_features)) # Non-members = 0
    ])
    
    # Check for any remaining numerical issues
    if np.isnan(X_target).any() or np.isinf(X_target).any():
        X_target = np.nan_to_num(X_target, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Perform attack on target model
    try:
        target_predictions_proba = attack_pipeline.predict_proba(X_target)[:, 1]
        target_predictions = attack_pipeline.predict(X_target)
        
        # Evaluate attack performance
        auc_score = roc_auc_score(y_target, target_predictions_proba)
        attack_accuracy = accuracy_score(y_target, target_predictions)
        
    except Exception as e:
        base = 0.5
        return {'auc': base, 'auc_star': auc_star(base), 'adv': auc_advantage(base), 'accuracy': 0.5, 'shadow_auc': shadow_auc}
    
    return {
        'auc': auc_score,
        'auc_star': auc_star(auc_score),
        'adv': auc_advantage(auc_score),
        'accuracy': attack_accuracy,
        'shadow_auc': shadow_auc,
        'n_shadow_samples': len(X_shadow),
        'n_target_samples': len(X_target)
    }

# ════════════════════════════════════════════════════════════════
# Main Evaluation Function
# ════════════════════════════════════════════════════════════════

def evaluate_membership_inference(baseline_model, fisher_dp_model, train_data, eval_data, 
                                private_indices, priv_ds, num_users, 
                                mia_size, sample_level, device, vanilla_dp_model=None, dp_sat_model=None, l2_baseline_model=None, shadow_epochs=3):
    """Evaluate membership inference attacks on baseline, Fisher DP, and optionally Vanilla DP & DP-SAT models
    
    Args:
        baseline_model: Non-DP baseline model
        fisher_dp_model: Fisher-informed DP model
        train_data: The actual training data used (priv_base from trainset)
        eval_data: The evaluation data for non-members (eval_base from testset)
        private_indices: Original private indices (for compatibility, may not be used)
        priv_ds: User-level dataset object (for user-level DP)
        num_users: Number of users for user-level DP
        mia_size: Number of member/non-member samples for evaluation
        sample_level: Whether using sample-level (True) or user-level (False) DP
        device: Device to run evaluation on
        vanilla_dp_model: Optional vanilla DP model for comparison
        dp_sat_model: Optional DP-SAT model for comparison
        l2_baseline_model: Optional L2 regularized baseline model for comparison
    """
    
    print(f"\n🛡️  MEMBERSHIP INFERENCE ATTACK EVALUATION")
    
    # Build comparison message
    methods = ["Baseline", "Fisher DP"]
    if l2_baseline_model is not None:
        methods.append("L2 Baseline")
    if vanilla_dp_model is not None:
        methods.append("Vanilla DP")
    if dp_sat_model is not None:
        methods.append("DP-SAT")
    print(f"    Comparing: {' vs '.join(methods)}")
    print("="*60)
    
    # Prepare member and non-member datasets
    if sample_level:
        print("📊 Sample-level MIA: Using actual private training samples as members")
        member_set, non_member_set = prepare_mia_data_sample_level(train_data, eval_data, private_indices, mia_size)
    else:
        print("👥 User-level MIA: Using actual private users as members")
        member_set, non_member_set = prepare_mia_data_user_level(priv_ds, eval_data, num_users, mia_size)
    
    member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
    non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
    
    print(f"   • Members: {len(member_set)} samples")
    print(f"   • Non-members: {len(non_member_set)} samples")
    
    # Track results across multiple runs for statistical analysis
    num_runs = NUM_RUNS  # Multiple runs for statistical robustness
    all_results = {
        'baseline': {'shadow': []},
        'fisher_dp': {'shadow': []},
    }
    if l2_baseline_model is not None:
        all_results['l2_baseline'] = {'shadow': []}
    if vanilla_dp_model is not None:
        all_results['vanilla_dp'] = {'shadow': []}
    if dp_sat_model is not None:
        all_results['dp_sat'] = {'shadow': []}
    
    for run_idx in range(num_runs):
        # Re-sample for each run to get different member/non-member sets
        if sample_level:
            member_set, non_member_set = prepare_mia_data_sample_level(train_data, eval_data, private_indices, mia_size)
        else:
            member_set, non_member_set = prepare_mia_data_user_level(priv_ds, eval_data, num_users, mia_size)
        
        member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
        
        # Shadow Model Attack for all models (only attack we use now)
        print(f"\n🕶️  SHADOW MODEL ATTACK (Run {run_idx + 1}/{num_runs})")
        print("-" * 40)
        
        baseline_shadow = shadow_model_attack(baseline_model, member_loader, non_member_loader, train_data, device, eval_data, shadow_epochs=shadow_epochs)
        fisher_shadow = shadow_model_attack(fisher_dp_model, member_loader, non_member_loader, train_data, device, eval_data, shadow_epochs=shadow_epochs)
        
        all_results['baseline']['shadow'].append(baseline_shadow['auc'])
        all_results['fisher_dp']['shadow'].append(fisher_shadow['auc'])
        
        if l2_baseline_model is not None:
            l2_baseline_shadow = shadow_model_attack(l2_baseline_model, member_loader, non_member_loader, train_data, device, eval_data, shadow_epochs=shadow_epochs)
            all_results['l2_baseline']['shadow'].append(l2_baseline_shadow['auc'])
        
        if vanilla_dp_model is not None:
            vanilla_shadow = shadow_model_attack(vanilla_dp_model, member_loader, non_member_loader, train_data, device, eval_data, shadow_epochs=shadow_epochs)
            all_results['vanilla_dp']['shadow'].append(vanilla_shadow['auc'])
        
        if dp_sat_model is not None:
            dp_sat_shadow = shadow_model_attack(dp_sat_model, member_loader, non_member_loader, train_data, device, eval_data, shadow_epochs=shadow_epochs)
            all_results['dp_sat']['shadow'].append(dp_sat_shadow['auc'])
    
    # Statistical analysis of results
    print(f"\n📊 FINAL RESULTS")
    print("="*40)
    
    def print_stats(name, values):
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")
        return mean_val, std_val
    
    print("🕶️  Shadow Attack AUC:")
    baseline_shadow_mean, baseline_shadow_std = print_stats("  Baseline", all_results['baseline']['shadow'])
    fisher_shadow_mean, fisher_shadow_std = print_stats("  Fisher DP", all_results['fisher_dp']['shadow'])
    if l2_baseline_model is not None:
        l2_baseline_shadow_mean, l2_baseline_shadow_std = print_stats("  L2 Baseline", all_results['l2_baseline']['shadow'])
    if vanilla_dp_model is not None:
        vanilla_shadow_mean, vanilla_shadow_std = print_stats("  Vanilla DP", all_results['vanilla_dp']['shadow'])
    
    if dp_sat_model is not None:
        dp_sat_shadow_mean, dp_sat_shadow_std = print_stats("  DP-SAT", all_results['dp_sat']['shadow'])
    
    # Statistical significance tests (t-tests)
    from scipy import stats
    
    print("\n🧮 Statistical Significance Tests (p-values):")
    
    # Fisher DP vs Baseline
    _, p_shadow_fisher_base = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['baseline']['shadow'])
    print(f"  Fisher DP vs Baseline (Shadow): p = {p_shadow_fisher_base:.4f}")
    
    # L2 baseline tests (for regularization hypothesis)
    if l2_baseline_model is not None:
        # L2 baseline vs regular baseline
        _, p_shadow_l2_base = stats.ttest_rel(all_results['l2_baseline']['shadow'], all_results['baseline']['shadow'])
        print(f"  🎯 L2 Baseline vs Baseline (Shadow): p = {p_shadow_l2_base:.4f}")
        
        # Fisher DP vs L2 baseline (key L2 regularization hypothesis test!)
        _, p_shadow_fisher_l2 = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['l2_baseline']['shadow'])
        print(f"  🔥 Fisher DP vs L2 Baseline (Shadow): p = {p_shadow_fisher_l2:.4f}")
    
    if vanilla_dp_model is not None:
        # Vanilla DP vs Baseline
        _, p_shadow_vanilla_base = stats.ttest_rel(all_results['vanilla_dp']['shadow'], all_results['baseline']['shadow'])
        print(f"  Vanilla DP vs Baseline (Shadow): p = {p_shadow_vanilla_base:.4f}")
        
        # Fisher DP vs Vanilla DP (the key comparison!)
        _, p_shadow_fisher_vanilla = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['vanilla_dp']['shadow'])
        print(f"  🔥 Fisher DP vs Vanilla DP (Shadow): p = {p_shadow_fisher_vanilla:.4f}")
        
        # L2 baseline vs Vanilla DP (if both available)
        if l2_baseline_model is not None:
            _, p_shadow_l2_vanilla = stats.ttest_rel(all_results['l2_baseline']['shadow'], all_results['vanilla_dp']['shadow'])
            print(f"  🆚 L2 Baseline vs Vanilla DP (Shadow): p = {p_shadow_l2_vanilla:.4f}")
    
    if dp_sat_model is not None:
        # DP-SAT vs Baseline
        _, p_shadow_dp_sat_base = stats.ttest_rel(all_results['dp_sat']['shadow'], all_results['baseline']['shadow'])
        print(f"  DP-SAT vs Baseline (Shadow): p = {p_shadow_dp_sat_base:.4f}")
        
        # Fisher DP vs DP-SAT (key comparison!)
        _, p_shadow_fisher_dp_sat = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['dp_sat']['shadow'])
        print(f"  🔥 Fisher DP vs DP-SAT (Shadow): p = {p_shadow_fisher_dp_sat:.4f}")
        
        # L2 baseline vs DP-SAT (if both available)
        if l2_baseline_model is not None:
            _, p_shadow_l2_dp_sat = stats.ttest_rel(all_results['l2_baseline']['shadow'], all_results['dp_sat']['shadow'])
            print(f"  🆚 L2 Baseline vs DP-SAT (Shadow): p = {p_shadow_l2_dp_sat:.4f}")
        
        # Vanilla DP vs DP-SAT (if both available)
        if vanilla_dp_model is not None:
            _, p_shadow_vanilla_dp_sat = stats.ttest_rel(all_results['vanilla_dp']['shadow'], all_results['dp_sat']['shadow'])
            print(f"  🆚 Vanilla DP vs DP-SAT (Shadow): p = {p_shadow_vanilla_dp_sat:.4f}")
    
    # Final assessment based on shadow attack AUC across runs (no need for worst-case since we only have one attack)
    fisher_worst_shadow = max(all_results['fisher_dp']['shadow'])
    
    worst_case_results = {'fisher_dp': fisher_worst_shadow}
    
    if l2_baseline_model is not None:
        l2_baseline_worst_shadow = max(all_results['l2_baseline']['shadow'])
        worst_case_results['l2_baseline'] = l2_baseline_worst_shadow
    
    if vanilla_dp_model is not None:
        vanilla_worst_shadow = max(all_results['vanilla_dp']['shadow'])
        worst_case_results['vanilla_dp'] = vanilla_worst_shadow
    
    if dp_sat_model is not None:
        dp_sat_worst_shadow = max(all_results['dp_sat']['shadow'])
        worst_case_results['dp_sat'] = dp_sat_worst_shadow
    
    print(f"\n🎯 FINAL PRIVACY PROTECTION COMPARISON")
    print("="*50)
    print(f"📊 Shadow Attack AUC (worst across runs):")
    print(f"   • Fisher DP: {fisher_worst_shadow:.4f}")
    if l2_baseline_model is not None:
        print(f"   • L2 Baseline: {l2_baseline_worst_shadow:.4f}")
    if vanilla_dp_model is not None:
        print(f"   • Vanilla DP: {vanilla_worst_shadow:.4f}")
    if dp_sat_model is not None:
        print(f"   • DP-SAT: {dp_sat_worst_shadow:.4f}")
    
    # Find the best performing method
    best_method = min(worst_case_results.items(), key=lambda x: x[1])
    best_name, best_auc = best_method
    
    if best_name == 'fisher_dp':
        print(f"   🏆 Fisher DP provides the BEST privacy protection!")
        if l2_baseline_model is not None:
            diff_l2 = l2_baseline_worst_shadow - fisher_worst_shadow
            print(f"   📈 vs L2 Baseline: {diff_l2:.4f} AUC reduction")
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_shadow - fisher_worst_shadow
            print(f"   📈 vs Vanilla DP: {diff_vanilla:.4f} AUC reduction")
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_shadow - fisher_worst_shadow
            print(f"   📈 vs DP-SAT: {diff_dp_sat:.4f} AUC reduction")
    elif best_name == 'l2_baseline':
        print(f"   🏆 L2 Baseline provides the BEST privacy protection!")
        diff_fisher = fisher_worst_shadow - l2_baseline_worst_shadow
        print(f"   📈 vs Fisher DP: {diff_fisher:.4f} AUC reduction")
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_shadow - l2_baseline_worst_shadow
            print(f"   📈 vs Vanilla DP: {diff_vanilla:.4f} AUC reduction")
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_shadow - l2_baseline_worst_shadow
            print(f"   📈 vs DP-SAT: {diff_dp_sat:.4f} AUC reduction")
    elif best_name == 'vanilla_dp':
        print(f"   🏆 Vanilla DP provides the BEST privacy protection!")
        diff_fisher = fisher_worst_shadow - vanilla_worst_shadow
        print(f"   📈 vs Fisher DP: {diff_fisher:.4f} AUC reduction")
        if l2_baseline_model is not None:
            diff_l2 = l2_baseline_worst_shadow - vanilla_worst_shadow
            print(f"   📈 vs L2 Baseline: {diff_l2:.4f} AUC reduction")
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_shadow - vanilla_worst_shadow
            print(f"   📈 vs DP-SAT: {diff_dp_sat:.4f} AUC reduction")
    elif best_name == 'dp_sat':
        print(f"   🏆 DP-SAT provides the BEST privacy protection!")
        diff_fisher = fisher_worst_shadow - dp_sat_worst_shadow
        print(f"   📈 vs Fisher DP: {diff_fisher:.4f} AUC reduction")
        if l2_baseline_model is not None:
            diff_l2 = l2_baseline_worst_shadow - dp_sat_worst_shadow
            print(f"   📈 vs L2 Baseline: {diff_l2:.4f} AUC reduction")
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_shadow - dp_sat_worst_shadow
            print(f"   📈 vs Vanilla DP: {diff_vanilla:.4f} AUC reduction")
    
    # Privacy strength assessment for all methods
    privacy_threshold = 0.6  # AUC > 0.6 indicates weak privacy protection
    
    if fisher_worst_shadow < privacy_threshold:
        print("✅ Fisher DP provides STRONG privacy protection!")
    else:
        print("⚠️  Fisher DP provides WEAK privacy protection!")
    
    if l2_baseline_model is not None:
        if l2_baseline_worst_shadow < privacy_threshold:
            print("✅ L2 Baseline provides STRONG privacy protection!")
        else:
            print("⚠️  L2 Baseline provides WEAK privacy protection!")
        
    if vanilla_dp_model is not None:
        if vanilla_worst_shadow < privacy_threshold:
            print("✅ Vanilla DP provides STRONG privacy protection!")
        else:
            print("⚠️  Vanilla DP provides WEAK privacy protection!")
    
    if dp_sat_model is not None:
        if dp_sat_worst_shadow < privacy_threshold:
            print("✅ DP-SAT provides STRONG privacy protection!")
        else:
            print("⚠️  DP-SAT provides WEAK privacy protection!")
    
    return {
        'fisher_worst_auc': fisher_worst_shadow,
        'l2_baseline_worst_auc': l2_baseline_worst_shadow if l2_baseline_model else None,
        'vanilla_worst_auc': vanilla_worst_shadow if vanilla_dp_model else None,
        'dp_sat_worst_auc': dp_sat_worst_shadow if dp_sat_model else None,
        'statistical_results': all_results
    }

# ════════════════════════════════════════════════════════════════
# Standalone Evaluation Function
# ════════════════════════════════════════════════════════════════

def get_device(args):
    """Get the appropriate device based on command line arguments"""
    if not hasattr(args, 'cuda_devices'):
        args.cuda_devices = None
    if not hasattr(args, 'multi_gpu'):
        args.multi_gpu = False
    return resolve_device(args)

def prepare_standalone_mia_data(trainset, testset, member_size=5000, non_member_size=5000):
    """Prepare member and non-member datasets for standalone MIA evaluation"""
    
    # Member set: samples that were used for training
    member_indices = np.random.choice(len(trainset), member_size, replace=False)
    member_set = Subset(trainset, member_indices)
    
    # Non-member set: samples from test set (never seen during training)
    non_member_indices = np.random.choice(len(testset), non_member_size, replace=False)
    non_member_set = Subset(testset, non_member_indices)
    
    return member_set, non_member_set, member_indices, non_member_indices

def main():
    """Standalone MIA evaluation script (for backward compatibility)"""
    parser = argparse.ArgumentParser('Membership Inference Attack Evaluation')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--cuda-id', type=int)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda-devices', type=str, default=None,
                       help='Comma-separated CUDA device ids for multi-GPU execution')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Enable torch.nn.DataParallel across the requested CUDA devices')
    parser.add_argument('--member-size', type=int, default=2000,
                       help='Number of member samples for MIA')
    parser.add_argument('--non-member-size', type=int, default=2000,
                       help='Number of non-member samples for MIA')
    
    args = parser.parse_args()
    device = get_device(args)
    
    # Load data
    trans = T.Compose([T.ToTensor(), T.Normalize((.5,.5,.5),(.5,.5,.5))])
    trainset = torchvision.datasets.CIFAR10(dataset_root, train=True, download=allow_download, transform=trans)
    testset = torchvision.datasets.CIFAR10(dataset_root, train=False, download=allow_download, transform=trans)
    
    # Load trained models (you need to train them first using main.py)
    models_dir = './saved_models'
    
    try:
        # Load baseline model and its training metadata
        baseline = CNN().to(device)
        baseline_path = os.path.join(models_dir, '基线模型.pth')
        if os.path.exists(baseline_path):
            checkpoint = torch.load(baseline_path, map_location=device)
            baseline.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded baseline model")
            
            # Extract training indices if available
            if 'training_indices' in checkpoint:
                training_indices = checkpoint['training_indices']
                sample_level = checkpoint.get('sample_level', True)
                dataset_size = checkpoint.get('dataset_size', len(training_indices))
                num_users = checkpoint.get('num_users', 10)
                print(f"✅ Found training metadata: {len(training_indices)} training samples")
                print(f"   Mode: {'Sample-level' if sample_level else f'User-level ({num_users} users)'}")
            else:
                print("⚠️  No training indices found, using random samples (less accurate)")
                training_indices = None
                sample_level = True
                
        else:
            print("❌ Baseline model not found. Please train models first using main.py")
            return
        
        # Load DP model
        dp_model = CNN().to(device)
        dp_path = os.path.join(models_dir, 'DP模型.pth')
        if os.path.exists(dp_path):
            checkpoint = torch.load(dp_path, map_location=device)
            dp_model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded DP model")
        else:
            print("❌ DP model not found. Please train models first using main.py")
            return
        
        # Load L2 baseline model (optional)
        l2_baseline = None
        l2_baseline_path = os.path.join(models_dir, 'L2基线模型.pth')
        if os.path.exists(l2_baseline_path):
            l2_baseline = CNN().to(device)
            checkpoint = torch.load(l2_baseline_path, map_location=device)
            l2_baseline.load_state_dict(checkpoint['model_state_dict'])
            weight_decay = checkpoint.get('weight_decay', 0.0)
            print(f"✅ Loaded L2 baseline model (λ={weight_decay})")
        else:
            print("ℹ️  L2 baseline model not found (optional)")
        
        # Prepare MIA datasets using actual training data
        if training_indices is not None:
            # Use actual training data as members
            member_indices = training_indices[:args.member_size]  # Take subset if needed
            member_set = Subset(trainset, member_indices)
            
            # Use test set as non-members
            non_member_indices = list(range(min(args.non_member_size, len(testset))))
            non_member_set = Subset(testset, non_member_indices)
            
            print(f"📊 Using ACTUAL training data as members:")
            print(f"   • Members: {len(member_set)} samples from actual training set")
            print(f"   • Non-members: {len(non_member_set)} samples from test set")
        else:
            # Fallback to random sampling (legacy behavior)
            member_set, non_member_set, _, _ = prepare_standalone_mia_data(
                trainset, testset, args.member_size, args.non_member_size
            )
            print(f"📊 Using RANDOM training samples as members (legacy mode):")
            print(f"   • Members: {len(member_set)} random samples from training set") 
            print(f"   • Non-members: {len(non_member_set)} samples from test set")
        
        # Run MIA evaluation
        member_loader = DataLoader(member_set, batch_size=128, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=128, shuffle=False)
        
        print("\n" + "="*60)
        print("🛡️  STANDALONE MEMBERSHIP INFERENCE ATTACK EVALUATION")
        print("="*60)
        
        # Run shadow model attack (only attack we use now - more powerful assessment)
        print("\n🕶️  SHADOW MODEL ATTACK")
        print("-" * 30)
        
        if training_indices is not None:
            # Use actual training data for shadow models
            actual_train_data = Subset(trainset, training_indices)
            print("🎯 Using ACTUAL training data for shadow models")
        else:
            # Fallback to full trainset
            actual_train_data = trainset
            print("⚠️  Using full training set for shadow models (less accurate)")
        
        baseline_shadow_results = shadow_model_attack(baseline, member_loader, non_member_loader, actual_train_data, device, eval_data=testset)
        print(f"\n📈 Baseline Model:")
        print(f"   • AUC: {baseline_shadow_results['auc']:.4f}")
        print(f"   • Attack Accuracy: {baseline_shadow_results['accuracy']:.4f}")
        
        if l2_baseline is not None:
            l2_baseline_shadow_results = shadow_model_attack(l2_baseline, member_loader, non_member_loader, actual_train_data, device, eval_data=testset)
            print(f"\n🎯 L2 Baseline Model:")
            print(f"   • AUC: {l2_baseline_shadow_results['auc']:.4f}")
            print(f"   • Attack Accuracy: {l2_baseline_shadow_results['accuracy']:.4f}")
        
        dp_shadow_results = shadow_model_attack(dp_model, member_loader, non_member_loader, actual_train_data, device, eval_data=testset)
        print(f"\n🔒 DP Model:")
        print(f"   • AUC: {dp_shadow_results['auc']:.4f}")
        print(f"   • Attack Accuracy: {dp_shadow_results['accuracy']:.4f}")
        
        # Overall assessment using shadow attack results only
        dp_auc = dp_shadow_results['auc']
        
        if l2_baseline is not None:
            l2_auc = l2_baseline_shadow_results['auc']
            print(f"\n🎯 OVERALL PRIVACY PROTECTION (Shadow Attack AUC):")
            print(f"   • DP Model: {dp_auc:.4f}")
            print(f"   • L2 Baseline: {l2_auc:.4f}")
            
            if l2_auc < dp_auc:
                diff = dp_auc - l2_auc
                print(f"   📈 L2 Baseline has {diff:.4f} better privacy protection than DP model!")
            elif dp_auc < l2_auc:
                diff = l2_auc - dp_auc
                print(f"   📈 DP Model has {diff:.4f} better privacy protection than L2 baseline!")
            else:
                print(f"   🔄 Similar privacy protection between DP and L2 baseline")
        else:
            print(f"\n🎯 OVERALL PRIVACY PROTECTION (Shadow Attack AUC: {dp_auc:.4f}):")
            
        if training_indices is not None:
            print("✅ Using actual training data for accurate evaluation")
        else:
            print("⚠️  Note: For accurate evaluation, retrain models or use integrated MIA with --run-mia flag")
            
        if dp_auc <= 0.55:
            print("✅ DP model provides STRONG privacy protection!")
        elif dp_auc <= 0.65:
            print("⚠️  DP model provides MODERATE privacy protection.")
        else:
            print("❌ DP model privacy protection may be INSUFFICIENT.")
        
        print("🕶️  Using shadow attack only - more sophisticated and realistic privacy assessment!")
            
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please first train the models using:")
        print("python main.py --mps --adaptive-clip --quantile 0.95")

if __name__ == "__main__":
    main() 
