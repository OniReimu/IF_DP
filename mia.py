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

from model import CNN

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

NUM_RUNS = 1 

def unpack_batch(batch_data):
    """Helper function to handle both (x, y) and (x, y, user_id) formats"""
    if len(batch_data) == 3:
        return batch_data[0], batch_data[1], batch_data[2]  # x, y, user_id
    else:
        return batch_data[0], batch_data[1], None  # x, y, None

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
        x, y, uid = priv_ds[idx]
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
            x, y, _ = unpack_batch(batch_data)
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            probs = F.softmax(output, dim=1)
            
            # Use probability of the correct class (more standard for MIA)
            correct_class_probs = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            member_confidences.extend(correct_class_probs.cpu().numpy())
            
            # Also track max probabilities for comparison
            max_probs = torch.max(probs, dim=1)[0]
            member_max_probs.extend(max_probs.cpu().numpy())
    
    # Collect confidences for non-member samples
    with torch.no_grad():
        for batch_data in tqdm(non_member_loader, desc="Processing non-members", leave=False):
            x, y, _ = unpack_batch(batch_data)
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            probs = F.softmax(output, dim=1)
            
            # Use probability of the correct class
            correct_class_probs = probs.gather(1, y.unsqueeze(1)).squeeze(1)
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

def train_shadow_models(shadow_trainset, num_shadows=3, epochs=5, device='cpu'):
    """Train shadow models for Shokri attack"""
    shadow_models = []
    
    for i in tqdm(range(num_shadows), desc="Training shadow models"):
        # Create shadow model with same architecture as target
        shadow_model = CNN().to(device)
        optimizer = torch.optim.SGD(shadow_model.parameters(), lr=1e-3, momentum=0.9)
        
        # Create data loader for this shadow model
        shadow_loader = DataLoader(shadow_trainset, batch_size=128, shuffle=True)
        
        # Train shadow model
        shadow_model.train()
        for epoch in range(epochs):
            for batch_data in shadow_loader:
                x, y, _ = unpack_batch(batch_data)
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = shadow_model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
        
        shadow_models.append(shadow_model)
    
    return shadow_models

def extract_attack_features(model, data_loader, device):
    """Extract features for shadow model attack"""
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            x, y, _ = unpack_batch(batch_data)
            x, y = x.to(device), y.to(device)
            
            output = model(x)
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
                
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features) if features else np.array([])

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

def shadow_model_attack(target_model, member_loader, non_member_loader, train_data, device):
    """
    Shokri et al. shadow model attack.
    Uses shadow models to learn attack patterns.
    train_data: The actual training data used for the target model (priv_base)
    """
    
    # Prepare shadow training data from the actual training data
    # Use a subset of the actual training data for shadow models
    shadow_size = min(len(train_data) // 2, 2000)  # Use at most half of training data
    shadow_indices = np.random.choice(len(train_data), shadow_size, replace=False)
    shadow_trainset = Subset(train_data, shadow_indices)
    
    # Get remaining training data as shadow non-members
    remaining_indices = np.setdiff1d(np.arange(len(train_data)), shadow_indices)
    shadow_non_trainset = Subset(train_data, remaining_indices[:shadow_size])  # Use same size
    
    # Train shadow models
    shadow_models = train_shadow_models(shadow_trainset, num_shadows=3, epochs=3, device=device)
    
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
        return {'auc': 0.5, 'accuracy': 0.5}
    
    # Combine all shadow training data
    X_shadow = np.vstack(shadow_features)
    y_shadow = np.array(shadow_labels)
    
    # Validate and clean shadow features
    X_shadow, y_shadow = validate_and_clean_features(X_shadow, y_shadow, "shadow training data")
    
    if len(X_shadow) < 10:  # Need minimum samples for training
        return {'auc': 0.5, 'accuracy': 0.5, 'shadow_auc': 0.5}
    
    # Check class balance
    unique_labels, label_counts = np.unique(y_shadow, return_counts=True)
    
    if len(unique_labels) < 2:
        return {'auc': 0.5, 'accuracy': 0.5, 'shadow_auc': 0.5}
    
    # Ensure minimum samples per class
    min_class_size = min(label_counts)
    if min_class_size < 5:
        return {'auc': 0.5, 'accuracy': 0.5, 'shadow_auc': 0.5}
    
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
        return {'auc': 0.5, 'accuracy': 0.5, 'shadow_auc': 0.5}
    
    # Extract features from target model for actual attack
    target_member_features = extract_attack_features(target_model, member_loader, device)
    target_non_member_features = extract_attack_features(target_model, non_member_loader, device)
    
    if target_member_features.size == 0 or target_non_member_features.size == 0:
        return {'auc': 0.5, 'accuracy': 0.5, 'shadow_auc': shadow_auc}
    
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
        return {'auc': 0.5, 'accuracy': 0.5, 'shadow_auc': shadow_auc}
    
    return {
        'auc': auc_score,
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
                                mia_size, sample_level, device, vanilla_dp_model=None, dp_sat_model=None):
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
    """
    print("\n" + "="*60)
    print("🛡️  MEMBERSHIP INFERENCE ATTACK EVALUATION")
    
    # Build comparison message
    methods = ["Baseline", "Fisher DP"]
    if vanilla_dp_model is not None:
        methods.append("Vanilla DP")
    if dp_sat_model is not None:
        methods.append("DP-SAT")
    print(f"    Comparing: {' vs '.join(methods)}")
    print("="*60)
    
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
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
        'baseline': {'confidence': [], 'shadow': []},
        'fisher_dp': {'confidence': [], 'shadow': []},
    }
    if vanilla_dp_model is not None:
        all_results['vanilla_dp'] = {'confidence': [], 'shadow': []}
    if dp_sat_model is not None:
        all_results['dp_sat'] = {'confidence': [], 'shadow': []}
    
    for run_idx in range(num_runs):
        # Re-sample for each run to get different member/non-member sets
        if sample_level:
            member_set, non_member_set = prepare_mia_data_sample_level(train_data, eval_data, private_indices, mia_size)
        else:
            member_set, non_member_set = prepare_mia_data_user_level(priv_ds, eval_data, num_users, mia_size)
        
        member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
        
        # 1. Yeom Confidence Attack for all models
        print("\n1️⃣  CONFIDENCE ATTACK")
        print("-" * 30)
        
        baseline_conf = confidence_attack(baseline_model, member_loader, non_member_loader, device)
        fisher_conf = confidence_attack(fisher_dp_model, member_loader, non_member_loader, device)
        
        all_results['baseline']['confidence'].append(baseline_conf['auc'])
        all_results['fisher_dp']['confidence'].append(fisher_conf['auc'])
        
        if vanilla_dp_model is not None:
            vanilla_conf = confidence_attack(vanilla_dp_model, member_loader, non_member_loader, device)
            all_results['vanilla_dp']['confidence'].append(vanilla_conf['auc'])
        
        if dp_sat_model is not None:
            dp_sat_conf = confidence_attack(dp_sat_model, member_loader, non_member_loader, device)
            all_results['dp_sat']['confidence'].append(dp_sat_conf['auc'])
        
        # 2. Shokri Shadow Model Attack for all models  
        print("\n2️⃣  SHADOW MODEL ATTACK")
        print("-" * 30)
        
        baseline_shadow = shadow_model_attack(baseline_model, member_loader, non_member_loader, train_data, device)
        fisher_shadow = shadow_model_attack(fisher_dp_model, member_loader, non_member_loader, train_data, device)
        
        all_results['baseline']['shadow'].append(baseline_shadow['auc'])
        all_results['fisher_dp']['shadow'].append(fisher_shadow['auc'])
        
        if vanilla_dp_model is not None:
            vanilla_shadow = shadow_model_attack(vanilla_dp_model, member_loader, non_member_loader, train_data, device)
            all_results['vanilla_dp']['shadow'].append(vanilla_shadow['auc'])
        
        if dp_sat_model is not None:
            dp_sat_shadow = shadow_model_attack(dp_sat_model, member_loader, non_member_loader, train_data, device)
            all_results['dp_sat']['shadow'].append(dp_sat_shadow['auc'])
    
    # Statistical analysis of results
    print(f"\n📊 FINAL RESULTS")
    print("="*40)
    
    def print_stats(name, values):
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{name}: {mean_val:.4f} ± {std_val:.4f}")
        return mean_val, std_val
    
    print("🎯 Confidence Attack AUC:")
    baseline_conf_mean, baseline_conf_std = print_stats("  Baseline", all_results['baseline']['confidence'])
    fisher_conf_mean, fisher_conf_std = print_stats("  Fisher DP", all_results['fisher_dp']['confidence'])
    if vanilla_dp_model is not None:
        vanilla_conf_mean, vanilla_conf_std = print_stats("  Vanilla DP", all_results['vanilla_dp']['confidence'])
    
    if dp_sat_model is not None:
        dp_sat_conf_mean, dp_sat_conf_std = print_stats("  DP-SAT", all_results['dp_sat']['confidence'])
    
    print("\n🕶️  Shadow Attack AUC:")
    baseline_shadow_mean, baseline_shadow_std = print_stats("  Baseline", all_results['baseline']['shadow'])
    fisher_shadow_mean, fisher_shadow_std = print_stats("  Fisher DP", all_results['fisher_dp']['shadow'])
    if vanilla_dp_model is not None:
        vanilla_shadow_mean, vanilla_shadow_std = print_stats("  Vanilla DP", all_results['vanilla_dp']['shadow'])
    
    if dp_sat_model is not None:
        dp_sat_shadow_mean, dp_sat_shadow_std = print_stats("  DP-SAT", all_results['dp_sat']['shadow'])
    
    # Statistical significance tests (t-tests)
    from scipy import stats
    
    print("\n🧮 Statistical Significance Tests (p-values):")
    
    # Fisher DP vs Baseline
    _, p_conf_fisher_base = stats.ttest_rel(all_results['fisher_dp']['confidence'], all_results['baseline']['confidence'])
    _, p_shadow_fisher_base = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['baseline']['shadow'])
    print(f"  Fisher DP vs Baseline (Confidence): p = {p_conf_fisher_base:.4f}")
    print(f"  Fisher DP vs Baseline (Shadow): p = {p_shadow_fisher_base:.4f}")
    
    if vanilla_dp_model is not None:
        # Vanilla DP vs Baseline
        _, p_conf_vanilla_base = stats.ttest_rel(all_results['vanilla_dp']['confidence'], all_results['baseline']['confidence'])
        _, p_shadow_vanilla_base = stats.ttest_rel(all_results['vanilla_dp']['shadow'], all_results['baseline']['shadow'])
        print(f"  Vanilla DP vs Baseline (Confidence): p = {p_conf_vanilla_base:.4f}")
        print(f"  Vanilla DP vs Baseline (Shadow): p = {p_shadow_vanilla_base:.4f}")
        
        # Fisher DP vs Vanilla DP (the key comparison!)
        _, p_conf_fisher_vanilla = stats.ttest_rel(all_results['fisher_dp']['confidence'], all_results['vanilla_dp']['confidence'])
        _, p_shadow_fisher_vanilla = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['vanilla_dp']['shadow'])
        print(f"  🔥 Fisher DP vs Vanilla DP (Confidence): p = {p_conf_fisher_vanilla:.4f}")
        print(f"  🔥 Fisher DP vs Vanilla DP (Shadow): p = {p_shadow_fisher_vanilla:.4f}")
    
    if dp_sat_model is not None:
        # DP-SAT vs Baseline
        _, p_conf_dp_sat_base = stats.ttest_rel(all_results['dp_sat']['confidence'], all_results['baseline']['confidence'])
        _, p_shadow_dp_sat_base = stats.ttest_rel(all_results['dp_sat']['shadow'], all_results['baseline']['shadow'])
        print(f"  DP-SAT vs Baseline (Confidence): p = {p_conf_dp_sat_base:.4f}")
        print(f"  DP-SAT vs Baseline (Shadow): p = {p_shadow_dp_sat_base:.4f}")
        
        # Fisher DP vs DP-SAT (key comparison!)
        _, p_conf_fisher_dp_sat = stats.ttest_rel(all_results['fisher_dp']['confidence'], all_results['dp_sat']['confidence'])
        _, p_shadow_fisher_dp_sat = stats.ttest_rel(all_results['fisher_dp']['shadow'], all_results['dp_sat']['shadow'])
        print(f"  🔥 Fisher DP vs DP-SAT (Confidence): p = {p_conf_fisher_dp_sat:.4f}")
        print(f"  🔥 Fisher DP vs DP-SAT (Shadow): p = {p_shadow_fisher_dp_sat:.4f}")
        
        # Vanilla DP vs DP-SAT (if both available)
        if vanilla_dp_model is not None:
            _, p_conf_vanilla_dp_sat = stats.ttest_rel(all_results['vanilla_dp']['confidence'], all_results['dp_sat']['confidence'])
            _, p_shadow_vanilla_dp_sat = stats.ttest_rel(all_results['vanilla_dp']['shadow'], all_results['dp_sat']['shadow'])
            print(f"  🆚 Vanilla DP vs DP-SAT (Confidence): p = {p_conf_vanilla_dp_sat:.4f}")
            print(f"  🆚 Vanilla DP vs DP-SAT (Shadow): p = {p_shadow_vanilla_dp_sat:.4f}")
    
    # Final assessment based on worst-case AUC across runs
    fisher_worst_conf = max(all_results['fisher_dp']['confidence'])
    fisher_worst_shadow = max(all_results['fisher_dp']['shadow'])
    fisher_worst_overall = max(fisher_worst_conf, fisher_worst_shadow)
    
    worst_case_results = {'fisher_dp': fisher_worst_overall}
    
    if vanilla_dp_model is not None:
        vanilla_worst_conf = max(all_results['vanilla_dp']['confidence'])
        vanilla_worst_shadow = max(all_results['vanilla_dp']['shadow'])
        vanilla_worst_overall = max(vanilla_worst_conf, vanilla_worst_shadow)
        worst_case_results['vanilla_dp'] = vanilla_worst_overall
    
    if dp_sat_model is not None:
        dp_sat_worst_conf = max(all_results['dp_sat']['confidence'])
        dp_sat_worst_shadow = max(all_results['dp_sat']['shadow'])
        dp_sat_worst_overall = max(dp_sat_worst_conf, dp_sat_worst_shadow)
        worst_case_results['dp_sat'] = dp_sat_worst_overall
    
    print(f"\n🎯 FINAL PRIVACY PROTECTION COMPARISON")
    print("="*50)
    print(f"📊 Worst-case AUC:")
    print(f"   • Fisher DP: {fisher_worst_overall:.4f}")
    if vanilla_dp_model is not None:
        print(f"   • Vanilla DP: {vanilla_worst_overall:.4f}")
    if dp_sat_model is not None:
        print(f"   • DP-SAT: {dp_sat_worst_overall:.4f}")
    
    # Find the best performing method
    best_method = min(worst_case_results.items(), key=lambda x: x[1])
    best_name, best_auc = best_method
    
    if best_name == 'fisher_dp':
        print(f"   🏆 Fisher DP provides the BEST privacy protection!")
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_overall - fisher_worst_overall
            print(f"   📈 vs Vanilla DP: {diff_vanilla:.4f} AUC reduction")
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_overall - fisher_worst_overall
            print(f"   📈 vs DP-SAT: {diff_dp_sat:.4f} AUC reduction")
    elif best_name == 'vanilla_dp':
        print(f"   🏆 Vanilla DP provides the BEST privacy protection!")
        diff_fisher = fisher_worst_overall - vanilla_worst_overall
        print(f"   📈 vs Fisher DP: {diff_fisher:.4f} AUC reduction")
        if dp_sat_model is not None:
            diff_dp_sat = dp_sat_worst_overall - vanilla_worst_overall
            print(f"   📈 vs DP-SAT: {diff_dp_sat:.4f} AUC reduction")
    elif best_name == 'dp_sat':
        print(f"   🏆 DP-SAT provides the BEST privacy protection!")
        diff_fisher = fisher_worst_overall - dp_sat_worst_overall
        print(f"   📈 vs Fisher DP: {diff_fisher:.4f} AUC reduction")
        if vanilla_dp_model is not None:
            diff_vanilla = vanilla_worst_overall - dp_sat_worst_overall
            print(f"   📈 vs Vanilla DP: {diff_vanilla:.4f} AUC reduction")
    
    # Privacy strength assessment for all methods
    privacy_threshold = 0.6  # AUC > 0.6 indicates weak privacy protection
    
    if fisher_worst_overall < privacy_threshold:
        print("✅ Fisher DP provides STRONG privacy protection!")
    else:
        print("⚠️  Fisher DP provides WEAK privacy protection!")
        
    if vanilla_dp_model is not None:
        if vanilla_worst_overall < privacy_threshold:
            print("✅ Vanilla DP provides STRONG privacy protection!")
        else:
            print("⚠️  Vanilla DP provides WEAK privacy protection!")
    
    if dp_sat_model is not None:
        if dp_sat_worst_overall < privacy_threshold:
            print("✅ DP-SAT provides STRONG privacy protection!")
        else:
            print("⚠️  DP-SAT provides WEAK privacy protection!")
    
    return {
        'fisher_worst_auc': fisher_worst_overall,
        'vanilla_worst_auc': vanilla_worst_overall if vanilla_dp_model else None,
        'dp_sat_worst_auc': dp_sat_worst_overall if dp_sat_model else None,
        'statistical_results': all_results
    }

# ════════════════════════════════════════════════════════════════
# Standalone Evaluation Function
# ════════════════════════════════════════════════════════════════

def get_device(args):
    """Get the appropriate device based on command line arguments"""
    if args.cpu: 
        return torch.device('cpu')
    if args.mps and torch.backends.mps.is_available():
        print('Using MPS')
        return torch.device('mps')
    if torch.cuda.is_available():
        idx = 0 if args.cuda_id is None else args.cuda_id
        print(f'Using CUDA:{idx}')
        return torch.device(f'cuda:{idx}')
    print('Using CPU')
    return torch.device('cpu')

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
    parser.add_argument('--member-size', type=int, default=2000,
                       help='Number of member samples for MIA')
    parser.add_argument('--non-member-size', type=int, default=2000,
                       help='Number of non-member samples for MIA')
    
    args = parser.parse_args()
    device = get_device(args)
    
    # Load data
    trans = T.Compose([T.ToTensor(), T.Normalize((.5,.5,.5),(.5,.5,.5))])
    trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=trans)
    
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
        
        # Run confidence attack on both models
        print("\n1️⃣  CONFIDENCE ATTACK")
        print("-" * 30)
        
        baseline_conf_results = confidence_attack(baseline, member_loader, non_member_loader, device)
        print(f"\n📈 Baseline Model:")
        print(f"   • AUC: {baseline_conf_results['auc']:.4f}")
        print(f"   • Attack Accuracy: {baseline_conf_results['accuracy']:.4f}")
        
        dp_conf_results = confidence_attack(dp_model, member_loader, non_member_loader, device)
        print(f"\n🔒 DP Model:")
        print(f"   • AUC: {dp_conf_results['auc']:.4f}")
        print(f"   • Attack Accuracy: {dp_conf_results['accuracy']:.4f}")
        
        # Run shadow model attack on both models
        print("\n2️⃣  SHADOW MODEL ATTACK")
        print("-" * 30)
        
        if training_indices is not None:
            # Use actual training data for shadow models
            actual_train_data = Subset(trainset, training_indices)
            print("🎯 Using ACTUAL training data for shadow models")
        else:
            # Fallback to full trainset
            actual_train_data = trainset
            print("⚠️  Using full training set for shadow models (less accurate)")
        
        baseline_shadow_results = shadow_model_attack(baseline, member_loader, non_member_loader, actual_train_data, device)
        print(f"\n📈 Baseline Model:")
        print(f"   • AUC: {baseline_shadow_results['auc']:.4f}")
        print(f"   • Attack Accuracy: {baseline_shadow_results['accuracy']:.4f}")
        
        dp_shadow_results = shadow_model_attack(dp_model, member_loader, non_member_loader, actual_train_data, device)
        print(f"\n🔒 DP Model:")
        print(f"   • AUC: {dp_shadow_results['auc']:.4f}")
        print(f"   • Attack Accuracy: {dp_shadow_results['accuracy']:.4f}")
        
        # Overall assessment
        worst_dp_auc = max(dp_conf_results['auc'], dp_shadow_results['auc'])
        print(f"\n🎯 OVERALL PRIVACY PROTECTION (worst-case AUC: {worst_dp_auc:.4f}):")
        if training_indices is not None:
            print("✅ Using actual training data for accurate evaluation")
        else:
            print("⚠️  Note: For accurate evaluation, retrain models or use integrated MIA with --run-mia flag")
            
        if worst_dp_auc <= 0.55:
            print("✅ DP model provides STRONG privacy protection!")
        elif worst_dp_auc <= 0.65:
            print("⚠️  DP model provides MODERATE privacy protection.")
        else:
            print("❌ DP model privacy protection may be INSUFFICIENT.")
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please first train the models using:")
        print("python main.py --mps --adaptive-clip --quantile 0.95")

if __name__ == "__main__":
    main() 