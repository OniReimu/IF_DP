#!/usr/bin/env python3
# ================================================================
# FAST ABLATION STUDY: Fisher DP-SGD (No Calibration Variants)
#    * Vanilla DP-SGD (baseline)
#    * Vanilla DP-SGD + DP-SAT 
#    * Fisher DP + Normal Optimizer
#    * Fisher DP + DP-SAT Optimizer  
# ================================================================

import os, glob, argparse, copy, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.autograd import grad  # Required for per-sample gradients

# Ensure project root on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project-specific helpers
from core.fisher_dp_sgd import compute_fisher, topk_eigh_with_floor, maha_clip
from core.dp_sgd import train_with_vanilla_dp
from core.dp_sat import train_with_dp_sat
from core.privacy_accounting import (
    get_privacy_params_for_target_epsilon, 
)
from models import available_models, create_model
from core.device_utils import resolve_device, maybe_wrap_model_for_multi_gpu
from data import DATASET_REGISTRY, DatasetConfig, build_dataset_builder
from core.mia import evaluate_membership_inference, confidence_attack, shadow_model_attack, prepare_mia_data_sample_level, prepare_mia_data_user_level
from core.influence_function import calibrate_model_research_protocol
from core.config import set_random_seeds, get_random_seed, get_dataset_location
from data.common import prepare_batch, SyntheticUserDataset, UserBatchSampler, move_to_device

AVAILABLE_DATASETS = tuple(DATASET_REGISTRY.keys())
AVAILABLE_MODELS = tuple(available_models())

set_random_seeds()  # Set reproducible random seeds
np.random.seed(get_random_seed())
models_dir = './saved_models'; os.makedirs(models_dir, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# Cache helpers
# ════════════════════════════════════════════════════════════════

def _sanitize_cache_key(value: str) -> str:
    """Make a string safe for filenames (best-effort, stable)."""
    return str(value).replace("/", "_").replace(" ", "_")


def build_pretrain_cache_path(models_dir: str, dataset_name: str, model_type: str, epochs: int, scope: str) -> str:
    """
    Cache naming scheme (dataset-scoped to avoid accidental reuse across datasets):
      Pretrain_{dataset}_{model}_{epochs}_{scope}.pth
    """
    ds = _sanitize_cache_key(dataset_name)
    return os.path.join(models_dir, f"Pretrain_{ds}_{model_type}_{int(epochs)}_{scope}.pth")

# ════════════════════════════════════════════════════════════════
# Diagnostic functions (moved from influence_function.py)
# ════════════════════════════════════════════════════════════════

def diagnose_calibration(model, critical_data, critical_targets, device):
    """
    Simple diagnostic function to measure critical slice performance.
    Replacement for the removed function in influence_function.py.
    """
    if critical_targets is None or critical_targets.numel() == 0:
        return {'loss': float('inf'), 'accuracy': 0.0, 'num_samples': 0}
    
    model.eval()
    with torch.no_grad():
        features = move_to_device(critical_data, device)
        targets = critical_targets.to(device)
        output = model(features)
        loss = F.cross_entropy(output, targets)
        predictions = output.argmax(dim=1)
        accuracy = (predictions == targets).float().mean()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy.item() * 100,
        'num_samples': targets.size(0)
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


def count_samples(loader):
    if loader is None:
        return 0
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        return len(dataset)
    batch_size = getattr(loader, "batch_size", None)
    if batch_size:
        return len(loader) * batch_size
    return len(loader)


def ensure_model_dataset_compatibility(model, dataset_task_type, dataset_name, model_type):
    """Ensure a model matches the modality expected by the dataset."""
    if dataset_task_type is None:
        return
    model_task_type = getattr(model, "task_type", None)
    if model_task_type is None:
        return
    if model_task_type != dataset_task_type:
        raise ValueError(
            f"Model '{model_type}' is a {model_task_type} architecture but dataset "
            f"'{dataset_name}' expects task type '{dataset_task_type}'. "
            f"Please pick a compatible model."
        )


def safe_torch_load(path, map_location=None):
    """
    Load checkpoints with weights_only=True when available to avoid pickle warnings.
    Falls back to legacy behavior on older PyTorch versions.
    """
    load_kwargs = {'map_location': map_location}
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def load_state_dict_forgiving(model, state_dict, description="model"):
    """
    Load a checkpoint while skipping parameters whose shapes do not match.
    Returns the list of skipped parameter keys.
    """
    # Normalize common wrapper prefixes to improve compatibility across branches.
    # Example: older/saber-style wrappers may save EfficientNet under "efficientnet.*"
    # while this repo wraps torchvision models under "backbone.*".
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    def _count_key_matches(sd):
        return sum(1 for k in sd.keys() if k in model_keys)

    # Try best-effort prefix remaps only when it actually increases key matches.
    if isinstance(state_dict, dict):
        base_matches = _count_key_matches(state_dict)
        candidates = []
        if any(k.startswith("backbone.") for k in model_keys):
            candidates.extend(
                [
                    ("efficientnet.", "backbone."),
                    ("resnet.", "backbone."),
                    ("vit.", "backbone."),
                ]
            )
        best_state = state_dict
        best_matches = base_matches
        best_rule = None
        for src_prefix, dst_prefix in candidates:
            if not any(k.startswith(src_prefix) for k in state_dict.keys()):
                continue
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith(src_prefix):
                    remapped[dst_prefix + k[len(src_prefix) :]] = v
                else:
                    remapped[k] = v
            matches = _count_key_matches(remapped)
            if matches > best_matches:
                best_state = remapped
                best_matches = matches
                best_rule = (src_prefix, dst_prefix)
        # Only apply if we meaningfully improve compatibility.
        if best_rule is not None and best_matches > base_matches:
            src_prefix, dst_prefix = best_rule
            print(f"   🔧 Remapped checkpoint keys for {description}: '{src_prefix}*' → '{dst_prefix}*' ({base_matches}→{best_matches} matching keys)")
            state_dict = best_state

    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if value.shape != target.shape:
            skipped.append(key)
            continue
        compatible_state[key] = value
    model.load_state_dict(compatible_state, strict=False)
    if not compatible_state and isinstance(state_dict, dict) and len(state_dict) > 0:
        print(f"   ⚠️  Loaded 0 parameters for {description}. The checkpoint is likely incompatible with this model wrapper/type.")
    if skipped:
        print(f"   ⚠️  Skipped {len(skipped)} incompatible parameters when loading {description}:")
        for name in skipped:
            print(f"      • {name}")
        print("      (This is expected when switching between datasets such as CIFAR-10 and CIFAR-100.)")
    return skipped


def build_calibration_loader(public_loader, args):
    dataset = getattr(public_loader, "dataset", None)
    if dataset is None:
        return public_loader
    total = len(dataset)
    take = min(args.calibration_subset, total)
    if take <= 0 or take >= total:
        return public_loader
    indices = torch.randperm(total)[:take].tolist()
    if isinstance(dataset, SyntheticUserDataset):
        base_subset = Subset(dataset.base, indices)
        synthetic = SyntheticUserDataset(base_subset, args.users)
        sampler = UserBatchSampler(synthetic.uid)
        return DataLoader(synthetic, batch_sampler=sampler)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=args.batch_size, shuffle=True)

# ════════════════════════════════════════════════════════════════
# Critical slice helpers (support both vision and language inputs)
# ════════════════════════════════════════════════════════════════

def _index_select_features(features, indices):
    if torch.is_tensor(features):
        if features.dim() == 0:
            expanded = features.unsqueeze(0)
            return expanded.index_select(0, indices)
        return features.index_select(0, indices)
    if isinstance(features, dict):
        return {k: _index_select_features(v, indices) for k, v in features.items()}
    if isinstance(features, tuple):
        return tuple(_index_select_features(v, indices) for v in features)
    if isinstance(features, list):
        as_list = indices.tolist()
        return [features[i] for i in as_list]
    raise TypeError(f"Unsupported feature type for slicing: {type(features)}")


def _concat_feature_chunks(chunks):
    if not chunks:
        return None
    first = chunks[0]
    if torch.is_tensor(first):
        return torch.cat(chunks, dim=0)
    if isinstance(first, dict):
        return {k: _concat_feature_chunks([chunk[k] for chunk in chunks]) for k in first}
    if isinstance(first, tuple):
        transposed = list(zip(*chunks))
        return tuple(_concat_feature_chunks(list(items)) for items in transposed)
    if isinstance(first, list):
        transposed = list(zip(*chunks))
        return [_concat_feature_chunks(list(items)) for items in transposed]
    raise TypeError(f"Unsupported feature type for concatenation: {type(first)}")


def build_critical_slice(eval_loader, target_class="all", label_mapping=None, max_samples_per_class=200):
    """
    Collect a balanced critical slice from the evaluation loader that works for
    both tensor (vision) and dictionary (language) features.
    """
    cpu = torch.device("cpu")
    use_all_classes = target_class == "all"
    target_value = None if use_all_classes else int(target_class)
    class_chunks = defaultdict(list)
    label_chunks = defaultdict(list)
    class_counts = defaultdict(int)

    for batch_data in eval_loader:
        features, labels, _ = prepare_batch(batch_data, cpu)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)

        candidate_labels = torch.unique(labels).tolist() if use_all_classes else [target_value]
        for cls in candidate_labels:
            if cls is None:
                continue
            mask = (labels == cls)
            if not mask.any():
                continue

            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            if use_all_classes:
                remaining = max_samples_per_class - class_counts[cls]
                if remaining <= 0:
                    continue
                if remaining < idx.numel():
                    idx = idx[:remaining]

            selected = _index_select_features(features, idx)
            selected_labels = labels.index_select(0, idx)
            class_chunks[cls].append(selected)
            label_chunks[cls].append(selected_labels)
            class_counts[cls] += idx.numel()


    def _describe(cls, count):
        if label_mapping and cls in label_mapping:
            return f"Class {cls} ({label_mapping[cls]}): {count} samples"
        return f"Class {cls}: {count} samples"

    if use_all_classes:
        if not class_counts:
            print("⚠️  No samples found in evaluation data")
            return None, torch.empty(0, dtype=torch.long)

        ordered = sorted(class_counts.keys())
        combined_features = []
        combined_labels = []
        total_samples = 0
        print("🎯 Using ALL CLASSES for calibration (general utility improvement)")
        for cls in ordered:
            combined_features.append(_concat_feature_chunks(class_chunks[cls]))
            combined_labels.append(torch.cat(label_chunks[cls], dim=0))
            total_samples += class_counts[cls]
            print(f"   • {_describe(cls, class_counts[cls])}")

        crit_features = _concat_feature_chunks(combined_features)
        crit_labels = torch.cat(combined_labels, dim=0)
        print(f"✅ Evaluation slice: {total_samples} samples across {len(ordered)} classes")
        return crit_features, crit_labels

    collected = class_counts.get(target_value, 0)
    if collected == 0:
        print(f"⚠️  No samples of class {target_value}")
        return None, torch.empty(0, dtype=torch.long)

    print(f"🎯 Using SINGLE CLASS {target_value} for calibration (targeted improvement)")
    print(f"   • {_describe(target_value, collected)}")
    crit_features = _concat_feature_chunks(class_chunks[target_value])
    crit_labels = torch.cat(label_chunks[target_value], dim=0)
    return crit_features, crit_labels

# ════════════════════════════════════════════════════════════════
# Optimized Calibration Functions (Line Search + Multi-Step)
# ════════════════════════════════════════════════════════════════

def _eval_slice_loss(model, critical_data, critical_targets, device):
    """Helper function to evaluate loss on critical slice.
    Paper mapping — used to compute L_crit(θ) for back-tracking line search (Step d(iii)).
    """
    if critical_targets is None or critical_targets.numel() == 0 or critical_data is None:
        return float('inf')
    
    model.eval()
    with torch.no_grad():
        features = move_to_device(critical_data, device)
        targets = critical_targets.to(device)
        output = model(features)
        loss = F.cross_entropy(output, targets)
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
    """
    TRICK 1: LINE SEARCH OPTIMIZATION
    Instead of applying the influence update directly with scale 1.0, we perform a 
    back-tracking line search over candidate scales α ∈ {0, 0.25, ..., 2.0} to find 
    the update size that minimizes loss on the critical slice. This ensures we don't 
    overshoot (or undershoot) the optimal calibration point.
    
    Enhanced calibration with line search optimization for optimal step size.
    
    Paper mapping — Step d(iii): Back-tracking line search over α ∈ {1, 1/2, 1/4, …}
    We:
      1) call calibrate_model_research_protocol to compute a base Δθ (Steps c–d(ii)),
      2) extract Δθ, then evaluate L_crit(θ + αΔθ) for a candidate set,
      3) pick α that yields the lowest L_crit on the critical slice.
    """
    
    print(f"🔍 Line Search Calibration:")
    print(f"   • Method: {method}")
    print(f"   • Eta: {eta}")
    print(f"   • Trust tau: {trust_tau}")
    print(f"   • Regularization: {reg}")
    
    # First get the standard influence update
    # Paper mapping — Steps 3–4(b): compute α(z), select top-η, form Δθ via influence vectors
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
    # Paper mapping — Step d(iii): try α ∈ {0, 1/4, 1/2, 3/4, 1, 5/4, 3/2, 2} and pick best
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

def calibrate_with_combined_optimization(model, pub_loader, priv_loader, critical_data, critical_targets, 
                                        device, method='linear', eta=100, trust_tau=0.01, reg=10.0, 
                                        strict=True, clean_model=None, max_steps=3, patience=2, 
                                        min_improvement=0.001):
    """
    TRICK 2: MULTI-STEP REFINEMENT
    We apply the calibration iteratively. At each step t, we re-evaluate the influence 
    scores at the current parameter θ_t (since the loss landscape is non-convex and 
    influence is local), select a new set of top-k samples, and apply a line-searched 
    update. This typically converges to a better solution than a single step.
    
    Combined line search + multi-step calibration optimization.
    
    Paper mapping — Step d(iv): T refinement rounds (typically T=3).
    Each round recomputes α(z) at the current θ̂^(t) (inside calibrate_model_research_protocol),
    reselects top-η, recomputes Δθ^(t), then performs a back-tracking line search to get θ̂^(t+1).
    """
    
    print(f"🚀 Combined Line Search + Multi-Step Calibration:")
    print(f"   • Method: {method}")
    print(f"   • Eta: {eta}")
    print(f"   • Trust tau: {trust_tau}")
    print(f"   • Max steps: {max_steps}")
    
    current_model = copy.deepcopy(model)
    best_model = copy.deepcopy(model)
    best_loss = _eval_slice_loss(model, critical_data, critical_targets, device)
    no_improvement_count = 0
    
    print(f"   📊 Initial loss: {best_loss:.4f}")
    
    for step in range(max_steps):
        print(f"   🔄 Combined Step {step + 1}/{max_steps}:")
        
        # Apply line search calibration for this step
        # Paper mapping — Step 4(d): recompute α(z) and Δθ at θ̂^(t), then line search to obtain θ̂^(t+1)
        step_calibrated = calibrate_with_line_search(
            current_model, pub_loader, priv_loader,
            critical_data, critical_targets, device=device,
            method=method, eta=eta, trust_tau=trust_tau,
            strict=strict, clean_model=clean_model, reg=reg
        )
        
        # Evaluate improvement
        step_loss = _eval_slice_loss(step_calibrated, critical_data, critical_targets, device)
        improvement = best_loss - step_loss
        
        print(f"     • Loss: {step_loss:.4f}")
        print(f"     • Improvement: {improvement:+.4f}")
        
        if improvement > min_improvement:
            print(f"     ✅ Improvement above threshold")
            best_model = copy.deepcopy(step_calibrated)
            best_loss = step_loss
            current_model = step_calibrated
            no_improvement_count = 0
        else:
            print(f"     ⚠️  Improvement below threshold")
            no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print(f"     🛑 Early stopping: {patience} steps without improvement")
                break
    
    print(f"   ✅ Combined optimization complete!")
    print(f"   • Best loss: {best_loss:.4f}")
    print(f"   • Steps completed: {step + 1}")
    
    return best_model

# ════════════════════════════════════════════════════════════════
# Synthetic users + batch sampler (reused from main.py)
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════
def get_device(args):
    return resolve_device(args)


def build_model_for_device(model_type, model_kwargs, args, device):
    model = create_model(model_type, **model_kwargs)
    model = maybe_wrap_model_for_multi_gpu(model, args)
    return model.to(device)

def accuracy(model, loader, device):
    model.eval(); tot=correct=0
    with torch.no_grad():
        for batch_data in loader:
            features, labels, _ = prepare_batch(batch_data, device)
            correct += (model(features).argmax(1)==labels).sum().item()
            tot += labels.size(0)
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
                                  use_dp_sat=False,
                                  optimizer_name="Normal", positive_noise_correlation=False,
                                  precomputed_lam=None, precomputed_U=None,
                                  dp_sat_mode="none", rho_sat=0.001, dp_param_count=None,
                                  dp_epochs=None):
    """
    TRICK 4: EXACT FISHER-AWARE DP-SAT
    Standard DP-SAT adds a gradient term, which is approximate. Here we implement 
    the *exact* version by physically perturbing the weights in the Fisher-whitened 
    direction (w_pert = w + δ), computing the gradient at w_pert, and then restoring w.
    This aligns the sharpness-aware step with the Fisher geometry used for privacy.
    
    Fisher DP-SGD with optional DP-SAT optimization.
    
    This function explores the synergy between:
    1. Fisher-informed noise (curvature-aware)
    2. DP-SAT optimization (sharpness-aware)
    
    Args:
        clip_radius: Target Euclidean sensitivity Δ₂ (same as vanilla DP-SGD for fair comparison).
                    This will be converted to appropriate Mahalanobis threshold internally.
        use_dp_sat: If True, apply DP-SAT flatness adjustment (legacy flag)
        dp_sat_mode: "none", "euclidean", "fisher"
        rho_sat: Perturbation radius for Exact DP-SAT
        optimizer_name: String identifier for logging purposes
        positive_noise_correlation: If False (default), use negatively correlated noise (noise ∝ 1/√λ).
                                   If True, use positively correlated noise (noise ∝ √λ).
        precomputed_lam: Pre-computed eigenvalues (if None, compute from fisher matrix)
        precomputed_U: Pre-computed eigenvectors (if None, compute from fisher matrix)
        
    Algorithm when dp_sat_mode != 'none':
        1. Compute per-sample gradients and clip them (standard DP-SGD)
        2. Add Fisher-informed noise → g_fisher_priv
        3. Apply DP-SAT flatness adjustment:
           - Perturb weights before gradient computation (Exact DP-SAT)
        4. Final update: θ ← θ - η g_fisher_priv
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # Fisher eigendecomposition (use pre-computed if available)
    if precomputed_lam is not None and precomputed_U is not None:
        print(f"   ✅ Using pre-computed Fisher eigendecomposition")
        lam, U = precomputed_lam, precomputed_U
        actual_k = len(lam)
    else:
        print(f"   🔍 Computing Fisher eigendecomposition...")
        lam, U = topk_eigh_with_floor(fisher, k=k, lam_floor=lam_floor)
        lam, U = lam.to(device), U.to(device)
        actual_k = len(lam)
        if actual_k != k:
            print(f"⚠️  Using k={actual_k} eigenpairs (requested {k}) due to matrix rank constraints")
    
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

    # Gather parameter objects
    def _match(name: str, layer: str) -> bool:
        # stricter match: prefix match to avoid accidental substring matches across deeper blocks
        return name.startswith(layer)

    # Gather parameter objects
    if dp_param_count is not None:
        # DP parameter budget mode: smart selection to maximize parameter usage
        print(f"   🎯 DP Parameter Budget Mode: selecting up to {dp_param_count} parameters")
        all_params = list(model.named_parameters())
        
        # Build list of (name, param, size, index) for knapsack optimization
        param_info = [(name, param, param.numel(), idx) 
                      for idx, (name, param) in enumerate(all_params)]
        
        # Greedy knapsack: select parameters that fit within budget
        # Prioritize by order (to maintain some semantic ordering)
        selected_indices = []
        total_selected = 0
        
        for name, param, size, idx in param_info:
            if total_selected + size <= dp_param_count:
                selected_indices.append(idx)
                total_selected += size
            elif total_selected < dp_param_count:
                # Would exceed budget - silently skip for cleaner logs
                continue
        
        # Extract selected parameters
        names = []
        params = []
        for idx in sorted(selected_indices):
            name, param = all_params[idx]
            names.append(name)
            params.append(param)
        
        param_dim = total_selected
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
            names = [n for n,_ in model.named_parameters()
                    if any(_match(n, l) for l in layers)]
        else:
            names = [n for n,_ in model.named_parameters()
                    if _match(n, target_layer)]
        params = [dict(model.named_parameters())[n] for n in names]
        param_dim = sum(p.numel() for p in params)

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
    if use_dp_sat or dp_sat_mode != "none":
        print(f"   • DP-SAT enabled: mode={dp_sat_mode}, ρ={rho_sat}")
        if dp_sat_mode == "fisher":
            print(f"   • ✨ Using Fisher-whitened weight perturbation (Fisher DP-SAT)")
        elif dp_sat_mode == "euclidean":
            print(f"   • ⚠️  Using Euclidean weight perturbation (Euclidean DP-SAT)")
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

    # Determine DP fine-tuning epochs
    if dp_epochs is None:
        dp_epochs = max(1, int(math.ceil(epochs / 10)))
    print(f"   • DP finetuning epochs: {dp_epochs} (requested {epochs})")

    for epoch in range(dp_epochs):
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Fisher DP + {opt_type} ({mode_str}) epoch {epoch+1}/{dp_epochs}", leave=False)):
            features, labels, user_ids = prepare_batch(batch_data, device)
            batch_size = labels.size(0)

            # ============================================================
            # DP-SAT Exact Mode: Weight Perturbation (Start of Step)
            # ============================================================
            perturbation = None
            if dp_sat_mode != "none" and g_fisher_priv_prev is not None:
                if dp_sat_mode == "euclidean":
                    # Euclidean perturbation: δ = ρ * g / ||g||₂
                    g_norm = g_fisher_priv_prev.norm() + 1e-8
                    perturbation = rho_sat * g_fisher_priv_prev / g_norm
                elif dp_sat_mode == "fisher":
                    # Fisher perturbation: δ = ρ * F⁻¹g / ||F⁻¹g||_F
                    alpha = U.T @ g_fisher_priv_prev
                    alpha_white = alpha * inv_sqrt_lam
                    alpha_white_norm = alpha_white.norm() + 1e-8
                    perturbation = rho_sat * (U @ (alpha_white / alpha_white_norm * inv_sqrt_lam))

                # Apply perturbation
                current_idx = 0
                for p in params:
                    n = p.numel()
                    p.data.add_(perturbation[current_idx:current_idx+n].view_as(p))
                    current_idx += n

            model.zero_grad()
            losses = F.cross_entropy(model(features), labels, reduction="none")

            if sample_level:
                # SAMPLE-LEVEL DP: Compute per-sample gradients
                per_g = []
                for i in range(batch_size):
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
            
            g_final = g_fisher_priv
            
            # 1. Restore weights if using Exact Mode
            if perturbation is not None:
                current_idx = 0
                for p in params:
                    n = p.numel()
                    p.data.sub_(perturbation[current_idx:current_idx+n].view_as(p))
                    current_idx += n

            # Store current Fisher-noisy gradient for next iteration (needed for Exact)
            if use_dp_sat or dp_sat_mode != "none":
                g_fisher_priv_prev = g_fisher_priv.clone().detach()

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
    if full_complement_noise:
        print(f"   • Last batch: Fisher={fisher_noise_norm:.1f}, Complement={complement_noise_norm:.1f}")
    else:
        print(f"   • Last batch: Fisher only={fisher_noise_norm:.1f} (complement disabled)")
    print(f"   • Privacy: (ε={epsilon}, δ={delta}) over {dp_epochs} DP fine-tuning epochs")
    print(f"   • ✅ FAIR COMPARISON: Same effective sensitivity Δ₂ as vanilla DP-SGD")

    return model

# ════════════════════════════════════════════════════════════════
# Ablation Study Main Function
# ════════════════════════════════════════════════════════════════

def run_ablation_study(args, device, priv_loader, eval_loader, priv_base, priv_idx, priv_ds, Fmat, pub_loader, pub_loader_calib, model_kwargs, dataset_task_type=None, label_mapping=None, eval_dataset=None):
    """
    FAST ablation study on core Fisher DP-SGD variants (no calibration).
    
    Variants tested:
    1. Vanilla DP-SGD (Non-Fisher)
    2. Vanilla DP-SGD + DP-SAT (Non-Fisher)
    3. Fisher DP + Normal Optimizer
    4. Fisher DP + DP-SAT Optimizer
    """
    
    print("\n" + "="*70)
    print("🚀  FAST ABLATION STUDY: Fisher DP-SGD (No Calibration Variants)")
    print("="*70)
    
    # ════════════════════════════════════════════════════════════════
    # PRE-COMPUTE FISHER EIGENDECOMPOSITION (Eliminate Redundancy)
    # ════════════════════════════════════════════════════════════════
    
    """
    TRICK 3: LOW-RANK FISHER PRE-COMPUTATION
    Calculating the Fisher eigendecomposition is expensive. Since the Fisher information 
    approximates the geometry at the *initial* (or converged) point, we pre-compute 
    the top-k eigenpairs once and reuse them across different runs (where applicable) 
    to save massive compute time during ablation studies.
    """
    
    print(f"\n🔍 Pre-computing Fisher eigendecomposition to avoid redundancy...")
    lam, U = topk_eigh_with_floor(Fmat, k=args.k, lam_floor=5e-1)  # Use consistent lam_floor
    lam, U = lam.to(device), U.to(device)
    actual_k = len(lam)
    
    if actual_k != args.k:
        print(f"⚠️  Using k={actual_k} eigenpairs (requested {args.k}) due to matrix rank constraints")
    
    print(f"✅ Fisher eigendecomposition complete: k={actual_k} eigenpairs")
    print(f"   • Eigenvalue range: [{lam.min().item():.3e}, {lam.max().item():.3e}]")
    
    # Load baseline model for initialization
    baseline = build_model_for_device(args.model_type, model_kwargs, args, device)
    ensure_model_dataset_compatibility(baseline, dataset_task_type, args.dataset_name, args.model_type)
    
    # Check for pretrained baseline model (saved for efficiency)
    pretrain_path = build_pretrain_cache_path(
        models_dir=models_dir,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        epochs=args.epochs,
        scope="public",
    )
    legacy_pretrain_path = os.path.join(models_dir, f"Pretrain_{args.model_type}_{args.epochs}_public.pth")

    loaded_baseline = False
    if os.path.exists(pretrain_path) and not args.clean:
        print(f'\n📥 Loading pretrained baseline from {pretrain_path}...')
        checkpoint = safe_torch_load(pretrain_path, map_location=device)
        ck_dataset = checkpoint.get("dataset_name") if isinstance(checkpoint, dict) else None
        if ck_dataset is not None and ck_dataset != args.dataset_name:
            print(f"   ⚠️  Cache dataset mismatch: checkpoint={ck_dataset} vs requested={args.dataset_name}. Retraining baseline.")
        else:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            load_state_dict_forgiving(baseline, state_dict, description="pretrained baseline")
            baseline_acc_cached = checkpoint.get('accuracy', 'N/A')
            print(f"   ✅ Loaded baseline (eval accuracy: {baseline_acc_cached})")
            loaded_baseline = True
    elif os.path.exists(legacy_pretrain_path) and not args.clean:
        print(f"\n📥 Found legacy pretrained baseline cache: {legacy_pretrain_path}")
        checkpoint = safe_torch_load(legacy_pretrain_path, map_location=device)
        legacy_dataset = checkpoint.get("dataset_name") if isinstance(checkpoint, dict) else None
        if legacy_dataset is None:
            print("   ⚠️  Legacy cache has no dataset metadata; skipping to avoid cross-dataset reuse.")
        elif legacy_dataset != args.dataset_name:
            print(f"   ⚠️  Legacy cache dataset mismatch: checkpoint={legacy_dataset} vs requested={args.dataset_name}. Skipping.")
        else:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            load_state_dict_forgiving(baseline, state_dict, description="pretrained baseline (legacy cache)")
            baseline_acc_cached = checkpoint.get('accuracy', 'N/A')
            print(f"   ✅ Loaded baseline from legacy cache (eval accuracy: {baseline_acc_cached})")
            loaded_baseline = True

    if not loaded_baseline:
        print(f'\n⚙️  Training baseline on PUBLIC data (Strict DP Setup)...')
        print(f"   • Public data size: {len(pub_loader.dataset)} samples")
        print(f"   • Model: {args.model_type}")
        print(f"   • Epochs: {args.epochs}")
        print(f"   • Private data will ONLY be used for DP-training selected layers ({args.dp_layer if not args.dp_param_count else f'budget={args.dp_param_count}'})")
        
        # Stronger from-scratch recipe for public pretrain (no ImageNet weights)
        base_lr = 0.1
        weight_decay = 5e-4
        momentum = 0.9
        opt_b = torch.optim.SGD(baseline.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        
        # Cosine schedule over public pretrain epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs)
        
        for epoch in tqdm(range(args.epochs), desc="Public baseline training"):
            baseline.train()
            for batch_data in pub_loader:
                features, labels, _ = prepare_batch(batch_data, device)
                opt_b.zero_grad()
                loss = F.cross_entropy(baseline(features), labels)
                loss.backward()
                opt_b.step()
            scheduler.step()
        
        # Evaluate baseline accuracy on eval set
        baseline.eval()
        baseline_acc = accuracy(baseline, eval_loader, device)
        
        # Save pretrained baseline (dataset-scoped cache filename)
        torch.save({
            'model_state_dict': baseline.state_dict(),
            'model_type': args.model_type,
            'dataset_name': args.dataset_name,
            'epochs': args.epochs,
            'accuracy': baseline_acc,
            'public_data_size': len(pub_loader.dataset),
            'timestamp': __import__('time').strftime('%Y%m%d_%H%M%S')
        }, pretrain_path)
        print(f"\n💾 Saved pretrained baseline to {pretrain_path}")
        print(f"   • Baseline accuracy: {baseline_acc:.2f}%")

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
        
        # sigma = noise_multiplier * args.clip_radius
        sigma = noise_multiplier
        display_epsilon = args.target_epsilon
        
        print(f"\n🔒 Privacy Accounting for Optimized Ablation Study:")
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
        epochs=args.epochs,
        dp_param_count=args.dp_param_count,
        dp_epochs=args.dp_epochs
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
        rho_sat=args.rho_sat,  # Use consistent perturbation radius
        dp_param_count=args.dp_param_count
    )
    
    # ════════════════════════════════════════════════════════════════
    # Variant 3: Fisher DP + Normal Optimizer (USING PRE-COMPUTED EIGENDECOMPOSITION)
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
        positive_noise_correlation=args.positive_noise_correlation,
        precomputed_lam=lam,  # Pass pre-computed eigendecomposition
        precomputed_U=U,
        dp_param_count=args.dp_param_count,
        dp_epochs=args.dp_epochs
    )
    
    # ════════════════════════════════════════════════════════════════
    # Variant 4: Fisher DP + DP-SAT Optimizer (USING PRE-COMPUTED EIGENDECOMPOSITION)
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*50}")
    print("🔺 VARIANT 4: Fisher DP + DP-SAT Optimizer")
    print(f"{'='*50}")

    # Determine DP-SAT mode for Fisher variants
    # If user didn't specify a mode, default to 'fisher' for the ablation study
    # to show the "Best vs Best" comparison (Vanilla+Euclidean vs Fisher+Fisher)
    effective_sat_mode = args.dp_sat_mode if args.dp_sat_mode != 'none' else 'fisher'
    print(f"   ℹ️  Using DP-SAT mode: {effective_sat_mode} for Fisher variants")
    
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
        use_dp_sat=True,  # Force DP-SAT enabled for this variant
        dp_sat_mode=effective_sat_mode,  # Use determined mode
        rho_sat=args.rho_sat,
        optimizer_name="DP-SAT",
        positive_noise_correlation=args.positive_noise_correlation,
        precomputed_lam=lam,  # Pass pre-computed eigendecomposition
        precomputed_U=U,
        dp_param_count=args.dp_param_count,
        dp_epochs=args.dp_epochs
    )
    
    # ════════════════════════════════════════════════════════════════
    # Evaluation and Comparison
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("📊 FAST ABLATION STUDY RESULTS")
    print(f"{'='*70}")
    
    # Compute accuracies
    baseline_acc = accuracy(baseline, eval_loader, device)
    vanilla_dp_acc = accuracy(vanilla_dp_model, eval_loader, device)
    vanilla_dpsat_acc = accuracy(vanilla_dpsat_model, eval_loader, device)
    fisher_normal_acc = accuracy(fisher_normal_model, eval_loader, device)
    fisher_dpsat_acc = accuracy(fisher_dpsat_model, eval_loader, device)
    
    ablation_results['baseline'] = baseline_acc
    ablation_results['vanilla_dp'] = vanilla_dp_acc
    ablation_results['vanilla_dpsat'] = vanilla_dpsat_acc
    ablation_results['fisher_normal'] = fisher_normal_acc
    ablation_results['fisher_dpsat'] = fisher_dpsat_acc
    
    dp_mode = "Sample-level" if args.sample_level else f"User-level ({args.users} users)"
    print(f"\n🎯 Accuracy Comparison ({dp_mode} DP):")
    print(f"   • Baseline (Public Only)          : {baseline_acc:6.2f}%")
    print(f"   • Vanilla DP-SGD                  : {vanilla_dp_acc:6.2f}%")
    print(f"   • Vanilla DP-SGD + DP-SAT          : {vanilla_dpsat_acc:6.2f}%")
    print(f"   • Fisher DP + Normal              : {fisher_normal_acc:6.2f}%")
    print(f"   • Fisher DP + DP-SAT              : {fisher_dpsat_acc:6.2f}%")
    
    # Compute improvements
    vanilla_dp_improvement = vanilla_dp_acc - baseline_acc
    vanilla_dpsat_improvement = vanilla_dpsat_acc - baseline_acc
    vanilla_dpsat_vs_vanilla = vanilla_dpsat_acc - vanilla_dp_acc
    fisher_vs_vanilla = fisher_normal_acc - vanilla_dp_acc
    normal_improvement = fisher_normal_acc - baseline_acc
    dpsat_improvement = fisher_dpsat_acc - baseline_acc
    synergy_gain = fisher_dpsat_acc - fisher_normal_acc
    
    print(f"\n📈 Improvement Analysis:")
    print(f"   • Vanilla DP-SGD:                 {vanilla_dp_improvement:+5.2f}% vs baseline")
    print(f"   • Vanilla DP-SGD + DP-SAT:        {vanilla_dpsat_improvement:+5.2f}% vs baseline")
    print(f"   • DP-SAT gain (Vanilla):          {vanilla_dpsat_vs_vanilla:+5.2f}% over vanilla DP")
    print(f"   • Fisher benefit:                 {fisher_vs_vanilla:+5.2f}% over vanilla DP")
    print(f"   • Fisher DP (Normal):             {normal_improvement:+5.2f}% vs baseline")
    print(f"   • Fisher DP (DP-SAT):             {dpsat_improvement:+5.2f}% vs baseline")
    print(f"   • Synergy Gain (DP-SAT):          {synergy_gain:+5.2f}% over normal Fisher DP")
    
    best_method = max([
        ('Vanilla DP-SGD', vanilla_dp_acc),
        ('Vanilla DP-SGD + DP-SAT', vanilla_dpsat_acc),
        ('Fisher Normal', fisher_normal_acc),
        ('Fisher DP-SAT', fisher_dpsat_acc),
    ], key=lambda x: x[1])
    
    print(f"   🏆 Best method: {best_method[0]} ({best_method[1]:.2f}%)")
    
    if synergy_gain > 0.5:  # Threshold for meaningful improvement
        print(f"   🎉 SYNERGY DETECTED: DP-SAT optimization provides meaningful benefit!")
    elif synergy_gain > 0:
        print(f"   ✅ SMALL SYNERGY: DP-SAT provides modest improvement")
    else:
        print(f"   ⚠️  NO SYNERGY: DP-SAT may not help with Fisher-informed noise")

    # ════════════════════════════════════════════════════════════════
    # Save Models for Further Analysis
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n💾 Saving ablation study models...")
    ds_tag = _sanitize_cache_key(args.dataset_name)
    
    # Save Vanilla DP-SGD
    vanilla_dp_path = os.path.join(models_dir, f'Vanilla_DP_{ds_tag}_Ablation.pth')
    torch.save({
        'model_state_dict': vanilla_dp_model.state_dict(),
        'model_type': 'vanilla_dp',
        'dataset_name': args.dataset_name,
        'accuracy': vanilla_dp_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'ablation_study': True
    }, vanilla_dp_path)
    print(f"✅ Saved Vanilla DP-SGD to {vanilla_dp_path}")
    
    # Save Vanilla DP-SGD + DP-SAT
    vanilla_dpsat_path = os.path.join(models_dir, f'Vanilla_DPSAT_{ds_tag}_Ablation.pth')
    torch.save({
        'model_state_dict': vanilla_dpsat_model.state_dict(),
        'model_type': 'vanilla_dp_dpsat',
        'dataset_name': args.dataset_name,
        'accuracy': vanilla_dpsat_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'lambda_flatness': args.lambda_flatness,
        'dpsat_gain_vanilla': vanilla_dpsat_vs_vanilla,
        'ablation_study': True
    }, vanilla_dpsat_path)
    print(f"✅ Saved Vanilla DP-SGD + DP-SAT to {vanilla_dpsat_path}")
    
    # Save Fisher DP + Normal
    fisher_normal_path = os.path.join(models_dir, f'Fisher_Normal_{ds_tag}_Ablation.pth')
    torch.save({
        'model_state_dict': fisher_normal_model.state_dict(),
        'model_type': 'fisher_dp_normal',
        'dataset_name': args.dataset_name,
        'accuracy': fisher_normal_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'k': args.k,
        'full_complement_noise': args.full_complement_noise,
        'ablation_study': True
    }, fisher_normal_path)
    print(f"✅ Saved Fisher DP + Normal to {fisher_normal_path}")
    
    # Save Fisher DP + DP-SAT
    fisher_dpsat_path = os.path.join(models_dir, f'Fisher_DPSAT_{ds_tag}_Ablation.pth')
    torch.save({
        'model_state_dict': fisher_dpsat_model.state_dict(),
        'model_type': 'fisher_dp_dpsat',
        'dataset_name': args.dataset_name,
        'accuracy': fisher_dpsat_acc,
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'k': args.k,
        'lambda_flatness': args.lambda_flatness,
        'dp_sat_mode': effective_sat_mode,
        'rho_sat': args.rho_sat,
        'full_complement_noise': args.full_complement_noise,
        'synergy_gain': synergy_gain,
        'ablation_study': True
    }, fisher_dpsat_path)
    print(f"✅ Saved Fisher DP + DP-SAT ({effective_sat_mode}) to {fisher_dpsat_path}")

    # ════════════════════════════════════════════════════════════════
    # Optional: Comprehensive 4-Way MIA Evaluation
    # ════════════════════════════════════════════════════════════════
    
    if args.run_mia:
        print(f"\n🛡️  Running MIA evaluation on core ablation variants (no calibration models)...")
        
        # Prepare all models for MIA evaluation
        # Optional non-DP comparator (trained on private data WITHOUT DP) for AUC reference
        non_dp_scope = "private"
        non_dp_path = build_pretrain_cache_path(
            models_dir=models_dir,
            dataset_name=args.dataset_name,
            model_type=args.model_type,
            epochs=args.epochs,
            scope=non_dp_scope,
        )
        legacy_non_dp_path = os.path.join(models_dir, f"Pretrain_{args.model_type}_{args.epochs}_{non_dp_scope}.pth")
        non_dp_private = build_model_for_device(args.model_type, model_kwargs, args, device)
        if os.path.exists(non_dp_path):
            print(f"\n💾 Loading cached NON-DP private comparator: {non_dp_path}")
            comparator_state = safe_torch_load(non_dp_path, map_location=device)
            ck_dataset = comparator_state.get("dataset_name") if isinstance(comparator_state, dict) else None
            if ck_dataset is not None and ck_dataset != args.dataset_name:
                print(f"   ⚠️  Cache dataset mismatch: checkpoint={ck_dataset} vs requested={args.dataset_name}. Retraining comparator.")
            else:
                if isinstance(comparator_state, dict) and 'model_state_dict' in comparator_state:
                    comparator_state = comparator_state['model_state_dict']
                load_state_dict_forgiving(non_dp_private, comparator_state, description="NON-DP comparator")
                comparator_state = None
        elif os.path.exists(legacy_non_dp_path):
            print(f"\n💾 Found legacy NON-DP private comparator cache: {legacy_non_dp_path}")
            comparator_state = safe_torch_load(legacy_non_dp_path, map_location=device)
            legacy_dataset = comparator_state.get("dataset_name") if isinstance(comparator_state, dict) else None
            if legacy_dataset is None:
                print("   ⚠️  Legacy comparator cache has no dataset metadata; skipping to avoid cross-dataset reuse.")
            elif legacy_dataset != args.dataset_name:
                print(f"   ⚠️  Legacy comparator cache dataset mismatch: checkpoint={legacy_dataset} vs requested={args.dataset_name}. Skipping.")
            else:
                if isinstance(comparator_state, dict) and 'model_state_dict' in comparator_state:
                    comparator_state = comparator_state['model_state_dict']
                load_state_dict_forgiving(non_dp_private, comparator_state, description="NON-DP comparator (legacy cache)")
                comparator_state = None
        else:
            print("\n⚠️  Training NON-DP comparator on PRIVATE data for AUC reference (not DP-safe)...")
            opt_non_dp = torch.optim.SGD(non_dp_private.parameters(), lr=1e-3, momentum=.9)
            for epoch in tqdm(range(args.epochs), desc="Training NON-DP private comparator"):
                non_dp_private.train()
                for batch_data in priv_loader:
                    features, labels, _ = prepare_batch(batch_data, device)
                    opt_non_dp.zero_grad()
                    F.cross_entropy(non_dp_private(features), labels).backward()
                    opt_non_dp.step()
            torch.save({
                "model_state_dict": non_dp_private.state_dict(),
                "model_type": args.model_type,
                "dataset_name": args.dataset_name,
                "epochs": args.epochs,
                "scope": non_dp_scope,
                "timestamp": __import__("time").strftime("%Y%m%d_%H%M%S"),
            }, non_dp_path)
            print(f"✅ Saved NON-DP private comparator to {non_dp_path}")

        models_to_evaluate = {
            'Baseline (Public Only)': baseline,  # strictly public-pretrained
            'Non-DP (Private, not DP-safe)': non_dp_private,  # reference AUC > 0.5 expected
            'Vanilla DP-SGD': vanilla_dp_model,
            'Vanilla DP-SGD + DP-SAT': vanilla_dpsat_model,
            'Fisher DP + Normal': fisher_normal_model,
            'Fisher DP + DP-SAT': fisher_dpsat_model,
        }
        
        eval_source = eval_dataset if eval_dataset is not None else getattr(eval_loader, "dataset", None)
        if eval_source is None:
            raise RuntimeError("Evaluation dataset is required for MIA sampling. Ensure dataset builder provides it.")
        # Prepare member and non-member datasets
        if args.sample_level:
            print("📊 Sample-level MIA: Using actual private training samples as members")
            member_set, non_member_set = prepare_mia_data_sample_level(priv_base, eval_source, priv_idx, args.mia_size)
        else:
            print("👥 User-level MIA: Using actual private users as members")
            member_set, non_member_set = prepare_mia_data_user_level(priv_ds, eval_source, args.users, args.mia_size)
        
        member_loader = DataLoader(member_set, batch_size=64, shuffle=False)
        non_member_loader = DataLoader(non_member_set, batch_size=64, shuffle=False)
        
        print(f"   • Members: {len(member_set)} samples")
        print(f"   • Non-members: {len(non_member_set)} samples")
        
        # Run comprehensive MIA evaluation on all models
        mia_results = {}
        
        print(f"\n🕶️  SHADOW MODEL ATTACK RESULTS:")
        print("-" * 50)
        
        for model_name, model in models_to_evaluate.items():
            print(f"   Evaluating {model_name}...")
            shadow_result = shadow_model_attack(model, member_loader, non_member_loader, priv_base, device, eval_source)
            mia_results[model_name] = {
                'shadow_auc': shadow_result['auc'],
                'shadow_acc': shadow_result['accuracy']
            }
            print(f"     • AUC: {shadow_result['auc']:.4f}, Accuracy: {shadow_result['accuracy']:.4f}")
        
        # Comprehensive analysis
        print(f"\n📊 COMPREHENSIVE MIA ANALYSIS")
        print("=" * 60)
        
        # Use shadow attack AUC directly (no need for worst-case since we only have one attack)
        shadow_aucs = {}
        for model_name in models_to_evaluate.keys():
            shadow_aucs[model_name] = mia_results[model_name]['shadow_auc']
        
        print(f"\n🎯 Shadow Attack AUC Comparison:")
        for model_name, auc in shadow_aucs.items():
            print(f"   • {model_name:30}: {auc:.4f}")
        
        # Identify best and worst models for privacy
        best_privacy_model = min(shadow_aucs.items(), key=lambda x: x[1])
        worst_privacy_model = max(shadow_aucs.items(), key=lambda x: x[1])
        
        print(f"\n🏆 Privacy Protection Ranking:")
        print(f"   🥇 BEST:  {best_privacy_model[0]} (AUC: {best_privacy_model[1]:.4f})")
        print(f"   🥴 WORST: {worst_privacy_model[0]} (AUC: {worst_privacy_model[1]:.4f})")
        
        # Privacy vs Accuracy tradeoff analysis
        print(f"\n⚖️  Privacy vs Accuracy Tradeoff:")
        print(f"   Model                          Accuracy  Privacy (1-AUC)")
        print(f"   {'─'*30} ─────────  ──────────────")
        for model_name in ['Vanilla DP-SGD', 'Vanilla DP-SGD + DP-SAT', 'Fisher DP + Normal', 'Fisher DP + DP-SAT']:
            if model_name == 'Vanilla DP-SGD':
                acc = vanilla_dp_acc
            elif model_name == 'Vanilla DP-SGD + DP-SAT':
                acc = vanilla_dpsat_acc
            elif model_name == 'Fisher DP + Normal':
                acc = fisher_normal_acc
            else:  # Fisher DP + DP-SAT
                acc = fisher_dpsat_acc
            
            privacy_score = 1.0 - shadow_aucs[model_name]  # Higher is better
            print(f"   {model_name:30} {acc:5.1f}%     {privacy_score:.3f}")
        
        # Key comparisons only
        print(f"\n🔒 Key Privacy Effects:")
        
        fisher_normal_auc = shadow_aucs['Fisher DP + Normal']
        fisher_dpsat_auc = shadow_aucs['Fisher DP + DP-SAT']
        
        dpsat_privacy_effect = fisher_normal_auc - fisher_dpsat_auc
        
        print(f"   • DP-SAT effect:      {dpsat_privacy_effect:+.4f} AUC")
        
        # Final recommendation
        print(f"\n🎯 Best Privacy Protection: {best_privacy_model[0]} (AUC: {best_privacy_model[1]:.4f})")
        non_dp_private_auc = shadow_aucs.get('Non-DP (Private, not DP-safe)', None)
        if non_dp_private_auc is not None:
            print(f"   • Non-DP private comparator AUC: {non_dp_private_auc:.4f} (expected >0.5, not DP-safe)")
        
        # Store results for return
        ablation_results['mia_results'] = {
            'shadow_aucs': shadow_aucs,
            'best_privacy_model': best_privacy_model,
            'detailed_results': mia_results,
            'privacy_effects': {
                'dpsat_effect': dpsat_privacy_effect,
            }
        }
        
        # Update legacy fields for compatibility
        ablation_results['mia_results']['fisher_worst_auc'] = fisher_normal_auc
        ablation_results['mia_results']['dp_sat_worst_auc'] = fisher_dpsat_auc
        
        print(f"\n✅ MIA evaluation complete! (Shadow attack only)")

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
    parser.add_argument('--cuda-devices', type=str, default=None,
                       help='Comma-separated CUDA device ids for multi-GPU execution (e.g., "0,1,2")')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Enable torch.nn.DataParallel across the requested CUDA devices')
    
    # Data arguments
    parser.add_argument('--dataset', '--dataset-name', dest='dataset_name',
                       choices=AVAILABLE_DATASETS, default='cifar10',
                       help='Dataset identifier registered in the data package (vision or language)')
    parser.add_argument('--dataset-size', type=int, default=5000,
                       help='Number of private samples to draw from the dataset (default: 5k, matching legacy ablation)')
    parser.add_argument('--public-ratio', type=float, default=1.0,
                       help='Fraction of the remaining training data reserved for the public split (default: all remaining samples)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Mini-batch size for private/public loaders')
    parser.add_argument('--eval-batch-size', type=int, default=256,
                       help='Batch size for evaluation loaders')
    parser.add_argument('--critical-label', type=int, default=None,
                       help='Optional class index for targeted calibration slices')
    parser.add_argument('--tokenizer-name', type=str, default='bert-base-uncased',
                       help='Tokenizer checkpoint for language datasets')
    parser.add_argument('--max-seq-length', type=int, default=512,
                       help='Maximum sequence length for language datasets')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dp-epochs', type=int, default=None,
                       help='Number of DP fine-tuning epochs. If None, uses max(1, ceil(epochs/10)). '
                            'Lower values may help DP-SAT perform better.')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='cnn',
                       choices=AVAILABLE_MODELS,
                       help='Model architecture registered in the models package')
    
    # DP layer/parameter selection (mutually exclusive)
    dp_selection = parser.add_mutually_exclusive_group()
    dp_selection.add_argument('--dp-layer', type=str, default='conv1',
                             help='Target layers for DP training (e.g., "conv1,conv2"). Mutually exclusive with --dp-param-count.')
    dp_selection.add_argument('--dp-param-count', type=int, default=None,
                             help='DP parameter budget: train first N parameters in model order. Mutually exclusive with --dp-layer.')
    
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
    parser.add_argument('--full-complement-noise', action='store_true',
                       help='Use full complement noise in orthogonal subspace')
    
    # Fisher DP noise scaling strategy (fixed to negatively correlated noise)
    parser.set_defaults(negatively_correlated_noise=True)
    
    # DP-SAT arguments
    parser.add_argument('--rho-sat', type=float, default=0.001,
                       help='Perturbation radius for Fisher/Euclidean DP-SAT (Exact Method)')
    parser.add_argument('--lambda-flatness', type=float, default=0.01,
                       help='Flatness coefficient for Vanilla DP-SAT (Original Paper Baseline)')
    parser.add_argument('--dp-sat-mode', type=str, default='none',
                       choices=['none', 'euclidean', 'fisher'],
                       help='DP-SAT mode: none (default), euclidean (Exact Euclidean), fisher (Exact Fisher)')
    
    # Adaptive clipping
    parser.add_argument('--adaptive-clip', action='store_true')
    parser.add_argument('--quantile', type=float, default=0.95)
    
    # DP mode
    parser.add_argument('--sample-level', action='store_true')
    parser.add_argument('--users', type=int, default=10)
    
    # Calibration arguments (now enabled)
    parser.add_argument('--method', type=str, default='linear',
                       choices=['linear', 'public-fisher'],
                       help='Calibration method: linear (fast regularization) or public-fisher (uses public data Fisher matrix)')
    parser.add_argument('--calibration-subset', type=int, default=5000,
                       help='Number of public samples used to build the calibration loader')
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
    
    # Optimized calibration arguments
    parser.add_argument('--combined-steps', type=int, default=3,
                       help='Maximum number of combined optimization steps for enhanced calibration (default: 3)')
    parser.add_argument('--patience', type=int, default=2,
                       help='Early stopping patience: stop after this many steps without improvement (default: 2)')
    parser.add_argument('--min-improvement', type=float, default=0.001,
                       help='Minimum loss improvement threshold to continue optimization (default: 0.001)')
    
    # MIA evaluation
    parser.add_argument('--run-mia', action='store_true')
    parser.add_argument('--mia-size', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Map new argument names for consistency  
    # Fixed noise strategy: negatively correlated noise (no positive option)
    args.positive_noise_correlation = False
    
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
        for f in glob.glob(os.path.join(models_dir,'*Ablation*.pth')):
            os.remove(f); print('  removed',f)
    
    # ════════════════════════════════════════════════════════════════
    # Data preparation (vision + language)
    # ════════════════════════════════════════════════════════════════
    
    dataset_root, allow_download = get_dataset_location(dataset_key=args.dataset_name)
    dataset_builder = build_dataset_builder(args.dataset_name)
    dataset_task_type = getattr(dataset_builder, "task_type", None)
    dataset_num_labels = getattr(dataset_builder, "num_labels", None)
    if not dataset_num_labels:
        raise ValueError(f"Dataset '{args.dataset_name}' did not specify num_labels; cannot build model.")
    model_kwargs = {'num_labels': dataset_num_labels}
    label_mapping = dataset_builder.get_label_mapping()
    dataset_config = DatasetConfig(
        dataset_root=dataset_root,
        allow_download=allow_download,
        dataset_size=args.dataset_size,
        public_ratio=args.public_ratio,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        sample_level=args.sample_level,
        num_users=args.users,
        critical_label=args.critical_label,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        seed=get_random_seed(),
    )
    loaders = dataset_builder.build(dataset_config)
    priv_loader = loaders.private
    pub_loader = loaders.public
    eval_loader = loaders.evaluation
    crit_loader = loaders.critical_eval
    calibration_loader = getattr(loaders, "calibration", None)
    priv_base = loaders.private_base
    priv_idx = loaders.private_indices
    priv_ds = None if args.sample_level else getattr(priv_loader, "dataset", None)
    eval_dataset = getattr(eval_loader, "dataset", None)
    
    if calibration_loader is None:
        calibration_loader = build_calibration_loader(pub_loader, args)
    pub_loader_calib = calibration_loader
    
    public_samples = count_samples(pub_loader)
    eval_samples = count_samples(eval_loader)
    calibration_samples = count_samples(pub_loader_calib)
    print(f'📊 Ablation data overview for {args.dataset_name}:')
    print(f'   • Private samples : {len(priv_base)}')
    print(f'   • Public samples  : {public_samples}')
    print(f'   • Eval samples    : {eval_samples}')
    print(f'   • Calibration subset: {calibration_samples} samples')
    
    if args.sample_level:
        print('📊 Using SAMPLE-level DP')
    else:
        print(f'👥 Using USER-level DP ({args.users} synthetic users)')
    
    # ════════════════════════════════════════════════════════════════
    # Fisher matrix computation
    # ════════════════════════════════════════════════════════════════
    
    print('\n🔍 Computing Fisher matrix for ablation study…')
    
    # Train a baseline model for Fisher computation
    fisher_baseline = build_model_for_device(args.model_type, model_kwargs, args, device)
    ensure_model_dataset_compatibility(fisher_baseline, dataset_task_type, args.dataset_name, args.model_type)
    fisher_opt = torch.optim.SGD(fisher_baseline.parameters(), lr=1e-3, momentum=.9)
    
    for epoch in tqdm(range(5), desc="Training Fisher baseline"):  # Fewer epochs for Fisher
        fisher_baseline.train()
        for batch_data in priv_loader:
            features, labels, _ = prepare_batch(batch_data, device)
            fisher_opt.zero_grad()
            F.cross_entropy(fisher_baseline(features), labels).backward()
            fisher_opt.step()
    
    Fmat, _ = compute_fisher(fisher_baseline, priv_loader, device,
                            target_layer=args.dp_layer, rho=1e-2,
                            dp_param_count=args.dp_param_count)
    
    # ════════════════════════════════════════════════════════════════
    # Run ablation study
    # ════════════════════════════════════════════════════════════════
    
    results = run_ablation_study(
        args,
        device,
        priv_loader,
        eval_loader,
        priv_base,
        priv_idx,
        priv_ds,
        Fmat,
        pub_loader,
        pub_loader_calib,
        model_kwargs,
        dataset_task_type=dataset_task_type,
        label_mapping=label_mapping,
        eval_dataset=eval_dataset,
    )
    
    # ════════════════════════════════════════════════════════════════
    # Final summary
    # ════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("🚀 FAST ABLATION STUDY SUMMARY")
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
    
    print(f"\n🏆 Overall Best Performance:")
    best_variant = max(results['vanilla_dp'], results['vanilla_dpsat'],
                      results['fisher_normal'], results['fisher_dpsat'])
    if best_variant == results['fisher_dpsat']:
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
    
    print(f"\n🔒 Key Insights:")
    print(f"   • Fisher-informed noise shapes noise according to loss curvature")
    print(f"   • DP-SAT guides optimization toward flatter minima")
    print(f"   • DP-SAT synergy: {synergy_gain:+.2f}% suggests {'beneficial' if synergy_gain > 0 else 'neutral'} interaction")
    
    if 'mia_results' in results:
        print(f"\n🛡️  Privacy Summary:")
        best_privacy = results['mia_results']['best_privacy_model']
        effects = results['mia_results']['privacy_effects']
        
        print(f"   • Best protection: {best_privacy[0]} (AUC: {best_privacy[1]:.4f})")
        print(f"   • DP-SAT privacy effect: {effects['dpsat_effect']:+.4f} AUC vs Fisher-Normal")
    
    print(f"\n📍 Key Findings:")
    print(f"   • DP-SAT synergy: {synergy_gain:+.2f}% accuracy improvement")
    if 'mia_results' in results:
        print(f"   • Privacy: DP-SAT privacy effect {results['mia_results']['privacy_effects']['dpsat_effect']:+.3f} AUC (lower is better)")
    
    print(f"\n✅ Fast ablation study complete! Models saved in {models_dir}/")

if __name__ == "__main__":
    main() 
