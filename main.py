#!/usr/bin/env python3
# ================================================================
# CIFAR-10  ·  curvature-aware USER-level DP-SGD demo
#    * K synthetic users  (--users K, default 10)
#    * one mini-batch  ≙  one user  (via custom BatchSampler)
# ================================================================

import os, glob, argparse, copy, math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, Sampler
from tqdm import tqdm

# project-specific helpers
from fisher_dp_sgd      import compute_fisher, train_with_dp
from dp_sgd             import train_with_vanilla_dp
from dp_sat             import train_with_dp_sat

from model              import CNN
from mia                import evaluate_membership_inference
from privacy_accounting import (
    get_privacy_params_for_target_epsilon, 
    compute_actual_epsilon,
    validate_privacy_comparison,
    print_privacy_summary
)

torch.manual_seed(42) ; np.random.seed(42)
models_dir = './saved_models'; os.makedirs(models_dir, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# Synthetic users + batch sampler
# ════════════════════════════════════════════════════════════════
class SyntheticUserDataset(torch.utils.data.Dataset):
    """Assigns each sample a user_id ∈ {0,…,K-1} (round-robin)."""
    def __init__(self, base_ds, num_users, perm=None):
        self.base = base_ds
        if perm is None: perm = np.arange(len(base_ds))
        self.uid  = torch.tensor(perm % num_users, dtype=torch.long)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x,y = self.base[idx]
        return x, y, self.uid[idx].item()

class UserBatchSampler(Sampler):
    """Yield all indices of exactly ONE user per iteration."""
    def __init__(self, user_ids, shuffle=True):
        self.by_user  = {}
        for idx,u in enumerate(user_ids):
            u_key = int(u)  # Convert to Python int to avoid numpy int64 issues
            self.by_user.setdefault(u_key, []).append(idx)
        self.uids     = list(self.by_user.keys())
        self.shuffle  = shuffle
    def __iter__(self):
        order = np.random.permutation(self.uids) if self.shuffle else self.uids
        for u in order: 
            u_key = int(u)  # Convert to Python int
            yield self.by_user[u_key]
    def __len__(self): return len(self.uids)

# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════
def get_device(a):
    if a.cpu: return torch.device('cpu')
    if a.mps and torch.backends.mps.is_available():
        print('Using MPS');  return torch.device('mps')
    if torch.cuda.is_available():
        idx = 0 if a.cuda_id is None else a.cuda_id
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
            tot     += y.size(0)
    return 100*correct/tot

# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════
p = argparse.ArgumentParser('Fisher DP-SGD on CIFAR-10')
p.add_argument('--mps', action='store_true')
p.add_argument('--cuda-id', type=int)
p.add_argument('--cpu', action='store_true')

p.add_argument('--dataset-size', type=int,   default=50000)
p.add_argument('--private-ratio', type=float, default=0.8)
p.add_argument('--epochs', type=int, default=10)

p.add_argument('--clean', action='store_true',
               help='Remove all saved models before training')

# DP-SGD hyper-params
p.add_argument('--delta',   type=float, default=1e-5)
p.add_argument('--clip-radius', type=float, default=1.0)
p.add_argument('--k', type=int, default=32)
p.add_argument('--dp-layer', type=str, default='conv1')
p.add_argument('--lambda-flatness', type=float, default=0.01,
               help='Flatness coefficient for DP-SAT (default: 0.01)')

# Adaptive clipping options
p.add_argument('--adaptive-clip', action='store_true',
               help='Use adaptive clipping based on gradient norm quantiles')
p.add_argument('--quantile', type=float, default=0.95,
               help='Quantile for adaptive clipping (default: 0.95 for 95th percentile)')

# DP mode selection
p.add_argument('--sample-level', action='store_true',
               help='Use sample-level DP-SGD (traditional) instead of user-level DP-SGD')
p.add_argument('--users', type=int,          default=10,
               help='Number of synthetic users in PRIVATE split')
p.add_argument('--full-complement-noise', action='store_true',
               help='Use full complement noise in orthogonal subspace')

# Fisher DP noise scaling strategy
noise_strategy_group = p.add_mutually_exclusive_group()
noise_strategy_group.add_argument('--negatively_correlated_noise', action='store_true', default=True,
                                 help='Fisher DP: noise inversely correlated with curvature (noise ∝ 1/√λ, less noise in high curvature directions, default)')
noise_strategy_group.add_argument('--positively_correlated_noise', action='store_true',
                                 help='Fisher DP: noise positively correlated with curvature (noise ∝ √λ, more noise in high curvature directions)')

# MIA evaluation flags
p.add_argument('--run-mia', action='store_true',
               help='Run membership inference attack evaluation after training')
p.add_argument('--mia-size', type=int, default=1000,
               help='Number of member/non-member samples for MIA (default: 1000)')

# Comparison flags
p.add_argument('--compare-others', action='store_true',
               help='Also train others for comparison with Fisher-informed DP-SGD')

# Privacy accounting arguments
privacy_group = p.add_mutually_exclusive_group()
privacy_group.add_argument('--use-legacy-accounting', action='store_true',
                          help='⚠️  Use legacy privacy accounting (NOT recommended for research - results will be scientifically invalid)')

# Separate epsilon arguments with clear usage
epsilon_group = p.add_mutually_exclusive_group()
epsilon_group.add_argument('--epsilon', type=float, default=None,
                          help='Privacy epsilon (legacy accounting only)')
epsilon_group.add_argument('--target-epsilon', type=float, default=None,
                          help='Target epsilon for DP (proper accounting)')

args = p.parse_args()

# Set defaults and validate privacy parameter combinations
if args.use_legacy_accounting:
    if args.epsilon is None:
        print("❌ Error: --use-legacy-accounting requires --epsilon parameter")
        print("   Usage: python main.py --use-legacy-accounting --epsilon 10.0")
        exit(1)
    if args.target_epsilon is not None:
        print("❌ Error: Cannot use --target-epsilon with --use-legacy-accounting")
        print("   Use --epsilon instead: python main.py --use-legacy-accounting --epsilon 10.0")
        exit(1)
else:
    # Default mode: proper accounting
    if args.epsilon is not None:
        print("❌ Error: --epsilon is only for legacy accounting")
        print("   Use --target-epsilon instead: python main.py --target-epsilon 10.0")
        print("   Or add --use-legacy-accounting to use legacy mode (not recommended)")
        exit(1)
    if args.target_epsilon is None:
        args.target_epsilon = 10.0  # Set default for proper accounting

device = get_device(args)

# ════════════════════════════════════════════════════════════════
# House-keeping
# ════════════════════════════════════════════════════════════════
if args.clean:
    print('Cleaning saved models…')
    for f in glob.glob(os.path.join(models_dir,'*.pth')):
        os.remove(f); print('  removed',f)

# ════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════
trans = T.Compose([T.ToTensor(), T.Normalize((.5,.5,.5),(.5,.5,.5))])
trainset = torchvision.datasets.CIFAR10('./data', train=True,
                                        download=True, transform=trans)
testset  = torchvision.datasets.CIFAR10('./data', train=False,
                                        download=True, transform=trans)

# Data partitioning for differential privacy:
# - priv_base: From trainset (50k) - used for baseline training, DP training, and MIA members
# - pub_base: From testset (10k) - used ONLY for calibration (currently commented out)  
# - remaining testset: Used for evaluation and MIA non-members

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

print(f'📊 Data split:')
print(f'   • Private data: {len(priv_base)} samples from trainset (for training)')
print(f'   • Public data: {len(pub_base)} samples from testset (for calibration)')
print(f'   • Evaluation data: {len(eval_base)} samples from testset (for evaluation & MIA non-members)')

# Choose DP mode: sample-level vs user-level
if args.sample_level:
    print('📊 Using SAMPLE-level DP-SGD (traditional)')
    # Sample-level: regular datasets and DataLoaders
    priv_loader = DataLoader(priv_base, batch_size=128, shuffle=True)
    pub_loader  = DataLoader(pub_base, batch_size=128, shuffle=True)
    priv_ds = None  # Not used in sample-level mode
else:
    print(f'👥 Using USER-level DP-SGD ({args.users} synthetic users)')
    # User-level: synthetic user datasets with custom batch sampler
    priv_ds  = SyntheticUserDataset(priv_base, args.users)
    pub_ds   = SyntheticUserDataset(pub_base,  1)  # For calibration only
    
    priv_loader = DataLoader(priv_ds,
                             batch_sampler=UserBatchSampler(priv_ds.uid))
    pub_loader  = DataLoader(pub_ds, batch_size=128, shuffle=True)

# Use evaluation subset for model testing
test_loader = DataLoader(eval_base, batch_size=128, shuffle=False)

cat_idx   = [i for i, (_, y) in enumerate(eval_base) if eval_base.dataset[eval_base.indices[i]][1] == 3]
crit_loader = DataLoader(Subset(eval_base, cat_idx),
                         batch_size=128, shuffle=False)

if not args.sample_level:
    print(f'▶  {args.users} synthetic users '
          f'({len(priv_base)//args.users:.1f} samples each)')

# ════════════════════════════════════════════════════════════════
# 1. Non-private baseline
# ════════════════════════════════════════════════════════════════
baseline = CNN().to(device)
opt_b = torch.optim.SGD(baseline.parameters(), lr=1e-3, momentum=.9)
print('\n⚙️  Training baseline…')
for epoch in tqdm(range(args.epochs)):
    baseline.train()
    for batch_data in priv_loader:
        if args.sample_level:
            x, y = batch_data  # Sample-level: (x, y)
        else:
            x, y, _ = batch_data  # User-level: (x, y, user_id)
        x, y = x.to(device), y.to(device)
        opt_b.zero_grad(); F.cross_entropy(baseline(x),y).backward(); opt_b.step()

# ════════════════════════════════════════════════════════════════
# 2. Fisher computation (chosen layers)
# ════════════════════════════════════════════════════════════════
print('\n🔍  Fisher matrix…')
Fmat,_ = compute_fisher(baseline, priv_loader, device,
                        target_layer=args.dp_layer, rho=1e-2)

# ════════════════════════════════════════════════════════════════
# 3. Fisher-informed DP-SGD
# ════════════════════════════════════════════════════════════════
print('\n🚀 Fisher-informed DP-SGD…')
fisher_dp_model = copy.deepcopy(baseline)

if args.use_legacy_accounting:
    print("\n⚠️  ⚠️  ⚠️  WARNING: USING LEGACY PRIVACY ACCOUNTING ⚠️  ⚠️  ⚠️")
    print("   This mode produces SCIENTIFICALLY INVALID results!")
    print("   Comparisons between DP methods will be meaningless!")
    print("   Only use this for reproducing old (incorrect) experiments.")
    print("   Remove --use-legacy-accounting for valid research results.")
    print("⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️")
    
    # Legacy approach - keep existing logic
    sigma = math.sqrt(2*math.log(1.25/args.delta)) / args.epsilon
    display_epsilon = args.epsilon
    total_steps = args.epochs * len(priv_loader)

else:
    print("\n🔒 Using Proper Privacy Accounting (Opacus RDP)")
    
    # Calculate proper noise parameters for target epsilon
    sample_rate = len(priv_loader) / len(priv_base)
    steps_per_epoch = len(priv_loader)
    
    noise_multiplier, total_steps = get_privacy_params_for_target_epsilon(
        target_epsilon=args.target_epsilon,
        target_delta=args.delta,
        sample_rate=sample_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    # Convert noise multiplier to sigma for both methods
    sigma = noise_multiplier * args.clip_radius
    
    print(f"\n🎯 Proper Privacy Accounting Setup:")
    print(f"   • Target (ε, δ): ({args.target_epsilon}, {args.delta})")
    print(f"   • Sample rate: {sample_rate:.4f}")
    print(f"   • Required noise multiplier: {noise_multiplier:.4f}")
    print(f"   • Sigma (for both methods): {sigma:.4f}")
    print(f"   • Total steps: {total_steps}")
    
    # Override epsilon parameter for display purposes
    display_epsilon = args.target_epsilon

fisher_dp_model = train_with_dp(fisher_dp_model, priv_loader, Fmat,
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
                                positive_noise_correlation=args.positively_correlated_noise)

# ════════════════════════════════════════════════════════════════
# 4. Vanilla DP-SGD (comparison baseline)
# ════════════════════════════════════════════════════════════════
vanilla_dp_model = None
if args.compare_others:
    print('\n📐 Vanilla DP-SGD (comparison)…')
    vanilla_dp_model = copy.deepcopy(baseline)
    
    if args.use_legacy_accounting:
        # Legacy approach
        vanilla_dp_model = train_with_vanilla_dp(vanilla_dp_model, priv_loader,
                                                 epsilon=args.epsilon, delta=args.delta,
                                                 clip_radius=args.clip_radius,
                                                 device=device,
                                                 target_layer=args.dp_layer,
                                                 adaptive_clip=args.adaptive_clip,
                                                 quantile=args.quantile,
                                                 sample_level=args.sample_level,
                                                 epochs=args.epochs)
    else:
        vanilla_dp_model = train_with_vanilla_dp(vanilla_dp_model, priv_loader,
                                                 epsilon=display_epsilon, delta=args.delta,
                                                 sigma=sigma,  # Use same sigma for fair comparison
                                                 clip_radius=args.clip_radius,
                                                 device=device,
                                                 target_layer=args.dp_layer,
                                                 adaptive_clip=args.adaptive_clip,
                                                 quantile=args.quantile,
                                                 sample_level=args.sample_level,
                                                 epochs=args.epochs)
        
        # Compute and validate actual achieved epsilons
        actual_fisher_epsilon = compute_actual_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=total_steps,
            target_delta=args.delta
        )
        
        actual_vanilla_epsilon = compute_actual_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=total_steps,
            target_delta=args.delta
        )
        
        # Validate fair comparison
        is_fair = validate_privacy_comparison(actual_fisher_epsilon, actual_vanilla_epsilon)
        
        print_privacy_summary(
            method_name="Fisher DP-SGD",
            target_epsilon=args.target_epsilon,
            actual_epsilon=actual_fisher_epsilon,
            delta=args.delta,
            noise_multiplier=noise_multiplier,
            steps=total_steps,
            sample_rate=sample_rate
        )
        
        print_privacy_summary(
            method_name="Vanilla DP-SGD",
            target_epsilon=args.target_epsilon,
            actual_epsilon=actual_vanilla_epsilon,
            delta=args.delta,
            noise_multiplier=noise_multiplier,
            steps=total_steps,
            sample_rate=sample_rate
        )
        
        if is_fair:
            print("\n✅ Fair privacy comparison: Both methods at same privacy level")
        else:
            print("\n⚠️  Privacy levels differ - comparison may not be fair!")

# ════════════════════════════════════════════════════════════════
# 5. DP-SAT (Sharpness-Aware Training comparison)
# ════════════════════════════════════════════════════════════════
dp_sat_model = None
if args.compare_others:
    print('\n🔺 DP-SAT: Sharpness-Aware Training (comparison)…')
    dp_sat_model = copy.deepcopy(baseline)
    
    if args.use_legacy_accounting:
        # Legacy approach
        dp_sat_model = train_with_dp_sat(dp_sat_model, priv_loader,
                                        epsilon=args.epsilon, delta=args.delta,
                                        clip_radius=args.clip_radius,
                                        device=device,
                                        target_layer=args.dp_layer,
                                        adaptive_clip=args.adaptive_clip,
                                        quantile=args.quantile,
                                        sample_level=args.sample_level,
                                        epochs=args.epochs,
                                        lambda_flatness=args.lambda_flatness)
    else:
        dp_sat_model = train_with_dp_sat(dp_sat_model, priv_loader,
                                        epsilon=display_epsilon, delta=args.delta,
                                        sigma=sigma,  # Use same sigma for fair comparison
                                        clip_radius=args.clip_radius,
                                        device=device,
                                        target_layer=args.dp_layer,
                                        adaptive_clip=args.adaptive_clip,
                                        quantile=args.quantile,
                                        sample_level=args.sample_level,
                                        epochs=args.epochs,
                                        lambda_flatness=args.lambda_flatness)
        
        # Compute actual achieved epsilon for DP-SAT (should be same as others)
        actual_dp_sat_epsilon = compute_actual_epsilon(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=total_steps,
            target_delta=args.delta
        )
        
        print_privacy_summary(
            method_name="DP-SAT",
            target_epsilon=args.target_epsilon,
            actual_epsilon=actual_dp_sat_epsilon,
            delta=args.delta,
            noise_multiplier=noise_multiplier,
            steps=total_steps,
            sample_rate=sample_rate
        )

# ════════════════════════════════════════════════════════════════
# 6. Evaluation
# ════════════════════════════════════════════════════════════════
def cat_acc(m): return accuracy(m, crit_loader, device)

dp_mode = "Sample-level" if args.sample_level else f"User-level ({args.users} users)"
print(f'\n📊  Accuracy summary ({dp_mode} DP)')
print(f' baseline         : {accuracy(baseline,test_loader,device):6.2f}% '
      f'(cat {cat_acc(baseline):5.2f}%)')
print(f' Fisher DP        : {accuracy(fisher_dp_model,test_loader,device):6.2f}% '
      f'(cat {cat_acc(fisher_dp_model):5.2f}%)')
if vanilla_dp_model is not None:
    print(f' Vanilla DP       : {accuracy(vanilla_dp_model,test_loader,device):6.2f}% '
          f'(cat {cat_acc(vanilla_dp_model):5.2f}%)')
if dp_sat_model is not None:
    print(f' DP-SAT           : {accuracy(dp_sat_model,test_loader,device):6.2f}% '
          f'(cat {cat_acc(dp_sat_model):5.2f}%)')

if vanilla_dp_model is not None or dp_sat_model is not None:
    # Calculate improvements
    fisher_acc = accuracy(fisher_dp_model, test_loader, device)
    
    if vanilla_dp_model is not None:
        vanilla_acc = accuracy(vanilla_dp_model, test_loader, device)
        improvement_vs_vanilla = fisher_acc - vanilla_acc
        print(f' Fisher vs Vanilla: {improvement_vs_vanilla:+5.2f}% improvement')
    
    if dp_sat_model is not None:
        dp_sat_acc = accuracy(dp_sat_model, test_loader, device)
        improvement_vs_dp_sat = fisher_acc - dp_sat_acc
        print(f' Fisher vs DP-SAT : {improvement_vs_dp_sat:+5.2f}% improvement')
        
        if vanilla_dp_model is not None:
            dp_sat_vs_vanilla = dp_sat_acc - vanilla_acc
            print(f' DP-SAT vs Vanilla: {dp_sat_vs_vanilla:+5.2f}% improvement')

# ════════════════════════════════════════════════════════════════
# 7. Save models for MIA evaluation
# ════════════════════════════════════════════════════════════════
print('\n💾 Saving models for MIA evaluation...')

# Save baseline model
baseline_path = os.path.join(models_dir, '基线模型.pth')
torch.save({
    'model_state_dict': baseline.state_dict(),
    'model_type': 'baseline',
    'accuracy': accuracy(baseline, test_loader, device),
    'cat_accuracy': cat_acc(baseline),
    'training_indices': priv_idx.tolist(),  # Save actual training indices
    'dataset_size': args.dataset_size,
    'sample_level': args.sample_level,
    'num_users': args.users if not args.sample_level else None
}, baseline_path)
print(f'✅ Saved baseline model to {baseline_path}')

# Save DP model
dp_path = os.path.join(models_dir, 'DP模型.pth')
torch.save({
    'model_state_dict': fisher_dp_model.state_dict(),
    'model_type': 'fisher_dp',
    'accuracy': accuracy(fisher_dp_model, test_loader, device),
    'cat_accuracy': cat_acc(fisher_dp_model),
    'epsilon': display_epsilon,
    'clip_radius': args.clip_radius,
    'adaptive_clip': args.adaptive_clip,
    'quantile': args.quantile if args.adaptive_clip else None,
    'training_indices': priv_idx.tolist(),  # Save actual training indices  
    'dataset_size': args.dataset_size,
    'sample_level': args.sample_level,
    'num_users': args.users if not args.sample_level else None
}, dp_path)
print(f'✅ Saved Fisher DP model to {dp_path}')

# Save Vanilla DP model  
if vanilla_dp_model is not None:
    vanilla_dp_path = os.path.join(models_dir, 'Vanilla_DP模型.pth')
    torch.save({
        'model_state_dict': vanilla_dp_model.state_dict(),
        'model_type': 'vanilla_dp',
        'accuracy': accuracy(vanilla_dp_model, test_loader, device),
        'cat_accuracy': cat_acc(vanilla_dp_model),
        'epsilon': args.epsilon,
        'clip_radius': args.clip_radius,
        'adaptive_clip': args.adaptive_clip,
        'quantile': args.quantile if args.adaptive_clip else None,
        'training_indices': priv_idx.tolist(),  # Save actual training indices
        'dataset_size': args.dataset_size,
        'sample_level': args.sample_level,
        'num_users': args.users if not args.sample_level else None
    }, vanilla_dp_path)
    print(f'✅ Saved Vanilla DP model to {vanilla_dp_path}')

# Save DP-SAT model
if dp_sat_model is not None:
    dp_sat_path = os.path.join(models_dir, 'DP_SAT模型.pth')
    torch.save({
        'model_state_dict': dp_sat_model.state_dict(),
        'model_type': 'dp_sat',
        'accuracy': accuracy(dp_sat_model, test_loader, device),
        'cat_accuracy': cat_acc(dp_sat_model),
        'epsilon': display_epsilon,
        'clip_radius': args.clip_radius,
        'adaptive_clip': args.adaptive_clip,
        'quantile': args.quantile if args.adaptive_clip else None,
        'lambda_flatness': args.lambda_flatness,
        'training_indices': priv_idx.tolist(),  # Save actual training indices
        'dataset_size': args.dataset_size,
        'sample_level': args.sample_level,
        'num_users': args.users if not args.sample_level else None
    }, dp_sat_path)
    print(f'✅ Saved DP-SAT model to {dp_sat_path}')

print(f'\n🛡️  To evaluate privacy protection, add --run-mia to your command:')
if vanilla_dp_model is not None or dp_sat_model is not None:
    methods = []
    if vanilla_dp_model is not None: methods.append("Vanilla")
    if dp_sat_model is not None: methods.append("DP-SAT")
    methods_str = " & ".join(methods)
    print(f'   Re-run with: --run-mia --mia-size 1000 (will compare Fisher vs {methods_str})')
else:
    print(f'   Re-run with: --run-mia --mia-size 1000')
    print(f'   Add --compare-others for Fisher vs Others (Vanilla + DP-SAT)')
    print(f'   Or use standalone: python mia.py --mps --member-size 2000 --non-member-size 2000')
    print(f'\n✅ Using proper privacy accounting by default for scientifically valid results')
    print(f'   Add --use-legacy-accounting only to reproduce old (incorrect) experiments')

# ════════════════════════════════════════════════════════════════
# 8. Evaluate membership inference attacks
# ════════════════════════════════════════════════════════════════
if args.run_mia:
    evaluate_membership_inference(baseline, fisher_dp_model, priv_base, eval_base, 
                                priv_idx, priv_ds, args.users, args.mia_size, 
                                args.sample_level, device, vanilla_dp_model, dp_sat_model)