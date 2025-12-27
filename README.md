# Fisher-Informed Differential Privacy (Fisher DP-SGD)

This repository implements curvature-aware differential privacy for CIFAR-10 training using Fisher information matrices. The key innovation is using Fisher-informed noise that adapts to the loss surface geometry, providing better privacy-utility tradeoffs compared to vanilla DP-SGD.

## 🎯 **Three-Method Research Platform**

This repository provides a comprehensive comparison of three major DP-SGD enhancement methods:

1. **Fisher DP-SGD**: Curvature-aware noise adaptation using Fisher information
2. **DP-SAT**: Differentially Private Sharpness-Aware Training ([Park et al., ICML 2023](https://proceedings.mlr.press/v202/park23g.html))
3. **Vanilla DP-SGD**: Standard baseline ([Abadi et al., CCS 2016](https://dl.acm.org/doi/abs/10.1145/2976749.2978318))

## 🚀 **Quick Start**

### Install Dependencies
```bash
uv pip install torch torchvision numpy scipy scikit-learn tqdm opacus
```

### Run Experiments
```bash
# Full ablation study (all variants including calibration)
uv run ablation.py --mps --model-type efficientnet --users 100 --dp-epochs 20 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --run-mia

# Fast ablation (no calibration variants, faster iteration)
uv run ablation_fast_no_calib.py --mps --model-type efficientnet --users 100 --dp-epochs 10 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --dp-sat-mode fisher --run-mia

# Non-IID split with public rehearsal
uv run ablation_fast_no_calib.py --mps --model-type efficientnet --non-iid --public-pretrain-exclude-classes 0,1 --users 100 --dp-epochs 10 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --rehearsal-lambda 0.1 --run-mia
```

## 🧱 Modular Architecture Overview

To keep experiments decoupled and extensible, the refactor introduces first-class packages in the repo root:

```
models/               # Vision + language backbones implementing ModelBase
data/                 # Dataset builders wrapping torchvision + HF datasets
training/             # (future) composable pipelines
core/registry.py      # Lightweight registries for models & datasets
```

- **ModelBase** (`models/base.py`) standardises the forward/loss interface and enables registry-based instantiation via `build_model`.
- **DatasetBuilder** (`data/base.py`) encapsulates data loading, preprocessing, and split construction through a consistent `build()` API that returns `DatasetLoaders`.
- Shared utilities (`data/common.py`) normalise batch structures (tensor tuples, Hugging Face dicts, user-level batches) so training, evaluation, and attacks can consume any modality transparently.

Adding a new dataset or model now requires dropping a single module that registers itself through the registry decorators—no changes to the main training script.

## ✅ Supported Models & Datasets

| Modality | Models | Datasets |
|----------|--------|----------|
| Vision   | `cnn`, `resnet18`, `efficientnet_b0`, `vit_b16` | `cifar10`, `fashion_mnist`, `cifar100` |
| Text / LLM | `bert`, `qwen`, `llama3.1-8b` | `ag_news`, `wildchat` |

Use `--model <name>` and `--dataset <name>` when running `main.py`. The dataset builder exposes label mappings and feeds the requested number of labels into the model factory automatically.

Example:

```bash
uv run main.py \
  --dataset ag_news \
  --model bert \
  --tokenizer-name bert-base-uncased \
  --batch-size 32 \
  --target-epsilon 8.0 \
  --compare-others
```

## 🛠 Singularity-First Execution

All validation and testing are containerised. A typical LUMI command looks like:

```bash
singularity exec ${SIF} \
  bash -lc "
    source ${ENV} && \
    python main.py \
      --dataset cifar10 \
      --model vit \
      --batch-size 256 \
      --target-epsilon 6.0 \
      --compare-others \
      --run-mia
  "
```

The provided `run_on_lumi.sh` script shows a complete SLURM submission that uses the same pattern. Swap out the python entry point (e.g., `ablation.py`) or pass a config file as needed—no direct `python` invocation outside Singularity is required.

## 🎯 **Method Overview**

### Fisher DP-SGD: Curvature-Aware Noise
**Problem**: Vanilla DP-SGD uses isotropic noise in anisotropic loss landscapes.
**Solution**: Shape noise according to Fisher information F = E[∇log p(y|x) ∇log p(y|x)ᵀ]

**Noise Scaling Strategies**:
- **Negatively Correlated**: `noise ∝ 1/√λ` - Less noise in high curvature directions

### DP-SAT: Flatter Minima
**Problem**: Sharp loss landscapes cause DP-SGD to fail.
**Solution**: Guide optimization toward flatter minima for better noise robustness.

### Vanilla DP-SGD: Standard Baseline
Standard per-sample gradient clipping + isotropic Gaussian noise.

## 🔒 **Strict Differential Privacy Guarantee**

This repository implements **Option 1: Frozen Backbone + DP Finetuning** to provide formal $(\epsilon, \delta)$-DP guarantees for the entire released model.

### Architecture

**Strict DP Setup**:
1. **Baseline Training (Public Data)**: The baseline model is pre-trained on **public data only**. This ensures the initial weights for frozen layers contain no private information.
2. **DP Finetuning (Private Data)**: Only selected layers (specified via `--dp-layer` or `--dp-param-count`) are trained with DP-SGD on the **private dataset**. All other layers are **frozen** (`requires_grad=False`).
3. **Formal Guarantee**: The entire released model is $(\epsilon, \delta)$-DP with respect to the private dataset, because:
   - Frozen layers depend only on public data (no privacy leakage).
   - DP-trained layers use formal DP-SGD mechanisms (Fisher DP-SGD or Vanilla DP-SGD).

### Data Splitting (Simulation Setup)

**For CIFAR-10 ablation studies**, we simulate a "large public corpus + small private dataset" scenario with two modes:

### IID Mode (Default)
- **Public pretraining data (~10k)**: Subset of CIFAR-10 train → baseline pretraining
- **Public calibration data (5k)**: Small subset of CIFAR-10 train → influence function calibration
- **Private data (30k)**: Large subset of CIFAR-10 train → DP finetuning  
  - Controlled by `--dataset-size` (default: 30,000 for IID mode)
- **Evaluation data (10k)**: Full CIFAR-10 test → final accuracy measurement

### Non-IID Mode (`--non-iid`)
- **Public pretraining data (~27k)**: Subset of CIFAR-10 train **excluding specified classes** → baseline pretraining
- **Public calibration data (5k)**: Small subset of CIFAR-10 train (unchanged) → influence function calibration
- **Private data (~18k)**: Includes **all samples from excluded classes** + rehearsal buffer from non-excluded classes → DP finetuning
  - Excluded classes specified via `--public-pretrain-exclude-classes` (e.g., `0,1`)
  - Rehearsal buffer automatically added to prevent catastrophic forgetting (keeps excluded classes ≤ 50% of private set)
- **Evaluation data (10k)**: Full CIFAR-10 test → final accuracy measurement

**Optimization**: The public data is split into pretraining and calibration subsets. This significantly speeds up influence function computation, as we only need to compute influences on 5k samples instead of the full public set, while still maintaining strong baseline performance.

**Note**: This treats most of the training set as "public" for simulation purposes. In real applications, public data would come from a genuinely public source (e.g., ImageNet pretraining, web-scraped images, etc.).


### Why This Approach?

Computing Fisher information matrices for all parameters is computationally expensive. By:
- Training the baseline on large public data (strong pretrain),
- Freezing most layers (no DP cost),
- Applying Fisher DP-SGD only to selected layers (manageable cost),

we achieve **strict DP** while maintaining computational feasibility and strong baseline performance.

### Implementation Details

- **Baseline**: Trained on public pretraining data in `ablation.py` and `ablation_fast_no_calib.py`.
  - **Caching**: Pretrained baselines are automatically saved as `Pretrain_{dataset}_{model}_{epochs}_public_{iid_mode}.pth`
    - `iid_mode` is `"iid"` or `"noniid"` to prevent cache reuse across different data distribution modes
  - Current recipe: strong-from-scratch SGD (lr=0.1, momentum=0.9, weight_decay=5e-4, cosine schedule)
  - On subsequent runs with same model/epochs/IID mode, the cached baseline is loaded (massive speedup!)
  - Use `--clean` flag to force retraining from scratch
- **Freezing**: All parameters NOT in `--dp-layer`/`--dp-param-count` are automatically frozen during DP training (`requires_grad=False`).
- **Verification**: The code logs `🔒 Strict DP: Frozen N parameter groups` to confirm the setup.
- **Public Rehearsal**: All DP training functions support optional public rehearsal via `--rehearsal-lambda`:
  - Combines DP private gradient with non-DP public gradient: `g_total = g_priv_DP + λ * g_public`
  - Privacy-free post-processing (public gradient uses only public data)
  - Helps prevent catastrophic forgetting in non-IID scenarios

**Example**:
```bash
# IID mode (default): 30k private, ~10k public pretrain, 5k calibration
uv run ablation.py --dp-param-count 20000 --target-epsilon 2.0

# Non-IID mode: exclude classes 0,1 from public pretrain
uv run ablation.py --non-iid --public-pretrain-exclude-classes 0,1 --dp-param-count 20000 --target-epsilon 2.0

# Force retrain baseline (ignore cached pretrained model)
uv run ablation.py --clean --dp-layer "conv1,conv2" --target-epsilon 2.0

# Cached baseline: Second run is much faster (skips 100 epochs of pretraining!)
uv run ablation.py --model-type resnet18 --epochs 100 --dp-layer "resnet.layer4" --target-epsilon 2.0
# → Creates: Pretrain_cifar10_resnet18_100_public_iid.pth
# Next run with same model/epochs/IID mode will load this cached baseline
```

This ensures the final model has a formal $(\epsilon, \delta)$-DP guarantee w.r.t. the private training dataset.

## 📈 Privacy Profile Plot (ε(δ))

Use the ablation-style arguments to plot an ε(δ) privacy profile with the same dataset splits and accounting logic:

```bash
# User-level DP (matches ablation-style args)
uv run scripts/plot_privacy_profile.py \
  --dataset cifar10 \
  --target-epsilon 0.5 \
  --delta 1e-5 \
  --dp-epochs 3 \
  --users 200 \
  --clip-radius 2.0 \
  --non-iid \
  --public-pretrain-exclude-classes 0,1

# Sample-level DP
uv run scripts/plot_privacy_profile.py \
  --dataset cifar10 \
  --target-epsilon 1.0 \
  --delta 1e-5 \
  --dp-epochs 3 \
  --sample-level \
  --batch-size 128
```

The script rebuilds the same private/public splits as the ablation runners, computes the implied sampling rate and steps, and then plots ε(δ) with a highlighted marker at the target δ.

Important accounting notes:
- The privacy profile uses only the subsampled Gaussian mechanism parameters (q, T, C, sigma). Extra implementation details such as `--full-complement-noise`, DP-SAT, Fisher shaping, or calibration do not change the accountant curve and can only make privacy stronger in practice.
- The plot is valid only if the implemented step matches the assumed mechanism. If `--full-complement-noise` is disabled, DP correctness requires subspace-only updates (or otherwise bounding complement sensitivity); if complement updates are allowed, enable full complement noise to keep the per-step sensitivity bounded by C.
- User-level subsampling nuance: for “one user per step” training, a standard Poisson-subsampling approximation would use `q_user = 1 / num_users` and `T = dp_epochs × num_users`. In this repo’s current ablation runners and plotter we instead use an “effective” rate `q_eff = len(priv_loader) / len(priv_base)` (typically `num_users / num_private_samples`). This can be larger than `q_user` and therefore yields a more conservative (higher-noise) accountant solution for a fixed target ε. If you want the most standard reporting, use Poisson user subsampling (`q_user`) and state it explicitly (recommended for papers).



## 🔧 **Configuration Options**

### Core Parameters
```bash
# Privacy levels
--target-epsilon 1.0    # High privacy
--target-epsilon 10.0   # Moderate privacy

# Dataset sizes (IID mode default)
--dataset-size 30000   # Size of PRIVATE dataset (from CIFAR-10 trainset, default: 30,000 for IID)
                        # Public pretrain ≈ 10,000 (calculated from public_ratio=0.667)
                        # Public calibration = fixed 5k (for efficient influence computation)
                        # Evaluation = full testset (10,000)
                        # Non-IID mode: default dataset-size=5000, public_ratio=1.0

# Model architecture
--model-type cnn              # Simple CNN (default)
--model-type resnet18         # ResNet-18
--model-type efficientnet_b0  # EfficientNet-B0

# Fisher strategies (fixed default)
--negatively_correlated_noise   # Less noise in high curvature (default; positive option removed)

# DP modes
--sample-level          # Traditional DP
--users 10             # User-level DP

# Fisher subspace
--k 64                 # Smaller subspace (faster)
--k 256                # Larger subspace (more accurate)
```

### User-Level vs Sample-Level DP

- **Sample-level DP (`--sample-level`)**  
  - Each **training example** is one DP unit.  
  - We compute and clip **per-sample gradients** and add noise at the sample level.  
  - Intuitively, this is equivalent to “user-level DP with one sample per user” (i.e., \#users = \#samples).

- **User-level DP (`--users K`, without `--sample-level`)**  
  - The DP unit is an entire **synthetic user**, i.e., all samples belonging to that user.  
  - In code, we create `K` synthetic users by assigning each training sample a user id in round-robin and clip **per-user gradients**.  
  - Each DP step processes **one user per batch** via `UserBatchSampler`, so changing `--users` changes the *training dynamics*:
    - **More users (`K↑`)** → **more DP steps per epoch** (because `len(priv_loader) == K`) but **fewer samples per user** (each user batch is smaller).
    - **Fewer users (`K↓`)** → **fewer DP steps per epoch** but **more samples per user** (each user batch is larger).
  - **Privacy accounting note (implementation detail)**: in the ablation scripts we currently pass an *effective* sampling rate derived from loader/dataset sizes (e.g., `len(priv_loader) / len(priv_base)`), so the noise multiplier can change with `--users`. This is an approximation of user-level accounting in this repo’s current setup.

**Important nuance in this repo**:
- In practice, the **effective harshness** of DP-SGD depends on both:
  - the sampling rate used in the RDP accountant (controls σ), and  
  - the **scale of the clipped gradients** (per-sample vs per-user sums).
- In our experiments:
  - Increasing `--users` can make fine-tuning **move much more** (more DP steps), which can *recover private-only classes* but also cause **catastrophic forgetting** on the remaining classes if the private split is highly skewed.
  - Decreasing `--users` reduces the number of DP steps, so models stay closer to the public baseline (often higher “rest” accuracy) but may fail to adapt to private-only classes.

Use **sample-level DP** if you care about individual examples, and **user-level DP** if you want to protect an entire user’s contribution (all their samples) as one DP unit.

### Parameter Relationship: `--k` vs `--dp-layer`
- **`--dp-layer` (Scope)**: Selects which parameters (total count $P$) are trained with DP. **All other parameters are frozen** (trained on public data only), ensuring strict DP.
- **`--k` (Fidelity)**: Sets the rank of the Fisher approximation within that scope.
  - **Constraint**: $k \le P$ (automatically clamped if larger).
  - **Tradeoff**: Higher $k$ captures more curvature information but requires more memory/compute. Lower $k$ is faster but uses a coarser approximation.
  - **Noise magnitude**: By default, Fisher DP adds noise **only in the Fisher subspace** (top-$k$ directions), so noise ℓ₂ norm scales like $\sqrt{k} \times \sigma \times C$ instead of $\sqrt{P} \times \sigma \times C$ (vanilla DP-SGD). This makes Fisher DP's noise **much smaller** (e.g., $\sqrt{512}$ vs $\sqrt{20000}$ ≈ **6× smaller**), which can lead to better utility but makes direct comparison with vanilla DP-SGD less fair.

**For fairer comparison with vanilla DP-SGD**: Use `--full-complement-noise` to add isotropic noise in the **orthogonal complement** (remaining $P-k$ dimensions) in addition to the Fisher subspace noise. This makes total noise magnitude similar to vanilla DP-SGD ($\sqrt{P} \times \sigma \times C$) while still preserving curvature-aware benefits in the Fisher subspace. The tradeoff: utility may drop closer to vanilla levels, but privacy accounting becomes more aligned with standard DP-SGD mechanisms.

**Important**: Fisher DP uses **norm calibration** to ensure fair comparison. The `--clip-radius` argument sets the target Euclidean sensitivity Δ₂ (same as vanilla DP-SGD). During training, the code automatically calibrates a Mahalanobis threshold (`actual_radius`) that achieves the same effective sensitivity in Fisher space. Both clipping and noise scaling use this calibrated `actual_radius` to ensure the same privacy-utility tradeoff as vanilla DP-SGD.

### Advanced Features
```bash
# Adaptive clipping (disabled by default)
--adaptive-clip --quantile 0.95

# DP-SAT configuration
--dp-sat-mode fisher      # Fisher DP-SAT (for Fisher DP variants) or euclidean (for Vanilla DP-SGD)
--rho-sat 0.001          # Perturbation radius for DP-SAT

# Public rehearsal (prevents catastrophic forgetting in non-IID scenarios)
--rehearsal-lambda 0.1   # Mixing weight: g_total = g_priv_DP + λ * g_public (default: 1.0, set to 0 to disable)

# Non-IID data splits
--non-iid                                # Enable non-IID mode (requires --public-pretrain-exclude-classes)
--public-pretrain-exclude-classes 0,1   # Exclude classes 0,1 from public pretrain (moved to private)

# MIA evaluation
--run-mia --shadow-epochs 3             # Run membership inference attack (default: 3 shadow epochs)

# Target specific layers (Strict DP: other layers are frozen)
--dp-layer "conv1,conv2"               # CNN: matches prefixes conv1, conv2, ...
--dp-layer "resnet.conv1"              # ResNet-18 stem only (smallest)
--dp-layer "resnet.conv1,resnet.bn1"   # ResNet-18 stem + BN
--dp-layer "resnet.conv1,resnet.layer1"  # add first block; expand with layer2, layer3, layer4 as needed

# OR: Use parameter budget (mutually exclusive with --dp-layer)
--dp-param-count 20000                 # Train up to 20,000 parameters with DP (head-first selection)
```

**Note**: When using `--dp-layer`, the specified layers are trained with DP on private data, while all other layers are **frozen** (pre-trained on public data). This ensures strict $(\epsilon, \delta)$-DP for the entire model.

When using `--dp-param-count N`, the code uses **head-first selection** to prioritize classifier/head parameters, then fills remaining budget with backbone parameters. This ensures:
- Classifier parameters are always included when budget allows (critical for learning new classes)
- Only complete parameters are selected (no partial layer training)
- Maximum utilization of the parameter budget
- Predictable behavior across different architectures

**Example**: With budget 20,000 and EfficientNet-B0:
- Prioritizes: classifier (12,810 params) + early backbone layers
- Ensures model can learn private-only classes in non-IID scenarios

**`--dp-layer` vs `--dp-param-count`**:
- These options are **mutually exclusive**
- `--dp-layer`: Select parameters by layer names (more semantic, substring matching)
- `--dp-param-count`: Select parameters by budget (architecture-agnostic, greedy knapsack)

**Layer naming (prefix match)**:
- **CNN**: `conv1`, `conv2`, `conv3`, `fc1`, `fc2`
- **ResNet-18**: `resnet.conv1`, `resnet.bn1`, `resnet.layer1`, `resnet.layer2`, `resnet.layer3`, `resnet.layer4`, `resnet.fc`
- **EfficientNet-B0**: `efficientnet.features.0`, `efficientnet.features.1`, …, `efficientnet.features.8`, `efficientnet.classifier`



## 🏗️ **Core Files**

### Main Entry Points
- **`ablation.py`**: Full ablation study with all variants (pretrain → DP finetune → MIA → optional IF calibration)
- **`ablation_fast_no_calib.py`**: Fast ablation study without calibration variants (pretrain → DP finetune → MIA only)
- **`training/main.py`**: Single experiment comparison script for quick testing

### Core DP Training Modules (`core/`)
- **`core/fisher_dp_sgd.py`**: Fisher estimation (`compute_fisher`) + low-rank Fisher-DP-SGD training loop (`train_with_dp`)
- **`core/dp_sat.py`**: DP-SAT-style optimizer component (`train_with_dp_sat`) - baseline Euclidean DP-SGD + post-processing flatness term
- **`core/dp_sgd.py`**: Vanilla DP-SGD implementation (`train_with_vanilla_dp`) - standard baseline with Euclidean clipping + isotropic noise
- **`core/influence_function.py`**: Influence-function calibration routines (iterative + line search) for post-hoc slice repair
- **`core/mia.py`**: Membership inference attack evaluation (shadow model attack, confidence attack, etc.)
- **`core/privacy_accounting.py`**: Privacy accounting utilities (RDP accountant, noise multiplier calculation, privacy tracking)
- **`core/param_selection.py`**: Shared parameter selection utility (`select_parameters_by_budget`) for consistent DP parameter budgeting across all methods
- **`config/config.py`**: Configuration utilities (random seeds, dataset location, rehearsal buffer constants)
- **`core/device_utils.py`**: Device resolution utilities (MPS/CUDA/CPU detection and multi-GPU support)

## 📚 **References**

- **DP-SAT**: Park, J., et al. (2023). Differentially Private Sharpness-Aware Training. *ICML 2023*. [Link](https://proceedings.mlr.press/v202/park23g.html)
- **Vanilla DP-SGD**: Abadi, M., et al. (2016). Deep learning with differential privacy. *CCS 2016*. [Link](https://dl.acm.org/doi/abs/10.1145/2976749.2978318)

## ⚖️ **License**

MIT License - see LICENSE file for details. 
