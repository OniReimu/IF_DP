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

### Run Single Experiments
```bash
# Compare all three methods (default: positively correlated noise)
uv run main.py --mps --compare-others --target-epsilon 10.0 --run-mia

# Noise strategy is fixed to negatively correlated (default); no flag needed
uv run main.py --mps --compare-others --target-epsilon 10.0

# Run ablation study with Fisher + DP-SAT synergy
uv run ablation.py --mps --target-epsilon 10.0 --k 64 --run-mia
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

**For CIFAR-10 ablation studies**, we simulate a "large public corpus + small private dataset" scenario:

- **Public pretraining data (40k)**: Large subset of CIFAR-10 train → baseline pretraining
- **Public calibration data (5k)**: Small subset of CIFAR-10 train → influence function calibration
- **Private data (5k)**: Small subset of CIFAR-10 train → DP finetuning  
  - Controlled by `--dataset-size` (default: 5,000)
- **Evaluation data (10k)**: Full CIFAR-10 test → final accuracy measurement

**Optimization**: The public data is split into pretraining (40k) and calibration (5k) subsets. This significantly speeds up influence function computation, as we only need to compute influences on 5k samples instead of 45k, while still maintaining strong baseline performance from 40k pretraining samples.

**Note**: This treats most of the training set as "public" for simulation purposes. In real applications, public data would come from a genuinely public source (e.g., ImageNet pretraining, web-scraped images, etc.).

### Why This Approach?

Computing Fisher information matrices for all parameters is computationally expensive. By:
- Training the baseline on large public data (strong pretrain),
- Freezing most layers (no DP cost),
- Applying Fisher DP-SGD only to selected layers (manageable cost),

we achieve **strict DP** while maintaining computational feasibility and strong baseline performance.

### Implementation Details

- **Baseline**: Trained on `pub_loader` (45k public samples) in `ablation.py`.
  - **Caching**: Pretrained baselines are automatically saved as `Pretrain_{model}_{epochs}_public.pth`
  - Current recipe: strong-from-scratch SGD (lr=0.1, momentum=0.9, weight_decay=5e-4, cosine schedule)
  - On subsequent runs with same model/epochs, the cached baseline is loaded (massive speedup!)
  - Use `--clean` flag to force retraining from scratch
- **Freezing**: All parameters NOT in `--dp-layer`/`--dp-param-count` are automatically frozen during DP training.
- **Verification**: The code logs `🔒 Strict DP: Frozen N parameter groups` to confirm the setup.

**Example**:
```bash
# Default: 40k pretrain + 5k calib public, 5k private (from 50k trainset)
uv run ablation.py --dp-param-count 20000 --target-epsilon 2.0

# Custom split: 35k pretrain + 5k calib, 10k private
uv run ablation.py --dataset-size 10000 --dp-param-count 20000 --target-epsilon 2.0

# Force retrain baseline (ignore cached pretrained model)
uv run ablation.py --clean --dp-layer "conv1,conv2" --target-epsilon 2.0

# Cached baseline: Second run is much faster (skips 300 epochs of pretraining!)
uv run ablation.py --model-type resnet18 --epochs 300 --dp-layer "resnet.layer4" --target-epsilon 2.0
# → Creates: Pretrain_resnet18_300_public.pth
# Next run with same model+epochs will load this cached baseline
```

This ensures the final model has a formal $(\epsilon, \delta)$-DP guarantee w.r.t. the private training dataset.



## 🔧 **Configuration Options**

### Core Parameters
```bash
# Privacy levels
--target-epsilon 1.0    # High privacy
--target-epsilon 10.0   # Moderate privacy

# Dataset sizes (Simulation Setup - Option B)
--dataset-size 5000     # Size of PRIVATE dataset (from CIFAR-10 trainset, default: 5,000)
                        # Public pretrain = 50k - dataset_size - 5k (default: 40,000)
                        # Public calibration = fixed 5k (for efficient influence computation)
                        # Evaluation = full testset (10,000)
                        # Example: --dataset-size 10000 → 35k pretrain, 5k calib, 10k private, 10k eval

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
  - The sampling rate for privacy accounting is approximately `1 / K` (one user per batch with `UserBatchSampler`), so:
    - For fixed target \((\epsilon,\delta)\), changing `--users` changes the required noise multiplier.  
    - Larger `K` → smaller per-step sample rate → Opacus can often use **smaller noise** for the same target ε, but each user also has fewer samples.

**Important nuance in this repo**:
- In practice, the **effective harshness** of DP-SGD depends on both:
  - the sampling rate used in the RDP accountant (controls σ), and  
  - the **scale of the clipped gradients** (per-sample vs per-user sums).
- In our experiments:
  - User-level with many users (e.g. `--users 100`) can require a **large σ** and has large per-user gradients → Vanilla DP-SGD accuracy can collapse (≈10%).  
  - User-level with fewer users (e.g. `--users 10`) yields smaller σ and milder clipping → accuracy improves.  
  - Sample-level DP uses much smaller sampling rate (batch_size / dataset_size) and per-sample gradients → the accountant chooses **smaller σ**, and accuracy can look closer to the “10-user” regime rather than the harsh 100-user case.

Use **sample-level DP** if you care about individual examples, and **user-level DP** if you want to protect an entire user’s contribution (all their samples) as one DP unit.

### Parameter Relationship: `--k` vs `--dp-layer`
- **`--dp-layer` (Scope)**: Selects which parameters (total count $P$) are trained with DP. **All other parameters are frozen** (trained on public data only), ensuring strict DP.
- **`--k` (Fidelity)**: Sets the rank of the Fisher approximation within that scope.
  - **Constraint**: $k \le P$ (automatically clamped if larger).
  - **Tradeoff**: Higher $k$ captures more curvature information but requires more memory/compute. Lower $k$ is faster but uses a coarser approximation.

### Advanced Features
```bash
# Adaptive clipping (disabled by default)
--adaptive-clip --quantile 0.95

# DP-SAT flatness
--lambda-flatness 0.01

# MIA evaluation
--run-mia --mia-size 1000

# Target specific layers (Strict DP: other layers are frozen)
--dp-layer "conv1,conv2"               # CNN: matches prefixes conv1, conv2, ...
--dp-layer "resnet.conv1"              # ResNet-18 stem only (smallest)
--dp-layer "resnet.conv1,resnet.bn1"   # ResNet-18 stem + BN
--dp-layer "resnet.conv1,resnet.layer1"  # add first block; expand with layer2, layer3, layer4 as needed

# OR: Use parameter budget (mutually exclusive with --dp-layer)
--dp-param-count 20000                 # Train up to 20,000 parameters with DP (smart selection)
```

**Note**: When using `--dp-layer`, the specified layers are trained with DP on private data, while all other layers are **frozen** (pre-trained on public data). This ensures strict $(\epsilon, \delta)$-DP for the entire model.

When using `--dp-param-count N`, the code smartly selects **complete parameters** (no partial layers) that maximize parameter usage within budget `N`. It selects parameters in model order, skipping any that would exceed the budget. This ensures:
- Only complete parameters are selected (no partial layer training)
- Maximum utilization of the parameter budget
- Predictable behavior across different architectures

**Example**: With budget 20,000 and layers [conv1: 896, conv2: 18,496, conv3: 73,728]:
- Selects: conv1 + conv2 = 19,392 parameters (96.96% efficiency)
- Skips: conv3 (would exceed budget)

**`--dp-layer` vs `--dp-param-count`**:
- These options are **mutually exclusive**
- `--dp-layer`: Select parameters by layer names (more semantic, substring matching)
- `--dp-param-count`: Select parameters by budget (architecture-agnostic, greedy knapsack)

**Layer naming (prefix match)**:
- **CNN**: `conv1`, `conv2`, `conv3`, `fc1`, `fc2`
- **ResNet-18**: `resnet.conv1`, `resnet.bn1`, `resnet.layer1`, `resnet.layer2`, `resnet.layer3`, `resnet.layer4`, `resnet.fc`
- **EfficientNet-B0**: `efficientnet.features.0`, `efficientnet.features.1`, …, `efficientnet.features.8`, `efficientnet.classifier`



## 🏗️ **Core Files**

- **`main.py`**: Single experiment comparison
- **`ablation.py`**: Fisher + DP-SAT synergy study
- **`fisher_dp_sgd.py`**: Fisher-informed DP-SGD implementation
- **`dp_sat.py`**: DP-SAT implementation
- **`mia.py`**: Membership inference attack evaluation

## 📚 **References**

- **DP-SAT**: Park, J., et al. (2023). Differentially Private Sharpness-Aware Training. *ICML 2023*. [Link](https://proceedings.mlr.press/v202/park23g.html)
- **Vanilla DP-SGD**: Abadi, M., et al. (2016). Deep learning with differential privacy. *CCS 2016*. [Link](https://dl.acm.org/doi/abs/10.1145/2976749.2978318)

## ⚖️ **License**

MIT License - see LICENSE file for details. 
