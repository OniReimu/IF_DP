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
pip install torch torchvision numpy scipy scikit-learn tqdm opacus
```

### Run Single Experiments
```bash
# Compare all three methods (default: positively correlated noise)
uv run main.py --mps --compare-others --target-epsilon 10.0 --run-mia

# Try negatively correlated noise strategy
uv run main.py --mps --compare-others --negatively_correlated_noise --target-epsilon 10.0

# Run ablation study with Fisher + DP-SAT synergy
uv run ablation.py --mps --target-epsilon 10.0 --k 64 --lambda-flatness 0.01 --run-mia
```

## 🎯 **Method Overview**

### Fisher DP-SGD: Curvature-Aware Noise
**Problem**: Vanilla DP-SGD uses isotropic noise in anisotropic loss landscapes.
**Solution**: Shape noise according to Fisher information F = E[∇log p(y|x) ∇log p(y|x)ᵀ]

**Noise Scaling Strategies**:
- **Positively Correlated**: `noise ∝ √λ` - More noise in high curvature directions
- **Negatively Correlated** (default): `noise ∝ 1/√λ` - Less noise in high curvature directions

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

- **Public data (45k)**: Large subset of CIFAR-10 train → baseline pretraining
- **Private data (5k)**: Small subset of CIFAR-10 train → DP finetuning  
  - Controlled by `--dataset-size` (default: 5,000)
- **Evaluation data (10k)**: Full CIFAR-10 test → final accuracy measurement

**Note**: This treats most of the training set as "public" for simulation purposes. In real applications, public data would come from a genuinely public source (e.g., ImageNet pretraining, web-scraped images, etc.).

### Why This Approach?

Computing Fisher information matrices for all parameters is computationally expensive. By:
- Training the baseline on large public data (strong pretrain),
- Freezing most layers (no DP cost),
- Applying Fisher DP-SGD only to selected layers (manageable cost),

we achieve **strict DP** while maintaining computational feasibility and strong baseline performance.

### Implementation Details

- **Baseline**: Trained on `pub_loader` (45k public samples) in `ablation.py`.
- **Freezing**: All parameters NOT in `--dp-layer`/`--dp-param-count` are automatically frozen during DP training.
- **Verification**: The code logs `🔒 Strict DP: Frozen N parameter groups` to confirm the setup.

**Example**:
```bash
# Default: 45k public, 5k private (from 50k trainset)
uv run ablation.py --dp-layer "conv1,conv2" --target-epsilon 2.0

# Custom split: 40k public, 10k private
uv run ablation.py --dataset-size 10000 --dp-layer "conv1,conv2" --target-epsilon 2.0
```

This ensures the final model has a formal $(\epsilon, \delta)$-DP guarantee w.r.t. the private training dataset.

## 📊 **Experimental Results (negatively related noise)**

We use the following configuration:
```bash
uv run ablation.py --mps --k 2048 --epochs 100 --dataset-size 50000 --target-epsilon 2.0 --delta 1e-5 --dp-layer conv1,conv2 --clip-radius 2.0 --run-mia --users 100 --calibration-k 200 --trust-tau 0.0005 --reg 10
```

### 100 Users
```
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  48.3%     0.509
   Vanilla DP-SGD + DP-SAT         50.2%     0.500
   Fisher DP + Normal              65.7%     0.498
   Fisher DP + DP-SAT              63.4%     0.493
   Fisher DP + Normal + Calib      66.8%     0.491
   Fisher DP + DP-SAT + Calib      64.2%     0.506
```

**Key Discovery**: Fisher DP-SGD shows significant improvements over vanilla DP-SGD, with Fisher DP + Normal + Calibration achieving the best accuracy of 66.8% while maintaining strong privacy protection (1-AUC = 0.491).

### Sample-Level DP

Using sample-level configuration:
```bash
uv run ablation.py --mps --k 2048 --epochs 100 --dataset-size 50000 --target-epsilon 5.0 --delta 1e-5 --dp-layer conv1,conv2 --clip-radius 1.0 --run-mia --sample-level --calibration-k 200 --trust-tau 0.0005 --reg 10
```

```
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  59.8%     0.473
   Vanilla DP-SGD + DP-SAT         59.4%     0.477
   Fisher DP + Normal              64.0%     0.464
   Fisher DP + DP-SAT              65.7%     0.534
   Fisher DP + Normal + Calib      64.6%     0.461
   Fisher DP + DP-SAT + Calib      66.4%     0.536
```

**Sample-Level Discovery**: Fisher DP + DP-SAT + Calibration achieves the highest accuracy of 66.4% in sample-level DP, demonstrating the effectiveness of combining all three techniques.

## 🔧 **Configuration Options**

### Core Parameters
```bash
# Privacy levels
--target-epsilon 1.0    # High privacy
--target-epsilon 10.0   # Moderate privacy

# Dataset sizes (Simulation Setup - Option B)
--dataset-size 5000     # Size of PRIVATE dataset (from CIFAR-10 trainset, default: 5,000)
                        # Public dataset = remaining trainset samples (default: 45,000)
                        # Evaluation dataset = full testset (10,000)
                        # Example: --dataset-size 10000 → 40k public, 10k private, 10k eval

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

## 🔬 **Research Applications**

### Compare Noise Strategies
```bash
# (positively correlated noise option removed)
uv run main.py --negatively_correlated_noise --target-epsilon 8.0 --compare-others --run-mia
```

### Privacy-Utility Tradeoffs
```bash
for eps in 1.0 5.0 10.0 20.0; do
    uv run main.py --target-epsilon $eps --compare-others --run-mia
done
```

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