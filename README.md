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
- **Positively Correlated** (default): `noise ∝ √λ` - More noise in high curvature directions
- **Negatively Correlated**: `noise ∝ 1/√λ` - Less noise in high curvature directions

### DP-SAT: Flatter Minima
**Problem**: Sharp loss landscapes cause DP-SGD to fail.
**Solution**: Guide optimization toward flatter minima for better noise robustness.

### Vanilla DP-SGD: Standard Baseline
Standard per-sample gradient clipping + isotropic Gaussian noise.

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

# Fisher strategies
--positively_correlated_noise   # More noise in high curvature (default)
--negatively_correlated_noise   # Less noise in high curvature

# DP modes
--sample-level          # Traditional DP
--users 10             # User-level DP

# Fisher subspace
--k 64                 # Smaller subspace (faster)
--k 256                # Larger subspace (more accurate)
```

### Parameter Relationship: `--k` vs `--dp-layer`
- **`--dp-layer` (Scope)**: Selects which parameters (total count $P$) are trained with DP.
- **`--k` (Fidelity)**: Sets the rank of the Fisher approximation within that scope.
  - **Constraint**: $k \le P$ (automatically clamped if larger).
  - **Tradeoff**: Higher $k$ captures more curvature information but requires more memory/compute. Lower $k$ is faster but uses a coarser approximation.

### Advanced Features
```bash
# Adaptive clipping
--adaptive-clip --quantile 0.95

# DP-SAT flatness
--lambda-flatness 0.01

# MIA evaluation
--run-mia --mia-size 1000

# Target specific layers
--dp-layer "conv1,conv2"
```

## 🔬 **Research Applications**

### Compare Noise Strategies
```bash
uv run main.py --positively_correlated_noise --target-epsilon 8.0 --compare-others --run-mia
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