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
python main.py --mps --compare-others --target-epsilon 10.0 --run-mia

# Try negatively correlated noise strategy
python main.py --mps --compare-others --negatively_correlated_noise --target-epsilon 10.0

# Run ablation study with Fisher + DP-SAT synergy
python ablation.py --mps --target-epsilon 10.0 --k 64 --lambda-flatness 0.01 --run-mia
```

### Run Systematic Validation Studies
```bash
# 1. List available experiment configurations
python visual_discovery_analysis.py --list-configs

# 2. Run user count sensitivity experiments (10 experiments total)
python visual_discovery_analysis.py --config validation_config_users_num.json

# 3. Run clip radius sensitivity experiments (10 experiments total)
python visual_discovery_analysis.py --config validation_config_clip_radius.json

# 4. Generate comprehensive plots from results
python visual_plotter.py --latest
```

## 🎨 **Configuration-Driven Validation System**

**NEW**: Systematic experiment framework with perfect reproducibility.

### Available Configurations
- **`validation_config_users_num.json`**: User count sensitivity (25, 50, 100, 200, 400 users)
- **`validation_config_clip_radius.json`**: Clip radius sensitivity (0.5, 1.0, 2.0, 5.0, 10.0)

### Results Structure
```
📁 validation_configs/           # Experiment configurations
📁 validation_results/           # Results with seed tracking
📁 validation_plots/             # Generated visualizations
```

### Create Custom Studies
Create new config file in `validation_configs/`:
```json
{
  "experiment_name": "epsilon_sensitivity_analysis",
  "common_args": {"k": 2048, "epochs": 50, "users": 100, "mps": true},
  "experiments": [
    {"name": "eps_1.0_positive", "target-epsilon": 1.0, "positively_correlated_noise": true},
    {"name": "eps_1.0_negative", "target-epsilon": 1.0, "negatively_correlated_noise": true}
  ]
}
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

## 📊 **Experimental Results**

We use the following configuration:
```bash
uv run ablation_optimized.py --mps --k 2048 --epochs 100 --dataset-size 50000 --target-epsilon 2.0 --delta 1e-5 --dp-layer conv1,conv2 --clip-radius 2.0 --run-mia --users 100 --calibration-k 200 --trust-tau 0.0005 --reg 10
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
uv run ablation_optimized.py --mps --k 2048 --epochs 100 --dataset-size 50000 --target-epsilon 5.0 --delta 1e-5 --dp-layer conv1,conv2 --clip-radius 1.0 --run-mia --sample-level --calibration-k 200 --trust-tau 0.0005 --reg 10
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
python main.py --positively_correlated_noise --target-epsilon 8.0 --compare-others --run-mia
python main.py --negatively_correlated_noise --target-epsilon 8.0 --compare-others --run-mia
```

### Privacy-Utility Tradeoffs
```bash
for eps in 1.0 5.0 10.0 20.0; do
    python main.py --target-epsilon $eps --compare-others --run-mia
done
```

### Systematic Validation
```bash
# Run comprehensive experiments
python visual_discovery_analysis.py --config validation_config_users_num.json
python visual_plotter.py --latest
```

## 🏗️ **Core Files**

- **`main.py`**: Single experiment comparison
- **`ablation.py`**: Fisher + DP-SAT synergy study
- **`visual_discovery_analysis.py`**: Systematic validation framework
- **`visual_plotter.py`**: Results visualization
- **`fisher_dp_sgd.py`**: Fisher-informed DP-SGD implementation
- **`dp_sat.py`**: DP-SAT implementation
- **`mia.py`**: Membership inference attack evaluation

## 📚 **References**

- **DP-SAT**: Park, J., et al. (2023). Differentially Private Sharpness-Aware Training. *ICML 2023*. [Link](https://proceedings.mlr.press/v202/park23g.html)
- **Vanilla DP-SGD**: Abadi, M., et al. (2016). Deep learning with differential privacy. *CCS 2016*. [Link](https://dl.acm.org/doi/abs/10.1145/2976749.2978318)

## ⚖️ **License**

MIT License - see LICENSE file for details. 