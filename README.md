# Fisher-Informed Differential Privacy (Fisher DP-SGD)

This repository implements curvature-aware differential privacy for CIFAR-10 training using Fisher information matrices. The key innovation is using Fisher-informed noise that adapts to the loss surface geometry, providing better privacy-utility tradeoffs compared to vanilla DP-SGD.

## 🎯 **Three-Method Research Platform**

This repository provides a comprehensive comparison of three major DP-SGD enhancement methods:

1. **Fisher DP-SGD**: Curvature-aware noise adaptation using Fisher information
2. **DP-SAT**: Differentially Private Sharpness-Aware Training ([Park et al., ICML 2023](https://proceedings.mlr.press/v202/park23g.html))
3. **Vanilla DP-SGD**: Standard baseline ([Abadi et al., CCS 2016](https://dl.acm.org/doi/abs/10.1145/2976749.2978318))

## 🚀 **Quick Start**

### 1. **Install Dependencies**
```bash
pip install torch torchvision numpy scipy scikit-learn tqdm opacus
```

### 2. **Run Comprehensive Comparison**
```bash
# Compare all three methods with proper privacy accounting
python main.py --mps --compare-others \
    --target-epsilon 10.0 --epochs 50 --adaptive-clip \
    --lambda-flatness 0.01 --run-mia --mia-size 1000

# User-level DP (synthetic users)
python main.py --mps --compare-others \
    --target-epsilon 10.0 --users 10 --run-mia
```

### 3. **Run Ablation Study (Fisher + DP-SAT Synergy)**
```bash
# Explore synergistic combination of Fisher DP and DP-SAT
python ablation.py --mps --target-epsilon 10.0 --epochs 20 \
    --k 64 --lambda-flatness 0.01 --run-mia --adaptive-clip
```

## 🏗️ **Core Components**

- **`fisher_dp_sgd.py`**: Fisher-informed DP-SGD with curvature-aware noise
- **`dp_sgd.py`**: Vanilla DP-SGD baseline ([Abadi et al., CCS 2016](https://dl.acm.org/doi/abs/10.1145/2976749.2978318))
- **`dp_sat.py`**: DP-SAT implementation ([Park et al., ICML 2023](https://proceedings.mlr.press/v202/park23g.html))
- **`ablation.py`**: Ablation study exploring Fisher + DP-SAT synergy
- **`privacy_accounting.py`**: Proper privacy accounting using Opacus RDP
- **`mia.py`**: Membership inference attack evaluation

## ✅ **Proper Privacy Accounting (Default)**

**This repository uses proper privacy accounting by default for scientifically valid results.**

### ✅ **Recommended Usage**
```bash
# Proper accounting (default) - ensures fair comparison
python main.py --compare-others --target-epsilon 10.0
```
- Uses Opacus RDP accountant for accurate composition
- Ensures all methods have identical privacy cost
- Valid scientific comparison

### ⚠️ **Legacy Mode (For Reproduction Only)**
```bash
# Legacy accounting (deprecated - use only to reproduce old experiments)
python main.py --compare-others --use-legacy-accounting --epsilon 10.0
```

## 🎯 **Method Descriptions**

### Fisher DP-SGD
Addresses the fundamental limitation of vanilla DP-SGD: **isotropic noise in anisotropic loss landscapes**.

**Key Innovation**: Shape noise according to Fisher information F = E[∇log p(y|x) ∇log p(y|x)ᵀ]

**Algorithm**:
- **Fisher subspace**: Anisotropic noise shaped by F⁻¹/²
- **Orthogonal complement**: Optional isotropic noise in remaining directions
- **Result**: Less noise in flat directions, more noise in steep directions

### DP-SAT (Differentially Private Sharpness-Aware Training)
Based on [Park et al., ICML 2023](https://proceedings.mlr.press/v202/park23g.html). Addresses the problem of **sharp loss landscapes** that cause DP-SGD to fail.

**Key Innovation**: Guide optimization toward flatter minima for better noise robustness.

**Algorithm**:
1. Standard DP-SGD: Compute clipped + noisy gradient `g_priv`
2. **Flatness adjustment**: `g_flat = λ * g_priv / ||g_priv||_2`
3. Final update: `θ ← θ - η(g_priv + g_flat)`

**Properties**:
- **No extra privacy cost**: Flatness adjustment is deterministic post-processing
- **Same accountant**: Uses identical noise and clipping as vanilla DP-SGD
- **Orthogonal to Fisher**: Can be combined with curvature-aware methods

### Vanilla DP-SGD
Standard baseline from [Abadi et al., CCS 2016](https://dl.acm.org/doi/abs/10.1145/2976749.2978318).

**Algorithm**:
1. Compute per-sample gradients
2. Clip to fixed L2 norm
3. Add isotropic Gaussian noise
4. Average and apply update

## 🔬 **Ablation Study: Fisher + DP-SAT + Calibration**

The `ablation.py` file explores the **comprehensive combination** of Fisher DP, DP-SAT, and Influence Function Calibration:

### Motivation
- **Fisher DP**: Shapes noise according to loss curvature (geometric)
- **DP-SAT**: Guides optimization toward flatter minima (optimization)  
- **Influence Function Calibration**: Adjusts model using public data (post-processing)
- **Hypothesis**: These three orthogonal approaches can be combined for enhanced performance

### Ablation Variants
1. **Fisher DP + Normal Optimizer** (baseline)
2. **Fisher DP + DP-SAT Optimizer** (synergistic combination)
3. **Fisher DP + Normal + Influence Function Calibration** (calibration baseline)
4. **Fisher DP + DP-SAT + Influence Function Calibration** (triple combination)

### Synergistic Algorithm
```
θ_{t+1} = θ_t - η(g_fisher_priv + λ * g_{t-1}^{fisher_priv} / ||g_{t-1}^{fisher_priv}||_2)
```

Where `g_fisher_priv` is the Fisher-informed noisy gradient and `g_{t-1}^{fisher_priv}` is from the previous step (following official DP-SAT implementation).

### Usage Examples
```bash
# Basic ablation study with calibration
python ablation.py --mps --target-epsilon 10.0 --epochs 20 \
    --k 64 --lambda-flatness 0.01 --efficient --method linear

# Fast calibration with linear approximation
python ablation.py --mps --efficient --method linear --calibration-k 50 \
    --target-epsilon 8.0 --run-mia

# Accurate calibration with original method (slower)
python ablation.py --mps --method original --calibration-k 200 \
    --target-epsilon 10.0 --epochs 15

# Sample-level vs User-level comparison with calibration
python ablation.py --mps --sample-level --target-epsilon 8.0 --efficient --run-mia
python ablation.py --mps --users 10 --target-epsilon 8.0 --efficient --run-mia

# Parameter sensitivity analysis
python ablation.py --mps --lambda-flatness 0.005 --calibration-k 100 --target-epsilon 10.0  # Conservative
python ablation.py --mps --lambda-flatness 0.02 --calibration-k 150 --target-epsilon 10.0   # Aggressive
```

### Expected Outcomes
```
🔬 Synergy Analysis:
   • Fisher DP + Normal:         76.20%
   • Fisher DP + DP-SAT:         78.45%
   • Synergy gain:               +2.25%

📐 Calibration Analysis:
   • Fisher DP + Normal + Calib: 77.80%
   • Fisher DP + DP-SAT + Calib: 80.10%
   • Calibration gain (Normal):  +1.60%
   • Calibration gain (DP-SAT):  +1.65%

🏆 Overall Best Performance:
   🥇 Fisher DP + DP-SAT + Calibration: 80.10%
   🎉 TRIPLE COMBINATION: All three techniques work together!
```

### Calibration Methods
- **Linear (`--method linear`)**: Fast gradient-based approximation (default)
- **Batch (`--method batch`)**: Diagonal Fisher approximation (medium speed)
- **Original (`--method original`)**: Full Hessian inverse computation (slow but accurate)

### Calibration Configuration
```bash
# Efficient calibration settings
--efficient --method linear --calibration-k 50-100    # Fast, good for experiments

# Balanced calibration settings  
--efficient --method batch --calibration-k 100-200    # Medium speed and accuracy

# High-quality calibration settings
--method original --calibration-k 200-500             # Slow but most accurate
```

### Influence Function Protocol
The calibration follows the experimental protocol exactly:

1. **Critical Slice Definition**: Extract target class samples (e.g., CIFAR-10 "cat" class)
2. **Influence Score Computation**: `α(z) = -∑_{s∈S_crit} ∇_θℓ(s,θ̂_DP)^T H^{-1} ∇_θℓ(z,θ̂_DP)`
3. **Sample Selection**: Choose top-k samples with lowest α(z) (most helpful)
4. **Deterministic Bias**: `Δθ_w = -1/n * H^{-1} ∑_{z∈P} w_z ∇_θℓ(z,θ̂_DP)`
5. **Model Calibration**: `θ̂_DP^* = θ̂_DP + Δθ_w`

### Target Class Configuration
```bash
# Default: CIFAR-10 "cat" class (class 3)
python ablation.py --target-class 3

# Other CIFAR-10 classes
python ablation.py --target-class 0  # airplane
python ablation.py --target-class 1  # automobile  
python ablation.py --target-class 2  # bird
python ablation.py --target-class 4  # deer
# ... etc
```

## 🛡️ **Privacy Evaluation**

### Membership Inference Attacks (MIA)
Comprehensive evaluation including:
- **Yeom Confidence Attack**: Uses model confidence scores
- **Shokri Shadow Model Attack**: Trains shadow models to learn membership patterns
- **Statistical significance testing**: Multiple runs with t-tests
- **Worst-case AUC analysis**: Across all attack types

### Three-Way MIA Comparison
```bash
# Compare all methods with MIA evaluation
python main.py --compare-others --target-epsilon 10.0 \
    --run-mia --mia-size 1000
```

## 📊 **Expected Results**

With proper privacy accounting at ε = 10.0:

| Method | Test Accuracy | Worst-case MIA AUC | Privacy Protection |
|--------|---------------|---------------------|-------------------|
| Baseline | ~85% | 0.95+ | None |
| Vanilla DP | ~75% | 0.60-0.65 | Moderate |
| DP-SAT | ~81% | 0.58-0.62 | Strong |
| Fisher DP | ~78% | 0.55-0.60 | Strong |

**Key Insights**:
- **DP-SAT**: +6% accuracy over Vanilla DP through flatter minima
- **Fisher DP**: Better privacy protection through curvature-aware noise
- **Synergy**: Fisher + DP-SAT combination may provide additional benefits

## 🔧 **Configuration Options**

### Privacy Parameters
```bash
# Different privacy levels
python main.py --target-epsilon 1.0    # High privacy
python main.py --target-epsilon 10.0   # Moderate privacy
python main.py --target-epsilon 20.0   # Lower privacy
```

### DP Modes
```bash
# Sample-level DP (traditional)
python main.py --sample-level --target-epsilon 10.0

# User-level DP (synthetic users)
python main.py --users 10 --target-epsilon 10.0
```

### Model Configuration
```bash
# Target specific layers
python main.py --dp-layer "conv1"        # Single layer
python main.py --dp-layer "conv1,conv2"  # Multiple layers
python main.py --dp-layer "all"          # All parameters

# Fisher subspace dimension
python main.py --k 32   # Lower dimensional (faster)
python main.py --k 256  # Higher dimensional (more accurate)

# DP-SAT flatness tuning
python main.py --lambda-flatness 0.01   # Default
python main.py --lambda-flatness 0.02   # More aggressive
```

### Experimental Features
```bash
# Adaptive clipping
python main.py --adaptive-clip --quantile 0.95

# Complement noise control
python main.py --full-complement-noise  # Add orthogonal noise (default: off)
```

## 🔬 **Research Applications**

### Fair Comparison Studies
1. Use default proper privacy accounting for valid results
2. All methods automatically use identical privacy parameters
3. Validation with `validate_privacy_comparison()` is automatic

### Privacy-Utility Tradeoffs
```bash
# Test different privacy levels
for eps in 1.0 5.0 10.0 20.0; do
    python main.py --target-epsilon $eps --compare-others --run-mia
done
```

### Ablation Analysis
```bash
# Clean previous results and run comprehensive ablation
python ablation.py --clean --target-epsilon 10.0 --epochs 30 \
    --dataset-size 15000 --run-mia --adaptive-clip
```

## 📚 **References**

- **DP-SAT**: Park, J., Kim, H., Choi, Y., & Lee, J. (2023). Differentially Private Sharpness-Aware Training. *Proceedings of the 40th International Conference on Machine Learning*, 202:27204-27224. [Link](https://proceedings.mlr.press/v202/park23g.html)

- **Vanilla DP-SGD**: Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). Deep learning with differential privacy. *Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security*, 308-318. [Link](https://dl.acm.org/doi/abs/10.1145/2976749.2978318)

## 🤝 **Contributing**

Contributions welcome! Key areas:
- Additional privacy accounting methods
- More sophisticated MIA attacks
- Support for other datasets/models
- Ablation study extensions

## ⚖️ **License**

MIT License - see LICENSE file for details. 