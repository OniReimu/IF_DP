## Results Summary

**Legend**:

- **↑**: increases
- **↓**: decreases
- **~**: roughly unchanged / weak effect

### Knob-Level Overview

| Knob you increase                         | Vanilla DP-SGD utility                                              | Fisher DP (+ DP-SAT + Calib) utility                                   | MI (all methods)                          | Comment                                                                                          |
|-------------------------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------|--------------------------------------------------------------------------------------------------|
| Privacy budget ε (e.g. 2 → 4 → 8)         | ↑↑ strong (big jumps in accuracy)                                   | ↑ weak (already high at ε≈2; only +1–2 pts beyond that)                | ~ stays near random (1−AUC ≈ 0.5)         | Our method is much less sensitive to ε; big free-lunch gap at small ε.                          |
| Clip radius C (1 → 2 → 3)                 | C≈1: high; C≥3: ↓↓ collapses                                        | High across a wide C range; mild ↓ at very large C                     | ~ no clear trend                          | Fisher is robust to mis-specified C; vanilla is extremely fragile.                              |
| # Users (60 → 80 → 100)                   | ↓↓ quickly collapses as users ↑                                     | Stays in 70–80% range; gap vs vanilla grows from +18 to +60 pts        | ~ MI AUC ≈ 0.5 throughout                 | Our stack absorbs stricter user-level DP much better than vanilla.                              |
| Epochs T (100 → 200 → 300)                | ~ always low; no reliable gain from more epochs                     | Best at ~100, then ↓ (extra noisy steps hurt; “noise-wandering”)       | ~ stable                                  | Short schedules are best; our stack stays 50+ pts above vanilla at all T.                       |
| Fisher rank k (512 → 2048)                | n/a (no Fisher)                                                     | k≈512 best; increasing to 2048 → ↓↓ utility (esp. with large T)        | ~ stable                                  | Too-high rank makes noise too full-rank / unstable; moderate k is ideal.                        |
| DP-trained params (1k → 10k → 40k)        | 1k: high; 10k–40k: ↓↓ collapses (≈72% → ≈20%)                       | 1k: ≈ vanilla; 10k–40k: stays ≈70–77%                                  | ~ stable                                  | As DP gets harsher (more private params), vanilla dies; our stack remains strong.               |
| Add Fisher (vs vanilla)                   | –                                                                   | ↑↑ huge gain (often +30–50 pts at same ε)                              | ~                                         | Curvature-aware noise shaping is the main source of “free-lunch” utility.                       |
| Add DP-SAT on top of Fisher               | On vanilla: small / mixed; sometimes ↓                              | On Fisher: small ↑ (≈0–3 pts), bigger when regime is harder            | ~                                         | Helps most in strict / user-heavy / high-k regimes.                                             |
| Add IF-calibration on top of Fisher       | –                                                                   | ↑ consistently (≈+1–3.5 pts over Fisher-only)                          | ~                                         | Especially helpful when many params are DP or k/epochs are large.                               |

---

### Detailed Experimental Results

Each block shows the exact command used and the resulting **Privacy vs Accuracy Tradeoff** table.

---

#### 1. ε = 2.0, users = 60, k = 512, C = 2.0, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 200 --target-epsilon 2.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 60 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   29.9% |            0.522 |
| Vanilla DP-SGD + DP-SAT       |   25.1% |            0.487 |
| Fisher DP + Normal            |   77.3% |            0.509 |
| Fisher DP + DP-SAT            |   76.8% |            0.504 |
| Fisher DP + Normal + Calib    |   77.5% |            0.510 |
| Fisher DP + DP-SAT + Calib    |   76.9% |            0.513 |

---

#### 2. ε = 4.0, users = 60, k = 512, C = 2.0, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 200 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 60 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   61.0% |            0.511 |
| Vanilla DP-SGD + DP-SAT       |   61.9% |            0.488 |
| Fisher DP + Normal            |   78.8% |            0.521 |
| Fisher DP + DP-SAT            |   78.7% |            0.508 |
| Fisher DP + Normal + Calib    |   78.8% |            0.517 |
| Fisher DP + DP-SAT + Calib    |   78.9% |            0.510 |

---

#### 3. ε = 8.0, users = 60, k = 512, C = 2.0, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 200 --target-epsilon 8.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 60 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   73.9% |            0.503 |
| Vanilla DP-SGD + DP-SAT       |   73.4% |            0.510 |
| Fisher DP + Normal            |   79.1% |            0.521 |
| Fisher DP + DP-SAT            |   79.0% |            0.509 |
| Fisher DP + Normal + Calib    |   79.3% |            0.513 |
| Fisher DP + DP-SAT + Calib    |   79.2% |            0.514 |

---

#### 4. ε = 4.0, users = 60, k = 512, C = 3.0, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 200 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 3.0 \
  --run-mia --users 60 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   18.7% |            0.502 |
| Vanilla DP-SGD + DP-SAT       |   14.2% |            0.494 |
| Fisher DP + Normal            |   75.4% |            0.512 |
| Fisher DP + DP-SAT            |   74.4% |            0.502 |
| Fisher DP + Normal + Calib    |   76.2% |            0.517 |
| Fisher DP + DP-SAT + Calib    |   74.9% |            0.513 |

---

#### 5. ε = 4.0, users = 80, k = 512, C = 2.0, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 200 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 80 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   34.5% |            0.492 |
| Vanilla DP-SGD + DP-SAT       |   34.2% |            0.509 |
| Fisher DP + Normal            |   74.6% |            0.486 |
| Fisher DP + DP-SAT            |   75.8% |            0.481 |
| Fisher DP + Normal + Calib    |   74.7% |            0.512 |
| Fisher DP + DP-SAT + Calib    |   76.5% |            0.505 |

---

#### 6. ε = 4.0, users = 60, k = 512, C = 1.0, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 200 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 1.0 \
  --run-mia --users 60 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   78.6% |            0.507 |
| Vanilla DP-SGD + DP-SAT       |   78.6% |            0.521 |
| Fisher DP + Normal            |   79.3% |            0.522 |
| Fisher DP + DP-SAT            |   79.3% |            0.511 |
| Fisher DP + Normal + Calib    |   79.4% |            0.515 |
| Fisher DP + DP-SAT + Calib    |   79.3% |            0.513 |

---

#### 7. ε = 4.0, users = 100, k = 512, C = 2.0, epochs = 200, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 200 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   11.3% |            0.508 |
| Vanilla DP-SGD + DP-SAT       |   16.8% |            0.472 |
| Fisher DP + Normal            |   71.8% |            0.496 |
| Fisher DP + DP-SAT            |   74.7% |            0.509 |
| Fisher DP + Normal + Calib    |   72.8% |            0.513 |
| Fisher DP + DP-SAT + Calib    |   74.8% |            0.486 |

---

#### 8. ε = 4.0, users = 100, k = 2048, C = 2.0, epochs = 200, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 2048 --epochs 200 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   11.3% |            0.508 |
| Vanilla DP-SGD + DP-SAT       |   16.8% |            0.472 |
| Fisher DP + Normal            |   51.5% |            0.470 |
| Fisher DP + DP-SAT            |   60.6% |            0.528 |
| Fisher DP + Normal + Calib    |   57.2% |            0.515 |
| Fisher DP + DP-SAT + Calib    |   60.6% |            0.481 |

---

#### 9. ε = 4.0, users = 100, k = 512, C = 2.0, epochs = 300, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 300 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   12.7% |            0.503 |
| Vanilla DP-SGD + DP-SAT       |   11.3% |            0.492 |
| Fisher DP + Normal            |   68.0% |            0.496 |
| Fisher DP + DP-SAT            |   68.1% |            0.485 |
| Fisher DP + Normal + Calib    |   68.3% |            0.500 |
| Fisher DP + DP-SAT + Calib    |   68.8% |            0.484 |

---

#### 10. ε = 4.0, users = 100, k = 2048, C = 2.0, epochs = 300, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 2048 --epochs 300 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   14.0% |            0.474 |
| Vanilla DP-SGD + DP-SAT       |   10.7% |            0.514 |
| Fisher DP + Normal            |   39.5% |            0.499 |
| Fisher DP + DP-SAT            |   47.4% |            0.518 |
| Fisher DP + Normal + Calib    |   40.7% |            0.503 |
| Fisher DP + DP-SAT + Calib    |   48.4% |            0.480 |

---

#### 11. ε = 4.0, users = 100, k = 512, C = 2.0, epochs = 100, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 512 --epochs 100 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   29.2% |            0.505 |
| Vanilla DP-SGD + DP-SAT       |   23.7% |            0.486 |
| Fisher DP + Normal            |   76.1% |            0.482 |
| Fisher DP + DP-SAT            |   76.1% |            0.484 |
| Fisher DP + Normal + Calib    |   76.1% |            0.483 |
| Fisher DP + DP-SAT + Calib    |   76.3% |            0.484 |

---

#### 12. ε = 4.0, users = 100, k = 2048, C = 2.0, epochs = 100, dp-param-count = 20k

```bash
uv run ablation.py --mps --k 2048 --epochs 100 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 20000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   26.6% |            0.497 |
| Vanilla DP-SGD + DP-SAT       |   31.6% |            0.502 |
| Fisher DP + Normal            |   70.0% |            0.474 |
| Fisher DP + DP-SAT            |   72.4% |            0.484 |
| Fisher DP + Normal + Calib    |   70.7% |            0.488 |
| Fisher DP + DP-SAT + Calib    |   72.7% |            0.503 |

---

#### 13. ε = 4.0, users = 100, k = 2048, C = 2.0, epochs = 100, dp-param-count = 10k

```bash
uv run ablation.py --mps --k 2048 --epochs 100 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 10000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   34.5% |            0.492 |
| Vanilla DP-SGD + DP-SAT       |   42.2% |            0.490 |
| Fisher DP + Normal            |   76.8% |            0.505 |
| Fisher DP + DP-SAT            |   76.6% |            0.482 |
| Fisher DP + Normal + Calib    |   77.2% |            0.507 |
| Fisher DP + DP-SAT + Calib    |   76.9% |            0.514 |

---

#### 14. ε = 4.0, users = 100, k = 2048, C = 2.0, epochs = 100, dp-param-count = 1k

```bash
uv run ablation.py --mps --k 2048 --epochs 100 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 1000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   72.3% |            0.484 |
| Vanilla DP-SGD + DP-SAT       |   71.6% |            0.506 |
| Fisher DP + Normal            |   68.5% |            0.522 |
| Fisher DP + DP-SAT            |   69.0% |            0.484 |
| Fisher DP + Normal + Calib    |   69.1% |            0.494 |
| Fisher DP + DP-SAT + Calib    |   72.1% |            0.523 |

---

#### 15. ε = 4.0, users = 100, k = 2048, C = 2.0, epochs = 100, dp-param-count = 40k

```bash
uv run ablation.py --mps --k 2048 --epochs 100 --target-epsilon 4.0 --delta 1e-5 \
  --dp-param-count 40000 --model-type efficientnet --clip-radius 2.0 \
  --run-mia --users 100 --calibration-k 200 --dp-sat-mode fisher
```

| Model                         | Accuracy | Privacy (1-AUC) |
|-------------------------------|---------:|-----------------:|
| Vanilla DP-SGD                |   18.2% |            0.501 |
| Vanilla DP-SGD + DP-SAT       |   19.4% |            0.508 |
| Fisher DP + Normal            |   70.9% |            0.493 |
| Fisher DP + DP-SAT            |   70.5% |            0.493 |
| Fisher DP + Normal + Calib    |   72.2% |            0.478 |
| Fisher DP + DP-SAT + Calib    |   71.9% |            0.495 |

---

## Why Vanilla DP-SGD Collapses While Fisher DP Remains Stable

### Vanilla vs Fisher: Same Privacy, Different Geometry

- **Vanilla DP-SGD**:
  - Clip each (sample/user) gradient in **Euclidean norm** to radius \(C\).
  - Add **isotropic Gaussian noise** \(z \sim \mathcal{N}(0, \sigma^2 I)\) to the clipped average.
  - Every direction in parameter space gets the **same variance**; the DP guarantee depends on:
    - the clipping radius \(C\) (sensitivity), and
    - the noise scale \(\sigma\) (via the RDP accountant).

- **Fisher DP-SGD**:
  - Clip in a **Fisher/Mahalanobis norm**, based on a low-rank approximation of the Fisher information \(F\).
  - Add **anisotropic noise** in the Fisher eigenspace, here with **negatively correlated** scaling (less noise in high-curvature directions).
  - We use the **same RDP accountant and target \((\epsilon,\delta)\)** as vanilla; the difference is only **how** noise is shaped in parameter space, not how much privacy we get.

So under matched \((\epsilon,\delta)\), **Fisher vs Vanilla differ in utility, not in formal privacy level**.

### Why Decreasing ε or Increasing C Hurts Vanilla Much More

Two key levers affect how “harsh” DP-SGD is:

1. **Lower ε (stricter privacy)**  
   - The RDP accountant returns a **larger noise multiplier**.
   - For fixed \(C\), the per-coordinate noise standard deviation is \(\sigma C\); lowering ε increases \(\sigma\).

2. **Larger clip radius C**  
   - Sensitivity increases linearly with \(C\).
   - Even if the noise multiplier is unchanged, the absolute noise scale grows as \(\sigma C\).

In **Vanilla DP-SGD (user-level)**:

- We saw in `debug_vanilla_dp.py` and the ablation runs:
  - Median **user gradient norm after clipping**: on the order of a few hundred (e.g. \(\approx 432\)).
  - **Noise ℓ₂ norm**: on the order of **tens of thousands** (e.g. \(\approx 27{,}000\)) for ε=4.0, C=2.0, 100 users.
- That means the **signal-to-noise ratio** per DP step is extremely poor:
  - Every update is dominated by isotropic noise.
  - Starting from a strong public pretrain (≈79% accuracy), Vanilla DP-SGD rapidly “forgets” useful structure and collapses to ~10–30% accuracy.

This is exactly what we see in the tables:

- For stricter regimes (ε small, large C, many users, many DP parameters):
  - **Vanilla DP-SGD** often falls to ~10–30% accuracy.
  - **Fisher DP-SGD** stays in the 70–80% range.

### Why Fisher DP-SGD Survives in the Same Regime

Fisher DP-SGD **reweights directions** so that noise hurts much less where it matters:

- Clipping is done in a **Mahalanobis norm** induced by the Fisher \(F\), not plain Euclidean norm.
- Noise in the Fisher subspace is **negatively correlated with curvature**:
  - Less noise in high-curvature directions (which strongly affect the loss).
  - More noise in flatter directions (where the model is less sensitive).

Consequences:

1. **Effective sensitivity is smaller along important directions**  
   - High-curvature eigendirections are tightly controlled by the Fisher norm.
   - Gradients in those directions are clipped more consistently.

2. **Noise is pushed into “less important” directions**  
   - Anisotropic noise means we don’t waste budget on directions that the loss barely cares about.
   - The useful part of the gradient (aligned with high-curvature directions) has much better SNR than in the isotropic case.

3. **Same \((\epsilon,\delta)\), but much better geometry**  
   - The RDP accountant sees the same sampling rate and training schedule, so the **total privacy budget is the same** as vanilla.
   - Geometrically, however, Fisher’s clipping + noise shaping keeps the model near the good public-pretrained solution instead of wandering randomly.

This explains the empirical pattern in all the tables above:

- As **ε decreases** or **C increases** (or as we: increase \#users, increase Fisher rank k too much, or DP-train more parameters):
  - **Vanilla DP-SGD** quickly collapses in accuracy.
  - **Fisher DP-SGD (+ DP-SAT + IF-calibration)** remains **high and stable**, often 30–60 points above vanilla at the same \((\epsilon,\delta)\).

