# TODOs for Ablation Experiments and Paper Draft

## 🔬 Ablation / Sweep Experiments

All commands below assume CIFAR-10, strict DP setup (frozen backbone + DP on `conv1,conv2`), and `target-epsilon 2.0` unless otherwise noted.

### 0. Base Configuration (Reference)

- [ ] **Base run (Fisher DP + Fisher DP-SAT, ε=2.0)**
  - Command:
    ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

### 1. Clip Radius Sweep (ε = 2.0)

- [ ] **clip_radius = 1.0**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 1.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **clip_radius = 1.5**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 1.5 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **clip_radius = 2.5**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.5 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **clip_radius = 3.0**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 3.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

### 2. Fisher Rank k Sweep (ε = 2.0, lower k)

- [ ] **k = 256**
  - ```bash
    uv run ablation.py --mps --k 256 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **k = 512**
  - ```bash
    uv run ablation.py --mps --k 512 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **k = 1024**
  - ```bash
    uv run ablation.py --mps --k 1024 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

### 3. Epoch Sweep (ε = 2.0, change training length)

- [ ] **epochs = 150**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 150 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **epochs = 200**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 200 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **(Optional combos)** – lower k with fewer epochs
  - Examples:
    - `--k 512 --epochs 150`
    - `--k 512 --epochs 200`

### 4. ρ_sat Sweep for Fisher DP-SAT (ε = 2.0)

- [ ] **ρ_sat = 0.0005**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.0005 --dp-sat-mode fisher
    ```

- [ ] **ρ_sat = 0.002**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.002 --dp-sat-mode fisher
    ```

- [ ] **ρ_sat = 0.005**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.005 --dp-sat-mode fisher
    ```

### 5. Slightly Deeper DP Layer Set (ε = 2.0)

- [ ] **DP on conv1, conv2, conv3 (if conv3 exists)**
  - ```bash
    uv run ablation.py --mps --k 2048 --epochs 300 --dataset-size 50000 \
      --target-epsilon 2.0 --delta 1e-5 \
      --dp-layer conv1,conv2,conv3 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

### 6. Optional: Looser Privacy (ε = 4.0)

- [ ] **ε = 4.0, k = 512, epochs = 300**
  - ```bash
    uv run ablation.py --mps --k 512 --epochs 300 --dataset-size 50000 \
      --target-epsilon 4.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

- [ ] **ε = 8.0, k = 512, epochs = 300**
  - ```bash
    uv run ablation.py --mps --k 512 --epochs 300 --dataset-size 50000 \
      --target-epsilon 8.0 --delta 1e-5 \
      --dp-layer conv1,conv2 --clip-radius 2.0 \
      --run-mia --users 100 --calibration-k 2000 \
      --rho_sat 0.001 --dp-sat-mode fisher
    ```

### Minimal Recommended Subset

If time is limited, prioritise:
- [ ] **k sweep**: `k = 256`, `k = 512`
- [ ] **clip-radius sweep**: `clip_radius = 1.5`, `clip_radius = 2.5`
- [ ] **ρ_sat tweaks**: `rho_sat = 0.0005`, `rho_sat = 0.002`