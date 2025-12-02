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


## Notes

Right now your relative story is quite nice (Fisher + DP-SAT + Calib is clearly best), but yeah: 51% vs 61.7% non-DP at ε=2, user-level, 100 users still feels sad.

The key thing: with user-level ε=2 on only 100 users, the bar for accuracy is much harsher than the usual sample-level CIFAR stuff. A ~10-point gap to non-DP might actually be close to the Pareto front. But there are knobs you can turn.

I’ll give you a concrete “tuning menu” with priorities.

⸻

1. Privacy knobs (the big levers)

(1) Try a looser ε: 4 or 8

Right now:
	•	ε = 2.0, δ = 1e-5, 100 users
	•	1-AUC is ~0.49–0.51 for all DP variants → MI is already at random.

That’s actually telling you: you have room to relax ε a bit without obviously harming MI robustness:
	•	Run the same ablation with:
	•	--target-epsilon 4.0
	•	and maybe --target-epsilon 8.0
	•	Keep everything else fixed (epochs, clip radius, users).

If at ε=4 your best model jumps from ~51% → ~55–58% and MI AUC still ~0.5, that’s a very defensible “accuracy vs privacy” plot:
	•	“At ε=2 we get 51.3% with near-random MI.”
	•	“At ε=4 we get X% with still near-random MI.”

If you must stick to ε=2 for the main claim, you can still include ε=4/8 curves as secondary ablations.

(2) Use a proper DP-SGD accountant for σ (This one has been fixed, can skip this) 

Your current code uses the “legacy” σ:

sigma_single_epoch = sqrt(2*log(1.25/delta)) / epsilon
sigma = sigma_single_epoch / sqrt(epochs)

This is a hacky composition bound, not the DP-SGD/RDP formula. In practice:
	•	A proper RDP / moments accountant will usually allow a smaller σ than that legacy bound for the same ε, given your actual sampling rate and number of steps.
	•	Smaller σ → less noise → higher accuracy for the same ε.

Concrete action:
	•	Compute σ with Opacus or your own RDP library based on:
	•	number of users (100),
	•	sampling scheme (batch of how many users per step),
	•	total steps (≈ epochs × steps_per_epoch).
	•	Then call train_with_dp(..., sigma=that_value) and ignore the legacy σ branch.

This is probably the single most “principled” way to get extra accuracy without cheating on ε.

⸻

2. Training hyperparameters under DP

At ε=2, you’re in a very noisy regime. Getting the optimiser schedule right matters a lot.

(3) Reduce epochs + increase / schedule LR

You used:

--epochs 300

DP-SGD behaviour is quite different from non-DP:
	•	Too many noisy updates at low LR can just wander, not converge.
	•	Often it’s better to:
	•	train for fewer epochs (e.g. 80–150),
	•	use a larger initial LR with a good schedule (cosine, step decay).

Concrete things to try:
	•	Halve epochs:
	•	--epochs 150 with LR 1e-3 (or a bit larger, e.g. 3e-3), cosine decay.
	•	Or:
	•	--epochs 100, LR 3e-3 with a step schedule.

DP-SAT in particular tends to like a reasonably high LR early on; otherwise the sharpness-aware perturbation is a tiny correction under huge noise.

(4) Clip radius sweep

You’re using:

--clip-radius 2.0

Even though you have adaptive/norm calibration, the base target still matters.

Easy sweep:
	•	--clip-radius {1.0, 1.5, 2.0, 3.0}

For each, record:
	•	fraction of gradients clipped,
	•	final accuracy.

Under DP, often the sweet spot is somewhere around “20–40% of per-user gradients clipped” — too small C → heavy clipping bias; too big C → huge sensitivity, you need more noise for the same ε and gradients get drowned.

You already have nice log stats in train_with_dp (Mahalanobis norms, etc.), so it’s easy to sanity-check.

⸻

3. Fisher / architecture knobs

(5) Reduce Fisher rank k

You’re currently using:

--k 2048
--calibration-k 2000
--dp-layer conv1,conv2

For conv1+conv2 only, k=2048 might be:
	•	larger than the actual parameter count of those layers, or
	•	so large that the Fisher noise becomes “almost isotropic” again (it covers nearly all directions).

Try:
	•	--k 64
	•	--k 128
	•	--k 256

This:
	•	keeps the main curvature directions,
	•	but lets the anisotropy actually do something, instead of just approximating full-covariance noise that’s numerically fragile.

You can keep --calibration-k large (used only for norm calibration), but the noise rank k is the important one.

(6) DP-layer choice: include a bit more of the network

Right now you’re DP-training only:

--dp-layer conv1,conv2

If those are the only layers updated on private data, they might be too “early” to absorb all the task-specific adaptation the dataset needs, especially with 100 users and a relatively hard task.

Two options:
	1.	Still strict DP on full model, but Fisher only on shallow layers
	•	Add the next conv layer to the DP set with either Fisher or vanilla DP:
	•	--dp-layer conv1,conv2,conv3
	•	In code:
	•	treat conv1+conv2 as Fisher block,
	•	treat conv3 (and maybe the head) with plain Euclidean DP-SGD (clip + isotropic noise).
	2.	Fisher on more layers, smaller k per layer
	•	E.g. all conv layers Fisher-DP, but with small per-layer k (like 32–64) to keep the Fisher computation manageable.
	•	This might let later features adapt better, pushing accuracy up.

Even a DP head only variant is worth checking:
	•	--dp-layer fc (or whatever your last linear is),
	•	Fisher or Euclidean DP there,
	•	frozen backbone from non-DP or public training.

That often gives you a large chunk of the non-DP accuracy back for a given ε.

⸻

4. DP-SAT-specific tweaks

Given your new results:
	•	Vanilla DP-SGD → +DP-SAT: 36.0 → 37.4
	•	Fisher DP → +DP-SAT: 46.0 → 49.9
	•	Fisher + Calib → +DP-SAT: 47.7 → 51.3

So DP-SAT is helping quite a bit now (≈ +4–5 points in the Fisher regime). Nice.

Still, you can try:

(7) ρ (rho_sat) sweep

If you’re using the default rho_sat=0.001, that’s arbitrary. Try:
	•	rho_sat ∈ {5e-4, 1e-3, 2e-3, 5e-3}

For each, record Fisher+DP-SAT accuracy (with and without calibration). You may find:
	•	one ρ gives ~1–2 extra points,
	•	too large ρ destabilizes training early.

(8) Turn off DP-SAT for vanilla, keep for Fisher

For the paper narrative, it’s nice to show:
	•	the incremental contribution of DP-SAT on vanilla vs Fisher:
	•	“Improves vanilla by +1.4 points”
	•	“Improves Fisher by +3.9 points”

But if your main concern is peak accuracy, you can decide:
	•	only use DP-SAT in the Fisher runs where it gives the biggest benefit.

⸻

5. Realistic expectations

Given:
	•	user-level ε=2, δ=1e-5,
	•	only 100 users,
	•	the task is non-trivial (non-DP baseline 61.7%),

then:
	•	getting 51–55% with random MI is already a solid result;
	•	expecting to hit 60%+ at that ε and N might be unrealistic.

So I’d aim for something like:
	•	At ε=2: best model around 52–55% (you’re already at 51.3, so +2–3 points via the knobs above is plausible).
	•	At ε=4: best model maybe 56–60%.
	•	Non-DP baseline: 61–62%.

That gives you a clean privacy–utility curve and a compelling “free-lunch utility” story: Fisher + DP-SAT + calibration consistently dominates vanilla DP-SGD at the same ε.