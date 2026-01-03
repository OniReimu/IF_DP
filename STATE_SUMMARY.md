# STATE_SUMMARY

Last updated: 2025-12-30 (local) — by Codex

## 1) One-paragraph summary
This repo is a DP-SGD “free-lunch utility” POC comparing (1) Fisher-informed DP-SGD, (2) DP-SAT, and (3) post-hoc IF calibration at fixed (ε,δ), with user-level DP emphasis. Paper-readiness work focused on DP correctness and audit hygiene: Fisher estimation and Fisher-radius calibration are public-only, BatchNorm running stats are frozen during DP fine-tuning, and MIA member/non-member sampling is transform-aligned to reduce distribution-shift artifacts. **Major update**: Fisher DP-SGD is now **mechanism-aligned** for standard subsampled-Gaussian accounting via **subspace-only private updates** (projected into span(U)), **F-norm clipping** (√λ in eigen-coordinates), and **F⁻¹-shaped noise** (1/√λ). Privacy reporting uses the standard accountant inputs (σ, q, T, δ); prior “σ_eff/ε_eff” Euclidean reinterpretations are removed. Legacy complement-noise support (and its noise-floor helpers) has been removed to avoid ambiguous accounting/claims.

## 2) Current status (high signal)
- ✅ Working: ablation runners (`ablation.py`, `ablation_fast_no_calib.py`), Fisher DP training, MIA-only runner (`training/mia_only.py`), privacy profile plotter (`scripts/plot_privacy_profile.py`).
- ⚠️ Known limitations: MIA can fall back to AUC=0.5 when shadow data is insufficient; user-level accounting has two modes (`user_poisson` vs `repo_q_eff`) and must be reported explicitly; Fisher DP-SGD currently supports only the mechanism-aligned (subspace-only, F-norm clip + F⁻¹ noise) variant for paper claims; local working tree has uncommitted changes.
- ❌ Broken / failing: none known.

## 3) How to run (copy/paste)
### Setup
- `uv pip install -r requirements.txt`

### Main workflow
- Full ablation (with calibration):
  - `uv run ablation.py --mps --model-type efficientnet --users 100 --dp-epochs 20 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --run-mia`
- Fast ablation (no calibration):
  - `uv run ablation_fast_no_calib.py --mps --model-type efficientnet --users 100 --dp-epochs 10 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --dp-sat-mode fisher --run-mia`
- MIA-only from cached models:
  - `uv run training/mia_only.py --dataset cifar10 --model-type efficientnet --epochs 100 --non-iid --public-pretrain-exclude-classes 0,1 --users 200 --mia-level user --mia-attack shadow --shadow-epochs 3`
- Privacy profile plot:
  - `uv run scripts/plot_privacy_profile.py --dataset cifar10 --target-epsilon 0.5 --delta 1e-5 --dp-epochs 3 --users 200 --clip-radius 2.0 --non-iid --public-pretrain-exclude-classes 0,1`

### Test / validation
- `python -m py_compile ablation.py ablation_fast_no_calib.py core/fisher_dp_sgd.py training/main.py scripts/plot_privacy_profile.py`
- `python -m unittest discover -s tests`

## 4) Key commands and flags
### CLI entrypoints
- `ablation.py`: full ablation (Fisher + DP-SAT + calibrated variants).
- `ablation_fast_no_calib.py`: ablation without calibration variants.
- `training/mia_only.py`: MIA audit from cached models (no retrain).
- `scripts/plot_privacy_profile.py`: ε(δ) profile with ablation-style args.
- `training/main.py`: legacy/demo single-run pipeline.

### Important flags
- `--dp-layer` / `--dp-param-count`: DP scope selection (non-DP params are frozen).
- `--k`: Fisher rank within the DP scope.
- `--users` / `--sample-level`: user-level vs sample-level DP mode.
- `--non-iid --public-pretrain-exclude-classes 0,1`: non-IID split.
- `--rehearsal-lambda`: public rehearsal mixing.
- `--mia-level` / `--mia-attack`: MIA granularity and attack type.

## 5) Architecture / modules (what to read first)
- `README.md`: usage, DP accounting notes, MIA audit notes.
- `core/fisher_dp_sgd.py`: Fisher DP training and clipping/calibration logic.
- `core/dp_sgd.py`, `core/dp_sat.py`: baselines and DP-SAT.
- `core/mia.py`: MIA audits (sample/user-level, matching logic, fallbacks).
- `core/influence_function.py`: public-only IF calibration.
- `data/vision.py`, `data/text.py`: dataset splits and non-IID logic.
- `scripts/plot_privacy_profile.py`: ε(δ) plotting tool.
- `training/mia_only.py`: cached MIA runner.
- `tests/`: MIA utils + Fisher noise-floor test.
- `config/config.py`: seed + dataset cache defaults.

## 6) Non-obvious decisions (avoid re-learning)
- Paper-readiness checklist (all done; migrated here from the old TODO file, updated for Fisher mechanism alignment):
  - Fisher estimation uses public-only data (no private Fisher).
  - Fisher DP-SGD paper-default is **mechanism-aligned**:
    - private updates are projected into `span(U)` (subspace-only) so complement sensitivity is 0,
    - clipping uses the **F norm** (√λ in eigen-coordinates),
    - noise uses covariance proportional to **F⁻¹** (1/√λ in eigen-coordinates),
    - standard subsampled-Gaussian accountant applies using (σ, q, T, δ).
  - Fisher-radius calibration uses public-only gradients and uses `--clip-radius` as the *vanilla Euclidean* Δ₂ target to match clip rates (fairness), not as the Fisher radius directly.
  - BatchNorm running stats are frozen during DP fine-tuning.
  - Sample-level accounting rate uses `q = batch_size / N_private`.
  - Critical slice for calibration is built from the public calibration split and aligned to eval transforms (avoid MIA non-member overlap).
- Fisher estimation and Fisher-radius calibration are **public-only** for DP correctness.
- BatchNorm running stats are frozen during DP fine-tuning to avoid private-data leakage via buffers.
- Calibration slices come from the **public calibration split** aligned to eval transforms to avoid MIA non-member overlap.
- Sample-level accounting uses `q = batch_size / N_private` (not len(loader)/N_private).
- Fisher DP accounting choices (important for paper claims):
  - Do **not** report “σ_eff/ε_eff” Euclidean reinterpretations; report privacy from the mechanism’s actual accountant inputs (σ, q, T, δ).
- MIA import hygiene: `core/mia.py` avoids import-time seeding/dataset resolution; seeding happens in calling scripts or `core.mia.main()` only.

## 7) Recent validation results
Record what was run and what it returned (keep concise).
- 2025-12-30: `python -m py_compile core/fisher_dp_sgd.py ablation.py ablation_fast_no_calib.py training/main.py` → PASS
- 2025-12-30: `python -m unittest discover -s tests -v` → PASS (includes Fisher mechanism-alignment unit tests)

## 8) Open issues / TODO next
- [ ] Decide on a single user-level accounting mode for paper reporting (`user_poisson` recommended) and ensure all scripts clearly log it.

## 9) Changelog (state-summary-only)
- 2025-12-30: Merged paper-readiness checklist (from the old TODO file) into this summary; updated noise-floor + MIA import-hygiene status.
- 2025-12-30: Switched Fisher DP-SGD to mechanism-aligned, subspace-only updates (F-norm clipping + F⁻¹ noise), removed σ_eff/ε_eff reporting, removed legacy complement-noise/noise-floor codepaths, and updated ablation/training scripts and README accordingly.
