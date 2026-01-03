# AGENTS.md

## Project scope (what this repo is)

This repository is a **proof-of-concept scientific project for academic research** on *free-lunch utility* for differentially private training.

- **Topic**: a unified framework for **utility-boosting DP-SGD** at fixed \((\varepsilon,\delta)\), with emphasis on **user-level DP**.
- **Core idea**: improve utility *without spending extra privacy budget* by combining:
  1. **Curvature-aware noise shaping** (our **Fisher-DP-SGD**: Mahalanobis clipping + anisotropic Gaussian noise in a Fisher subspace).
  2. **Optimizer plug-ins** (DP-SAT-style sharpness-aware optimization, treated as a component—not the primary contribution).
  3. **Post-hoc slice repair** (our **Influence-Function calibration** using only DP model outputs + public data, hence privacy-free post-processing).
- **Primary evaluation**: accuracy vs empirical membership-inference (MIA) strength, plus slice-level utility where applicable.

## What is `*.egg-info/` and do we need it?

Any `*.egg-info/` directory (e.g., `free_lunch_dp.egg-info/`) is **generated packaging metadata** produced by `setuptools` / editable installs (e.g., `uv pip install -e .`).

- **We do not need it in version control**: it is build/install output, not source code.
- **Action**: keep `*.egg-info/` ignored/untracked; delete it safely if it appears.

## Project structure (where things live)

Current layout of this project:

### Main Entry Points
- **`ablation.py`**: Full ablation study with all variants (pretrain → DP finetune → MIA → optional IF calibration).
- **`ablation_fast_no_calib.py`**: Fast ablation study without calibration variants (pretrain → DP finetune → MIA only).
- **`training/main.py`**: Single experiment comparison script for quick testing.

### Core DP Training Modules (`core/`)
- **`core/fisher_dp_sgd.py`**: Fisher estimation (`compute_fisher`) + low-rank Fisher-DP-SGD training loop (`train_with_dp`).
- **`core/dp_sat.py`**: DP-SAT-style optimizer component (`train_with_dp_sat`) - baseline Euclidean DP-SGD + post-processing flatness term.
- **`core/dp_sgd.py`**: Vanilla DP-SGD implementation (`train_with_vanilla_dp`) - standard baseline with Euclidean clipping + isotropic noise.
- **`core/influence_function.py`**: Influence-function calibration routines (iterative + line search) for post-hoc slice repair.
- **`core/mia.py`**: Membership inference attack evaluation (shadow model attack, confidence attack, etc.).
- **`core/privacy_accounting.py`**: Privacy accounting utilities (RDP accountant, noise multiplier calculation, privacy tracking).
- **`core/param_selection.py`**: Shared parameter selection utility (`select_parameters_by_budget`) for consistent DP parameter budgeting across all methods.
- **`core/config.py`**: Configuration utilities (random seeds, dataset location, rehearsal buffer constants).
- **`core/device_utils.py`**: Device resolution utilities (MPS/CUDA/CPU detection and multi-GPU support).
- **`core/registry.py`**: Lightweight registry system for models and datasets.

### Models (`models/`)
- **`models/base.py`**: `ModelBase` abstract class standardizing forward/loss interface.
- **`models/registry.py`**: Model registry for dynamic instantiation.
- **`models/vision/`**: Vision model implementations:
  - `efficientnet.py`: EfficientNet-B0 classifier
  - `resnet.py`: ResNet-18 classifier
  - `cnn.py`: Simple CNN baseline
  - `vit.py`: Vision Transformer (ViT-B16)
- **`models/language/`**: Language model implementations:
  - `bert.py`: BERT-based text classification
  - `qwen.py`: Qwen model support
  - `llama.py`: LLaMA model support
- **`models/utils.py`**: Model utilities (loss computation, etc.).

### Data (`data/`)
- **`data/base.py`**: `DatasetBuilder` abstract class + `DatasetConfig` + `split_private_public_calibration_indices` utility.
- **`data/common.py`**: Shared batch utilities (batch unpacking, device movement, `SyntheticUserDataset`, `UserBatchSampler` for user-level DP).
- **`data/vision.py`**: Vision dataset builders (CIFAR-10, Fashion-MNIST, CIFAR-100) with non-IID split support and rehearsal buffer logic.
- **`data/text.py`**: Text dataset builders (AG News, etc.) using HuggingFace datasets.
- **`data/registry.py`**: Dataset registry for dynamic instantiation.

### Additional Features
- **Public Rehearsal**: All DP training functions (`train_with_vanilla_dp`, `train_with_dp_sat`, `train_with_dp`) support optional public rehearsal via `public_loader` and `rehearsal_lambda` parameters. This combines DP private gradients with non-DP public gradients: `g_total = g_priv_DP + λ * g_public`.
- **IID/Non-IID Data Splits**: Support for both IID and non-IID dataset splits via `--non-iid` flag. Non-IID mode excludes specified classes from public pretrain and moves them to private, with optional rehearsal buffer to prevent catastrophic forgetting.

## Do / Don’t

- **Do**:
  - **Use `uv` everywhere** for reproducible runs: `uv run ...`, `uv pip ...`.
  - Keep experiments **auditable**: log seeds, \((\varepsilon,\delta)\), accountant settings, sampling rate \(q\), total steps \(T\), and noise multiplier \(\sigma\).
  - Treat **user-level DP carefully**: define the "user" unit explicitly (grouping strategy, samples-per-user distribution). Use `--users K` for user-level DP, `--sample-level` for sample-level DP.
  - Prefer **public-only** data for: Fisher estimation (if intended), calibration pools, public rehearsal, and hyperparameter sweeps.
  - Use **`--dp-param-count`** for architecture-agnostic parameter budgeting (head-first selection ensures classifier parameters are prioritized).
  - Use **`--rehearsal-lambda`** to control public rehearsal strength in non-IID scenarios (prevents catastrophic forgetting).
  - Use **`--non-iid`** flag when testing non-IID data distributions (requires `--public-pretrain-exclude-classes`).

- **Don’t**:
  - Don’t commit generated artifacts: `*.egg-info/`, `__pycache__/`, `.venv/`, cached models, downloaded datasets.
  - Don’t mix public/private splits silently. If private data is used, it must flow **only** through the DP mechanism.

## Safety, permissions, and operational rules

When acting as an automated agent (or when reviewing automated changes):

- **Ask first before**:
  - Installing/upgrading packages or changing `pyproject.toml`
  - Changing default privacy parameters (accountant, \((\varepsilon,\delta)\), clipping, sampling rate assumptions)
  - Rewriting experimental protocols in a way that changes the meaning of “user-level DP” (e.g., changing user batching, steps, or sampling)
  - Creating git commits unless explicitly requested

- **Never do**:
  - Never log or export raw private examples.
  - Never save “private” intermediate artifacts that can leak membership (e.g., per-user gradients, per-example losses) unless explicitly DP-protected or derived from public data.
  - Never claim privacy from **empirical MIA AUC** alone. The DP guarantee comes from the accountant + mechanism.

- **DP correctness checklist (before reporting results)**:
  - Clipping is applied at the intended granularity (sample vs user).
  - Noise is added once per step with the accountant-derived \(\sigma\).
  - Any additional terms (DP-SAT, calibration, public rehearsal) are **post-processing** or use **public** data only.
  - If only a subset of parameters is DP-trained (via `--dp-param-count` or `--dp-layer`), ensure *all* private-dependent updates are confined to that subset (others frozen via `requires_grad=False`), or explicitly state the privacy scope.
  - Public rehearsal (`g_total = g_priv_DP + λ * g_public`) is privacy-free post-processing since `g_public` uses only public data.
  - Fisher DP noise scaling should use the **calibrated Mahalanobis threshold** (`actual_radius`), not the raw Euclidean target, after norm calibration completes.

## Quick start (examples, `uv`-first)

### Example 1 — Full ablation study (EfficientNet-B0 + CIFAR-10, user-level DP)

```bash
uv run ablation.py --mps --model-type efficientnet --dataset-size 30000 --users 100 --dp-epochs 20 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --run-mia
```

### Example 2 — Fast ablation (no calibration variants)

```bash
uv run ablation_fast_no_calib.py --mps --model-type efficientnet --users 100 --dp-epochs 10 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --dp-sat-mode fisher --run-mia
```

### Example 3 — Non-IID split with public rehearsal

```bash
uv run ablation_fast_no_calib.py --mps --model-type efficientnet --non-iid --public-pretrain-exclude-classes 0,1 --users 100 --dp-epochs 10 --target-epsilon 20.0 --delta 1e-5 --clip-radius 1.0 --dp-param-count 20000 --k 512 --rehearsal-lambda 0.1 --run-mia
```

### Example 4 — Fisher + DP-SAT + IF-calibration (full “free-lunch stack”)

```bash
uv run ablation.py --mps --model-type efficientnet --users 100 --dp-epochs 20 --target-epsilon 4.0 --delta 1e-5 --clip-radius 2.0 --dp-param-count 20000 --k 512 --dp-sat-mode fisher --rho-sat 0.001 --calibration-k 200 --run-mia
```

## Project state documentation

Keep a lightweight running log of “what we learned” from ablations and code changes.

- **`STATE_SUMMARY.md`**: Summary of the current experimental state and key takeaways, including:
  - Best-performing regimes (accuracy / MI AUC / slice metrics)
  - What settings are brittle (epochs, rank `k`, clip, #users, dp-param-count)
  - Known pitfalls (AUC inversion, distribution-shift artifacts, adaptive clipping leakage)
  - Outstanding issues / hypotheses to test next

## Notes for contributors

- When adding a new utility technique, explicitly state whether it is:
  - **privacy-mechanism changing** (affects the accountant / sensitivity), or
  - **post-processing / public-data-only** (privacy-free at fixed \((\varepsilon,\delta)\)).
- Prefer improvements that preserve the **one noisy query per step** DP-SGD template.
- If you change the definition of “user” or sampling scheme, update the documentation and re-run the key sanity checks:
  - baseline MIA on a model that never saw “members” should be near chance,
  - report `max(AUC, 1-AUC)` or advantage `|AUC-0.5|` alongside raw AUC when appropriate.
- Keep the repo runnable with `uv` and avoid introducing heavy dependencies without discussion.
