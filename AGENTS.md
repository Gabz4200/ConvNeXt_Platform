# AGENTS.md

Context and technical guidelines for AI coding agents working in the **ConvNeXt Platform** repository.

---

## 1. Project Overview

**ConvNeXt Platform** is a research and training codebase for training ConvNeXt vision backbones and visual representations for **platformer games** (reinforcement learning & vision-based control), featuring DINOv3-compatible pre-trained weight loading, intermediate layer feature extraction, and PyTorch Lightning + Hydra training workflows.

### Core Tech Stack
- **Deep Learning Framework:** PyTorch >= 2.0.0, PyTorch Lightning >= 2.0.0
- **Configuration & CLI:** Hydra 1.3 (`hydra-core`, `hydra-colorlog`, `hydra-optuna-sweeper`), OmegaConf
- **Vision Backbones & Weights:** Custom ConvNeXt (DINOv3-compatible with intermediate layer extraction), HuggingFace Hub, `safetensors`, `timm`
- **Data & Datasets:** HuggingFace `datasets`, `torchvision`, `torchmetrics`
- **Dependency & Environment Management:** `uv`, `pyproject.toml`
- **Code Quality & Testing:** `pytest`, `pre-commit` (Black 99 col, isort, flake8, docformatter, interrogate)

---

## 2. Directory Structure & Key Files

```
ConvNeXt_Platform/
├── .project-root               # Root indicator used by rootutils for PYTHONPATH resolution
├── pyproject.toml              # Build metadata, dependencies, pytest & uv index config
├── Makefile                    # Standard automation targets (train, test, clean, format)
├── configs/                    # Hydra configuration hierarchy
│   ├── train.yaml              # Main training config entrypoint (defaults: data=cifar10, model=convnext)
│   ├── eval.yaml               # Main evaluation config entrypoint
│   ├── model/                  # Model configurations (convnext.yaml, mnist.yaml)
│   ├── data/                   # DataModule configs (cifar10.yaml, mnist.yaml)
│   ├── trainer/                # Trainer configs (default.yaml, cpu.yaml, gpu.yaml, ddp.yaml)
│   ├── callbacks/              # Lightning callback configs (model_checkpoint.yaml, early_stopping.yaml)
│   ├── logger/                 # Logger configs (tensorboard.yaml, wandb.yaml, csv.yaml, etc.)
│   ├── debug/                  # Debugging presets (default.yaml -> fast_dev_run, limit.yaml, overfit.yaml)
│   ├── experiment/             # Experiment overrides (version-controlled recipe presets)
│   └── hparams_search/         # Hyperparameter search configs (Optuna sweeper)
├── src/
│   ├── train.py                # Main training entrypoint script (@hydra.main train.yaml)
│   ├── eval.py                 # Main evaluation entrypoint script (@hydra.main eval.yaml)
│   ├── models/
│   │   ├── convnext_module.py  # ConvNeXtLitModule (LightningModule for classification)
│   │   ├── mnist_module.py     # MNISTLitModule (baseline reference)
│   │   └── components/
│   │       ├── convnext.py     # Pure PyTorch ConvNeXt (DINOv3 weights & intermediate layers)
│   │       └── simple_dense_net.py
│   ├── data/
│   │   ├── cifar10_datamodule.py # CIFAR10DataModule (loads uoft-cs/cifar10 from HF Hub)
│   │   └── mnist_datamodule.py
│   └── utils/                  # Rich logging, instantiators, hyperparameter trackers
└── tests/
    ├── conftest.py             # Shared fixtures (cfg_train, cfg_eval)
    ├── test_configs.py         # Config validation & instantiation tests
    ├── test_datamodules.py     # DataModule batch shape and loading tests
    ├── test_dinov3_ricl_embeds.py # DINOv3 weight loading & timm equivalence tests (@pytest.mark.slow)
    ├── test_eval.py            # End-to-end train + eval test
    ├── test_sweeps.py          # Hydra Optuna sweeper integration tests
    └── test_train.py           # Training step and fast_dev_run integration tests
```

---

## 3. Setup & Environment

### Package Management with `uv`

The environment is managed using `uv`. CPU PyTorch wheels are configured by default in `pyproject.toml` via `[[tool.uv.index]]`.

```bash
# Create virtual environment and install all dependencies (including dev)
uv sync --extra dev

# Or install editable package in active environment
uv pip install -e ".[dev]"
```

### Environment Variables & Project Root

Every entrypoint (`src/train.py`, `src/eval.py`) calls `rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)`:
- Guarantees the repository root is in `PYTHONPATH`.
- Exposes `PROJECT_ROOT` environment variable used in `configs/paths/default.yaml`.
- Loads optional `.env` file from the workspace root.

---

## 4. Development Workflow & Commands

### Training

```bash
# Train default setup (ConvNeXt-Tiny on CIFAR-10)
python src/train.py

# Train on CPU explicitly
python src/train.py trainer=cpu

# Train on GPU / Multi-GPU (DDP)
python src/train.py trainer=gpu trainer.devices=1
python src/train.py trainer=ddp trainer.devices=2

# Override hyperparameters via CLI
python src/train.py model.optimizer.lr=0.001 data.batch_size=64 trainer.max_epochs=20

# Run specific experiment config
python src/train.py experiment=example

# Resume training from checkpoint
python src/train.py ckpt_path=/path/to/checkpoint.ckpt
```

### Evaluation

```bash
# Evaluate checkpoint on test set
python src/eval.py ckpt_path=/path/to/best_model.ckpt

# Evaluate on CPU
python src/eval.py ckpt_path=/path/to/best_model.ckpt trainer=cpu
```

### Debugging Presets

Use Hydra `debug` configs for rapid diagnostics:

```bash
# Fast dev run: runs 1 train, 1 val, 1 test batch (disables checkpointing and logging)
python src/train.py debug=default

# Limit batches: runs only 10% of batches per epoch
python src/train.py debug=limit

# Overfit batches: overfits on 10 batches to test model convergence
python src/train.py debug=overfit

# Profiler: PyTorch profiler enabled
python src/train.py debug=profiler
```

### Hyperparameter Search (Optuna)

```bash
# Run Optuna hyperparameter sweep (multirun mode -m)
python src/train.py -m hparams_search=mnist_optuna
```

---

## 5. Testing Instructions

### Test Suites and Markers

Tests use `pytest`. Markers are configured in `pyproject.toml`.

- **Fast tests (unit & config checks):** Exclude network / heavy downloads.
- **Slow tests (`@pytest.mark.slow`):** Include HuggingFace Hub downloads and DINOv3 weight loading.

```bash
# Run standard fast test suite (skipping slow tests)
pytest -k "not slow"
# or via Makefile
make test

# Run full test suite (including slow tests)
pytest
# or via Makefile
make test-full

# Run specific test file
pytest tests/test_configs.py
pytest tests/test_datamodules.py
pytest tests/test_train.py

# Run specific test case with verbose output
pytest tests/test_configs.py::test_train_config -vv -s

# Run DINOv3 Hub weight loading verification (requires internet access)
pytest tests/test_dinov3_ricl_embeds.py -s
```

---

## 6. Code Style & Standards

### Formatting and Linters

Adhere to the configured `pre-commit` hooks:
- **Line Length:** `99` characters (Black, Flake8, docformatter).
- **Import Sorting:** `isort` with `--profile black`.
- **Docstrings:** Sphinx style formatting (`docformatter`), minimum 80% coverage enforced by `interrogate`.
- **Typing:** Strict type annotations for function signatures, return types, and module arguments.

```bash
# Run all pre-commit hooks manually
pre-commit run -a
# or via Makefile
make format
```

### Engineering Rules
1. **Fail Fast:** Do not catch generic exceptions (`except Exception:`) or hide failures behind silent fallbacks. Allow invalid configurations and missing attributes to raise immediately.
2. **Single Source of Truth:** Infer dependent parameters (e.g. `num_classes` in `ConvNeXtLitModule` inferred from `net.head.out_features`) rather than duplicating them across configs.
3. **No Dead Comments / Obvious Restatements:** Avoid narrative storytelling, decorative ASCII separators, or comments that merely restate symbol names.
4. **Hydra Interpolation:** Tie scheduler lengths dynamically to trainer configurations using variable interpolation (`T_max: ${trainer.max_epochs}`).

---

## 7. Key Architecture & Implementation Patterns

### 1. Backbone Component (`src/models/components/convnext.py`)
- Pure `nn.Module` implementation of ConvNeXt.
- Supports both classification heads (`num_classes > 0`) and feature extraction (`num_classes = 0`).
- Provides `get_intermediate_layers()` and `forward_features()` with per-stage normalization compatible with DINOv3 evaluation protocols.
- Weight loading utilities (`load_dinov3_weights`) map both `timm` format and Facebook HF format checkpoints to the internal module structure.

### 2. LightningModule (`src/models/convnext_module.py`)
- Standardized lifecycle: `model_step`, `training_step`, `validation_step`, `test_step`, and `configure_optimizers`.
- Tracks loss via `MeanMetric`, accuracy via `Accuracy(task="multiclass", num_classes=...)`, and peak validation accuracy via `MaxMetric`.
- Supports `torch.compile` during `setup(stage)` when enabled in config (`compile: true`).

### 3. DataModule (`src/data/cifar10_datamodule.py`)
- Wraps HuggingFace dataset splits (`uoft-cs/cifar10`) using a lightweight `Dataset` class (`CIFAR10HFDataset`) applying torchvision transforms on-the-fly.
- **Deferred Imports:** Keeps `datasets` and heavy imports inside `prepare_data()` and `setup()` to ensure top-level module import remains fast.
- **Explicit None Checks:** Uses `if self.data_train is not None:` instead of truthiness testing (`if self.data_train:`), as custom datasets can evaluate to `False` when empty.

---

## 8. Debugging & Common Pitfalls

| Issue | Cause | Solution |
|---|---|---|
| `UnpicklingError: Weights only load failed` | PyTorch 2.6+ defaults `torch.load(..., weights_only=True)`. | Use `torch.serialization.add_safe_globals` or load state dicts via safetensors/timm. |
| Spatial dimension too small in ConvNeXt | ConvNeXt stem (stride 4) + 3 downsampling stages (stride 2 each) = 32x reduction. Input smaller than 32x32 collapses to 0x0. | Ensure input image resolution is at least 32x32 (e.g. resize MNIST 28x28 -> 32x32). |
| Empty dataset evaluation error | Direct truthiness checks `if not dataset:` trigger `len(dataset) == 0`. | Always guard with explicit identity checks: `if dataset is None:`. |
| Missing logger directory / rank_zero error | Running without logger config defaults to console-only logging. | Pass `logger=csv` or `logger=tensorboard` if metric persistence is required. |
