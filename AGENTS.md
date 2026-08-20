# AGENTS.md

Context and technical guidelines for AI coding agents working in the **ConvNeXt Platform** repository.

---

## 1. Project Overview

**ConvNeXt Platform** is a research and training codebase for vision-temporal models combining **ConvNeXt** vision backbones (with DINOv3 self-supervised visual priors) and **RWKV-7** linear attention recurrent blocks. It is designed for **gameplay action prediction and platformer control**, predicting 21-D gamepad vectors (17 boolean buttons + 2 dual-axis joysticks) from raw video frames and action trajectories, with support for offline sequence training, dataset streaming from NVIDIA NitroGen, Super Mario Bros (SMB) behavioral cloning, and real-time online recurrent streaming inference.

> **MANDATORY RULE — PyTorch Lightning, not plain PyTorch:** Every training, evaluation, and inference
> path in this codebase MUST be driven by PyTorch Lightning (`LightningModule` + `Trainer`). Never
> write manual training loops, raw `torch.no_grad()` inference pipelines, or bare `optimizer.step()`
> code. Pure-`nn.Module` backbones (e.g. `src/models/components/convnext_rwkv7.py`) exist only as model
> definitions and MUST always be wrapped in a `LightningModule` (e.g. `ConvNeXtRWKV7GamepadLitModule`)
> and run through `Trainer.fit`, `Trainer.test`, or `Trainer.predict`. Plain PyTorch is allowed only for
> (a) defining `nn.Module` layer logic and (b) state-dict weight loading/saving I/O.

### Core Tech Stack
- **Deep Learning Framework:** PyTorch >= 2.0.0, PyTorch Lightning >= 2.0.0
- **Vision Backbone:** ConvNeXt (Tiny/Small/Base/Large) with DINOv3 weight ingestion (`facebook/dinov3-convnext-*-pretrain-lvd1689m`)
- **Temporal Modeling:** RWKV-7 (Goose) linear attention recurrent blocks with fast parallel scans and recurrent state caching
- **Spatial & Temporal Pooling:** `AdaptiveLearnedPool2d`, `LearnedWeightedGAP`, `CausalConv1d`, `CausalAdaptiveLearnedPool`
- **Configuration & CLI:** Hydra 1.3 (`hydra-core`, `hydra-colorlog`, `hydra-optuna-sweeper`), OmegaConf
- **Data & Streaming:** HuggingFace `datasets`, `huggingface-hub`, `pyarrow`, `pandas`, `pillow`, `torchvision`
- **Packaging & Environment:** `uv`, `pyproject.toml` (package: `convnext-platform`), `MANIFEST.in`
- **Code Quality & Testing:** `pytest`, `pyrefly`, `pre-commit` (Black 99 col, isort, flake8, docformatter, interrogate)

---

## 2. Directory Structure & Key Files

```
ConvNeXt_Platform/
├── .project-root               # Root indicator used by rootutils for PYTHONPATH resolution
├── pyproject.toml              # Build metadata, dependencies, console scripts & uv config
├── MANIFEST.in                 # Packaging manifest (includes src/configs, python modules)
├── Makefile                    # Standard automation targets (train, test, clean, format)
├── pyrefly.toml                # Type checking configuration
├── src/
│   ├── configs/                # Hierarchical Hydra configurations (packaged inside src)
│   │   ├── train.yaml          # Main training config entrypoint (defaults: data=cifar10, model=convnext)
│   │   ├── eval.yaml           # Main evaluation config entrypoint
│   │   ├── model/              # Model configs (convnext.yaml, convnext_rwkv7_gamepad*.yaml, mnist.yaml)
│   │   ├── data/               # DataModule configs (cifar10.yaml, nitrogen.yaml, smb.yaml, mnist.yaml)
│   │   ├── trainer/            # Trainer configs (default.yaml, cpu.yaml, gpu.yaml, ddp.yaml)
│   │   ├── callbacks/          # Lightning callback configs (model_checkpoint.yaml, early_stopping.yaml)
│   │   ├── logger/             # Logger configs (tensorboard.yaml, wandb.yaml, csv.yaml, etc.)
│   │   ├── debug/              # Debugging presets (default.yaml -> fast_dev_run, limit.yaml, overfit.yaml)
│   │   ├── experiment/         # Experiment overrides (nitrogen_gamepad.yaml, smb_gamepad.yaml, example.yaml)
│   │   └── hparams_search/     # Hyperparameter search configs (Optuna sweeper)
│   ├── train.py                # Generic training CLI entrypoint (delegates to run_train_task)
│   ├── train_nitrogen.py       # NitroGen training entrypoint (CLI + standalone train() API)
│   ├── train_smb.py            # Super Mario Bros secondary training entrypoint (CLI + train() API)
│   ├── eval.py                 # Evaluation CLI entrypoint
│   ├── models/
│   │   ├── convnext_rwkv7_module.py # ConvNeXtRWKV7GamepadLitModule (BCE + MSE gamepad loss & metrics)
│   │   ├── convnext_module.py  # ConvNeXtLitModule (classification + feature extraction)
│   │   ├── mnist_module.py     # Baseline MNIST module
│   │   └── components/
│   │       ├── convnext_rwkv7.py # ConvNeXtRWKV7Gamepad, GamepadHead, _InputNormalize, GamepadStreamingState
│   │       ├── convnext.py     # Pure PyTorch ConvNeXt + DINOv3 weight loader
│   │       ├── poolers.py      # AdaptiveLearnedPool2d, LearnedWeightedGAP, CausalConv1d, CausalAdaptiveLearnedPool
│   │       ├── rwkv7.py        # RWKV7Block, RWKV7BlockState, fast CPU/GPU parallel scans
│   │       └── simple_dense_net.py
│   ├── data/
│   │   ├── smb_datamodule.py      # SMBDataModule (Super Mario Bros frames & 8-to-21 action mapping)
│   │   ├── nitrogen_datamodule.py # NitroGenDataModule (streaming actions + video frames)
│   │   ├── cifar10_datamodule.py  # CIFAR10DataModule (loads uoft-cs/cifar10 from HF Hub)
│   │   ├── mnist_datamodule.py    # MNISTDataModule
│   │   └── components/
│   │       ├── smb_dataset.py      # SMBDataset & SMBStreamingDataset (.npz frames, action mapping)
│   │       ├── nitrogen_dataset.py # NitroGenDataset (streaming tar.gz, load-or-skip, shuffle buffer)
│   │       └── cifar10_dataset.py  # CIFAR10HFDataset
│   └── utils/
│       ├── trainer.py          # Shared run_train_task() lifecycle runner
│       └── ...                 # Logging, instantiators, hyperparameter tracking
└── tests/
    ├── conftest.py             # Shared fixtures (cfg_train, cfg_eval)
    ├── test_smb.py             # Super Mario Bros dataset, action mapping, and training tests
    ├── test_convnext_rwkv7.py  # Backbone shape, 4D/5D, stem-bypass, streaming, gradient tests
    ├── test_nitrogen.py        # NitroGen streaming, parquet parsing, real video loading, shuffle tests
    ├── test_poolers.py         # Pooler forward, causality, gradient flow, and streaming tests
    ├── test_rwkv7.py           # RWKV-7 block recurrence, state persistence, and gradient tests
    ├── test_configs.py         # Config validation & instantiation tests
    ├── test_datamodules.py     # DataModule batch shape and loading tests
    ├── test_dinov3_ricl_embeds.py # DINOv3 weight loading & timm equivalence tests (@pytest.mark.slow)
    ├── test_eval.py            # End-to-end evaluation pipeline tests
    ├── test_sweeps.py          # Hydra Optuna sweeper tests
    └── test_train.py           # Training pipeline integration tests
```

---

## 3. Setup & Environment

### Package Management with `uv`

The environment is managed using `uv`. CPU PyTorch wheels are configured by default in `pyproject.toml`.

```bash
# Create virtual environment and install all dependencies (including dev tools)
uv sync --extra dev

# Or install editable package in active environment
uv pip install -e ".[dev]"
```

### Kaggle & Remote Installation

The repository is packaged with setuptools and standard entry points:

```bash
# Install directly from GitHub in Kaggle notebooks or cloud environments
pip install git+https://github.com/YourUsername/ConvNeXt_Platform.git

# Available console scripts post-install:
train-command          # General training (src.train:main)
train-nitrogen         # NitroGen gamepad training (src.train_nitrogen:main)
train-smb              # Super Mario Bros gamepad training (src.train_smb:main)
eval-command           # Model evaluation (src.eval:main)
```

---

## 4. Development Workflow & Commands

### Training Workflows

```bash
# 1. Secondary Task: Super Mario Bros (SMB) Training (Easier benchmark for architecture testing)
python src/train.py experiment=smb_gamepad
python src/train_smb.py experiment=smb_gamepad

# 2. Main Task: NitroGen Gamepad Training (Large-scale behavioral cloning)
python src/train.py experiment=nitrogen_gamepad
python src/train_nitrogen.py experiment=nitrogen_gamepad

# 3. NitroGen with real gameplay videos from disk
python src/train.py experiment=nitrogen_gamepad data.video_dir=/path/to/gameplay_frames

# 4. Unfreeze ConvNeXt backbone with differential learning rate
python src/train.py model=convnext_rwkv7_gamepad_unfrozen model.convnext_lr=1e-5

# 5. Stem-bypass ablation (pooler directly feeds ConvNeXt stage 0)
python src/train.py model=convnext_rwkv7_gamepad_bypass_stem

# 6. Vision Classification Baseline (CIFAR-10)
python src/train.py data=cifar10 model=convnext

# 7. Hardware Accelerators
python src/train.py trainer=gpu trainer.devices=1
python src/train.py trainer=ddp trainer.devices=2
```

### Python API Training (No Hydra / Kaggle-Friendly)

```python
# Super Mario Bros Training
from src.train_smb import train as train_smb

smb_results = train_smb(
    data_dir="data/smb",
    batch_size=32,
    max_samples=10000,
    pretrained_dinov3=True,
    freeze_convnext=True,
    max_epochs=10,
    accelerator="auto",
)

# NitroGen Large-Scale Training
from src.train_nitrogen import train as train_nitrogen

nitrogen_results = train_nitrogen(
    video_dir="/kaggle/input/gameplay-videos",
    batch_size=32,
    max_samples=100000,
    steps_per_sample=16,
    single_step=True,
    pretrained_dinov3=True,
    freeze_convnext=True,
    max_epochs=10,
    accelerator="auto",
)
```

---

## 5. Testing Instructions

```bash
# Run all fast tests
pytest -k "not slow"
# or via Makefile
make test

# Run full test suite including slow tests
pytest
# or via Makefile
make test-full

# Run specific domain test suites
pytest tests/test_smb.py
pytest tests/test_convnext_rwkv7.py
pytest tests/test_nitrogen.py
pytest tests/test_poolers.py
pytest tests/test_rwkv7.py
```

### Type Checking

```bash
uv run pyrefly check src/ tests/
```

---

## 6. Architecture & Implementation Patterns

### 1. Gamepad Model Architecture (`src/models/components/convnext_rwkv7.py`)
- **Input Normalization:** `_InputNormalize` applies DINOv3 `DINOv3ViTImageProcessorFast` preprocessing (rescales uint8 $[0, 255] \to [0, 1]$ when needed, normalizes with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`).
- **Adaptive Spatial Pooling:** `AdaptiveLearnedPool2d` downsamples arbitrary-resolution inputs to a fixed resolution ($224 \times 224$ standard, or $56 \times 56$ when bypassing stem).
- **ConvNeXt Backbone:** DINOv3-compatible ConvNeXt backbone with pre-trained weights loaded from `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`.
- **Decoupled Autograd Flow:** When `freeze_convnext=True`, parameters have `requires_grad=False` and `convnext.eval()` is maintained. PyTorch autograd graph computation is preserved through frozen convolutions, allowing backpropagation gradients to flow back into `AdaptiveLearnedPool2d`.
- **Spatial Feature Aggregation:** `LearnedWeightedGAP(kernel_size=3, num_output=1, concat_gap=True)` pools 2D spatial feature maps using learned spatial attention weights combined with uniform global average pooling.
- **Temporal 1D Convolution:** `CausalConv1d(kernel_size=3)` with residual shortcut and LayerNorm.
- **RWKV-7 Recurrent Reasoning:** 4x `RWKV7Block` layers with residual streams and linear attention recurrence.
- **Gamepad Projection Head:** `GamepadHead` mapping representations to 21 outputs: 17 unconstrained button logits ($L_{\text{BCE}}$) and 4 continuous joystick coordinates ($L_{\text{MSE}}$) bounded to $[-1.0, 1.0]$ via `Tanh`.
- **Online Recurrent Streaming:** `init_streaming_state(batch_size, device, dtype)` and `step(x_t, state)` caching `CausalConv1d` receptive field buffers and `RWKV7BlockState` recurrent states for both single 4D frames $(B, C, H, W)$ and N-frame 5D chunks $(B, T, C, H, W)$.

### 2. Super Mario Bros Dataset (`src/data/components/smb_dataset.py`)
- **Format:** Loads $(224, 256, 3)$ uint8 frames and 8-element NES action vectors `[Up, Down, Left, Right, A, B, Start, Select]` from `DylanRiden/smb-worldmodel-data` (`smb_frames.zip`).
- **Action Mapping:** `map_nes_action_to_gamepad_21()` maps NES buttons into standard 21-D gamepad targets:
  - D-pad: `Up -> dpad_up`, `Down -> dpad_down`, `Left -> dpad_left`, `Right -> dpad_right`.
  - Buttons: `A -> south`, `B -> west`, `Start -> start`, `Select -> back`.
  - Joysticks: `j_left_x = Right - Left`, `j_left_y = Down - Up`.
- Allows zero-shot / fine-tuning transfer using the exact same 21-D `ConvNeXtRWKV7Gamepad` architecture.

### 3. NitroGen Streaming Dataset (`src/data/components/nitrogen_dataset.py`)
- **Action Tables:** Streams `actions_processed.parquet` and `metadata.json` from `nvidia/NitroGen` tar.gz shards on HuggingFace Hub via `HfFileSystem`.
- **Single-step Unrolling:** When `single_step=True`, each 16-step sequence window is unrolled into 16 individual single-frame forward passes $(3, H, W) \to (21,)$.
- **Real Video Loading ("Load or Skip"):** Loads real video frame images from `video_dir`. Missing video chunks are cleanly skipped and logged (`logger.info`), enabling training on partial video libraries.
- **Episode-Level Shuffling:** Preserves strict chronological frame sequence ($t_0 \to t_1 \to \dots \to t_{15}$) within every 16-step episode for continuous RWKV-7 temporal mixing, while shuffling episode order via a reservoir shuffle buffer (`shuffle_buffer_size=1000`).

### 4. Shared Task Runner (`src/utils/trainer.py`)
- `run_train_task(cfg, task_name)`: Deduplicated Hydra train/test lifecycle runner used by `src/train.py`, `src/train_nitrogen.py`, and `src/train_smb.py`.

---

## 7. Engineering Rules & Best Practices

1. **Fail Fast:** Do not catch generic exceptions (`except Exception:`) or hide failures behind silent fallbacks. Allow invalid configurations and missing attributes to raise immediately.
2. **Single Source of Truth:** Infer dependent parameters (e.g. `num_classes` in `ConvNeXtLitModule` inferred from `net.head.out_features`) rather than duplicating them across configs.
3. **No Dead Comments / Obvious Restatements:** Avoid narrative storytelling, decorative ASCII separators, or comments that merely restate symbol names.
4. **Hydra Interpolation:** Tie scheduler lengths dynamically to trainer configurations using variable interpolation (`T_max: ${trainer.max_epochs}`).
5. **PyTorch Lightning, Not Plain PyTorch:** Drive ALL training, evaluation, and inference through Lightning (`LightningModule` + `Trainer`).
6. **Chronological Frame Integrity:** Never shuffle individual frames within a gameplay video episode. Shuffling operates at the episode/window buffer level to maintain continuous temporal mixing for RWKV-7.
7. **Sphinx Docstrings:** Docstring parameter tags must have a space after the colon (`:param name: desc`, `:return: desc`).

---

## 8. Debugging & Common Pitfalls

| Issue | Cause | Solution |
|---|---|---|
| `UnpicklingError: Weights only load failed` | PyTorch 2.6+ defaults `torch.load(..., weights_only=True)`. | Use `torch.serialization.add_safe_globals` or load state dicts via safetensors/timm. |
| Spatial dimension too small in ConvNeXt | ConvNeXt stem (stride 4) + 3 downsampling stages (stride 2 each) = 32x reduction. Input smaller than 32x32 collapses to 0x0. | Ensure input image resolution is at least 32x32, or use `AdaptiveLearnedPool2d` with `output_size=(224, 224)`. |
| Missing video frames on disk | Training on a subset of downloaded videos. | Configure `data.video_dir`. NitroGen dataset cleanly skips and logs missing chunks with full visibility. |
| MultilabelAccuracy shape error on sequences | Torchmetrics MultilabelAccuracy validates `dim=1 == num_labels`. | Reshape sequence tensors to `(-1, 17)` before passing to metric. |
| Empty dataset evaluation error | Direct truthiness checks `if not dataset:` trigger `len(dataset) == 0`. | Always guard with explicit identity checks: `if dataset is None:`. |
| Missing logger directory / rank_zero error | Running without logger config defaults to console-only logging. | Pass `logger=csv` or `logger=tensorboard` if metric persistence is required. |
