<div align="center">

# ConvNeXt Platform

[![python](https://img.shields.io/badge/-Python_3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

Training **ConvNeXt** vision backbones and visual representations for **platformer games** (reinforcement learning & vision-based control), featuring **DINOv3** representation loading, intermediate feature extraction, and modular PyTorch Lightning + Hydra training workflows.

</div>

<br>

---

## 📌 Overview

**ConvNeXt Platform** is dedicated to training and fine-tuning ConvNeXt visual encoders for playing **platformer games** via vision-based reinforcement learning and representation learning.

By leveraging modern pure-convolutional architectures (ConvNeXt) and self-supervised visual priors (such as DINOv3 embeddings and intermediate feature layers), the codebase provides the visual representation pipeline and training infrastructure required to perceive game scenes and drive platformer policies.

### Key Capabilities

- 🎮 **Platformer Vision & RL Backbone:** ConvNeXt backbone engineered to process visual game frames, extract spatial hierarchy features, and serve as the visual encoder for downstream reinforcement learning agents.
- 🧬 **DINOv3 Representation & Intermediate Layers:** Pure PyTorch ConvNeXt supporting flexible depths, stage dimensions, LayerScale, stochastic depth (DropPath), and per-stage intermediate layer extraction matching DINOv3 evaluation protocols.
- 🔄 **Pre-trained Weight Ingestion:** Native loaders for DINOv3 ConvNeXt weights hosted on Hugging Face Hub in both `timm` format (`timm/convnext_tiny.dinov3_lvd1689m`) and official Facebook format (`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`).
- 📦 **Hugging Face Hub DataModules:** Integrated streaming and batch loading for vision datasets (such as CIFAR-10 `uoft-cs/cifar10`) with deferred loading and torchvision transforms.
- ⚙️ **Hydra 1.3 Configuration Engine:** Dynamic composition, CLI overrides, variable interpolation, and experiment presets.
- ⚡ **PyTorch Lightning 2.x Ecosystem:** Multi-GPU (DDP), CPU, MPS, mixed precision (AMP 16-bit/bf16), gradient accumulation, and `torch.compile` support.
- 📊 **Experiment Tracking & MLOps:** Seamless integration with TensorBoard, Weights & Biases, MLflow, CSVLogger, Comet, and Neptune.
- 🎯 **Automated Hyperparameter Optimization:** Optuna sweeps via `hydra-optuna-sweeper`.

<br>

---

## 📁 Directory Structure

```
ConvNeXt_Platform/
├── configs/                     # Hierarchical Hydra configurations
│   ├── train.yaml               # Default training entry config (data: cifar10, model: convnext)
│   ├── eval.yaml                # Default evaluation entry config
│   ├── model/                   # Model architectures (convnext.yaml, mnist.yaml)
│   ├── data/                    # Dataset configs (cifar10.yaml, mnist.yaml)
│   ├── trainer/                 # Lightning Trainer configurations (default, cpu, gpu, ddp, mps)
│   ├── callbacks/               # Checkpointing, early stopping, progress bars
│   ├── logger/                  # Experiment loggers (wandb, tensorboard, csv, mlflow, etc.)
│   ├── debug/                   # Debugging presets (default/fast_dev_run, limit, overfit)
│   ├── experiment/              # Version-controlled experiment recipe overrides
│   └── hparams_search/          # Hyperparameter search configs (Optuna)
├── src/
│   ├── train.py                 # Training entrypoint script
│   ├── eval.py                  # Evaluation entrypoint script
│   ├── models/
│   │   ├── convnext_module.py   # ConvNeXtLitModule (LightningModule for training & evaluation)
│   │   ├── mnist_module.py      # Baseline module
│   │   └── components/
│   │       ├── convnext.py      # Pure PyTorch ConvNeXt + DINOv3 weight loader
│   │       └── simple_dense_net.py
│   ├── data/
│   │   ├── cifar10_datamodule.py# Hugging Face Hub CIFAR-10 DataModule
│   │   └── mnist_datamodule.py  # MNIST DataModule
│   └── utils/                   # Logging, instantiators, hyperparameter tracking
├── tests/                       # Pytest test suite
│   ├── test_configs.py          # Config validation & instantiation tests
│   ├── test_datamodules.py      # DataModule tests
│   ├── test_dinov3_ricl_embeds.py # DINOv3 weight loading & timm equivalence tests
│   ├── test_eval.py             # Evaluation pipeline integration tests
│   └── test_train.py            # Training pipeline integration tests
├── pyproject.toml               # Project metadata, dependencies, and tool settings
└── Makefile                     # Automation commands (train, test, format, clean)
```

<br>

---

## 🚀 Quickstart & Installation

### 1. Environment Setup with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/YourUsername/ConvNeXt_Platform.git
cd ConvNeXt_Platform

# Create virtual environment and install all dependencies (including dev tools)
uv sync --extra dev
```

### 2. Conda / Pip Alternative

```bash
# Using Conda
conda env create -f environment.yaml -n convnext
conda activate convnext

# Using Pip in editable mode
pip install -e ".[dev]"
```

<br>

---

## 💻 Training & Running Experiments

The platform uses Hydra for configuration composition. Any parameter can be overridden directly from the command line.

### Basic Training

```bash
# Train default model (ConvNeXt-Tiny on CIFAR-10)
python src/train.py

# Or via Makefile
make train
```

### Hardware Accelerators

```bash
# Train on CPU
python src/train.py trainer=cpu

# Train on 1 GPU
python src/train.py trainer=gpu

# Train on Multi-GPU with Distributed Data Parallel (DDP)
python src/train.py trainer=ddp trainer.devices=2

# Train on Apple Silicon (MPS)
python src/train.py trainer=mps
```

### Hyperparameter Overrides

```bash
# Modify learning rate, batch size, and maximum epochs
python src/train.py model.optimizer.lr=1e-3 data.batch_size=64 trainer.max_epochs=50

# Train with Automatic Mixed Precision (FP16 or BF16)
python src/train.py trainer=gpu +trainer.precision=16-mixed

# Enable PyTorch 2.0 graph compilation
python src/train.py model.compile=true
```

### Using Experiment Configs

Version-controlled experiment recipes reside in [`configs/experiment/`](configs/experiment/):

```bash
# Run a specific experiment recipe
python src/train.py experiment=example
```

### Resuming from Checkpoints

```bash
# Resume training from a saved checkpoint
python src/train.py ckpt_path="/path/to/checkpoint.ckpt"
```

<br>

---

## 📈 Evaluation

Evaluate trained checkpoints against test splits:

```bash
# Evaluate checkpoint on test set
python src/eval.py ckpt_path="/path/to/checkpoint.ckpt"

# Evaluate on CPU
python src/eval.py ckpt_path="/path/to/checkpoint.ckpt" trainer=cpu
```

<br>

---

## 🔍 Debugging & Diagnostics

Rapid diagnostic presets are available under [`configs/debug/`](configs/debug/):

```bash
# Fast dev run: 1 train batch, 1 val batch, 1 test batch (bypasses loggers/checkpoints)
python src/train.py debug=default

# Limit batches: train on only 10% of the dataset
python src/train.py debug=limit

# Overfit verification: overfit on 10 batches to verify gradient flow and convergence
python src/train.py debug=overfit

# Profile execution bottlenecks with PyTorch Profiler
python src/train.py debug=profiler

# Detect tensor anomalies (NaNs / infinities)
python src/train.py +trainer.detect_anomaly=true
```

<br>

---

## 📊 Experiment Tracking & Logging

PyTorch Lightning supports various experiment tracking platforms:

```bash
# Weights & Biases
python src/train.py logger=wandb

# TensorBoard
python src/train.py logger=tensorboard

# CSV Logging
python src/train.py logger=csv

# Multiple loggers simultaneously
python src/train.py logger=many_loggers
```

Configure logger entities, project names, and credentials in [`configs/logger/`](configs/logger/).

<br>

---

## 🎯 Hyperparameter Sweeps (Optuna)

Execute automated hyperparameter optimization using Hydra's Optuna sweeper:

```bash
# Run multi-trial Optuna sweep across learning rates and batch sizes
python src/train.py -m hparams_search=mnist_optuna
```

Results are saved to `logs/task_name/multiruns/`.

<br>

---

## 🧪 Testing

The test suite covers configuration validation, model construction, DINOv3 weight loading equivalence, and training execution:

```bash
# Run fast tests (skips slow downloads)
pytest -k "not slow"
# or via Makefile
make test

# Run full test suite including DINOv3 Hub weight loading verification
pytest
# or via Makefile
make test-full

# Run individual test modules
pytest tests/test_configs.py
pytest tests/test_datamodules.py
pytest tests/test_dinov3_ricl_embeds.py
```

<br>

---

## 🛠️ Code Quality & Standards

Pre-commit hooks ensure consistent formatting and type hygiene:

```bash
# Run formatting and linting checks across all files
pre-commit run -a
# or via Makefile
make format
```

- **Formatting:** Black (`line-length = 99`), `isort` (`--profile black`).
- **Docstrings:** Sphinx-style formatting via `docformatter`, coverage verified with `interrogate`.
- **Linting:** Flake8, Bandit security checks.

<br>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
