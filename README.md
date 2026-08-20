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

Vision-temporal gamepad controller models combining **ConvNeXt** visual backbones (with **DINOv3** self-supervised priors) and **RWKV-7** linear attention recurrent blocks for **platformer gameplay and behavioral cloning**, with **NVIDIA NitroGen** dataset streaming and real-time online recurrent streaming inference.

</div>

<br>

---

## 📌 Overview

**ConvNeXt Platform** is a research and production codebase for training vision-temporal policies and controllers for gameplay and platformer games.

By fusing modern pure-convolutional visual encoders (**ConvNeXt** with **DINOv3** self-supervised representations), learned adaptive spatial pooling, and fast linear-attention recurrent reasoning (**RWKV-7** Goose blocks), the model predicts full 21-D standard gamepad states (17 boolean buttons + 2 dual-axis joysticks bounded in $[-1.0, 1.0]$) directly from raw video streams and temporal action sequences.

```
┌─────────────────────────┐
│ Input Frame (B, 3, H, W)│ (uint8 [0, 255] or float32 [0, 1])
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     _InputNormalize     │ (DINOv3 ImageNet mean/std normalization)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  AdaptiveLearnedPool2d  │ (adaptive learned downsampling to 224x224)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     DINOv3 ConvNeXt     │ (pre-trained visual hierarchy, frozen or fine-tuned)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    LearnedWeightedGAP   │ (spatial attention + global average pooling)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      CausalConv1d       │ (temporal 1D convolution + residual shortcut)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     4x RWKV-7 Blocks    │ (linear attention temporal mixing & recurrent state)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│       GamepadHead       │
└────────────┬────────────┘
             │
             ├──────────────────────────────────────────┐
             ▼                                          ▼
┌───────────────────────────┐             ┌───────────────────────────┐
│ 17 Binary Button Logits   │             │ 4 Continuous Joystick Axes│
│ (D-pad, Face, Shoulders)  │             │ (Left & Right Sticks in   │
│ Loss: BCEWithLogitsLoss   │             │  [-1.0, 1.0] via Tanh)    │
│ Metric: MultilabelAccuracy│             │ Loss: MSELoss             │
└───────────────────────────┘             └───────────────────────────┘
```

### Key Capabilities

- 🎮 **21-D Gamepad Controller Head:** Direct multi-task prediction of 17 boolean gamepad buttons (D-pad, Face, Triggers, Bumpers, Sticks, Menu) and 4 continuous joystick axes ($[-1.0, 1.0]$ via `Tanh`).
- ⚡ **RWKV-7 Recurrent Reasoning:** Temporal mixing across video frames using RWKV-7 (Goose) blocks with fast parallel scans for training and $O(1)$ recurrent step updates for inference.
- 🧬 **DINOv3 Pre-trained Visual Priors:** Native weight loading from `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` and timm checkpoints with intermediate feature extraction.
- 🔄 **Decoupled Autograd Flow:** Trainable `AdaptiveLearnedPool2d` receives backpropagation gradients through frozen ConvNeXt weights (`freeze_convnext=True`), enabling spatial pooling adaptation without corrupting pre-trained features. Supports differential learning rates (`convnext_lr`) for full fine-tuning.
- 📺 **NitroGen Dataset Streaming:** Direct streaming of action annotations from Hugging Face Hub (`nvidia/NitroGen` tar.gz shards) paired with real gameplay video frames from disk via a robust **"Load or Skip"** policy with full logging.
- ⏱️ **Episode-Level Shuffling:** Preserves strict chronological frame sequence ($t_0 \to t_1 \to \dots \to t_{15}$) within every 16-step episode window for continuous temporal mixing, while decorrelating batches across different videos via a streaming reservoir buffer.
- 🚀 **Online Recurrent Streaming Inference:** Native `init_streaming_state()` and `step(x_t, state)` API maintaining rolling causal convolution buffers and RWKV-7 state tuples for live frame-by-frame and chunk-by-chunk deployment.
- ⚙️ **Hydra 1.3 + Lightning 2.x:** Compositional YAML configs, CLI overrides, multi-GPU (DDP), mixed precision (AMP 16-bit/bf16), and `torch.compile` support.

<br>

---

## 📁 Directory Structure

```
ConvNeXt_Platform/
├── pyproject.toml               # Package configuration (convnext-platform) & CLI scripts
├── MANIFEST.in                 # Packaging manifest (includes configs and assets)
├── Makefile                    # Automation targets (train, test, format, clean)
├── src/
│   ├── configs/                # Hierarchical Hydra configurations (packaged inside src)
│   │   ├── train.yaml          # Default training config
│   │   ├── eval.yaml           # Evaluation config
│   │   ├── model/              # convnext_rwkv7_gamepad*.yaml, convnext.yaml, mnist.yaml
│   │   ├── data/               # nitrogen.yaml, cifar10.yaml, mnist.yaml
│   │   ├── trainer/            # default.yaml, cpu.yaml, gpu.yaml, ddp.yaml
│   │   └── experiment/         # nitrogen_gamepad.yaml, example.yaml
│   ├── train.py                # Main training CLI entrypoint
│   ├── train_nitrogen.py       # NitroGen training entrypoint (CLI + programmatic train() API)
│   ├── eval.py                 # Model evaluation script
│   ├── models/
│   │   ├── convnext_rwkv7_module.py # ConvNeXtRWKV7GamepadLitModule (BCE + MSE gamepad loss)
│   │   ├── convnext_module.py  # ConvNeXtLitModule (vision classifier & embeddings)
│   │   └── components/
│   │       ├── convnext_rwkv7.py # ConvNeXtRWKV7Gamepad, GamepadHead, _InputNormalize
│   │       ├── convnext.py     # Pure PyTorch ConvNeXt (DINOv3 weight ingestion)
│   │       ├── poolers.py      # AdaptiveLearnedPool2d, LearnedWeightedGAP, CausalConv1d
│   │       └── rwkv7.py        # RWKV7Block, parallel scans & recurrent states
│   ├── data/
│   │   ├── nitrogen_datamodule.py # NitroGenDataModule (streaming actions + video frames)
│   │   ├── cifar10_datamodule.py  # CIFAR10DataModule
│   │   └── components/
│   │       └── nitrogen_dataset.py # NitroGenDataset (streaming tar.gz, load-or-skip, shuffle buffer)
│   └── utils/
│       ├── trainer.py          # Shared run_train_task() lifecycle runner
│       └── ...
└── tests/                      # Behavioral pytest test suite (279 passed)
```

<br>

---

## 🚀 Installation

### 1. Local Setup with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/YourUsername/ConvNeXt_Platform.git
cd ConvNeXt_Platform

# Install all dependencies with uv
uv sync --extra dev
```

### 2. Kaggle & Cloud Installation

Install directly from GitHub into Kaggle notebooks, Colab, or remote cloud servers:

```bash
pip install git+https://github.com/YourUsername/ConvNeXt_Platform.git

# Available console scripts:
train-command          # General training (src.train:main)
train-nitrogen         # NitroGen gamepad training (src.train_nitrogen:main)
eval-command           # Model evaluation (src.eval:main)
```

<br>

---

## 💻 Training Workflows

### 1. Training NitroGen Gamepad via Hydra

```bash
# Train on NitroGen dataset with default frozen DINOv3 ConvNeXt
python src/train.py experiment=nitrogen_gamepad

# Train on real video frames from disk
python src/train.py experiment=nitrogen_gamepad data.video_dir=/path/to/gameplay_frames

# Unfreeze ConvNeXt backbone with differential learning rate
python src/train.py model=convnext_rwkv7_gamepad_unfrozen model.convnext_lr=1e-5

# Stem-bypass ablation (pooler directly feeds ConvNeXt stage 0)
python src/train.py model=convnext_rwkv7_gamepad_bypass_stem

# Train on GPU / Multi-GPU
python src/train.py experiment=nitrogen_gamepad trainer=gpu trainer.devices=1
python src/train.py experiment=nitrogen_gamepad trainer=ddp trainer.devices=2
```

### 2. Python API (Kaggle & Programmatic Usage)

Train directly without Hydra using `train()`:

```python
from src.train_nitrogen import train

results = train(
    video_dir="/kaggle/input/gameplay-videos",
    batch_size=32,
    max_samples=100000,          # Bounded sample count (or None for full dataset)
    steps_per_sample=16,         # 16-step sequence windows
    single_step=True,            # 1 gamepad state per forward pass
    shuffle=True,                # Episode-level shuffle buffer
    shuffle_buffer_size=1000,
    pretrained_dinov3=True,      # Loads DINOv3 pre-trained weights
    freeze_convnext=True,        # Freezes ConvNeXt while training pooler & RWKV-7
    convnext_lr=1e-4,            # Differential LR when unfrozen
    lr=1e-3,
    max_epochs=10,
    accelerator="auto",
)

model = results["model"]
metrics = results["metrics"]
```

### 3. Training Vision Classification Baseline (CIFAR-10)

```bash
# Train ConvNeXt-Tiny classifier on CIFAR-10
python src/train.py data=cifar10 model=convnext
```

<br>

---

## ⏱️ Real-Time Online Recurrent Streaming Inference

The model supports frame-by-frame and chunk-by-chunk live inference, caching causal convolution buffers and RWKV-7 recurrent states:

```python
import torch
from src.models.components.convnext_rwkv7 import ConvNeXtRWKV7Gamepad

model = ConvNeXtRWKV7Gamepad(pretrained_dinov3=True).eval()

# Initialize recurrent state for 1 stream
state = model.init_streaming_state(batch_size=1, device=torch.device("cpu"))

# Stream incoming live game frames (3, 224, 224)
for frame in live_game_frame_stream():
    frame_tensor = frame.unsqueeze(0)  # Shape: (1, 3, 224, 224)
    (gamepad_21, buttons_17, joysticks_4), state = model.step(frame_tensor, state)
    
    # Process predictions
    button_presses = (buttons_17.sigmoid() > 0.5).squeeze(0)
    left_stick = joysticks_4[0, :2]    # (X, Y) in [-1.0, 1.0]
    right_stick = joysticks_4[0, 2:]   # (X, Y) in [-1.0, 1.0]
```

<br>

---

## 🧪 Testing & Validation

The comprehensive test suite covers backbone forward passes, 4D/5D sequence handling, stem-bypass ablation, frozen autograd flow, real video frame loading, load-or-skip error handling, and Hydra configuration validation:

```bash
# Run all fast tests (skips slow downloads)
pytest -k "not slow"
# or via Makefile
make test

# Run full test suite including DINOv3 weight loading verification
pytest
# or via Makefile
make test-full

# Run specific domain test suites
pytest tests/test_convnext_rwkv7.py
pytest tests/test_nitrogen.py
pytest tests/test_poolers.py
pytest tests/test_rwkv7.py
```

### Type Checking

```bash
uv run pyrefly check src/ tests/
```

<br>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
