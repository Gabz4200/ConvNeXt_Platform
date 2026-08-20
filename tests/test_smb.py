"""Behavioral and integration tests for Super Mario Bros (SMB) dataset and training pipeline."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer

from src.data.components.smb_dataset import (
    SMBDataset,
    SMBStreamingDataset,
    map_nes_action_to_gamepad_21,
)
from src.data.smb_datamodule import SMBDataModule
from src.models.convnext_rwkv7_module import ConvNeXtRWKV7GamepadLitModule
from src.train_smb import train


def create_mock_smb_npz_files(data_dir: Path, num_frames: int = 20) -> list[Path]:
    """Create mock SMB .npz files containing (224, 256, 3) frames and (8,) action vectors."""
    frames_dir = data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    for i in range(num_frames):
        frame = np.random.randint(0, 256, size=(224, 256, 3), dtype=np.uint8)
        action = np.random.randint(0, 2, size=(8,)).astype(np.float32)

        file_path = frames_dir / f"frame_{i:06d}.npz"
        np.savez(file_path, frame=frame, action=action)
        created_paths.append(file_path)

    return created_paths


def create_mock_smb_zip(zip_path: Path, num_frames: int = 10) -> None:
    """Create in-memory zip archive containing mock SMB .npz frame files."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(num_frames):
            frame = np.random.randint(0, 256, size=(224, 256, 3), dtype=np.uint8)
            action = np.random.randint(0, 2, size=(8,)).astype(np.float32)

            buf = io.BytesIO()
            np.savez(buf, frame=frame, action=action)
            z.writestr(f"frames/frame_{i:06d}.npz", buf.getvalue())


def test_map_nes_action_to_gamepad_21() -> None:
    """Test mapping 8-button NES actions to 21-D standard gamepad targets."""
    # [Up, Down, Left, Right, A, B, Start, Select]
    action_jump_right = np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=np.float32)  # Right + A
    target = map_nes_action_to_gamepad_21(action_jump_right)

    assert isinstance(target, torch.Tensor)
    assert target.shape == (21,)

    # D-pad right (index 2) and South/A (index 10)
    assert target[2].item() == 1.0
    assert target[10].item() == 1.0

    # Left joystick X should be +1.0 (Right)
    assert target[17].item() == 1.0
    # Left joystick Y should be 0.0
    assert target[18].item() == 0.0

    # Test Down + Left + B
    action_duck_left_b = np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype=np.float32)
    target2 = map_nes_action_to_gamepad_21(action_duck_left_b)

    assert target2[0].item() == 1.0  # dpad_down
    assert target2[1].item() == 1.0  # dpad_left
    assert target2[11].item() == 1.0  # B (west)
    assert target2[17].item() == -1.0  # j_left_x (Left)
    assert target2[18].item() == 1.0  # j_left_y (Down)


def test_smb_dataset_loading(tmp_path: Path) -> None:
    """Test SMBDataset loading .npz files, resizing frames, and returning 21-D targets."""
    data_dir = tmp_path / "smb_test"
    create_mock_smb_npz_files(data_dir, num_frames=20)

    dataset = SMBDataset(
        data_dir=data_dir,
        download=False,
        split="train",
        val_ratio=0.2,
        test_ratio=0.1,
        image_size=(224, 224),
    )

    # 20 * 0.7 = 14 train samples
    assert len(dataset) == 14

    img, target = dataset[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert target.shape == (21,)


def test_smb_streaming_dataset(tmp_path: Path) -> None:
    """Test SMBStreamingDataset streaming directly from smb_frames.zip."""
    zip_path = tmp_path / "smb_frames.zip"
    create_mock_smb_zip(zip_path, num_frames=10)

    dataset = SMBStreamingDataset(
        zip_path=zip_path,
        split="train",
        val_ratio=0.2,
        image_size=(224, 224),
    )

    samples = list(dataset)
    assert len(samples) == 8  # 10 * 0.8 = 8
    for img, target in samples:
        assert img.shape == (3, 224, 224)
        assert target.shape == (21,)


def test_smb_datamodule(tmp_path: Path) -> None:
    """Test SMBDataModule dataloader creation and batch shapes."""
    data_dir = tmp_path / "smb_data"
    create_mock_smb_npz_files(data_dir, num_frames=16)

    dm = SMBDataModule(
        data_dir=data_dir,
        download=False,
        batch_size=4,
        val_ratio=0.25,
        test_ratio=0.25,
        num_workers=0,
    )

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    imgs, targets = batch

    assert imgs.shape == (4, 3, 224, 224)
    assert targets.shape == (4, 21)


def test_train_smb_execution(tmp_path: Path) -> None:
    """Test train_smb train() execution with custom hyperparameters on mock SMB data."""
    data_dir = tmp_path / "smb_train_data"
    create_mock_smb_npz_files(data_dir, num_frames=12)

    result = train(
        data_dir=data_dir,
        download=False,
        batch_size=2,
        max_epochs=1,
        pretrained_dinov3=False,
        freeze_convnext=True,
        rwkv_dim=64,
        rwkv_head_size=32,
        rwkv_layers=1,
        fast_dev_run=True,
        accelerator="cpu",
        run_test=False,
    )

    assert "metrics" in result
    assert "model" in result
    assert "datamodule" in result
    assert "trainer" in result
    assert isinstance(result["model"], ConvNeXtRWKV7GamepadLitModule)


def test_smb_hydra_configs(tmp_path: Path) -> None:
    """Test Hydra composition and instantiation for SMB data and experiment configs."""
    with initialize(version_base="1.3", config_path="../src/configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "data=smb",
                f"data.data_dir={tmp_path}",
                "data.download=false",
                "model=convnext_rwkv7_gamepad",
                "model.net.pretrained_dinov3=false",
                "trainer=cpu",
            ],
        )
        HydraConfig().set_config(cfg)
        dm = hydra.utils.instantiate(cfg.data)
        model = hydra.utils.instantiate(cfg.model)
        trainer = hydra.utils.instantiate(cfg.trainer)

        assert isinstance(dm, SMBDataModule)
        assert isinstance(model, ConvNeXtRWKV7GamepadLitModule)
        assert isinstance(trainer, Trainer)
