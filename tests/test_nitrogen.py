"""Behavioral and integration tests for NitroGen dataset streaming and training pipeline."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import patch

import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer

from src.data.components.nitrogen_dataset import (
    BUTTON_COLUMNS,
    NitroGenDataset,
    generate_synthetic_game_frame,
    parse_parquet_gamepad_actions,
)
from src.data.nitrogen_datamodule import NitroGenDataModule
from src.models.convnext_rwkv7_module import ConvNeXtRWKV7GamepadLitModule
from src.train_nitrogen import train


def create_mock_parquet_bytes(num_frames: int = 32) -> bytes:
    """Create in-memory parquet table bytes containing mock NitroGen action columns."""
    data: dict[str, Any] = {}
    for col in BUTTON_COLUMNS:
        data[col] = np.random.randint(0, 2, size=num_frames).astype(np.int32)

    data["j_left"] = [
        np.random.uniform(-1.0, 1.0, size=2).astype(np.float32).tolist() for _ in range(num_frames)
    ]
    data["j_right"] = [
        np.random.uniform(-1.0, 1.0, size=2).astype(np.float32).tolist() for _ in range(num_frames)
    ]

    table = pa.Table.from_pydict(data)
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


def test_parse_parquet_gamepad_actions() -> None:
    """Test parsing parquet action tables into 21-D gamepad target tensors."""
    num_frames = 32
    raw_bytes = create_mock_parquet_bytes(num_frames=num_frames)
    actions = parse_parquet_gamepad_actions(raw_bytes)

    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (num_frames, 21)
    assert actions.dtype == torch.float32

    # Buttons in [0, 1]
    btns = actions[:, :17]
    assert torch.all((btns == 0.0) | (btns == 1.0))

    # Joysticks in [-1.0, 1.0]
    joys = actions[:, 17:]
    assert joys.shape == (num_frames, 4)
    assert torch.all((joys >= -1.0) & (joys <= 1.0))


def test_generate_synthetic_game_frame() -> None:
    """Test synthetic game frame generation and bounding box area rendering."""
    meta = {
        "bbox_game_area": {"xtl": 0.1, "ytl": 0.1, "xbr": 0.9, "ybr": 0.9},
    }
    frame = generate_synthetic_game_frame(
        frame_idx=5,
        total_frames=100,
        meta=meta,
        image_size=(224, 224),
        seed=123,
    )

    assert isinstance(frame, torch.Tensor)
    assert frame.shape == (3, 224, 224)
    assert frame.dtype == torch.float32
    assert not torch.isnan(frame).any()


def test_nitrogen_dataset_unrolling_single_step() -> None:
    """Test that NitroGenDataset with single_step=True unrolls 16-step windows into single steps."""
    mock_parquet = create_mock_parquet_bytes(num_frames=32)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, None)],
    ):
        dataset = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            shards=[0],
            steps_per_sample=16,
            single_step=True,
            image_size=(224, 224),
            val_ratio=0.0,
        )

        samples = list(dataset)
        # 32 frames with steps_per_sample=16 -> 2 windows -> 2 * 16 = 32 single-step samples
        assert len(samples) == 32
        for img, action in samples:
            assert img.shape == (3, 224, 224)
            assert action.shape == (21,)


def test_nitrogen_dataset_sequence_mode() -> None:
    """Test that NitroGenDataset with single_step=False yields 5D sequence chunks."""
    mock_parquet = create_mock_parquet_bytes(num_frames=32)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, None)],
    ):
        dataset = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            shards=[0],
            steps_per_sample=16,
            single_step=False,
            image_size=(224, 224),
            val_ratio=0.0,
        )

        samples = list(dataset)
        # 32 frames with steps_per_sample=16 -> 2 sequence chunk samples
        assert len(samples) == 2
        for frames, actions in samples:
            assert frames.shape == (16, 3, 224, 224)
            assert actions.shape == (16, 21)


def test_nitrogen_dataset_max_samples_bound() -> None:
    """Test that max_samples parameter strictly limits yielded items."""
    mock_parquet = create_mock_parquet_bytes(num_frames=64)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, None)],
    ):
        dataset = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            shards=[0],
            max_samples=10,
            steps_per_sample=16,
            single_step=True,
            val_ratio=0.0,
        )

        samples = list(dataset)
        assert len(samples) == 10


def test_nitrogen_datamodule_dataloaders() -> None:
    """Test NitroGenDataModule setup and DataLoader iteration."""
    mock_parquet = create_mock_parquet_bytes(num_frames=32)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, None)],
    ):
        dm = NitroGenDataModule(
            batch_size=4,
            max_samples=16,
            val_samples=8,
            test_samples=8,
            steps_per_sample=16,
            single_step=True,
            image_size=(224, 224),
            num_workers=0,
        )

        dm.setup("fit")
        assert dm.data_train is not None
        assert dm.data_val is not None

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        images, actions = batch
        assert images.shape == (4, 3, 224, 224)
        assert actions.shape == (4, 21)


def test_train_nitrogen_custom_hparams() -> None:
    """Test train_nitrogen train() function execution with custom hyperparameters."""
    mock_parquet = create_mock_parquet_bytes(num_frames=32)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, None)],
    ):
        result = train(
            max_samples=8,
            val_samples=4,
            test_samples=4,
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


def test_nitrogen_hydra_configs() -> None:
    """Test Hydra composition and instantiation for nitrogen data and experiment configs."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "data=nitrogen",
                "model=convnext_rwkv7_gamepad",
                "model.net.pretrained_dinov3=false",
                "trainer=cpu",
            ],
        )
        HydraConfig().set_config(cfg)
        dm = hydra.utils.instantiate(cfg.data)
        model = hydra.utils.instantiate(cfg.model)
        trainer = hydra.utils.instantiate(cfg.trainer)

        assert isinstance(dm, NitroGenDataModule)
        assert isinstance(model, ConvNeXtRWKV7GamepadLitModule)
        assert isinstance(trainer, Trainer)
