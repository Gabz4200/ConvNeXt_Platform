"""Behavioral and integration tests for NitroGen dataset streaming and real video training pipeline."""

from __future__ import annotations

import io
from pathlib import Path
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
from PIL import Image

from src.data.components.nitrogen_dataset import (
    BUTTON_COLUMNS,
    NitroGenDataset,
    load_frame,
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


def create_mock_video_frames(
    video_dir: Path,
    video_id: str,
    num_frames: int = 32,
    image_size: tuple[int, int] = (64, 64),
) -> None:
    """Create mock video frame image files on disk for real video frame loading tests."""
    frame_dir = video_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        img = Image.new("RGB", image_size, color=(i * 7 % 255, i * 13 % 255, i * 17 % 255))
        img.save(frame_dir / f"frame_{i:06d}.jpg")


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


def test_load_frame_from_disk_success(tmp_path: Path) -> None:
    """Test load_frame successfully loads and resizes real image frames from video_dir."""
    video_dir = tmp_path / "videos"
    video_id = "test_video_123"
    create_mock_video_frames(video_dir, video_id, num_frames=10)

    meta = {
        "original_video": {
            "video_id": video_id,
            "start_frame": 0,
        },
        "chunk_id": "0000",
        "uuid": f"{video_id}_chunk_0000_actions",
    }

    frame = load_frame(
        frame_idx=5,
        meta=meta,
        video_dir=video_dir,
        image_size=(224, 224),
    )

    assert frame is not None
    assert isinstance(frame, torch.Tensor)
    assert frame.shape == (3, 224, 224)
    assert frame.dtype == torch.float32
    assert 0.0 <= frame.min() and frame.max() <= 1.0


def test_load_frame_missing_returns_none(tmp_path: Path) -> None:
    """Test load_frame returns None when frame is missing so the pipeline skips with logging."""
    video_dir = tmp_path / "empty_videos"
    video_dir.mkdir()

    meta = {
        "original_video": {
            "video_id": "missing_video",
            "start_frame": 0,
        },
        "chunk_id": "0000",
        "uuid": "missing_video_chunk_0000_actions",
    }

    frame = load_frame(
        frame_idx=10,
        meta=meta,
        video_dir=video_dir,
        image_size=(224, 224),
    )
    assert frame is None


def test_nitrogen_dataset_load_or_skip_chunks(tmp_path: Path) -> None:
    """Test NitroGenDataset loads video chunks when present on disk and cleanly skips missing ones with logging."""
    video_dir = tmp_path / "video_library"
    video_id_present = "vid_present"
    create_mock_video_frames(video_dir, video_id_present, num_frames=16)

    meta_present = {
        "original_video": {"video_id": video_id_present, "start_frame": 0},
        "chunk_id": "0000",
        "uuid": f"{video_id_present}_chunk_0000_actions",
    }
    meta_missing = {
        "original_video": {"video_id": "vid_missing", "start_frame": 0},
        "chunk_id": "0001",
        "uuid": "vid_missing_chunk_0001_actions",
    }

    mock_actions_1 = parse_parquet_gamepad_actions(create_mock_parquet_bytes(num_frames=16))
    mock_actions_2 = parse_parquet_gamepad_actions(create_mock_parquet_bytes(num_frames=16))

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions_1, meta_present), (mock_actions_2, meta_missing)],
    ):
        dataset = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            video_dir=video_dir,
            shards=[0],
            steps_per_sample=16,
            single_step=True,
            image_size=(224, 224),
            val_ratio=0.0,
            shuffle=False,
        )

        samples = list(dataset)
        # Only the 16 frames from the present video chunk are yielded; missing video chunk is skipped!
        assert len(samples) == 16
        for img, action in samples:
            assert img.shape == (3, 224, 224)
            assert action.shape == (21,)


def test_nitrogen_dataset_unrolling_single_step(tmp_path: Path) -> None:
    """Test that NitroGenDataset with single_step=True unrolls 16-step windows into single steps."""
    video_dir = tmp_path / "videos"
    video_id = "test_video_unroll"
    create_mock_video_frames(video_dir, video_id, num_frames=32)

    meta = {
        "original_video": {"video_id": video_id, "start_frame": 0},
        "chunk_id": "0000",
        "uuid": f"{video_id}_chunk_0000_actions",
    }

    mock_parquet = create_mock_parquet_bytes(num_frames=32)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, meta)],
    ):
        dataset = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            video_dir=video_dir,
            shards=[0],
            steps_per_sample=16,
            single_step=True,
            image_size=(224, 224),
            val_ratio=0.0,
            shuffle=False,
        )

        samples = list(dataset)
        assert len(samples) == 32
        for img, action in samples:
            assert img.shape == (3, 224, 224)
            assert action.shape == (21,)


def test_nitrogen_dataset_sequence_mode(tmp_path: Path) -> None:
    """Test that NitroGenDataset with single_step=False yields 5D sequence chunks with real frames."""
    video_dir = tmp_path / "videos"
    video_id = "test_video_seq"
    create_mock_video_frames(video_dir, video_id, num_frames=32)

    meta = {
        "original_video": {"video_id": video_id, "start_frame": 0},
        "chunk_id": "0000",
        "uuid": f"{video_id}_chunk_0000_actions",
    }

    mock_parquet = create_mock_parquet_bytes(num_frames=32)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, meta)],
    ):
        dataset = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            video_dir=video_dir,
            shards=[0],
            steps_per_sample=16,
            single_step=False,
            image_size=(224, 224),
            val_ratio=0.0,
            shuffle=False,
        )

        samples = list(dataset)
        assert len(samples) == 2
        for frames, actions in samples:
            assert frames.shape == (16, 3, 224, 224)
            assert actions.shape == (16, 21)


def test_nitrogen_dataset_max_samples_bound(tmp_path: Path) -> None:
    """Test that max_samples parameter strictly limits yielded items."""
    video_dir = tmp_path / "videos"
    video_id = "test_video_max"
    create_mock_video_frames(video_dir, video_id, num_frames=64)

    meta = {
        "original_video": {"video_id": video_id, "start_frame": 0},
        "chunk_id": "0000",
        "uuid": f"{video_id}_chunk_0000_actions",
    }

    mock_parquet = create_mock_parquet_bytes(num_frames=64)
    mock_actions = parse_parquet_gamepad_actions(mock_parquet)

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, meta)],
    ):
        dataset = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            video_dir=video_dir,
            shards=[0],
            max_samples=10,
            steps_per_sample=16,
            single_step=True,
            val_ratio=0.0,
            shuffle=False,
        )

        samples = list(dataset)
        assert len(samples) == 10


def test_nitrogen_dataset_streaming_shuffle_buffer(tmp_path: Path) -> None:
    """Test streaming shuffle buffer randomly reorders episode windows while preserving frames."""
    video_dir = tmp_path / "videos"
    video_id_1 = "test_vid_1"
    video_id_2 = "test_vid_2"
    create_mock_video_frames(video_dir, video_id_1, num_frames=32)
    create_mock_video_frames(video_dir, video_id_2, num_frames=32)

    meta_1 = {"original_video": {"video_id": video_id_1, "start_frame": 0}, "chunk_id": "0000"}
    meta_2 = {"original_video": {"video_id": video_id_2, "start_frame": 0}, "chunk_id": "0001"}

    mock_actions_1 = parse_parquet_gamepad_actions(create_mock_parquet_bytes(num_frames=32))
    mock_actions_2 = parse_parquet_gamepad_actions(create_mock_parquet_bytes(num_frames=32))

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions_1, meta_1), (mock_actions_2, meta_2)],
    ):
        dataset_unshuffled = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            video_dir=video_dir,
            shards=[0],
            steps_per_sample=16,
            single_step=True,
            shuffle=False,
            val_ratio=0.0,
        )
        dataset_shuffled = NitroGenDataset(
            repo_id="nvidia/NitroGen",
            video_dir=video_dir,
            shards=[0],
            steps_per_sample=16,
            single_step=True,
            shuffle=True,
            shuffle_buffer_size=16,
            val_ratio=0.0,
            seed=42,
        )

        unshuffled = list(dataset_unshuffled)
        shuffled = list(dataset_shuffled)

        assert len(unshuffled) == 64
        assert len(shuffled) == 64
        unshuffled_actions = torch.stack([a for _, a in unshuffled])
        shuffled_actions = torch.stack([a for _, a in shuffled])
        assert not torch.equal(unshuffled_actions, shuffled_actions)


def test_nitrogen_datamodule_dataloaders(tmp_path: Path) -> None:
    """Test NitroGenDataModule setup and DataLoader iteration with real frame loading."""
    video_dir = tmp_path / "videos"
    video_id = "test_vid_dm"
    create_mock_video_frames(video_dir, video_id, num_frames=32)

    meta = {"original_video": {"video_id": video_id, "start_frame": 0}, "chunk_id": "0000"}
    mock_actions = parse_parquet_gamepad_actions(create_mock_parquet_bytes(num_frames=32))

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, meta)],
    ):
        dm = NitroGenDataModule(
            video_dir=video_dir,
            batch_size=4,
            max_samples=16,
            val_samples=8,
            test_samples=8,
            steps_per_sample=16,
            single_step=True,
            image_size=(224, 224),
            num_workers=0,
            shuffle=False,
        )

        dm.setup("fit")
        assert dm.data_train is not None
        assert dm.data_val is not None

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        images, actions = batch
        assert images.shape == (4, 3, 224, 224)
        assert actions.shape == (4, 21)


def test_train_nitrogen_custom_hparams(tmp_path: Path) -> None:
    """Test train_nitrogen train() function execution with real frame loading and custom hparams."""
    video_dir = tmp_path / "videos"
    video_id = "test_vid_train"
    create_mock_video_frames(video_dir, video_id, num_frames=32)

    meta = {"original_video": {"video_id": video_id, "start_frame": 0}, "chunk_id": "0000"}
    mock_actions = parse_parquet_gamepad_actions(create_mock_parquet_bytes(num_frames=32))

    with patch.object(
        NitroGenDataset,
        "_stream_shard_chunks",
        return_value=[(mock_actions, meta)],
    ):
        result = train(
            video_dir=video_dir,
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
            shuffle=False,
        )

        assert "metrics" in result
        assert "model" in result
        assert "datamodule" in result
        assert "trainer" in result
        assert isinstance(result["model"], ConvNeXtRWKV7GamepadLitModule)


def test_nitrogen_hydra_configs(tmp_path: Path) -> None:
    """Test Hydra composition and instantiation for nitrogen data and experiment configs."""
    with initialize(version_base="1.3", config_path="../src/configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                "data=nitrogen",
                f"data.video_dir={tmp_path}",
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
