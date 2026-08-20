"""NitroGen streaming dataset for gamepad action learning.

Streams action annotations and metadata from HuggingFace Hub (nvidia/NitroGen)
or local archives, extracting 17 boolean button states and 4 continuous joystick
axes in [-1, 1], paired with vision frames.
"""

from __future__ import annotations

import io
import json
import logging
import math
import tarfile
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

BUTTON_COLUMNS: list[str] = [
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "dpad_up",
    "left_shoulder",
    "left_thumb",
    "left_trigger",
    "right_shoulder",
    "right_thumb",
    "right_trigger",
    "south",
    "west",
    "east",
    "north",
    "back",
    "start",
    "guide",
]

JOYSTICK_COLUMNS: list[str] = ["j_left", "j_right"]


def parse_parquet_gamepad_actions(parquet_bytes: bytes) -> Tensor:
    """Parse parquet table bytes into a 21-D gamepad tensor (17 buttons + 4 joystick axes).

    :param parquet_bytes: Raw bytes of actions_processed.parquet or actions_raw.parquet.
    :return: Float32 Tensor of shape `(num_frames, 21)`.
    """
    table = pq.read_table(io.BytesIO(parquet_bytes))

    btn_arrays = [table[col].to_numpy().astype(np.float32) for col in BUTTON_COLUMNS]
    btns = torch.from_numpy(np.stack(btn_arrays, axis=1))

    j_left = np.array(table["j_left"].to_pylist(), dtype=np.float32)
    j_right = np.array(table["j_right"].to_pylist(), dtype=np.float32)
    joys_np = np.concatenate([j_left, j_right], axis=-1)
    joys = torch.from_numpy(joys_np)

    return torch.cat([btns, joys], dim=-1)


def generate_synthetic_game_frame(
    frame_idx: int,
    total_frames: int,
    meta: dict[str, Any] | None,
    image_size: tuple[int, int] = (224, 224),
    seed: int = 42,
) -> Tensor:
    """Generate a structured synthetic game frame when raw video is not downloaded locally.

    Produces normalized float32 tensor of shape `(3, H, W)` with gameplay visual textures,
    game area bounds, and controller overlay positions derived from chunk metadata.

    **Note:** The phase offset uses `(seed % 100)`, so visual patterns repeat every 100 frames.
    This is intended only for local testing; replace with real decoded video frames or a learned
    visual encoder for production training.

    :param frame_idx: Frame index within the chunk (0 to total_frames - 1).
    :param total_frames: Total number of frames in the chunk.
    :param meta: Parsed metadata dict from metadata.json, if available.
    :param image_size: Target `(height, width)` tuple. Default: (224, 224).
    :param seed: Random seed for visual feature consistency. Default: 42.
    :return: Normalized image tensor of shape `(3, H, W)` in ImageNet color space.
    """
    height, width = image_size
    t_ratio = frame_idx / max(total_frames, 1)

    y_coords = torch.linspace(0.0, 1.0, height).unsqueeze(1).expand(height, width)
    x_coords = torch.linspace(0.0, 1.0, width).unsqueeze(0).expand(height, width)

    phase = 2.0 * math.pi * (t_ratio + (seed % 100) / 100.0)
    channel_r = 0.5 + 0.3 * torch.sin(x_coords * 4.0 + phase)
    channel_g = 0.5 + 0.3 * torch.cos(y_coords * 4.0 - phase)
    channel_b = 0.5 + 0.3 * torch.sin((x_coords + y_coords) * 3.0 + phase * 0.5)

    if meta is not None and "bbox_game_area" in meta:
        bbox = meta["bbox_game_area"]
        x_min = int(bbox.get("xtl", 0.0) * width)
        y_min = int(bbox.get("ytl", 0.0) * height)
        x_max = int(bbox.get("xbr", 1.0) * width)
        y_max = int(bbox.get("ybr", 1.0) * height)
        mask = torch.zeros(height, width, dtype=torch.float32)
        mask[max(0, y_min) : min(height, y_max), max(0, x_min) : min(width, x_max)] = 1.0
        channel_r = channel_r * mask + 0.1 * (1.0 - mask)
        channel_g = channel_g * mask + 0.1 * (1.0 - mask)
        channel_b = channel_b * mask + 0.1 * (1.0 - mask)

    frame = torch.stack([channel_r, channel_g, channel_b], dim=0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (frame - mean) / std


class NitroGenDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Streaming IterableDataset for the NitroGen dataset on HuggingFace Hub.

    Streams action annotations from tar.gz shards in `nvidia/NitroGen`, extracting 21-D gamepad
    targets (17 binary buttons and 4 joystick axes in [-1, 1]) paired with visual frame tensors.

    When `single_step=True`, each 16-step sequence window is unrolled so each forward pass
    processes 1 frame `(3, H, W)` and produces 1 Gamepad State `(21,)`, making each 16-step
    sample yield 16 individual step samples.

    :param repo_id: HuggingFace Hub repository ID. Default: 'nvidia/NitroGen'.
    :param split: Dataset split: 'train', 'val', or 'test'. Default: 'train'.
    :param shards: Optional list of integer shard indices to stream (0 to 99).
    :param max_shards: Maximum number of shards to process. Default: None (all available).
    :param max_chunks_per_shard: Maximum chunks to read per shard. Default: None (all chunks).
    :param max_samples: Maximum total samples to yield before stopping. Default: None.
    :param steps_per_sample: Number of temporal steps per chunk window. Default: 16.
    :param single_step: If True, each forward pass produces 1 Gamepad State instead of all 16,
        unrolling each 16-step sequence into 16 individual single-frame samples. Default: True.
    :param image_size: Target image resolution `(height, width)`. Default: (224, 224).
    :param val_ratio: Fraction of chunks reserved for validation when streaming. Default: 0.1.
    :param seed: Random seed for deterministic sample generation and splitting. Default: 42.
    """

    def __init__(
        self,
        repo_id: str = "nvidia/NitroGen",
        split: str = "train",
        shards: list[int] | None = None,
        max_shards: int | None = None,
        max_chunks_per_shard: int | None = None,
        max_samples: int | None = None,
        steps_per_sample: int = 16,
        single_step: bool = True,
        image_size: tuple[int, int] = (224, 224),
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.split = split
        self.max_shards = max_shards
        self.max_chunks_per_shard = max_chunks_per_shard
        self.max_samples = max_samples
        self.steps_per_sample = steps_per_sample
        self.single_step = single_step
        self.image_size = image_size
        self.val_ratio = val_ratio
        self.seed = seed

        if shards is not None:
            self.shards = shards
        else:
            total_shards = 100
            if max_shards is not None:
                total_shards = min(total_shards, max_shards)
            self.shards = list(range(total_shards))

    def _get_worker_shards(self) -> list[int]:
        """Partition shard indices across DataLoader multi-processing workers."""
        worker_info = get_worker_info()
        if worker_info is None:
            return self.shards
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        return [s for i, s in enumerate(self.shards) if i % num_workers == worker_id]

    def _is_chunk_in_split(self, chunk_idx: int) -> bool:
        """Deterministically determine if a chunk belongs to current split."""
        hash_val = (chunk_idx * 2654435761 + self.seed) % 1000 / 1000.0
        if self.split == "train":
            return hash_val >= self.val_ratio
        if self.split in ("val", "test"):
            return hash_val < self.val_ratio
        return True

    def _extract_tar_members(
        self, tar: tarfile.TarFile
    ) -> Iterator[tuple[Tensor, dict[str, Any] | None]]:
        """Extract parquet actions and metadata from tar members."""
        current_chunk_dir: str | None = None
        parquet_bytes: bytes | None = None
        metadata_dict: dict[str, Any] | None = None
        chunk_counter = 0

        for member in tar:
            if not member.isfile():
                continue

            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            chunk_dir = "/".join(parts[:-1])

            if current_chunk_dir is not None and chunk_dir != current_chunk_dir:
                if parquet_bytes is not None:
                    actions = parse_parquet_gamepad_actions(parquet_bytes)
                    yield actions, metadata_dict
                    chunk_counter += 1
                    if (
                        self.max_chunks_per_shard is not None
                        and chunk_counter >= self.max_chunks_per_shard
                    ):
                        return
                parquet_bytes = None
                metadata_dict = None

            current_chunk_dir = chunk_dir

            if member.name.endswith("actions_processed.parquet") or (
                member.name.endswith("actions_raw.parquet") and parquet_bytes is None
            ):
                extracted = tar.extractfile(member)
                if extracted is not None:
                    parquet_bytes = extracted.read()
            elif member.name.endswith("metadata.json"):
                extracted = tar.extractfile(member)
                if extracted is not None:
                    metadata_dict = json.load(extracted)

        if parquet_bytes is not None:
            actions = parse_parquet_gamepad_actions(parquet_bytes)
            yield actions, metadata_dict

    def _stream_shard_chunks(
        self, shard_idx: int
    ) -> Iterator[tuple[Tensor, dict[str, Any] | None]]:
        """Stream parsed actions and metadata from a single tar.gz shard on HF Hub."""
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        shard_path = f"datasets/{self.repo_id}/actions/SHARD_{shard_idx:04d}.tar.gz"

        # pyrefly cannot type the huggingface_hub fileobj returned by HfFileSystem.open,
        # so we cast it to Any to satisfy tarfile's stricter _Fileobj protocol.
        with fs.open(shard_path, "rb") as file_obj, tarfile.open(
            fileobj=cast(Any, file_obj), mode="r|gz"
        ) as tar:
            yield from self._extract_tar_members(tar)

    def _unroll_window_samples(
        self,
        start_t: int,
        target_window: Tensor,
        total_frames: int,
        meta: dict[str, Any] | None,
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield unrolled single-step samples for one window."""
        for step_i in range(self.steps_per_sample):
            frame_idx = start_t + step_i
            img_t = generate_synthetic_game_frame(
                frame_idx=frame_idx,
                total_frames=total_frames,
                meta=meta,
                image_size=self.image_size,
                seed=self.seed + frame_idx,
            )
            action_t = target_window[step_i]
            yield img_t, action_t

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over dataset samples, unrolling 16-step windows into single steps if enabled."""
        worker_shards = self._get_worker_shards()
        samples_yielded = 0
        chunk_idx = 0

        for shard_idx in worker_shards:
            for actions, meta in self._stream_shard_chunks(shard_idx):
                chunk_idx += 1
                if not self._is_chunk_in_split(chunk_idx):
                    continue

                total_frames = len(actions)
                if total_frames < self.steps_per_sample:
                    continue

                for start_t in range(
                    0, total_frames - self.steps_per_sample + 1, self.steps_per_sample
                ):
                    end_t = start_t + self.steps_per_sample
                    target_window = actions[start_t:end_t]

                    if self.single_step:
                        for sample in self._unroll_window_samples(
                            start_t, target_window, total_frames, meta
                        ):
                            yield sample
                            samples_yielded += 1
                            if (
                                self.max_samples is not None
                                and samples_yielded >= self.max_samples
                            ):
                                return
                    else:
                        frames = torch.stack([
                            generate_synthetic_game_frame(
                                frame_idx=start_t + step_i,
                                total_frames=total_frames,
                                meta=meta,
                                image_size=self.image_size,
                                seed=self.seed + start_t + step_i,
                            )
                            for step_i in range(self.steps_per_sample)
                        ], dim=0)

                        yield frames, target_window
                        samples_yielded += 1
                        if (
                            self.max_samples is not None
                            and samples_yielded >= self.max_samples
                        ):
                            return


__all__ = [
    "BUTTON_COLUMNS",
    "JOYSTICK_COLUMNS",
    "NitroGenDataset",
    "generate_synthetic_game_frame",
    "parse_parquet_gamepad_actions",
]
