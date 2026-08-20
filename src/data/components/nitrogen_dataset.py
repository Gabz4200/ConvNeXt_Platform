"""NitroGen streaming dataset for gamepad action learning.

Streams action annotations and metadata from HuggingFace Hub (nvidia/NitroGen)
or local archives, extracting 17 boolean button states and 4 continuous joystick
axes in [-1, 1], paired with vision frames.
"""

from __future__ import annotations

import io
import json
import logging
import random
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.functional as TF
from PIL import Image
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


def load_frame(
    frame_idx: int,
    meta: dict[str, Any] | None,
    video_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
) -> Tensor | None:
    """Attempt to load and resize a real video frame from disk, returning None if missing.

    :param frame_idx: Frame index within the chunk.
    :param meta: Metadata dictionary from metadata.json.
    :param video_dir: Base directory containing gameplay video files or extracted frames.
    :param image_size: Target `(height, width)` tuple. Default: (224, 224).
    :return: Image tensor in [0, 1] range if found, or None if missing.
    """
    if meta is None:
        return None

    base_dir = Path(video_dir)
    orig_video = meta.get("original_video", {})
    video_id = orig_video.get("video_id", "")
    uuid = meta.get("uuid", "")
    chunk_id = meta.get("chunk_id", "")
    start_frame = orig_video.get("start_frame", 0)
    abs_frame_idx = start_frame + frame_idx

    candidate_paths = [
        base_dir / video_id / f"frame_{abs_frame_idx:06d}.jpg",
        base_dir / video_id / f"frame_{abs_frame_idx:06d}.png",
        base_dir / video_id / f"{abs_frame_idx:06d}.jpg",
        base_dir / video_id / f"{abs_frame_idx:06d}.png",
        base_dir / uuid / f"frame_{frame_idx:06d}.jpg",
        base_dir / uuid / f"frame_{frame_idx:06d}.png",
        base_dir / f"{video_id}_chunk_{chunk_id}" / f"frame_{frame_idx:06d}.jpg",
        base_dir / f"{video_id}_chunk_{chunk_id}" / f"frame_{frame_idx:06d}.png",
    ]

    for path in candidate_paths:
        if path.is_file():
            img = Image.open(path).convert("RGB")
            tensor = TF.to_tensor(img)
            return TF.resize(tensor, list(image_size))

    return None


class NitroGenDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Streaming IterableDataset for the NitroGen dataset on HuggingFace Hub.

    Streams action annotations from tar.gz shards in `nvidia/NitroGen`, extracting 21-D gamepad
    targets (17 binary buttons and 4 joystick axes in [-1, 1]) paired with real visual video frames.

    When `single_step=True`, each 16-step sequence window is unrolled so each forward pass
    processes 1 frame `(3, H, W)` and produces 1 Gamepad State `(21,)`, making each 16-step
    sample yield 16 individual step samples.

    :param video_dir: Base directory with real gameplay video frames. Chunks whose frames are not
        found in `video_dir` are skipped with informational logging.
    :param repo_id: HuggingFace Hub repository ID. Default: 'nvidia/NitroGen'.
    :param split: Dataset split: 'train', 'val', or 'test'. Default: 'train'.
    :param shards: Optional list of integer shard indices to stream (0 to 99).
    :param max_shards: Maximum number of shards to process. Default: None (all available).
    :param max_chunks_per_shard: Maximum chunks to read per shard. Default: None (all chunks).
    :param max_samples: Maximum total samples to yield before stopping. Default: None.
    :param steps_per_sample: Number of temporal steps per chunk window. Default: 16.
    :param single_step: If True, each forward pass produces 1 Gamepad State instead of all 16,
        unrolling each 16-step sequence into 16 individual single-frame samples. Default: True.
    :param shuffle: Whether to shuffle shards and maintain a streaming shuffle buffer. Default: True.
    :param shuffle_buffer_size: Number of samples in the streaming reservoir shuffle buffer. Default: 1000.
    :param image_size: Target image resolution `(height, width)`. Default: (224, 224).
    :param val_ratio: Fraction of chunks reserved for validation when streaming. Default: 0.1.
    :param seed: Random seed for deterministic sample generation and splitting. Default: 42.
    """

    def __init__(
        self,
        video_dir: str | Path,
        repo_id: str = "nvidia/NitroGen",
        split: str = "train",
        shards: list[int] | None = None,
        max_shards: int | None = None,
        max_chunks_per_shard: int | None = None,
        max_samples: int | None = None,
        steps_per_sample: int = 16,
        single_step: bool = True,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        image_size: tuple[int, int] = (224, 224),
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not video_dir:
            raise ValueError(
                "video_dir must be specified to load real gameplay video frames for NitroGenDataset."
            )
        self.video_dir = Path(video_dir)
        self.repo_id = repo_id
        self.split = split
        self.max_shards = max_shards
        self.max_chunks_per_shard = max_chunks_per_shard
        self.max_samples = max_samples
        self.steps_per_sample = steps_per_sample
        self.single_step = single_step
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
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
        shards = list(self.shards)
        if self.shuffle and self.split == "train":
            random.Random(self.seed).shuffle(shards)
        worker_info = get_worker_info()
        if worker_info is None:
            return shards
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        return [s for i, s in enumerate(shards) if i % num_workers == worker_id]

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

    def _raw_windows(self) -> Iterator[list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor]]:
        """Extract sequence windows across all worker shards, preserving chronological order within each window."""
        worker_shards = self._get_worker_shards()
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

                    window_frames: list[Tensor] = []
                    missing = False
                    orig_vid = meta.get("original_video", {}) if meta else {}
                    vid_id = orig_vid.get("video_id", "unknown")
                    cid = meta.get("chunk_id", "unknown") if meta else "unknown"

                    for step_i in range(self.steps_per_sample):
                        f = load_frame(
                            frame_idx=start_t + step_i,
                            meta=meta,
                            video_dir=self.video_dir,
                            image_size=self.image_size,
                        )
                        if f is None:
                            missing = True
                            break
                        window_frames.append(f)

                    if missing:
                        logger.info(
                            f"Skipping video '{vid_id}' chunk '{cid}' window [{start_t}..{end_t}]: "
                            f"frames not found in video_dir '{self.video_dir}'"
                        )
                        continue

                    logger.info(
                        f"Loaded video '{vid_id}' chunk '{cid}' window [{start_t}..{end_t}] "
                        f"({len(window_frames)} frames) from '{self.video_dir}'"
                    )

                    if self.single_step:
                        episode = [
                            (window_frames[step_i], target_window[step_i])
                            for step_i in range(self.steps_per_sample)
                        ]
                        yield episode
                    else:
                        yield (torch.stack(window_frames, dim=0), target_window)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over dataset samples with episode-level streaming shuffle buffer.

        Preserves strict chronological frame sequence within each gameplay episode/window
        so that RWKV-7 and CausalConv1d execute continuous temporal mixing.
        """
        samples_yielded = 0
        raw_windows = self._raw_windows()

        if not self.shuffle or self.shuffle_buffer_size <= 1:
            for window in raw_windows:
                if isinstance(window, list):
                    for sample in window:
                        yield sample
                        samples_yielded += 1
                        if self.max_samples is not None and samples_yielded >= self.max_samples:
                            return
                else:
                    yield window
                    samples_yielded += 1
                    if self.max_samples is not None and samples_yielded >= self.max_samples:
                        return
            return

        rng = random.Random(self.seed)
        buffer: list[list[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor]] = []

        for window in raw_windows:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(window)
            else:
                idx = rng.randint(0, len(buffer) - 1)
                selected_window = buffer[idx]
                buffer[idx] = window

                if isinstance(selected_window, list):
                    for sample in selected_window:
                        yield sample
                        samples_yielded += 1
                        if self.max_samples is not None and samples_yielded >= self.max_samples:
                            return
                else:
                    yield selected_window
                    samples_yielded += 1
                    if self.max_samples is not None and samples_yielded >= self.max_samples:
                        return

        rng.shuffle(buffer)
        for window in buffer:
            if isinstance(window, list):
                for sample in window:
                    yield sample
                    samples_yielded += 1
                    if self.max_samples is not None and samples_yielded >= self.max_samples:
                        return
            else:
                yield window
                samples_yielded += 1
                if self.max_samples is not None and samples_yielded >= self.max_samples:
                    return


__all__ = [
    "BUTTON_COLUMNS",
    "JOYSTICK_COLUMNS",
    "NitroGenDataset",
    "load_frame",
    "parse_parquet_gamepad_actions",
]
