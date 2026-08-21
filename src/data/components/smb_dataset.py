"""Super Mario Bros (SMB) world model dataset loader and action mapping.

Loads NES gameplay frames (224, 256, 3) and 8-button action vectors
from DylanRiden/smb-worldmodel-data, mapping actions to the standard
21-D gamepad layout (17 buttons + 4 continuous joystick axes).
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

# NES 8-button action indices: [Up, Down, Left, Right, A, B, Start, Select]
NES_ACTION_NAMES: list[str] = [
    "Up",
    "Down",
    "Left",
    "Right",
    "A",
    "B",
    "Start",
    "Select",
]


def map_nes_action_to_gamepad_21(action_8: Tensor | np.ndarray) -> Tensor:
    """Map an 8-element NES action vector to a standard 21-D gamepad target tensor.

    Layout of 21-D target:
    - Indices 0..16 (17 buttons):
      - 0: dpad_down (Down)
      - 1: dpad_left (Left)
      - 2: dpad_right (Right)
      - 3: dpad_up (Up)
      - 10: south (A button)
      - 11: west (B button)
      - 14: back (Select button)
      - 15: start (Start button)
      - Others: 0.0
    - Indices 17..20 (4 joystick axes):
      - 17: j_left_x (Right - Left in [-1.0, 1.0])
      - 18: j_left_y (Down - Up in [-1.0, 1.0])
      - 19, 20: j_right_x, j_right_y (0.0)

    :param action_8: Float tensor or array of shape `(8,)` [Up, Down, Left, Right, A, B, Start, Select].
    :return: Float32 Tensor of shape `(21,)`.
    """
    if isinstance(action_8, np.ndarray):
        action_8 = torch.from_numpy(action_8).float()
    else:
        action_8 = action_8.float()

    btns = torch.zeros(17, dtype=torch.float32)
    joys = torch.zeros(4, dtype=torch.float32)

    # D-pad buttons
    btns[3] = action_8[0]  # Up -> dpad_up
    btns[0] = action_8[1]  # Down -> dpad_down
    btns[1] = action_8[2]  # Left -> dpad_left
    btns[2] = action_8[3]  # Right -> dpad_right

    # Face & Menu buttons
    btns[10] = action_8[4]  # A -> south
    btns[11] = action_8[5]  # B -> west
    btns[15] = action_8[6]  # Start -> start
    btns[14] = action_8[7]  # Select -> back

    # Left joystick directional projection in [-1.0, 1.0]
    joys[0] = action_8[3] - action_8[2]  # Right - Left
    joys[1] = action_8[1] - action_8[0]  # Down - Up

    return torch.cat([btns, joys], dim=-1)


class SMBDataset(Dataset[tuple[Tensor, Tensor]]):
    """Super Mario Bros gameplay dataset loading .npz frame/action files.

    Loads RGB gameplay frames of shape `(224, 256, 3)` and maps 8-button NES actions
    to standard 21-D gamepad targets for behavioral cloning and representation learning.

    :param data_dir: Base directory to extract and load `.npz` files from. Default: 'data/smb'.
    :param repo_id: HuggingFace Hub dataset repository ID. Default: 'DylanRiden/smb-worldmodel-data'.
    :param filename: Archive filename on HF Hub. Default: 'smb_frames.zip'.
    :param download: If True, automatically downloads and extracts archive if not present. Default: True.
    :param split: Split to load: 'train', 'val', or 'test'. Default: 'train'.
    :param val_ratio: Fraction of files reserved for validation. Default: 0.1.
    :param test_ratio: Fraction of files reserved for testing. Default: 0.1.
    :param max_samples: Optional maximum number of samples to load. Default: None.
    :param image_size: Target image resolution `(height, width)`. Default: (224, 224).
    :param target_mode: 'gamepad_21' (standard 21-D target) or 'nes_8' (raw 8-D target). Default: 'gamepad_21'.
    :param seed: Random seed for deterministic train/val/test splits. Default: 3407.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/smb",
        repo_id: str = "DylanRiden/smb-worldmodel-data",
        filename: str = "smb_frames.zip",
        download: bool = True,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_samples: int | None = None,
        image_size: tuple[int, int] = (224, 224),
        target_mode: str = "gamepad_21",
        seed: int = 3407,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.repo_id = repo_id
        self.filename = filename
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_samples = max_samples
        self.image_size = image_size
        self.target_mode = target_mode
        self.seed = seed

        if download and not (self.data_dir / "frames").exists():
            self.download_and_extract()

        self.file_paths = self._collect_files()

    def download_and_extract(self) -> None:
        """Download smb_frames.zip from Hugging Face Hub and extract into data_dir."""
        from huggingface_hub import hf_hub_download

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {self.filename} from {self.repo_id}...")
        archive_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type="dataset",
            local_dir=str(self.data_dir),
        )

        logger.info(f"Extracting {archive_path} into {self.data_dir}...")
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(self.data_dir)

    def _extract_frame_num(self, path: Path) -> int:
        """Extract frame sequence integer from file path."""
        match = re.search(r"frame_(\d+)\.npz", path.name)
        return int(match.group(1)) if match else 0

    def _collect_files(self) -> list[Path]:
        """Collect and sort all .npz frame files for the requested split."""
        frames_dir = self.data_dir / "frames" if (self.data_dir / "frames").exists() else self.data_dir
        all_files = sorted(frames_dir.glob("**/*.npz"), key=self._extract_frame_num)

        if not all_files:
            return []

        total_files = len(all_files)
        test_count = int(total_files * self.test_ratio)
        val_count = int(total_files * self.val_ratio)
        train_count = total_files - test_count - val_count

        if self.split == "train":
            split_files = all_files[:train_count]
        elif self.split == "val":
            split_files = all_files[train_count : train_count + val_count]
        elif self.split == "test":
            split_files = all_files[train_count + val_count :]
        else:
            split_files = all_files

        if self.max_samples is not None:
            split_files = split_files[: self.max_samples]

        return split_files

    def __len__(self) -> int:
        """Return total number of files in the split."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Load and return frame tensor and gamepad target vector.

        :param idx: Sample index.
        :return: Tuple `(image_tensor, target_tensor)` where `image_tensor` has shape `(3, H, W)`
            in range `[0, 1]` and `target_tensor` has shape `(21,)` or `(8,)`.
        """
        file_path = self.file_paths[idx]
        with np.load(file_path) as data:
            frame_np = data["frame"]  # (224, 256, 3) uint8
            action_np = data["action"]  # (8,) float32

        # Convert uint8 (H, W, C) -> float32 (C, H, W) in [0, 1]
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0

        # Resize to target image size if needed
        if frame_tensor.shape[1:] != self.image_size:
            frame_tensor = TF.resize(frame_tensor, list(self.image_size))

        if self.target_mode == "gamepad_21":
            target = map_nes_action_to_gamepad_21(action_np)
        else:
            target = torch.from_numpy(action_np).float()

        return frame_tensor, target


class SMBStreamingDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Streaming IterableDataset for SMB worldmodel data streaming directly from zip archives.

    :param zip_path: Path to smb_frames.zip archive on disk.
    :param split: Split to load: 'train', 'val', or 'test'. Default: 'train'.
    :param val_ratio: Fraction of files reserved for validation. Default: 0.1.
    :param max_samples: Optional maximum number of samples. Default: None.
    :param image_size: Target image resolution `(height, width)`. Default: (224, 224).
    :param target_mode: 'gamepad_21' or 'nes_8'. Default: 'gamepad_21'.
    """

    def __init__(
        self,
        zip_path: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        max_samples: int | None = None,
        image_size: tuple[int, int] = (224, 224),
        target_mode: str = "gamepad_21",
    ) -> None:
        super().__init__()
        self.zip_path = Path(zip_path)
        self.split = split
        self.val_ratio = val_ratio
        self.max_samples = max_samples
        self.image_size = image_size
        self.target_mode = target_mode

    def _extract_frame_num(self, name: str) -> int:
        match = re.search(r"frame_(\d+)\.npz", name)
        return int(match.group(1)) if match else 0

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over .npz members directly inside the zip archive."""
        if not self.zip_path.is_file():
            raise FileNotFoundError(f"Zip archive not found at {self.zip_path}")

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with zipfile.ZipFile(self.zip_path, "r") as z:
            all_members = sorted(
                [n for n in z.namelist() if n.endswith(".npz")],
                key=self._extract_frame_num,
            )

            total = len(all_members)
            val_count = int(total * self.val_ratio)
            train_count = total - val_count

            if self.split == "train":
                members = all_members[:train_count]
            else:
                members = all_members[train_count:]

            worker_members = [m for i, m in enumerate(members) if i % num_workers == worker_id]

            for yielded, name in enumerate(worker_members, start=1):
                with z.open(name) as f:
                    data = np.load(io.BytesIO(f.read()))
                    frame_np = data["frame"]
                    action_np = data["action"]

                frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
                if frame_tensor.shape[1:] != self.image_size:
                    frame_tensor = TF.resize(frame_tensor, list(self.image_size))

                if self.target_mode == "gamepad_21":
                    target = map_nes_action_to_gamepad_21(action_np)
                else:
                    target = torch.from_numpy(action_np).float()

                yield frame_tensor, target

                if self.max_samples is not None and yielded >= self.max_samples:
                    return


__all__ = [
    "NES_ACTION_NAMES",
    "SMBDataset",
    "SMBStreamingDataset",
    "map_nes_action_to_gamepad_21",
]
