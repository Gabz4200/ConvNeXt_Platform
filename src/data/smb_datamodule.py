"""Super Mario Bros (SMB) LightningDataModule."""

from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.components.smb_dataset import SMBDataset


class SMBDataModule(LightningDataModule):
    """`LightningDataModule` for Super Mario Bros worldmodel dataset.

    Loads .npz gameplay frames and 8-button action vectors from `DylanRiden/smb-worldmodel-data`,
    mapping actions to standard 21-D gamepad targets for behavioral cloning.

    :param data_dir: Base directory to extract and load `.npz` files from. Default: 'data/smb'.
    :param repo_id: HuggingFace dataset repo ID. Default: 'DylanRiden/smb-worldmodel-data'.
    :param filename: Archive filename on HF Hub. Default: 'smb_frames.zip'.
    :param download: If True, downloads and extracts archive if not present. Default: True.
    :param batch_size: Number of samples per batch. Default: 32.
    :param val_ratio: Fraction of files reserved for validation. Default: 0.1.
    :param test_ratio: Fraction of files reserved for testing. Default: 0.1.
    :param max_samples: Optional maximum number of training samples. Default: None.
    :param image_size: Target image resolution `(height, width)`. Default: (224, 224).
    :param target_mode: 'gamepad_21' or 'nes_8'. Default: 'gamepad_21'.
    :param num_workers: Number of DataLoader subprocesses. Default: 0.
    :param pin_memory: Whether to copy Tensors into CUDA pinned memory. Default: False.
    :param seed: Random seed for deterministic train/val/test splits. Default: 42.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/smb",
        repo_id: str = "DylanRiden/smb-worldmodel-data",
        filename: str = "smb_frames.zip",
        download: bool = True,
        batch_size: int = 32,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_samples: int | None = None,
        image_size: tuple[int, int] = (224, 224),
        target_mode: str = "gamepad_21",
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: SMBDataset | None = None
        self.data_val: SMBDataset | None = None
        self.data_test: SMBDataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download and extract smb_frames.zip if download=True and frames are absent."""
        if self.hparams.download:
            frames_dir = Path(self.hparams.data_dir) / "frames"
            if not frames_dir.exists():
                dataset = SMBDataset(
                    data_dir=self.hparams.data_dir,
                    repo_id=self.hparams.repo_id,
                    filename=self.hparams.filename,
                    download=True,
                )
                del dataset

    def setup(self, stage: str | None = None) -> None:
        """Initialize train, val, and test dataset splits."""
        if self.data_train is not None and self.data_val is not None and self.data_test is not None:
            return

        if stage in ("fit", None):
            self.data_train = SMBDataset(
                data_dir=self.hparams.data_dir,
                repo_id=self.hparams.repo_id,
                filename=self.hparams.filename,
                download=False,
                split="train",
                val_ratio=self.hparams.val_ratio,
                test_ratio=self.hparams.test_ratio,
                max_samples=self.hparams.max_samples,
                image_size=self.hparams.image_size,
                target_mode=self.hparams.target_mode,
                seed=self.hparams.seed,
            )
            self.data_val = SMBDataset(
                data_dir=self.hparams.data_dir,
                repo_id=self.hparams.repo_id,
                filename=self.hparams.filename,
                download=False,
                split="val",
                val_ratio=self.hparams.val_ratio,
                test_ratio=self.hparams.test_ratio,
                max_samples=self.hparams.max_samples,
                image_size=self.hparams.image_size,
                target_mode=self.hparams.target_mode,
                seed=self.hparams.seed,
            )

        if stage in ("test", None):
            self.data_test = SMBDataset(
                data_dir=self.hparams.data_dir,
                repo_id=self.hparams.repo_id,
                filename=self.hparams.filename,
                download=False,
                split="test",
                val_ratio=self.hparams.val_ratio,
                test_ratio=self.hparams.test_ratio,
                max_samples=self.hparams.max_samples,
                image_size=self.hparams.image_size,
                target_mode=self.hparams.target_mode,
                seed=self.hparams.seed,
            )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Create and return the training dataloader."""
        if self.data_train is None:
            self.setup("fit")
        assert self.data_train is not None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Create and return the validation dataloader."""
        if self.data_val is None:
            self.setup("fit")
        assert self.data_val is not None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Create and return the test dataloader."""
        if self.data_test is None:
            self.setup("test")
        assert self.data_test is not None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


__all__ = ["SMBDataModule"]
