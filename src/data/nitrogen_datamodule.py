"""NitroGen DataModule streaming gamepad action sequences and frames from HuggingFace Hub."""

from __future__ import annotations

from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.components.nitrogen_dataset import NitroGenDataset


class NitroGenDataModule(LightningDataModule):
    """`LightningDataModule` for streaming NitroGen gamepad dataset from HuggingFace Hub.

    Streams parquet action tables and metadata from `nvidia/NitroGen`, extracting 21-D gamepad
    targets (17 binary buttons + 4 continuous joystick axes in [-1, 1]) and visual frames.

    When `single_step=True`, each 16-step sequence window is unrolled into 16 individual
    step samples so each forward pass processes 1 frame `(3, H, W)` and produces 1 Gamepad State `(21,)`.

    **Note:** The train/val split is computed per-chunk using a deterministic hash, so individual
    shards can contain both train and val chunks. This is intentional for streaming reproducibility
    but means shard-level filtering (e.g. `shards=[0]`) does not guarantee a single split.

    :param repo_id: HuggingFace repository ID. Default: 'nvidia/NitroGen'.
    :param batch_size: Number of samples per batch. Default: 32.
    :param max_samples: Total training samples before epoch completion. Default: None.
    :param val_samples: Validation samples per evaluation cycle. Default: 100.
    :param test_samples: Test samples for evaluation. Default: 100.
    :param steps_per_sample: Number of temporal steps per chunk window. Default: 16.
    :param single_step: If True, each forward pass produces 1 Gamepad State, unrolling 16 steps
        into 16 samples. Default: True.
    :param shuffle: Whether to shuffle shards and maintain a streaming shuffle buffer. Default: True.
    :param shuffle_buffer_size: Size of streaming reservoir shuffle buffer. Default: 1000.
    :param shards: Optional list of integer shard indices to stream. Default: None.
    :param max_shards: Maximum number of shards to process. Default: None.
    :param max_chunks_per_shard: Maximum chunks to read per shard. Default: None.
    :param image_size: Target image resolution `(height, width)`. Default: (224, 224).
    :param val_ratio: Fraction of chunks reserved for validation. Default: 0.1.
    :param num_workers: Number of DataLoader subprocesses. Default: 0.
    :param pin_memory: Whether to copy Tensors into CUDA pinned memory. Default: False.
    :param seed: Random seed for deterministic sample generation and splitting. Default: 42.
    """

    def __init__(
        self,
        repo_id: str = "nvidia/NitroGen",
        batch_size: int = 32,
        max_samples: int | None = None,
        val_samples: int = 100,
        test_samples: int = 100,
        steps_per_sample: int = 16,
        single_step: bool = True,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        shards: list[int] | None = None,
        max_shards: int | None = None,
        max_chunks_per_shard: int | None = None,
        image_size: tuple[int, int] = (224, 224),
        val_ratio: float = 0.1,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: NitroGenDataset | None = None
        self.data_val: NitroGenDataset | None = None
        self.data_test: NitroGenDataset | None = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Verify repository access and HuggingFace Hub connectivity."""
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        if not fs.exists(f"datasets/{self.hparams.repo_id}/actions"):
            raise FileNotFoundError(
                f"Could not find actions directory in datasets/{self.hparams.repo_id}"
            )

    def setup(self, stage: str | None = None) -> None:
        """Initialize streaming datasets for train, val, and test splits."""
        if self.data_train is not None and self.data_val is not None and self.data_test is not None:
            return

        if stage in ("fit", None):
            self.data_train = NitroGenDataset(
                repo_id=self.hparams.repo_id,
                split="train",
                shards=self.hparams.shards,
                max_shards=self.hparams.max_shards,
                max_chunks_per_shard=self.hparams.max_chunks_per_shard,
                max_samples=self.hparams.max_samples,
                steps_per_sample=self.hparams.steps_per_sample,
                single_step=self.hparams.single_step,
                shuffle=self.hparams.shuffle,
                shuffle_buffer_size=self.hparams.shuffle_buffer_size,
                image_size=self.hparams.image_size,
                val_ratio=self.hparams.val_ratio,
                seed=self.hparams.seed,
            )
            self.data_val = NitroGenDataset(
                repo_id=self.hparams.repo_id,
                split="val",
                shards=self.hparams.shards,
                max_shards=self.hparams.max_shards,
                max_chunks_per_shard=self.hparams.max_chunks_per_shard,
                max_samples=self.hparams.val_samples,
                steps_per_sample=self.hparams.steps_per_sample,
                single_step=self.hparams.single_step,
                shuffle=False,
                shuffle_buffer_size=0,
                image_size=self.hparams.image_size,
                val_ratio=self.hparams.val_ratio,
                seed=self.hparams.seed + 1,
            )

        if stage in ("test", None):
            self.data_test = NitroGenDataset(
                repo_id=self.hparams.repo_id,
                split="test",
                shards=self.hparams.shards,
                max_shards=self.hparams.max_shards,
                max_chunks_per_shard=self.hparams.max_chunks_per_shard,
                max_samples=self.hparams.test_samples,
                steps_per_sample=self.hparams.steps_per_sample,
                single_step=self.hparams.single_step,
                shuffle=False,
                shuffle_buffer_size=0,
                image_size=self.hparams.image_size,
                val_ratio=self.hparams.val_ratio,
                seed=self.hparams.seed + 2,
            )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Create and return the training streaming dataloader."""
        if self.data_train is None:
            self.setup("fit")
        assert self.data_train is not None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Create and return the validation streaming dataloader."""
        if self.data_val is None:
            self.setup("fit")
        assert self.data_val is not None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor]]:
        """Create and return the test streaming dataloader."""
        if self.data_test is None:
            self.setup("test")
        assert self.data_test is not None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


__all__ = ["NitroGenDataModule"]
