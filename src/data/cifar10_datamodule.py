"""CIFAR-10 DataModule loading from HuggingFace Hub (uoft-cs/cifar10)."""

from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class CIFAR10HFDataset(Dataset):
    """Thin wrapper around a HuggingFace CIFAR-10 split that applies torchvision transforms."""

    def __init__(self, hf_split: Any, transform: Any) -> None:
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.hf_split[idx]
        return self.transform(item["img"]), item["label"]


class CIFAR10DataModule(LightningDataModule):
    """`LightningDataModule` for CIFAR-10 loaded from the HuggingFace Hub.

    Dataset: https://huggingface.co/datasets/uoft-cs/cifar10
    50,000 train images / 10,000 test images, 32x32 RGB, 10 classes.
    """

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    def __init__(
        self,
        data_dir: str = "data/",
        val_size: int = 5_000,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `CIFAR10DataModule`.

        :param data_dir: Directory used as the HuggingFace datasets cache.
        :param val_size: Number of training examples (count, not fraction) held out for validation.
        :param batch_size: Batch size for all dataloaders.
        :param num_workers: Number of DataLoader worker processes.
        :param pin_memory: Whether to pin memory in DataLoader (recommended when using GPU).
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return 10

    def prepare_data(self) -> None:
        """Download the dataset to the cache directory.

        Called once on the main process (not replicated across DDP workers).
        """
        import datasets as hf_datasets  # type: ignore[import-untyped]

        hf_datasets.load_dataset(
            "uoft-cs/cifar10",
            cache_dir=self.hparams["data_dir"],
        )

    def setup(self, stage: str | None = None) -> None:
        """Load and split the dataset.

        :param stage: One of `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.data_train is not None and self.data_val is not None and self.data_test is not None:
            return

        import datasets as hf_datasets  # type: ignore[import-untyped]

        raw = hf_datasets.load_dataset(
            "uoft-cs/cifar10",
            cache_dir=self.hparams["data_dir"],
        )

        split = raw["train"].train_test_split(
            test_size=self.hparams["val_size"],
            seed=42,
            shuffle=True,
        )

        self.data_train = CIFAR10HFDataset(split["train"], self.train_transform)
        self.data_val = CIFAR10HFDataset(split["test"], self.eval_transform)
        self.data_test = CIFAR10HFDataset(raw["test"], self.eval_transform)

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""
        assert self.data_train is not None, "Call setup() before train_dataloader()."
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        assert self.data_val is not None, "Call setup() before val_dataloader()."
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        assert self.data_test is not None, "Call setup() before test_dataloader()."
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )
