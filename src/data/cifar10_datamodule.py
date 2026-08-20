"""CIFAR-10 DataModule loading from HuggingFace Hub (uoft-cs/cifar10)."""

from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.components.cifar10_dataset import CIFAR10HFDataset


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
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if (
            self.data_train is not None
            and self.data_val is not None
            and self.data_test is not None
        ):
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

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        if self.data_train is None:
            self.setup("fit")
        assert self.data_train is not None, "data_train must be set before train_dataloader()"
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        if self.data_val is None:
            self.setup("fit")
        assert self.data_val is not None, "data_val must be set before val_dataloader()"
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        if self.data_test is None:
            self.setup("test")
        assert self.data_test is not None, "data_test must be set before test_dataloader()"
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )


__all__ = ["CIFAR10DataModule", "CIFAR10HFDataset"]
