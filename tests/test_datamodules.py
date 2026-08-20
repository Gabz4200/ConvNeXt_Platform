"""Tests for dataset components and PyTorch Lightning DataModules."""

from pathlib import Path

import pytest
import torch
from torchvision import transforms

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.data.components.cifar10_dataset import CIFAR10HFDataset
from src.data.mnist_datamodule import MNISTDataModule


def test_cifar10_hf_dataset() -> None:
    """Test CIFAR10HFDataset wrapper."""
    fake_split = [
        {"img": torch.zeros((3, 32, 32)), "label": 0},
        {"img": torch.ones((3, 32, 32)), "label": 1},
    ]
    transform = transforms.Lambda(lambda x: x + 1.0)
    dataset = CIFAR10HFDataset(fake_split, transform=transform)

    assert len(dataset) == 2
    img, label = dataset[0]
    assert label == 0
    assert torch.allclose(img, torch.ones((3, 32, 32)))


def test_cifar10_datamodule_properties() -> None:
    """Test basic properties of CIFAR10DataModule."""
    dm = CIFAR10DataModule(data_dir="data/", batch_size=64)
    assert dm.num_classes == 10
    assert dm.hparams.batch_size == 64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert dm.data_train is None and dm.data_val is None and dm.data_test is None
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train is not None and dm.data_val is not None and dm.data_test is not None
    assert (
        hasattr(dm.data_train, "__len__")
        and hasattr(dm.data_val, "__len__")
        and hasattr(dm.data_test, "__len__")
    )
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
