"""Data package."""

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.data.mnist_datamodule import MNISTDataModule
from src.data.nitrogen_datamodule import NitroGenDataModule

__all__ = ["CIFAR10DataModule", "MNISTDataModule", "NitroGenDataModule"]
