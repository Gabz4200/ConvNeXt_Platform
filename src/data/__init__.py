"""Data package."""

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.data.mnist_datamodule import MNISTDataModule

__all__ = ["CIFAR10DataModule", "MNISTDataModule"]
