"""CIFAR-10 HuggingFace Dataset wrapper."""

from typing import Any

import torch
from torch.utils.data import Dataset


class CIFAR10HFDataset(Dataset):
    """Thin wrapper around a HuggingFace CIFAR-10 split that applies torchvision transforms.

    :param hf_split: A HuggingFace Dataset split object containing "img" and "label" features.
    :param transform: A callable transform applied to the raw image (PIL / Tensor).
    """

    def __init__(self, hf_split: Any, transform: Any) -> None:
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.hf_split[idx]
        return self.transform(item["img"]), item["label"]
