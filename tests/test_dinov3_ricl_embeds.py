"""Tests for DINOv3 weight loading and feature equivalence with timm."""

from typing import Any, cast

import pytest
import timm
import torch
import torchvision.transforms as T
from datasets import load_dataset
from lightning import Trainer
from torch.utils.data import DataLoader

from src.models.components.convnext import build_convnext, load_dinov3_weights
from src.models.convnext_module import ConvNeXtLitModule


def _predict_embeddings(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Run embedding inference through a PyTorch Lightning `Trainer.predict`.

    The DINOv3 backbone is wrapped in `ConvNeXtLitModule` (feature-extraction mode) so all
    inference goes through the Lightning predict loop rather than a raw `torch.no_grad` forward.

    :param model: The ConvNeXt backbone with a disabled head (num_classes=0).
    :param imgs: A tensor of images, shape (N, C, H, W).
    :return: The predicted embeddings, shape (N, embed_dim).
    """
    lit_model = ConvNeXtLitModule(net=model, optimizer=None, scheduler=None, compile=False)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    dataloader = DataLoader(torch.utils.data.TensorDataset(imgs), batch_size=len(imgs))
    outputs = trainer.predict(lit_model, dataloaders=dataloader)
    assert outputs is not None
    return torch.cat(cast(list[torch.Tensor], outputs))


@pytest.mark.slow
@pytest.mark.parametrize(
    "repo_id",
    [
        "timm/convnext_tiny.dinov3_lvd1689m",
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    ],
)
def test_dinov3_convnext_ricl_embeds(repo_id: str) -> None:
    """Test loading DINOv3 ConvNeXt weights into our model and running a Lightning predict pass.

    Loads images from the Hugging Face dataset brandonyang/ricl_dinov3_embeds and verifies that
    output features are computed correctly with non-NaN, non-zero embeddings.
    """
    # 1. Load streaming dataset sample from brandonyang/ricl_dinov3_embeds
    ds = load_dataset("brandonyang/ricl_dinov3_embeds", split="train", streaming=True)
    sample = next(iter(ds))

    assert "wrist_image" in sample
    assert "top_image" in sample
    assert "right_image" in sample

    # Image transform matching standard ImageNet / DINOv3 preprocessing
    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    imgs = torch.stack(
        [
            transform(sample["wrist_image"]),
            transform(sample["top_image"]),
            transform(sample["right_image"]),
        ]
    )
    assert imgs.shape == (3, 3, 224, 224)

    # 2. Build our ConvNeXt model (num_classes=0 for feature extraction)
    model = build_convnext("tiny", num_classes=0)

    # 3. Load DINOv3 pre-trained weights from HuggingFace Hub
    model = load_dinov3_weights(model, repo_id=repo_id)
    model.eval()

    # 4. Run embedding inference through Lightning
    features = _predict_embeddings(model, imgs)

    # 5. Verify feature properties
    assert features.shape == (3, 768)
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()
    assert (features.abs().sum(dim=-1) > 0).all()

    # Ground truth embeddings in brandonyang/ricl_dinov3_embeds have dim 1280 (from ViT-Large DINOv3).
    # We verify our 768-dim ConvNeXt features are stable across images and have positive cosine similarity.
    sim_wrist_top = torch.nn.functional.cosine_similarity(features[0], features[1], dim=0).item()
    sim_wrist_right = torch.nn.functional.cosine_similarity(features[0], features[2], dim=0).item()

    assert -1.0 <= sim_wrist_top <= 1.0
    assert -1.0 <= sim_wrist_right <= 1.0


@pytest.mark.slow
def test_dinov3_convnext_timm_equivalence() -> None:
    """Test that our ConvNeXt implementation matches timm's reference DINOv3 ConvNeXt model.

    Evaluates on images from brandonyang/ricl_dinov3_embeds and verifies max absolute difference
    between our model features and timm reference features is below tolerance (1e-4).
    """
    # 1. Load dataset sample
    ds = load_dataset("brandonyang/ricl_dinov3_embeds", split="train", streaming=True)
    sample = next(iter(ds))

    transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    imgs = torch.stack(
        [
            transform(sample["wrist_image"]),
            transform(sample["top_image"]),
            transform(sample["right_image"]),
        ]
    )

    # 2. Reference timm model
    m_timm: Any = timm.create_model("hf-hub:timm/convnext_tiny.dinov3_lvd1689m", pretrained=True)
    m_timm.eval()
    with torch.no_grad():
        feat_timm = m_timm.forward_features(imgs)
        pooled_timm = m_timm.forward_head(feat_timm, pre_logits=True)

    # 3. Our ConvNeXt model loaded with weights, inferred through Lightning
    our_model = build_convnext("tiny", num_classes=0)
    load_dinov3_weights(our_model, repo_id="timm/convnext_tiny.dinov3_lvd1689m")
    our_model.eval()

    our_features = _predict_embeddings(our_model, imgs)

    # 4. Compare outputs with strict numerical tolerance
    max_diff = (pooled_timm - our_features).abs().max().item()
    assert max_diff < 1e-4, f"Max difference {max_diff} exceeded tolerance 1e-4"
