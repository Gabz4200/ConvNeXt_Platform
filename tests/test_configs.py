from typing import cast

import hydra
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_convnext_feature_extraction_predict() -> None:
    """Feature-extraction (num_classes=0) model config runs embeddings via Lightning predict.

    Instantiates `model=convnext_embeds` through Hydra and drives inference with `Trainer.predict`,
    verifying the DINOv3-compatible backbone is used via Lightning.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model=convnext_embeds"],
        )

    HydraConfig().set_config(cfg)

    model = hydra.utils.instantiate(cfg.model)
    assert not model.is_classifier
    assert model.embed_dim == 768

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    dataloader = DataLoader(torch.utils.data.TensorDataset(torch.randn(2, 3, 64, 64)))
    outputs = trainer.predict(model, dataloaders=dataloader)
    assert outputs is not None
    features = torch.cat(cast(list[torch.Tensor], outputs))
    assert features.shape == (2, 768)
