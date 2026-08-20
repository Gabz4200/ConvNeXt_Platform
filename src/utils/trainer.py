"""Shared training task runner used by both src/train.py and src/train_nitrogen.py."""

from __future__ import annotations

from typing import Any

import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    pylogger,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def run_train_task(
    cfg: DictConfig,
    *,
    task_name: str = "training",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the common train/test lifecycle from a Hydra ``DictConfig``.

    Instantiates datamodule, model, callbacks, loggers, and trainer from the config,
    then executes ``trainer.fit`` followed by ``trainer.test``. The resulting
    ``object_dict`` is logged to all active loggers via :func:`log_hyperparameters`.

    :param cfg: Composed Hydra config with ``data``, ``model``, ``trainer``, and
        optional ``callbacks``, ``logger``, ``ckpt_path``, ``train``, ``test`` keys.
    :param task_name: Human-readable label used in log messages. Default: ``"training"``.
    :return: Tuple ``(metric_dict, object_dict)`` where ``metric_dict`` merges
        train and test metrics, and ``object_dict`` contains all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = L.utilities.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = L.utilities.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = L.utilities.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train", True):
        log.info(f"Starting {task_name}!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test", True):
        log.info(f"Starting {task_name} testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


__all__ = ["run_train_task"]
