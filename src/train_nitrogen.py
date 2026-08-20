"""Training script and entry point for NitroGen dataset streaming with ConvNeXt-RWKV7 Gamepad."""

from __future__ import annotations

import os
from functools import partial
from typing import Any

import lightning as L
import torch
from lightning import Trainer
from omegaconf import DictConfig

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import hydra
import rootutils
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.nitrogen_datamodule import NitroGenDataModule
from src.models.components.convnext_rwkv7 import ConvNeXtRWKV7Gamepad
from src.models.convnext_rwkv7_module import ConvNeXtRWKV7GamepadLitModule
from src.utils import RankedLogger
from src.utils.trainer import run_train_task

log = RankedLogger(__name__, rank_zero_only=True)


def train(
    # Data hyperparameters
    repo_id: str = "nvidia/NitroGen",
    batch_size: int = 32,
    max_samples: int | None = None,
    val_samples: int = 100,
    test_samples: int = 100,
    steps_per_sample: int = 16,
    single_step: bool = True,
    shards: list[int] | None = None,
    max_shards: int | None = None,
    max_chunks_per_shard: int | None = None,
    image_size: tuple[int, int] = (224, 224),
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    # Model hyperparameters
    convnext_size: str = "tiny",
    convnext_dims: list[int] | None = None,
    convnext_depths: list[int] | None = None,
    convnext_drop_path_rate: float = 0.0,
    convnext_layer_scale_init_value: float = 1e-6,
    pretrained_dinov3: bool = True,
    dinov3_repo_id: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    bypass_stem: bool = False,
    freeze_convnext: bool = True,
    convnext_lr: float | None = 1e-4,
    gap_kernel_size: int = 3,
    gap_concat: bool = True,
    causal_conv_kernel_size: int = 3,
    rwkv_dim: int = 256,
    rwkv_head_size: int = 64,
    rwkv_layers: int = 4,
    rwkv_dim_ffn: int | None = None,
    head_hidden_dim: int = 256,
    joystick_loss_weight: float = 1.0,
    compile: bool = False,
    # Optimizer & Scheduler hyperparameters
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    # Trainer hyperparameters
    max_epochs: int = 10,
    max_steps: int = -1,
    limit_train_batches: float | None = None,
    limit_val_batches: float | None = None,
    val_check_interval: float | None = None,
    accelerator: str = "auto",
    devices: int | str = 1,
    precision: Any = "32-true",
    fast_dev_run: bool | int = False,
    seed: int | None = 42,
    ckpt_path: str | None = None,
    logger_type: str | None = "csv",
    log_dir: str = "logs",
    run_test: bool = True,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Train ConvNeXtRWKV7Gamepad on streamed NitroGen action dataset with PyTorch Lightning.

    :param repo_id: HuggingFace repository ID. Default: 'nvidia/NitroGen'.
    :param batch_size: Number of samples per batch. Default: 32.
    :param max_samples: Total training samples before epoch completion. Default: None (full dataset).
    :param val_samples: Validation samples per evaluation cycle. Default: 100.
    :param test_samples: Test samples for evaluation. Default: 100.
    :param steps_per_sample: Number of temporal steps per chunk window. Default: 16.
    :param single_step: If True, each forward pass produces 1 Gamepad State, unrolling 16 steps
        into 16 samples. Default: True.
    :param shards: Optional list of integer shard indices to stream. Default: None.
    :param max_shards: Maximum number of shards to process. Default: None.
    :param max_chunks_per_shard: Maximum chunks to read per shard. Default: None.
    :param image_size: Target image resolution `(height, width)`. Default: (224, 224).
    :param val_ratio: Fraction of chunks reserved for validation. Default: 0.1.
    :param num_workers: Number of DataLoader subprocesses. Default: 0.
    :param pin_memory: Whether to copy Tensors into CUDA pinned memory. Default: False.
    :param convnext_size: Named ConvNeXt size string ('tiny', etc.). Default: 'tiny'.
    :param convnext_dims: Channel widths per ConvNeXt stage. Default: None.
    :param convnext_depths: Blocks per ConvNeXt stage. Default: None.
    :param convnext_drop_path_rate: Stochastic depth rate. Default: 0.0.
    :param convnext_layer_scale_init_value: LayerScale init value. Default: 1e-6.
    :param pretrained_dinov3: Whether to load pre-trained DINOv3 weights from HF Hub. Default: True.
    :param dinov3_repo_id: HF Hub repository ID for DINOv3 weights.
        Default: 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'.
    :param bypass_stem: If True, pooler outputs stage-0 channels directly. Default: False.
    :param freeze_convnext: If True, freeze ConvNeXt backbone weights. Default: True.
    :param convnext_lr: Differential learning rate for ConvNeXt when unfrozen. Default: 1e-4.
    :param gap_kernel_size: Kernel size for LearnedWeightedGAP. Default: 3.
    :param gap_concat: Whether to concatenate standard GAP features. Default: True.
    :param causal_conv_kernel_size: Kernel size for CausalConv1d. Default: 3.
    :param rwkv_dim: Hidden dimension for RWKV-7 blocks. Default: 256.
    :param rwkv_head_size: Head dimension for RWKV-7 attention. Default: 64.
    :param rwkv_layers: Number of RWKV-7 blocks. Default: 4.
    :param rwkv_dim_ffn: FFN dimension for RWKV-7 blocks. Default: None.
    :param head_hidden_dim: Hidden dimension for GamepadHead. Default: 256.
    :param joystick_loss_weight: Multiplier weight for joystick MSE loss. Default: 1.0.
    :param compile: Whether to compile backbone with torch.compile. Default: False.
    :param lr: Learning rate for optimizer. Default: 1e-3.
    :param weight_decay: Weight decay for optimizer. Default: 0.01.
    :param max_epochs: Maximum training epochs. Default: 10.
    :param max_steps: Maximum training steps (-1 for unbounded by steps). Default: -1.
    :param limit_train_batches: Limit training batches per epoch. Default: None.
    :param limit_val_batches: Limit validation batches per epoch. Default: None.
    :param val_check_interval: Validation check interval. Default: None.
    :param accelerator: Accelerator type ('cpu', 'cuda', 'auto'). Default: 'auto'.
    :param devices: Number of devices or device IDs. Default: 1.
    :param precision: Precision configuration ('32-true', '16-mixed'). Default: '32-true'.
    :param fast_dev_run: Runs 1 or N batches for quick verification. Default: False.
    :param seed: Random seed for reproducibility. Default: 42.
    :param ckpt_path: Checkpoint path to resume from. Default: None.
    :param logger_type: Logger type ('csv', 'tensorboard', or None). Default: 'csv'.
    :param log_dir: Base directory for experiment logs. Default: 'logs'.
    :param run_test: Whether to evaluate on test set after training. Default: True.
    :param callbacks: Optional list of Lightning Callbacks. Default: None.
    :return: Dictionary containing `metrics`, `model`, `datamodule`, and `trainer`.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    # 1. Instantiate DataModule
    datamodule = NitroGenDataModule(
        repo_id=repo_id,
        batch_size=batch_size,
        max_samples=max_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        steps_per_sample=steps_per_sample,
        single_step=single_step,
        shards=shards,
        max_shards=max_shards,
        max_chunks_per_shard=max_chunks_per_shard,
        image_size=image_size,
        val_ratio=val_ratio,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed or 42,
    )

    # 2. Instantiate Neural Network Backbone
    net = ConvNeXtRWKV7Gamepad(
        in_chans=3,
        convnext_size=convnext_size,
        convnext_dims=convnext_dims,
        convnext_depths=convnext_depths,
        convnext_drop_path_rate=convnext_drop_path_rate,
        convnext_layer_scale_init_value=convnext_layer_scale_init_value,
        pretrained_dinov3=pretrained_dinov3,
        dinov3_repo_id=dinov3_repo_id,
        bypass_stem=bypass_stem,
        freeze_convnext=freeze_convnext,
        gap_kernel_size=gap_kernel_size,
        gap_concat=gap_concat,
        causal_conv_kernel_size=causal_conv_kernel_size,
        rwkv_dim=rwkv_dim,
        rwkv_head_size=rwkv_head_size,
        rwkv_layers=rwkv_layers,
        rwkv_dim_ffn=rwkv_dim_ffn,
        head_hidden_dim=head_hidden_dim,
    )

    # 3. Instantiate LightningModule
    optimizer_partial = partial(torch.optim.AdamW, lr=lr, weight_decay=weight_decay)
    scheduler_partial = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=max_epochs)

    effective_convnext_lr = convnext_lr if not freeze_convnext else None
    model = ConvNeXtRWKV7GamepadLitModule(
        net=net,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
        compile=compile,
        joystick_loss_weight=joystick_loss_weight,
        convnext_lr=effective_convnext_lr,
    )

    # 4. Configure Loggers
    loggers: list[Any] = []
    if logger_type == "csv":
        loggers.append(CSVLogger(save_dir=log_dir, name="nitrogen_gamepad"))
    elif logger_type == "tensorboard":
        loggers.append(TensorBoardLogger(save_dir=log_dir, name="nitrogen_gamepad"))

    # 5. Instantiate Trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        max_steps=max_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        precision=precision,
        fast_dev_run=fast_dev_run,
        logger=loggers if loggers else None,
        callbacks=callbacks or [],
    )

    # 6. Fit model
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    train_metrics = trainer.callback_metrics

    # 7. Evaluate on test set
    test_metrics: dict[str, Any] = {}
    if run_test and not fast_dev_run:
        trainer.test(model=model, datamodule=datamodule)
        test_metrics = trainer.callback_metrics

    metrics = {**train_metrics, **test_metrics}
    return {
        "metrics": metrics,
        "model": model,
        "datamodule": datamodule,
        "trainer": trainer,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Hydra CLI entry point."""
    metric_dict, _ = run_train_task(cfg, task_name="NitroGen training")
    metric_value = metric_dict.get("val/loss_best", None)
    return float(metric_value) if metric_value is not None else None


if __name__ == "__main__":
    main()


__all__ = ["main", "train"]
