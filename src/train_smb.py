"""Super Mario Bros (SMB) training script and entry point."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.smb_datamodule import SMBDataModule
from src.models.components.convnext_rwkv7 import ConvNeXtRWKV7Gamepad
from src.models.convnext_rwkv7_module import ConvNeXtRWKV7GamepadLitModule
from src.utils import RankedLogger
from src.utils.trainer import run_train_task

log = RankedLogger(__name__, rank_zero_only=True)


def train(
    # Data hyperparameters
    data_dir: str | Path = "data/smb",
    repo_id: str = "DylanRiden/smb-worldmodel-data",
    filename: str = "smb_frames.zip",
    download: bool = True,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int | None = None,
    image_size: tuple[int, int] = (224, 224),
    target_mode: str = "gamepad_21",
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
    seed: int | None = 3407,
    ckpt_path: str | None = None,
    logger_type: str | None = "csv",
    log_dir: str = "logs",
    run_test: bool = True,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Train ConvNeXtRWKV7Gamepad on Super Mario Bros worldmodel dataset.

    :param data_dir: Local directory with extracted SMB frames. Default: 'data/smb'.
    :param repo_id: Hugging Face dataset ID. Default: 'DylanRiden/smb-worldmodel-data'.
    :param filename: Archive filename on HF Hub. Default: 'smb_frames.zip'.
    :param download: Automatically download from HF Hub if not present. Default: True.
    :param batch_size: Training batch size. Default: 32.
    :param val_ratio: Fraction reserved for validation. Default: 0.1.
    :param test_ratio: Fraction reserved for testing. Default: 0.1.
    :param max_samples: Optional limit on training samples. Default: None.
    :param image_size: Input resolution `(height, width)`. Default: (224, 224).
    :param target_mode: Action target mapping ('gamepad_21' or 'nes_8'). Default: 'gamepad_21'.
    :param num_workers: DataLoader subprocess count. Default: 0.
    :param pin_memory: Pinned CUDA memory. Default: False.
    :param convnext_size: ConvNeXt size string ('tiny', etc.). Default: 'tiny'.
    :param convnext_dims: ConvNeXt stage channel dims. Default: None.
    :param convnext_depths: ConvNeXt stage block depths. Default: None.
    :param convnext_drop_path_rate: Stochastic depth rate. Default: 0.0.
    :param convnext_layer_scale_init_value: LayerScale init value. Default: 1e-6.
    :param pretrained_dinov3: Load DINOv3 pre-trained weights from HF Hub. Default: True.
    :param dinov3_repo_id: DINOv3 Hub repository ID.
    :param bypass_stem: Route pooler directly to stage 0. Default: False.
    :param freeze_convnext: Freeze ConvNeXt backbone weights. Default: True.
    :param convnext_lr: Differential learning rate for ConvNeXt. Default: 1e-4.
    :param gap_kernel_size: LearnedWeightedGAP kernel size. Default: 3.
    :param gap_concat: Concatenate uniform GAP features. Default: True.
    :param causal_conv_kernel_size: CausalConv1d kernel size. Default: 3.
    :param rwkv_dim: RWKV-7 hidden dimension. Default: 256.
    :param rwkv_head_size: Head dimension for RWKV-7. Default: 64.
    :param rwkv_layers: Number of RWKV-7 blocks. Default: 4.
    :param rwkv_dim_ffn: RWKV-7 FFN dimension. Default: None.
    :param head_hidden_dim: GamepadHead hidden dimension. Default: 256.
    :param joystick_loss_weight: Weight for joystick MSE loss. Default: 1.0.
    :param compile: Compile model with torch.compile. Default: False.
    :param lr: Learning rate. Default: 1e-3.
    :param weight_decay: Weight decay. Default: 0.01.
    :param max_epochs: Training epoch count. Default: 10.
    :param max_steps: Max training steps. Default: -1.
    :param limit_train_batches: Batches per epoch limit. Default: None.
    :param limit_val_batches: Validation batches limit. Default: None.
    :param val_check_interval: Validation check interval. Default: None.
    :param accelerator: Accelerator type ('cpu', 'cuda', 'auto'). Default: 'auto'.
    :param devices: Device count or IDs. Default: 1.
    :param precision: Precision mode ('32-true', '16-mixed'). Default: '32-true'.
    :param fast_dev_run: Run 1 batch for quick test. Default: False.
    :param seed: Random seed. Default: 3407.
    :param ckpt_path: Checkpoint to resume from. Default: None.
    :param logger_type: Logger type ('csv', 'tensorboard', or None). Default: 'csv'.
    :param log_dir: Base directory for logs. Default: 'logs'.
    :param run_test: Evaluate on test set after training. Default: True.
    :param callbacks: Optional Lightning callbacks. Default: None.
    :return: Dictionary containing `metrics`, `model`, `datamodule`, and `trainer`.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    # 1. Instantiate DataModule
    datamodule = SMBDataModule(
        data_dir=data_dir,
        repo_id=repo_id,
        filename=filename,
        download=download,
        batch_size=batch_size,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        max_samples=max_samples,
        image_size=image_size,
        target_mode=target_mode,
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
        loggers.append(CSVLogger(save_dir=log_dir, name="smb_gamepad"))
    elif logger_type == "tensorboard":
        loggers.append(TensorBoardLogger(save_dir=log_dir, name="smb_gamepad"))

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


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Hydra CLI entry point for Super Mario Bros training."""
    metric_dict, _ = run_train_task(cfg, task_name="Super Mario Bros training")
    metric_value = metric_dict.get("val/loss_best", None)
    return float(metric_value) if metric_value is not None else None


if __name__ == "__main__":
    main()


__all__ = ["main", "train"]
