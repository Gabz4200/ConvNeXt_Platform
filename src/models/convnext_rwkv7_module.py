"""LightningModule for ConvNeXt-RWKV7 Gamepad model."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.classification import MultilabelAccuracy


class ConvNeXtRWKV7GamepadLitModule(LightningModule):
    """LightningModule wrapping ConvNeXtRWKV7Gamepad for gamepad control training and evaluation.

    Tracks multilabel accuracy on 17 button outputs, MSE on 4 continuous joystick axes,
    and supports differential learning rates when fine-tuning the ConvNeXt backbone.

    :param net: The `ConvNeXtRWKV7Gamepad` neural network backbone.
    :param optimizer: Partial optimizer factory (e.g. AdamW).
    :param scheduler: Partial learning rate scheduler factory (e.g. CosineAnnealingLR).
    :param compile: Whether to compile the backbone with `torch.compile`.
    :param joystick_loss_weight: Multiplier weight for joystick MSE loss relative to BCE button loss.
        Default: 1.0.
    :param convnext_lr: Optional distinct learning rate for ConvNeXt backbone when fine-tuning unfrozen.
        Default: None.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: Any = None,
        scheduler: Any = None,
        compile: bool = False,
        joystick_loss_weight: float = 1.0,
        convnext_lr: float | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_btn_acc = MultilabelAccuracy(num_labels=17)
        self.val_btn_acc = MultilabelAccuracy(num_labels=17)
        self.test_btn_acc = MultilabelAccuracy(num_labels=17)

        self.train_joy_mse = MeanMetric()
        self.val_joy_mse = MeanMetric()
        self.test_joy_mse = MeanMetric()

        self.val_loss_best = MinMetric()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Perform forward pass through `self.net`.

        :param x: Image or video tensor of shape `(B, C, H, W)` or `(B, T, C, H, W)`.
        :return: Tuple `(full_gamepad, buttons_logits, joysticks_values)`.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_btn_acc.reset()
        self.val_joy_mse.reset()
        self.val_loss_best.reset()

    def model_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: Input data batch. Either `(images, target_gamepad_21)` or
            `(images, target_buttons_17, target_joysticks_4)`.
        :return: Tuple containing:
            - `loss`: Combined scalar loss.
            - `preds`: Prediction tuple `(full_gamepad, buttons_logits, joysticks_values)`.
            - `target`: Combined target tensor of shape `(..., 21)`.
        """
        x = batch[0]
        full_gamepad, btn_logits, joy_pred = self.forward(x)

        if len(batch) == 2:
            target = batch[1]
            y_btn = target[..., :17]
            y_joy = target[..., 17:]
        else:
            y_btn = batch[1]
            y_joy = batch[2]
            target = torch.cat([y_btn, y_joy], dim=-1)

        loss_btn = F.binary_cross_entropy_with_logits(btn_logits, y_btn.float())
        loss_joy = F.mse_loss(joy_pred, y_joy.float())
        loss = loss_btn + self.hparams.joystick_loss_weight * loss_joy

        return loss, (full_gamepad, btn_logits, joy_pred), target

    def training_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
        _batch_idx: int,
    ) -> Tensor:
        """Perform a single training step.

        :param batch: Training data batch.
        :param _batch_idx: Batch index.
        :return: Scalar training loss.
        """
        loss, (_, btn_logits, joy_pred), target = self.model_step(batch)
        y_btn = target[..., :17]
        y_joy = target[..., 17:]

        self.train_loss(loss)
        self.train_btn_acc(btn_logits.reshape(-1, 17), y_btn.reshape(-1, 17).long())
        self.train_joy_mse(F.mse_loss(joy_pred, y_joy.float()))

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/btn_acc", self.train_btn_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/joy_mse", self.train_joy_mse, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
        _batch_idx: int,
    ) -> None:
        """Perform a single validation step.

        :param batch: Validation data batch.
        :param _batch_idx: Batch index.
        """
        loss, (_, btn_logits, joy_pred), target = self.model_step(batch)
        y_btn = target[..., :17]
        y_joy = target[..., 17:]

        self.val_loss(loss)
        self.val_btn_acc(btn_logits.reshape(-1, 17), y_btn.reshape(-1, 17).long())
        self.val_joy_mse(F.mse_loss(joy_pred, y_joy.float()))

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/btn_acc", self.val_btn_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/joy_mse", self.val_joy_mse, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called when a validation epoch ends."""
        current_val_loss = self.val_loss.compute()
        self.val_loss_best(current_val_loss)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self,
        batch: tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor],
        _batch_idx: int,
    ) -> None:
        """Perform a single test step.

        :param batch: Test data batch.
        :param _batch_idx: Batch index.
        """
        loss, (_, btn_logits, joy_pred), target = self.model_step(batch)
        y_btn = target[..., :17]
        y_joy = target[..., 17:]

        self.test_loss(loss)
        self.test_btn_acc(btn_logits.reshape(-1, 17), y_btn.reshape(-1, 17).long())
        self.test_joy_mse(F.mse_loss(joy_pred, y_joy.float()))

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/btn_acc", self.test_btn_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/joy_mse", self.test_joy_mse, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: Tensor | tuple[Tensor, ...],
        _batch_idx: int,
    ) -> Tensor:
        """Perform a single predict step.

        :param batch: Input image/video tensor or tuple containing images.
        :param _batch_idx: Batch index.
        :return: Full gamepad prediction tensor of shape `(..., 21)`.
        """
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        full_gamepad, _, _ = self.forward(x)
        return full_gamepad

    def setup(self, stage: str) -> None:
        """Lightning hook called at beginning of fit, validate, test, or predict.

        :param stage: Stage string ('fit', 'validate', 'test', 'predict').
        """
        if self.hparams.compile and stage == "fit":
            self.net = cast(nn.Module, torch.compile(self.net))

    def configure_optimizers(self) -> Any:
        """Configure optimizers and learning-rate schedulers.

        Supports differential learning rate for ConvNeXt parameters via `convnext_lr`.

        :return: Optimizer dictionary compatible with PyTorch Lightning.
        """
        convnext_module = getattr(self.net, "convnext", None)
        convnext_params = (
            [p for p in convnext_module.parameters() if p.requires_grad]
            if convnext_module is not None
            else []
        )

        if self.hparams.convnext_lr is not None and convnext_params:
            convnext_param_set = set(convnext_params)
            other_params = [
                p
                for p in self.net.parameters()
                if p.requires_grad and p not in convnext_param_set
            ]
            param_groups = [
                {"params": other_params},
                {"params": convnext_params, "lr": self.hparams.convnext_lr},
            ]
            optimizer = self.hparams.optimizer(params=param_groups)
        else:
            trainable_params = [p for p in self.net.parameters() if p.requires_grad]
            optimizer = self.hparams.optimizer(params=trainable_params)

        if self.hparams.scheduler is not None:
            scheduler_factory: Any = self.hparams.scheduler
            lr_scheduler = scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([
        ConvNeXtRWKV7GamepadLitModule,
    ])
