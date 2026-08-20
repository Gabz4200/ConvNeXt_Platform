"""Behavioral and integration tests for ConvNeXt-RWKV7 Gamepad model and LitModule."""

from __future__ import annotations

from typing import Any, cast

import hydra
import pytest
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.models.components.convnext_rwkv7 import (
    ConvNeXtRWKV7Gamepad,
    GamepadHead,
    GamepadStreamingState,
)
from src.models.convnext_rwkv7_module import ConvNeXtRWKV7GamepadLitModule


@pytest.fixture
def small_model_kwargs() -> dict[str, Any]:
    """Provide lightweight parameters for fast CPU unit testing."""
    return {
        "in_chans": 3,
        "pool_intermediate_features": 8,
        "convnext_dims": [16, 32, 64, 128],
        "convnext_depths": [1, 1, 1, 1],
        "pretrained_dinov3": False,
        "freeze_convnext": True,
        "gap_kernel_size": 3,
        "gap_concat": True,
        "causal_conv_kernel_size": 3,
        "rwkv_dim": 64,
        "rwkv_head_size": 32,
        "rwkv_layers": 4,
        "head_hidden_dim": 32,
        "num_buttons": 17,
        "num_joysticks": 2,
    }


class TestGamepadHead:
    """Tests for GamepadHead component."""

    def test_output_shapes_and_bounds(self) -> None:
        head = GamepadHead(in_features=64, hidden_dim=32, num_buttons=17, num_joysticks=2)
        x = torch.randn(4, 64)
        full_gamepad, buttons, joysticks = head(x)

        assert full_gamepad.shape == (4, 21)
        assert buttons.shape == (4, 17)
        assert joysticks.shape == (4, 4)
        assert (joysticks >= -1.0).all() and (joysticks <= 1.0).all()
        assert torch.equal(full_gamepad, torch.cat([buttons, joysticks], dim=-1))

    def test_gradient_flow(self) -> None:
        head = GamepadHead(in_features=32, hidden_dim=16)
        x = torch.randn(2, 32, requires_grad=True)
        full_gamepad, _, _ = head(x)
        loss = full_gamepad.sum()
        loss.backward()

        assert x.grad is not None
        for param in head.parameters():
            assert param.grad is not None


class TestConvNeXtRWKV7Gamepad:
    """Tests for ConvNeXtRWKV7Gamepad backbone."""

    def test_forward_4d_shapes(self, small_model_kwargs: dict[str, Any]) -> None:
        model = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        x = torch.randn(2, 3, 40, 40)
        full_gamepad, buttons, joysticks = model(x)

        assert full_gamepad.shape == (2, 21)
        assert buttons.shape == (2, 17)
        assert joysticks.shape == (2, 4)
        assert (joysticks >= -1.0).all() and (joysticks <= 1.0).all()

    def test_forward_5d_shapes(self, small_model_kwargs: dict[str, Any]) -> None:
        model = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        x = torch.randn(2, 5, 3, 40, 40)
        full_gamepad, buttons, joysticks = model(x)

        assert full_gamepad.shape == (2, 5, 21)
        assert buttons.shape == (2, 5, 17)
        assert joysticks.shape == (2, 5, 4)
        assert (joysticks >= -1.0).all() and (joysticks <= 1.0).all()

    def test_invalid_input_dimension(self, small_model_kwargs: dict[str, Any]) -> None:
        model = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        x_3d = torch.randn(2, 3, 40)
        with pytest.raises(ValueError, match="Expected 4D .* or 5D .* input"):
            model(x_3d)

    @pytest.mark.parametrize("bypass_stem", [False, True])
    def test_bypass_stem_ablation(
        self, bypass_stem: bool, small_model_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(small_model_kwargs)
        kwargs["bypass_stem"] = bypass_stem
        model = ConvNeXtRWKV7Gamepad(**kwargs)

        if bypass_stem:
            assert model.pooler.out_features == cast(list[int], kwargs["convnext_dims"])[0]
            assert model.pooler.output_size == (56, 56)
        else:
            assert model.pooler.out_features == kwargs["in_chans"]
            assert model.pooler.output_size == (224, 224)

        x = torch.randn(2, 3, 36, 36)
        full_gamepad, _, _ = model(x)
        assert full_gamepad.shape == (2, 21)

    @pytest.mark.parametrize("kernel_size", [2, 3])
    def test_gap_kernel_sizes(
        self, kernel_size: int, small_model_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(small_model_kwargs)
        kwargs["gap_kernel_size"] = kernel_size
        model = ConvNeXtRWKV7Gamepad(**kwargs)
        x = torch.randn(2, 3, 32, 32)
        full_gamepad, _, _ = model(x)
        assert full_gamepad.shape == (2, 21)

    def test_gradient_flow_frozen_convnext(
        self, small_model_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(small_model_kwargs)
        kwargs["freeze_convnext"] = True
        kwargs["bypass_stem"] = True
        model = ConvNeXtRWKV7Gamepad(**kwargs)

        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        full_gamepad, _, _ = model(x)
        loss = full_gamepad.sum()
        loss.backward()

        # Input x must receive gradients through the entire frozen network
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Pooler parameters must receive gradients
        pooler_grads = [p.grad is not None for p in model.pooler.parameters()]
        assert all(pooler_grads)

        # ConvNeXt parameters must NOT receive gradients (frozen)
        convnext_grads = [p.grad is not None for p in model.convnext.parameters()]
        assert not any(convnext_grads)

        # Post-ConvNeXt layers must receive gradients
        rwkv_grads = [p.grad is not None for p in model.rwkv_blocks.parameters()]
        assert any(rwkv_grads)
        head_grads = [p.grad is not None for p in model.gamepad_head.parameters()]
        assert all(head_grads)

    def test_gradient_flow_unfrozen_convnext(
        self, small_model_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(small_model_kwargs)
        kwargs["freeze_convnext"] = False
        model = ConvNeXtRWKV7Gamepad(**kwargs)

        x = torch.randn(2, 3, 48, 48, requires_grad=True)
        full_gamepad, _, _ = model(x)
        loss = full_gamepad.sum()
        loss.backward()

        # ConvNeXt parameters must now receive gradients
        convnext_grads = [p.grad is not None for p in model.convnext.parameters()]
        assert any(convnext_grads)

    def test_online_streaming_step(self, small_model_kwargs: dict[str, Any]) -> None:
        model = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        batch_size = 2
        device = torch.device("cpu")
        dtype = torch.float32

        state: GamepadStreamingState | None = model.init_streaming_state(batch_size, device, dtype)
        assert state is not None
        assert state.conv_state.shape == (batch_size, cast(int, small_model_kwargs["rwkv_dim"]), 2)
        assert len(state.rwkv_states) == cast(int, small_model_kwargs["rwkv_layers"])

        # Stream 3 consecutive frames
        for _ in range(3):
            frame = torch.randn(batch_size, 3, 32, 32)
            (full_gp, btns, joys), state = model.step(frame, state)
            assert full_gp.shape == (batch_size, 21)
            assert btns.shape == (batch_size, 17)
            assert joys.shape == (batch_size, 4)
            assert (joys >= -1.0).all() and (joys <= 1.0).all()

    def test_online_streaming_chunk_step(self, small_model_kwargs: dict[str, Any]) -> None:
        """Verify streaming step works on multi-frame N-frame chunks."""
        model = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        batch_size = 2
        chunk_len = 3

        state = model.init_streaming_state(batch_size, torch.device("cpu"), torch.float32)

        # Stream two consecutive 3-frame chunks
        chunk_1 = torch.randn(batch_size, chunk_len, 3, 32, 32)
        (full_gp_1, btns_1, joys_1), state = model.step(chunk_1, state)
        assert full_gp_1.shape == (batch_size, chunk_len, 21)
        assert btns_1.shape == (batch_size, chunk_len, 17)
        assert joys_1.shape == (batch_size, chunk_len, 4)

        chunk_2 = torch.randn(batch_size, chunk_len, 3, 32, 32)
        (full_gp_2, btns_2, joys_2), state = model.step(chunk_2, state)
        assert full_gp_2.shape == (batch_size, chunk_len, 21)
        assert btns_2.shape == (batch_size, chunk_len, 17)
        assert joys_2.shape == (batch_size, chunk_len, 4)

    def test_streaming_temporal_dependency(self, small_model_kwargs: dict[str, Any]) -> None:
        """Verify that past frames affect interpretation of subsequent frames via recurrent state."""
        model = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        model.eval()

        frame_0_a = torch.randn(1, 3, 32, 32)
        frame_0_b = torch.randn(1, 3, 32, 32) + 5.0
        frame_1 = torch.randn(1, 3, 32, 32)

        # Stream history A -> frame 1
        _, state_a = model.step(frame_0_a)
        (pred_1_a, _, _), _ = model.step(frame_1, state_a)

        # Stream history B -> frame 1
        _, state_b = model.step(frame_0_b)
        (pred_1_b, _, _), _ = model.step(frame_1, state_b)

        # Predictions on identical frame_1 must differ due to distinct historical context
        assert not torch.allclose(pred_1_a, pred_1_b, atol=1e-4)


class TestConvNeXtRWKV7GamepadLitModule:
    """Tests for ConvNeXtRWKV7GamepadLitModule."""

    def test_training_step_unified_target(
        self, small_model_kwargs: dict[str, Any]
    ) -> None:
        net = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        lit_module = ConvNeXtRWKV7GamepadLitModule(net=net)

        x = torch.randn(2, 3, 32, 32)
        y_btns = (torch.rand(2, 17) > 0.5).long()
        y_joys = torch.rand(2, 4) * 2 - 1
        y_unified = torch.cat([y_btns.float(), y_joys], dim=-1)

        loss = lit_module.training_step((x, y_unified), 0)
        assert loss is not None
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_training_step_5d_sequence(
        self, small_model_kwargs: dict[str, Any]
    ) -> None:
        """Verify training step computes valid combined loss on 5D temporal sequences."""
        net = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        lit_module = ConvNeXtRWKV7GamepadLitModule(net=net)

        b, t = 2, 4
        x_5d = torch.randn(b, t, 3, 32, 32)
        y_btns = (torch.rand(b, t, 17) > 0.5).long()
        y_joys = torch.rand(b, t, 4) * 2 - 1
        y_unified = torch.cat([y_btns.float(), y_joys], dim=-1)

        loss = lit_module.training_step((x_5d, y_unified), 0)
        assert loss is not None
        assert not torch.isnan(loss)
        assert loss.item() > 0
        assert lit_module.train_btn_acc.compute().item() >= 0.0

    def test_training_step_split_target(
        self, small_model_kwargs: dict[str, Any]
    ) -> None:
        net = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        lit_module = ConvNeXtRWKV7GamepadLitModule(net=net)

        x = torch.randn(2, 3, 32, 32)
        y_btns = (torch.rand(2, 17) > 0.5).long()
        y_joys = torch.rand(2, 4) * 2 - 1

        loss = lit_module.training_step((x, y_btns, y_joys), 0)
        assert loss is not None
        assert not torch.isnan(loss)

    def test_validation_and_test_step(
        self, small_model_kwargs: dict[str, Any]
    ) -> None:
        net = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        lit_module = ConvNeXtRWKV7GamepadLitModule(net=net)

        x = torch.randn(2, 3, 32, 32)
        y = torch.cat([(torch.rand(2, 17) > 0.5).float(), torch.rand(2, 4) * 2 - 1], dim=-1)

        lit_module.on_train_start()
        lit_module.validation_step((x, y), 0)
        lit_module.on_validation_epoch_end()
        lit_module.test_step((x, y), 0)

        assert lit_module.val_loss.compute() > 0
        assert lit_module.test_loss.compute() > 0

    def test_predict_step(self, small_model_kwargs: dict[str, Any]) -> None:
        net = ConvNeXtRWKV7Gamepad(**small_model_kwargs)
        lit_module = ConvNeXtRWKV7GamepadLitModule(net=net)

        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        imgs = torch.randn(4, 3, 32, 32)
        loader = DataLoader(TensorDataset(imgs), batch_size=2)
        preds = trainer.predict(lit_module, dataloaders=loader)

        assert preds is not None
        stacked_preds = torch.cat(cast(list[torch.Tensor], preds))
        assert stacked_preds.shape == (4, 21)

    def test_differential_learning_rates(
        self, small_model_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(small_model_kwargs)
        kwargs["freeze_convnext"] = False
        net = ConvNeXtRWKV7Gamepad(**kwargs)

        opt_partial = torch.optim.AdamW
        lit_module = ConvNeXtRWKV7GamepadLitModule(
            net=net,
            optimizer=opt_partial,  # type: ignore[arg-type]
            convnext_lr=1e-5,
        )
        opt_dict = lit_module.configure_optimizers()
        optimizer = opt_dict["optimizer"]

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[1]["lr"] == 1e-5


class TestHydraInstantiation:
    """Tests Hydra YAML configuration instantiation for new gamepad models."""

    @pytest.mark.parametrize(
        "model_config",
        [
            "convnext_rwkv7_gamepad",
            "convnext_rwkv7_gamepad_unfrozen",
            "convnext_rwkv7_gamepad_bypass_stem",
        ],
    )
    def test_instantiate_gamepad_configs(self, model_config: str) -> None:
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=[
                    f"model={model_config}",
                    "model.net.pretrained_dinov3=false",
                ],
            )

        HydraConfig().set_config(cfg)
        model = hydra.utils.instantiate(cfg.model)
        assert isinstance(model, ConvNeXtRWKV7GamepadLitModule)
        assert isinstance(model.net, ConvNeXtRWKV7Gamepad)
