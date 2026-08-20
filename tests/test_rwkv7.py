"""Behavioral tests for RWKV-7 block/model."""

from __future__ import annotations

import math

import pytest
import torch

from src.models.components.rwkv7 import RWKV7Block, RWKV7BlockState, RWKV7Model


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


class TestRWKV7Block:
    def test_output_shape(self, device: torch.device, dtype: torch.dtype) -> None:
        block = RWKV7Block(dim=64, head_size=16, layer_id=0, total_layers=2).to(
            device=device, dtype=dtype
        )
        x = torch.randn(2, 8, 64, device=device, dtype=dtype)
        state = None
        out, new_state = block(x, state)
        assert out.shape == (2, 8, 64)
        assert new_state.att_x_prev.shape == (2, 64)
        assert new_state.att_state.shape == (2, 4, 16, 16)
        assert new_state.ffn_x_prev.shape == (2, 64)

    def test_recurrent_consistency(self, device: torch.device, dtype: torch.dtype) -> None:
        torch.manual_seed(0)
        block = RWKV7Block(dim=64, head_size=16, layer_id=0, total_layers=2).to(
            device=device, dtype=dtype
        )
        block.eval()
        x = torch.randn(2, 8, 64, device=device, dtype=dtype)

        out_full, state_full = block(x, None)

        state = None
        outs = []
        for t in range(x.shape[1]):
            out_t, state = block(x[:, t : t + 1], state)
            outs.append(out_t)
        out_recurrent = torch.cat(outs, dim=1)

        if dtype == torch.bfloat16:
            out_full = out_full.float()
            out_recurrent = out_recurrent.float()
        diff = (out_full - out_recurrent).abs().max().item()
        assert diff < 1e-2, f"Recurrent consistency failed: max diff {diff}"

    def test_state_persistence(self, device: torch.device, dtype: torch.dtype) -> None:
        block = RWKV7Block(dim=64, head_size=16, layer_id=0, total_layers=2).to(
            device=device, dtype=dtype
        )
        block.eval()
        x1 = torch.randn(2, 4, 64, device=device, dtype=dtype)
        x2 = torch.randn(2, 4, 64, device=device, dtype=dtype)

        _, state = block(x1, None)
        out2, state2 = block(x2, state)

        assert out2.shape == (2, 4, 64)
        assert state2.att_x_prev.shape == (2, 64)

    def test_gradient_flow(self, device: torch.device, dtype: torch.dtype) -> None:
        block = RWKV7Block(dim=64, head_size=16, layer_id=0, total_layers=2).to(
            device=device, dtype=dtype
        )
        block.train()
        x = torch.randn(2, 4, 64, device=device, dtype=dtype, requires_grad=True)
        out, _ = block(x, None)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not math.isnan(x.grad.abs().sum().item())

    def test_device_agnostic_cpu(self) -> None:
        block = RWKV7Block(dim=32, head_size=8, layer_id=0, total_layers=2).cpu()
        x = torch.randn(1, 4, 32)
        block.eval()
        out, state = block(x, None)
        assert out.device == x.device
        assert state.att_x_prev.device == x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_device_agnostic_cuda(self) -> None:
        block = RWKV7Block(dim=32, head_size=8, layer_id=0, total_layers=2).cuda()
        x = torch.randn(1, 4, 32, device="cuda")
        block.eval()
        out, state = block(x, None)
        assert out.device == x.device
        assert state.att_x_prev.device == x.device

    def test_v_first_residual_connection(self, device: torch.device, dtype: torch.dtype) -> None:
        torch.manual_seed(42)
        block0 = RWKV7Block(dim=64, head_size=16, layer_id=0, total_layers=2).to(
            device=device, dtype=dtype
        )
        torch.manual_seed(7)
        block1 = RWKV7Block(dim=64, head_size=16, layer_id=1, total_layers=2).to(
            device=device, dtype=dtype
        )
        block0.eval()
        block1.eval()
        x = torch.randn(2, 4, 64, device=device, dtype=dtype)

        _, _, v_first = block0._forward_impl(
            x, block0.initial_state(2, device, dtype), v_first=None
        )
        assert v_first is not None
        assert v_first.shape == x.shape

        _, _, returned_v_first = block1._forward_impl(
            x, block1.initial_state(2, device, dtype), v_first=v_first
        )
        assert torch.allclose(returned_v_first, v_first)


class TestRWKV7Model:
    def test_forward_shape(self, device: torch.device, dtype: torch.dtype) -> None:
        model = RWKV7Model(dim=64, head_size=16, n_layer=2, vocab_size=32).to(
            device=device, dtype=dtype
        )
        tokens = torch.randint(0, 32, (2, 8), device=device)
        logits, state = model(tokens)
        assert logits.shape == (2, 8, 32)
        assert len(state) == 2

    def test_recurrent_model_consistency(self, device: torch.device, dtype: torch.dtype) -> None:
        torch.manual_seed(0)
        model = RWKV7Model(dim=64, head_size=16, n_layer=2, vocab_size=32).to(
            device=device, dtype=dtype
        )
        model.eval()
        tokens = torch.randint(0, 32, (2, 8), device=device)

        logits_full, state_full = model(tokens)

        state = None
        logits_recurrent = []
        for t in range(tokens.shape[1]):
            logit_t, state = model(tokens[:, t : t + 1], state)
            logits_recurrent.append(logit_t)
        logits_recurrent = torch.cat(logits_recurrent, dim=1)

        if dtype == torch.bfloat16:
            logits_full = logits_full.float()
            logits_recurrent = logits_recurrent.float()
        diff = (logits_full - logits_recurrent).abs().max().item()
        assert diff < 1e-2, f"Model recurrent consistency failed: max diff {diff}"

    def test_state_reuse_across_calls(self, device: torch.device, dtype: torch.dtype) -> None:
        model = RWKV7Model(dim=64, head_size=16, n_layer=2, vocab_size=32).to(
            device=device, dtype=dtype
        )
        model.eval()
        tokens1 = torch.randint(0, 32, (1, 4), device=device)
        tokens2 = torch.randint(0, 32, (1, 4), device=device)

        _, state = model(tokens1)
        logits2, state2 = model(tokens2, state)

        assert logits2.shape == (1, 4, 32)
        assert len(state2) == 2

    def test_gradient_flow(self, device: torch.device, dtype: torch.dtype) -> None:
        model = RWKV7Model(dim=64, head_size=16, n_layer=2, vocab_size=32).to(
            device=device, dtype=dtype
        )
        model.train()
        tokens = torch.randint(0, 32, (2, 4), device=device)
        logits, _ = model(tokens)
        loss = logits.sum()
        loss.backward()
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm().item() ** 2
        assert math.sqrt(total_norm) > 0

    def test_cpu_and_cuda_same_result(self) -> None:
        torch.manual_seed(0)
        model_cpu = RWKV7Model(dim=32, head_size=8, n_layer=2, vocab_size=16).cpu().float()
        model_cpu.eval()
        tokens = torch.randint(0, 16, (1, 4))

        logits_cpu, _ = model_cpu(tokens)

        if torch.cuda.is_available():
            model_cuda = RWKV7Model(dim=32, head_size=8, n_layer=2, vocab_size=16).cuda().float()
            model_cuda.eval()
            model_cuda.load_state_dict(model_cpu.state_dict())
            tokens_cuda = tokens.cuda()
            logits_cuda, _ = model_cuda(tokens_cuda)
            diff = (logits_cpu - logits_cuda.cpu()).abs().max().item()
            assert diff < 1e-5, f"CPU/CUDA mismatch: {diff}"
