"""
Unit tests for the from-scratch LoRA implementation.

These tests use a small standalone Conv1D layer (no GPT-2 download) so they
run fast and offline. They verify the core mathematical guarantees of LoRA.
"""
import math

import pytest
import torch
from transformers.pytorch_utils import Conv1D

from src.lora.lora import LoRAConv1D, get_trainable_params


@pytest.fixture
def conv1d_layer():
    # Conv1D(nf=out_features, nx=in_features); weight shape is (nx, nf) = (d_in, d_out)
    torch.manual_seed(0)
    return Conv1D(16, 8)  # d_in=8, d_out=16


def test_delta_is_zero_at_init(conv1d_layer):
    """
    Because B is initialized to zero, the LoRA delta (B @ A) must be exactly
    zero at initialization — the adapted layer should match the original
    layer's output before any training.
    """
    lora = LoRAConv1D(conv1d_layer, rank=4, alpha=8, dropout=0.0)
    lora.eval()  # disable dropout

    x = torch.randn(2, 5, 8)  # (batch, seq, d_in)
    base_out = conv1d_layer(x)
    lora_out = lora(x)

    assert torch.allclose(base_out, lora_out, atol=1e-6), \
        "LoRA output must equal base output when B=0 at init"


def test_b_is_zero_a_is_not(conv1d_layer):
    lora = LoRAConv1D(conv1d_layer, rank=4, alpha=8, dropout=0.0)
    assert torch.count_nonzero(lora.lora_B) == 0, "B must be initialized to zero"
    assert torch.count_nonzero(lora.lora_A) > 0, "A must be non-zero (Kaiming init)"


def test_delta_nonzero_after_perturbing_b(conv1d_layer):
    """Once B is non-zero, the adapter must change the output."""
    lora = LoRAConv1D(conv1d_layer, rank=4, alpha=8, dropout=0.0)
    lora.eval()
    with torch.no_grad():
        lora.lora_B.add_(torch.randn_like(lora.lora_B))

    x = torch.randn(2, 5, 8)
    base_out = conv1d_layer(x)
    lora_out = lora(x)
    assert not torch.allclose(base_out, lora_out, atol=1e-4)


def test_only_lora_params_trainable(conv1d_layer):
    """The frozen original weight/bias must not require grad; only A and B do."""
    lora = LoRAConv1D(conv1d_layer, rank=4, alpha=8, dropout=0.0)

    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad
    assert not lora.original.weight.requires_grad
    if lora.original.bias is not None:
        assert not lora.original.bias.requires_grad


def test_trainable_param_count(conv1d_layer):
    """
    Trainable params must equal rank*(d_in + d_out) — the size of A plus B.
    For rank=4, d_in=8, d_out=16: 4*(8+16) = 96.
    """
    rank = 4
    lora = LoRAConv1D(conv1d_layer, rank=rank, alpha=8, dropout=0.0)
    trainable, total = get_trainable_params(lora)

    expected = rank * (8 + 16)
    assert trainable == expected, f"expected {expected} trainable params, got {trainable}"


def test_scaling_factor(conv1d_layer):
    """scaling must equal alpha / rank."""
    lora = LoRAConv1D(conv1d_layer, rank=4, alpha=8, dropout=0.0)
    assert lora.scaling == pytest.approx(8 / 4)


def test_output_shape_preserved(conv1d_layer):
    """The adapter must not change the output shape vs the original layer."""
    lora = LoRAConv1D(conv1d_layer, rank=4, alpha=8, dropout=0.0)
    x = torch.randn(3, 7, 8)
    out = lora(x)
    assert out.shape == (3, 7, 16)


def test_gradients_flow_to_lora_only(conv1d_layer):
    """After a backward pass, A and B get grads; the frozen weight does not."""
    lora = LoRAConv1D(conv1d_layer, rank=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        lora.lora_B.add_(0.1)  # make delta non-zero so grad flows

    x = torch.randn(2, 5, 8)
    out = lora(x)
    out.sum().backward()

    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.original.weight.grad is None
