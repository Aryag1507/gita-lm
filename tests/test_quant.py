"""
Unit tests for from-scratch absmax quantization.

Verify the round-trip stays within the theoretical error bound, that per-channel
scaling beats per-tensor on outlier-heavy weights, that 4-bit is lossier than
8-bit, and that the quantized GPT-2 Conv1D produces nearly the same output while
using far less weight memory.
"""
import pytest
import torch
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from src.quant.quantize import (
    QuantizedConv1D,
    dequantize,
    quantization_error,
    quantize_absmax,
    quantize_model,
)


def test_absmax_roundtrip_within_scale():
    """Reconstruction error per element is bounded by half the quant step."""
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    q, scale = quantize_absmax(w, bits=8, per_channel=True)
    w_hat = dequantize(q, scale)
    # max error <= scale/2 per column (rounding to nearest level)
    max_err = (w - w_hat).abs().amax(dim=0, keepdim=True)
    assert torch.all(max_err <= scale.squeeze(0).unsqueeze(0) / 2 + 1e-6)


def test_int8_codes_in_range():
    w = torch.randn(32, 32) * 10
    q, _ = quantize_absmax(w, bits=8)
    assert q.dtype == torch.int8
    assert q.abs().max().item() <= 127


def test_4bit_range():
    w = torch.randn(32, 32)
    q, _ = quantize_absmax(w, bits=4)
    assert q.abs().max().item() <= 7


def test_4bit_lossier_than_8bit():
    torch.manual_seed(1)
    w = torch.randn(128, 128)
    err8 = quantization_error(w, bits=8)
    err4 = quantization_error(w, bits=4)
    assert err4 > err8


def test_per_channel_beats_per_tensor_with_outlier():
    """A single huge column should hurt per-tensor scaling far more."""
    torch.manual_seed(2)
    w = torch.randn(64, 16)
    w[:, 0] *= 100.0  # outlier column
    err_pc = quantization_error(w, bits=8, per_channel=True)
    err_pt = quantization_error(w, bits=8, per_channel=False)
    assert err_pc < err_pt


def test_quantized_conv1d_matches_original():
    """Quantized Conv1D output should be close to the fp32 original."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    conv = model.transformer.h[0].mlp.c_fc
    x = torch.randn(2, 5, model.config.n_embd)
    with torch.no_grad():
        ref = conv(x)
        qconv = QuantizedConv1D(conv, bits=8, per_channel=True)
        out = qconv(x)
    # relative error should be small for 8-bit per-channel
    rel = (ref - out).abs().mean() / ref.abs().mean()
    assert rel < 0.02


def test_quantize_model_reports_compression():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    _, report = quantize_model(model, bits=8, per_channel=True)
    assert report["modules_quantized"] > 0
    # int8 weights are ~4x smaller than fp32 (plus small scale/bias overhead)
    assert report["compression_ratio"] > 3.0
