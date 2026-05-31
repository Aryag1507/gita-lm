"""
Post-training quantization for GPT-2 — from scratch. (QLoRA-style intuition)

Goal: shrink the memory footprint of the frozen base weights by storing them in
low precision, then dequantize on the fly during the forward pass. This is the
core trick behind running large models on small hardware: keep weights in int8
(or 4-bit) instead of fp32, cutting weight memory ~4x (int8) or ~8x (4-bit).

We implement *absmax* quantization, the scheme used by LLM.int8() and QLoRA:

  - For a weight tensor W, pick a quantization granularity (per-tensor or, better,
    per-output-channel). For each group compute the scale:
        scale = absmax(W_group) / q_max
    where q_max = 127 for int8, 7 for symmetric 4-bit.
  - Quantize:   q = round(W / scale)   (clamped to [-q_max, q_max])
  - Dequantize: W_hat = q * scale

Per-channel scales matter: GPT-2's Conv1D weight has shape (d_in, d_out); we keep
one scale per output column so a single outlier column doesn't blow up the scale
for the whole tensor (the classic LLM.int8() outlier problem).

The quantized linear stores int8 weights + fp32 scales and reconstructs W_hat at
call time. This trades a little compute for a big memory saving and lets us
measure the quality (perplexity) vs memory tradeoff empirically.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D


@dataclass
class QuantStats:
    bits: int
    per_channel: bool
    num_groups: int


def quantize_absmax(
    weight: torch.Tensor, bits: int = 8, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric absmax quantization.

    weight: (d_in, d_out) for GPT-2 Conv1D (per-channel = per output column).
    Returns (q_int, scale) where q_int is the integer codes (stored as int8 when
    bits<=8) and scale broadcasts against weight to dequantize: W_hat = q * scale.
    """
    q_max = 2 ** (bits - 1) - 1  # 127 for int8, 7 for 4-bit symmetric

    if per_channel:
        # one scale per output column (dim 1); keepdim for broadcasting
        absmax = weight.abs().amax(dim=0, keepdim=True)
    else:
        absmax = weight.abs().amax()

    scale = absmax / q_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # avoid div by 0

    q = torch.round(weight / scale).clamp(-q_max, q_max)
    # int8 storage holds both 8-bit and 4-bit codes (4-bit just uses a smaller range)
    q = q.to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct the float weight: W_hat = q * scale."""
    return q.to(torch.float32) * scale


class QuantizedConv1D(nn.Module):
    """
    Drop-in replacement for GPT-2's Conv1D that stores weights quantized.

    Conv1D computes:  y = x @ W + b,  W of shape (d_in, d_out).
    We keep `weight_q` (int8) + `scale` (fp32) as buffers and dequantize before
    the matmul. The bias is kept in fp32 (it's tiny). The dequantized weight is
    never stored, so the resident memory is the int8 codes plus per-channel scales.
    """
    def __init__(self, conv: Conv1D, bits: int = 8, per_channel: bool = True):
        super().__init__()
        self.nf = conv.nf  # number of output features
        self.bits = bits
        self.per_channel = per_channel

        weight = conv.weight.data  # (d_in, d_out)
        q, scale = quantize_absmax(weight, bits=bits, per_channel=per_channel)
        self.register_buffer("weight_q", q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", conv.bias.data.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_hat = dequantize(self.weight_q, self.scale)
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), w_hat)
        return x.view(size_out)


def _module_memory_bytes(module: nn.Module) -> int:
    """Resident bytes of a module's parameters + buffers (by dtype size)."""
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total


def quantize_model(
    model: nn.Module, bits: int = 8, per_channel: bool = True, skip_lm_head: bool = True
) -> tuple[nn.Module, dict]:
    """
    Replace every Conv1D in GPT-2 with a QuantizedConv1D and report the savings.

    Returns (model, report) where report has fp32/quantized byte counts and the
    compression ratio over the quantized Conv1D weights.
    """
    before = 0
    after = 0
    replaced = 0

    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                before += _module_memory_bytes(child)
                q = QuantizedConv1D(child, bits=bits, per_channel=per_channel)
                after += _module_memory_bytes(q)
                setattr(parent, name, q)
                replaced += 1

    report = {
        "bits": bits,
        "per_channel": per_channel,
        "modules_quantized": replaced,
        "conv1d_fp32_bytes": before,
        "conv1d_quantized_bytes": after,
        "compression_ratio": (before / after) if after else 0.0,
    }
    print(
        f"Quantized {replaced} Conv1D modules to {bits}-bit "
        f"({'per-channel' if per_channel else 'per-tensor'}): "
        f"{before / 1e6:.1f}MB -> {after / 1e6:.1f}MB "
        f"({report['compression_ratio']:.2f}x)"
    )
    return model, report


def quantization_error(weight: torch.Tensor, bits: int = 8, per_channel: bool = True) -> float:
    """Mean absolute reconstruction error introduced by quantizing `weight`."""
    q, scale = quantize_absmax(weight, bits=bits, per_channel=per_channel)
    w_hat = dequantize(q, scale)
    return (weight - w_hat).abs().mean().item()
