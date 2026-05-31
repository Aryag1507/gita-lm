"""
LoRA (Low-Rank Adaptation) implemented from scratch in PyTorch.

For a pretrained weight matrix W ∈ R^(d_out × d_in), instead of updating W
directly we inject two trainable matrices:
    A ∈ R^(rank × d_in)   — initialized with Kaiming uniform
    B ∈ R^(d_out × rank)  — initialized to zero (so ΔW = 0 at step 0)

The adapted forward pass becomes:
    output = x @ W.T + x @ A.T @ B.T * (alpha / rank)

W is frozen throughout training. Only A and B are updated.
This reduces trainable parameters from d_in*d_out to rank*(d_in + d_out),
which for GPT-2's attention projections is ~99% fewer parameters.
"""

import math

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from config import LoRAConfig


class LoRAConv1D(nn.Module):
    """
    LoRA adapter for HuggingFace's Conv1D layer used in GPT-2.

    Conv1D stores weights as (d_in, d_out) — the transpose of nn.Linear.
    Its forward pass is: output = x @ weight + bias
    So our LoRA delta is: x @ (A.T @ B.T).T = x @ (B @ A)
    where A ∈ R^(rank × d_in), B ∈ R^(d_out × rank)
    """

    def __init__(
        self,
        original: Conv1D,
        rank: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        # Conv1D weight shape is (d_in, d_out)
        d_in, d_out = original.weight.shape

        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout)

        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        # Conv1D: output = x @ W where W is (d_in, d_out)
        # LoRA delta: x @ (B @ A).T * scaling  →  (x @ A.T) @ B.T * scaling
        lora_delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora_delta

    def extra_repr(self) -> str:
        d_in, d_out = self.original.weight.shape
        return (
            f"d_in={d_in}, d_out={d_out}, rank={self.rank}, "
            f"scaling={self.scaling:.3f}, "
            f"params_original={d_in * d_out:,}, "
            f"params_lora={self.rank * (d_in + d_out):,}"
        )


def inject_lora(model: GPT2LMHeadModel, config: LoRAConfig) -> GPT2LMHeadModel:
    # Freeze entire model first; LoRAConv1D will unfreeze only A and B
    for param in model.parameters():
        param.requires_grad_(False)

    """
    Walk the model and replace target Conv1D layers with LoRAConv1D.
    For GPT-2, target_modules ["c_attn", "c_proj"] are the combined
    QKV projection and output projection in each attention block.
    """
    replaced = 0
    for name, module in model.named_modules():
        for target in config.target_modules:
            if name.endswith(target) and isinstance(module, Conv1D):
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                lora_layer = LoRAConv1D(
                    original=module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(parent, child_name, lora_layer)
                replaced += 1

    print(f"Injected LoRA into {replaced} layers")
    return model


def get_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_param_summary(model: nn.Module) -> None:
    trainable, total = get_trainable_params(model)
    print(f"Trainable parameters : {trainable:>12,}  ({100 * trainable / total:.2f}%)")
    print(f"Frozen parameters    : {total - trainable:>12,}")
    print(f"Total parameters     : {total:>12,}")


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA weights — not the full model."""
    lora_state = {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save(lora_state, path)
    print(f"Saved {len(lora_state)} LoRA weight tensors to {path}")


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA weights back into an already-injected model."""
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    print(f"Loaded LoRA weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
    return model
