"""
(IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations.
Implemented from scratch for GPT-2. (Liu et al., 2022)

Idea: instead of adding low-rank matrices (LoRA), (IA)³ learns element-wise
rescaling vectors that multiply intermediate activations. For each adapted
projection it learns a vector ℓ initialized to ones, and computes:

    h = (W x) ⊙ ℓ

Because ℓ starts at 1, the model is unchanged at init. Only ℓ is trained, so the
parameter count is tiny — one scalar per output feature, vs LoRA's rank·(d_in+d_out).

For GPT-2 we rescale three points per block:
  - the key projection output      (inhibit/amplify attention keys)
  - the value projection output    (inhibit/amplify attention values)
  - the FFN intermediate activation (inhibit/amplify the MLP hidden units)

GPT-2 packs Q,K,V into a single Conv1D (`c_attn`) of width 3*d_model, so the K
and V scaling vectors act on the corresponding slices of its output.
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D


class IA3Attention(nn.Module):
    """
    Wraps GPT-2's c_attn Conv1D and rescales the K and V slices of its output.
    Output layout of c_attn is [Q | K | V], each of width d_model.
    """
    def __init__(self, c_attn: Conv1D, d_model: int):
        super().__init__()
        self.c_attn = c_attn
        self.d_model = d_model
        # One scaling factor per feature for K and for V (init to 1 -> identity)
        self.l_k = nn.Parameter(torch.ones(d_model))
        self.l_v = nn.Parameter(torch.ones(d_model))

        for p in self.c_attn.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.c_attn(x)  # (..., 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)
        k = k * self.l_k
        v = v * self.l_v
        return torch.cat([q, k, v], dim=-1)


class IA3FFN(nn.Module):
    """Wraps GPT-2's MLP c_fc Conv1D and rescales its intermediate activation."""
    def __init__(self, c_fc: Conv1D, d_ff: int):
        super().__init__()
        self.c_fc = c_fc
        self.l_ff = nn.Parameter(torch.ones(d_ff))
        for p in self.c_fc.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_fc(x) * self.l_ff


def inject_ia3(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    """Replace c_attn and mlp.c_fc in every block with (IA)³-wrapped versions."""
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.config.n_embd
    d_ff = 4 * d_model  # GPT-2 MLP expands 4x
    replaced = 0

    for block in model.transformer.h:
        block.attn.c_attn = IA3Attention(block.attn.c_attn, d_model)
        block.mlp.c_fc = IA3FFN(block.mlp.c_fc, d_ff)
        replaced += 2

    print(f"Injected (IA)³ into {replaced} projections across {len(model.transformer.h)} blocks")
    return model


def ia3_state_dict(model: nn.Module) -> dict:
    """Extract only the (IA)³ scaling vectors for saving."""
    return {
        name: param
        for name, param in model.state_dict().items()
        if name.endswith((".l_k", ".l_v", ".l_ff"))
    }
