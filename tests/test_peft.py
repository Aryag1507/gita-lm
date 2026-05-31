"""
Unit tests for the PEFT methods: (IA)³ and prefix tuning.

These verify the parameter-efficiency invariants (base model frozen, only the
adapter trains), the identity-at-init property where it applies, and that the
injected modules produce correctly shaped, differentiable outputs. They load the
real GPT-2 small so they are slower than the pure-tensor tests but still offline.
"""
import pytest
import torch
from transformers import GPT2LMHeadModel

from src.peft.ia3 import IA3Attention, IA3FFN, inject_ia3, ia3_state_dict
from src.peft.prefix_tuning import (
    PrefixEncoder,
    inject_prefix_tuning,
    prefix_state_dict,
)


@pytest.fixture(scope="module")
def gpt2():
    return GPT2LMHeadModel.from_pretrained("gpt2")


# --------------------------- (IA)³ ---------------------------

def test_ia3_is_identity_at_init(gpt2):
    """Scaling vectors init to 1 -> wrapped output equals original c_attn output."""
    block = gpt2.transformer.h[0]
    original = block.attn.c_attn
    x = torch.randn(2, 5, gpt2.config.n_embd)
    with torch.no_grad():
        before = original(x)
        wrapped = IA3Attention(original, gpt2.config.n_embd)
        after = wrapped(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_ia3_ffn_is_identity_at_init(gpt2):
    block = gpt2.transformer.h[0]
    original = block.mlp.c_fc
    d_ff = 4 * gpt2.config.n_embd
    x = torch.randn(2, 5, gpt2.config.n_embd)
    with torch.no_grad():
        before = original(x)
        after = IA3FFN(original, d_ff)(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_ia3_scaling_changes_output(gpt2):
    """Perturbing the scaling vector must change the K/V slices."""
    n_embd = gpt2.config.n_embd
    wrapped = IA3Attention(gpt2.transformer.h[0].attn.c_attn, n_embd)
    x = torch.randn(1, 3, n_embd)
    with torch.no_grad():
        base = wrapped(x)
        wrapped.l_k.add_(0.5)
        perturbed = wrapped(x)
    # Q slice (first n_embd) unchanged; K slice changed
    assert torch.allclose(base[..., :n_embd], perturbed[..., :n_embd], atol=1e-6)
    assert not torch.allclose(base[..., n_embd:2 * n_embd], perturbed[..., n_embd:2 * n_embd])


def test_ia3_only_scaling_vectors_train(gpt2):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = inject_ia3(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all(n.endswith((".l_k", ".l_v", ".l_ff")) for n in trainable)


def test_ia3_state_dict_only_vectors(gpt2):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = inject_ia3(model)
    sd = ia3_state_dict(model)
    assert len(sd) == 3 * gpt2.config.n_layer  # l_k, l_v, l_ff per block
    assert all(k.endswith((".l_k", ".l_v", ".l_ff")) for k in sd)


def test_ia3_param_count(gpt2):
    """Each block adds 2*d_model (k,v) + 4*d_model (ff) scaling scalars."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = inject_ia3(model)
    n_embd = gpt2.config.n_embd
    expected = gpt2.config.n_layer * (2 * n_embd + 4 * n_embd)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == expected


# ----------------------- Prefix tuning -----------------------

def test_prefix_encoder_output_shape(gpt2):
    cfg = gpt2.config
    T = 16
    enc = PrefixEncoder(cfg.n_layer, cfg.n_embd, cfg.n_head, num_virtual_tokens=T)
    past = enc(batch_size=3, device=torch.device("cpu"))
    assert len(past) == cfg.n_layer
    key, value = past[0]
    head_dim = cfg.n_embd // cfg.n_head
    assert key.shape == (3, cfg.n_head, T, head_dim)
    assert value.shape == (3, cfg.n_head, T, head_dim)


def test_prefix_only_encoder_trains(gpt2):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    wrapped = inject_prefix_tuning(model, num_virtual_tokens=8)
    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert all(n.startswith("prefix_encoder.") for n in trainable)


def test_prefix_forward_runs_and_is_differentiable(gpt2):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    wrapped = inject_prefix_tuning(model, num_virtual_tokens=8)
    input_ids = torch.randint(0, gpt2.config.vocab_size, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    labels = input_ids.clone()

    out = wrapped(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert out.loss.requires_grad
    out.loss.backward()

    grads = [p.grad for p in wrapped.trainable_parameters() if p.grad is not None]
    assert len(grads) > 0


def test_prefix_state_dict_only_encoder(gpt2):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    wrapped = inject_prefix_tuning(model, num_virtual_tokens=8)
    sd = prefix_state_dict(wrapped)
    assert len(sd) > 0
    assert all(k.startswith("prefix_encoder.") for k in sd)
