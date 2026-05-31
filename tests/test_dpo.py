"""
Unit tests for the DPO implementation.

These use small synthetic tensors so they run fast and offline — verifying the
mathematical properties of the DPO loss and the sequence log-prob computation.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.dpo.dpo import dpo_loss, sequence_logprob
from src.dpo.preference_data import build_preference_pairs


def test_sequence_logprob_masks_ignored_tokens():
    """Tokens labeled -100 must not contribute to the summed log-prob."""
    torch.manual_seed(0)
    batch, seq, vocab = 1, 5, 10
    logits = torch.randn(batch, seq, vocab)

    # All-masked labels -> log-prob must be exactly 0
    labels = torch.full((batch, seq), -100)
    lp = sequence_logprob(logits, labels)
    assert torch.allclose(lp, torch.zeros(batch), atol=1e-7)


def test_sequence_logprob_matches_manual():
    """Cross-check against a manual gather on a single unmasked position."""
    logits = torch.tensor([[[0.0, 1.0, 2.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.0]]])  # (1,3,3)
    # labels shifted: position 1 predicts label at index 2 (=token 0)
    labels = torch.tensor([[-100, -100, 0]])  # only last token counts
    lp = sequence_logprob(logits, labels)

    # Manually: predict labels[2]=0 from logits[1]
    expected = F.log_softmax(logits[0, 1], dim=-1)[0]
    assert torch.allclose(lp[0], expected, atol=1e-6)


def test_dpo_loss_zero_when_policy_equals_reference():
    """
    If policy == reference, both log-ratios are 0, so the DPO logits are 0 and
    loss = -log σ(0) = log 2.
    """
    z = torch.zeros(4)
    loss, chosen_r, rejected_r = dpo_loss(z, z, z, z, beta=0.1)
    assert torch.allclose(loss, torch.tensor(math.log(2)), atol=1e-6)
    assert torch.allclose(chosen_r, torch.zeros(4))
    assert torch.allclose(rejected_r, torch.zeros(4))


def test_dpo_loss_decreases_when_chosen_preferred():
    """
    When the policy raises chosen log-prob above rejected (relative to ref),
    the loss must be lower than the equal-preference baseline (log 2).
    """
    policy_chosen = torch.tensor([2.0, 2.0])
    policy_rejected = torch.tensor([-2.0, -2.0])
    ref = torch.zeros(2)
    loss, _, _ = dpo_loss(policy_chosen, policy_rejected, ref, ref, beta=0.5)
    assert loss.item() < math.log(2)


def test_dpo_loss_higher_when_rejected_preferred():
    """If the policy wrongly prefers rejected, loss must exceed log 2."""
    policy_chosen = torch.tensor([-2.0])
    policy_rejected = torch.tensor([2.0])
    ref = torch.zeros(1)
    loss, _, _ = dpo_loss(policy_chosen, policy_rejected, ref, ref, beta=0.5)
    assert loss.item() > math.log(2)


def test_dpo_loss_is_differentiable():
    policy_chosen = torch.tensor([1.0], requires_grad=True)
    policy_rejected = torch.tensor([0.0], requires_grad=True)
    ref = torch.zeros(1)
    loss, _, _ = dpo_loss(policy_chosen, policy_rejected, ref, ref, beta=0.1)
    loss.backward()
    assert policy_chosen.grad is not None
    # Increasing chosen log-prob should decrease loss -> negative gradient
    assert policy_chosen.grad.item() < 0


def test_reward_margin_sign():
    """Chosen reward should exceed rejected reward when policy prefers chosen."""
    _, chosen_r, rejected_r = dpo_loss(
        torch.tensor([3.0]), torch.tensor([-1.0]),
        torch.zeros(1), torch.zeros(1), beta=0.2,
    )
    assert (chosen_r > rejected_r).all()


def test_build_preference_pairs_picks_best_and_worst():
    prompts = ["verse A"]
    # generator returns 3 fixed candidates; scorer = length proxy
    candidates = {"verse A": ["short", "medium text", "the longest candidate here"]}

    def generate_fn(p):
        return candidates[p]

    def score_fn(text):
        return float(len(text))

    pairs = build_preference_pairs(prompts, generate_fn, score_fn)
    assert len(pairs) == 1
    assert pairs[0].chosen == "the longest candidate here"
    assert pairs[0].rejected == "short"
    assert pairs[0].chosen_score > pairs[0].rejected_score


def test_build_preference_pairs_respects_min_margin():
    prompts = ["v"]

    def generate_fn(p):
        return ["aaaa", "aaab"]  # nearly equal scores

    def score_fn(text):
        return float(sum(c == "a" for c in text))

    # margin between best(4) and worst(3) is 1; require 2 -> no pair kept
    pairs = build_preference_pairs(prompts, generate_fn, score_fn, min_margin=2.0)
    assert pairs == []
