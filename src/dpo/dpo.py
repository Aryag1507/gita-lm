"""
Direct Preference Optimization (DPO) — implemented from scratch.

DPO aligns a language model to preferences without a separate RL loop. Given
preference pairs (x, y_chosen, y_rejected), it directly optimizes the policy so
that chosen completions become more likely than rejected ones, while staying
close to a frozen reference model.

The DPO loss (Rafailov et al., 2023):

    L = -E[ log σ( β · ( r_θ(x, y_w) - r_θ(x, y_l) ) ) ]

where the implicit reward is the log-ratio against the reference policy:

    r_θ(x, y) = log π_θ(y | x) - log π_ref(y | x)

Intuition: we push up the policy's log-prob on the chosen completion and push
down on the rejected one, but the reference-model term regularizes the update
so the model doesn't drift far from its fine-tuned starting point. β controls
how hard we deviate from the reference.

This module:
  - builds preference pairs from the data + a scorer (chosen = real commentary,
    rejected = a lower-quality generation), see preference_data.py
  - computes per-sequence log-probabilities with prompt masking
  - implements the DPO loss and a training loop
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Sum log-probabilities of the *completion* tokens for each sequence.

    logits: (batch, seq_len, vocab)   — model outputs
    labels: (batch, seq_len)          — target ids, with prompt + padding set to
                                        ignore_index so they don't contribute

    Returns: (batch,) summed log-prob over non-masked tokens.

    Standard causal-LM shift: token at position t is predicted from logits at
    position t-1, so we align logits[:-1] with labels[1:].
    """
    # Shift so that tokens < n predict token n
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels != ignore_index
    # Replace ignore_index with 0 so gather is valid; masked out below anyway.
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask  # zero out prompt/padding positions

    return token_logp.sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the DPO loss for a batch of preference pairs.

    All inputs are (batch,) summed sequence log-probs.

    Returns:
        loss            : scalar mean loss
        chosen_reward   : (batch,) implicit reward β·(logπ_θ - logπ_ref) for chosen
        rejected_reward : (batch,) implicit reward for rejected

    The reward margin (chosen_reward - rejected_reward) is a useful training
    signal: it should grow positive as alignment proceeds.
    """
    # Implicit rewards: log-ratio of policy to reference
    chosen_logratio = policy_chosen_logp - ref_chosen_logp
    rejected_logratio = policy_rejected_logp - ref_rejected_logp

    # DPO objective: -log σ(β · (chosen_logratio - rejected_logratio))
    logits = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * chosen_logratio.detach()
    rejected_reward = beta * rejected_logratio.detach()
    return loss, chosen_reward, rejected_reward


@dataclass
class DPOBatch:
    """A batch of tokenized preference pairs."""
    chosen_input_ids: torch.Tensor
    chosen_labels: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_labels: torch.Tensor
    rejected_attention_mask: torch.Tensor

    def to(self, device: str) -> "DPOBatch":
        return DPOBatch(
            chosen_input_ids=self.chosen_input_ids.to(device),
            chosen_labels=self.chosen_labels.to(device),
            chosen_attention_mask=self.chosen_attention_mask.to(device),
            rejected_input_ids=self.rejected_input_ids.to(device),
            rejected_labels=self.rejected_labels.to(device),
            rejected_attention_mask=self.rejected_attention_mask.to(device),
        )


def compute_batch_logps(
    model: nn.Module,
    batch: DPOBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model on both chosen and rejected, returning their seq log-probs."""
    chosen_out = model(
        input_ids=batch.chosen_input_ids,
        attention_mask=batch.chosen_attention_mask,
    )
    rejected_out = model(
        input_ids=batch.rejected_input_ids,
        attention_mask=batch.rejected_attention_mask,
    )
    chosen_logp = sequence_logprob(chosen_out.logits, batch.chosen_labels)
    rejected_logp = sequence_logprob(rejected_out.logits, batch.rejected_labels)
    return chosen_logp, rejected_logp


def dpo_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    batch: DPOBatch,
    beta: float = 0.1,
) -> dict:
    """
    One DPO forward pass. The reference model runs under no_grad (it's frozen).

    Returns a metrics dict including the loss tensor (with grad) and detached
    diagnostics (reward margin, accuracy).
    """
    policy_chosen_logp, policy_rejected_logp = compute_batch_logps(policy_model, batch)

    with torch.no_grad():
        ref_chosen_logp, ref_rejected_logp = compute_batch_logps(ref_model, batch)

    loss, chosen_reward, rejected_reward = dpo_loss(
        policy_chosen_logp,
        policy_rejected_logp,
        ref_chosen_logp,
        ref_rejected_logp,
        beta=beta,
    )

    reward_margin = (chosen_reward - rejected_reward).mean().item()
    # "accuracy": fraction of pairs where chosen is correctly preferred
    reward_accuracy = (chosen_reward > rejected_reward).float().mean().item()

    return {
        "loss": loss,
        "reward_margin": reward_margin,
        "reward_accuracy": reward_accuracy,
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
