"""
Build preference pairs for DPO.

Design: this is RLAIF-style (Reinforcement Learning from AI Feedback) preference
construction. Rather than collecting human labels, we generate several candidate
commentaries per verse from the policy model, score each with the reward model
(or the heuristic scorer it was trained on), and form a preference pair:

    chosen   = highest-scoring candidate
    rejected = lowest-scoring candidate

This reuses the reward model from the RLHF arc and produces on-policy negatives,
which is what makes DPO effective (the rejected samples come from the model's own
distribution, so the gradient is informative).

The candidate generator is injected so this module is testable without loading
GPT-2.
"""
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import GPT2Tokenizer

from src.dpo.dpo import DPOBatch


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float


def build_preference_pairs(
    prompts: list[str],
    generate_fn: Callable[[str], list[str]],
    score_fn: Callable[[str], float],
    min_margin: float = 0.0,
) -> list[PreferencePair]:
    """
    For each prompt, generate candidates, score them, and pair best vs worst.

    generate_fn(prompt) -> list[str]   candidate completions
    score_fn(text)      -> float       quality score (higher is better)
    min_margin: skip pairs whose score gap is below this (avoids near-ties that
                give a noisy training signal)
    """
    pairs: list[PreferencePair] = []
    for prompt in prompts:
        candidates = generate_fn(prompt)
        if len(candidates) < 2:
            continue
        scored = sorted(candidates, key=score_fn, reverse=True)
        chosen, rejected = scored[0], scored[-1]
        c_score, r_score = score_fn(chosen), score_fn(rejected)
        if c_score - r_score < min_margin:
            continue
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_score=c_score,
                rejected_score=r_score,
            )
        )
    return pairs


def _tokenize_pair_side(
    tokenizer: GPT2Tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize prompt+completion, returning input_ids, attention_mask, and labels
    with the prompt tokens masked (-100) so DPO log-probs cover only completion.
    """
    full = prompt + completion + tokenizer.eos_token
    enc = tokenizer(
        full,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)

    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def collate_preference_batch(
    pairs: list[PreferencePair],
    tokenizer: GPT2Tokenizer,
    max_length: int = 512,
) -> DPOBatch:
    """Tokenize a list of preference pairs into a single batched DPOBatch."""
    c_ids, c_mask, c_labels = [], [], []
    r_ids, r_mask, r_labels = [], [], []

    for pair in pairs:
        ci, cm, cl = _tokenize_pair_side(tokenizer, pair.prompt, pair.chosen, max_length)
        ri, rm, rl = _tokenize_pair_side(tokenizer, pair.prompt, pair.rejected, max_length)
        c_ids.append(ci); c_mask.append(cm); c_labels.append(cl)
        r_ids.append(ri); r_mask.append(rm); r_labels.append(rl)

    return DPOBatch(
        chosen_input_ids=torch.stack(c_ids),
        chosen_labels=torch.stack(c_labels),
        chosen_attention_mask=torch.stack(c_mask),
        rejected_input_ids=torch.stack(r_ids),
        rejected_labels=torch.stack(r_labels),
        rejected_attention_mask=torch.stack(r_mask),
    )
