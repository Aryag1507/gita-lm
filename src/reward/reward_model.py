"""
Reward Model for ranking generated commentaries.

Architecture: GPT-2 with a scalar regression head on top of the final
hidden state. Trained on pairs of (commentary, score) where scores are
derived from commentary length, theological vocabulary density, and
structural similarity to Prabhupada's writing style.

At inference time: generate N candidate commentaries, score each with the
reward model, return the highest-scoring one. This is a simplified version
of the reranking step in RLHF (Reinforcement Learning from Human Feedback).

We approximate human preference with heuristic scoring since we don't have
human preference labels. The reward model still learns a continuous scoring
function and demonstrates the architecture pattern.
"""

import re
import sqlite3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Tokenizer

from config import DataConfig, RewardConfig

# Theological vocabulary characteristic of Prabhupada's commentary style
PRABHUPADA_VOCAB = {
    "krishna", "supreme", "lord", "devotee", "bhakti", "yoga", "spiritual",
    "material", "consciousness", "transcendental", "vedic", "arjuna", "soul",
    "eternal", "liberation", "devotional", "service", "purport", "conditioned",
    "gita", "vedanta", "brahman", "atma", "karma", "dharma", "maya", "lila",
}


def heuristic_score(text: str) -> float:
    """
    Approximate commentary quality using three signals:
    1. Theological vocabulary density (Prabhupada-specific terms)
    2. Length normalized to expected range (100-800 words)
    3. Structural markers (verse citations, Sanskrit terms)
    """
    words = text.lower().split()
    if not words:
        return 0.0

    # Signal 1: vocab density
    vocab_hits = sum(1 for w in words if re.sub(r"[^a-z]", "", w) in PRABHUPADA_VOCAB)
    vocab_score = min(vocab_hits / max(len(words), 1) * 10, 1.0)

    # Signal 2: length (optimal ~200-600 words)
    length = len(words)
    if length < 50:
        length_score = length / 50
    elif length <= 600:
        length_score = 1.0
    else:
        length_score = max(0.5, 1.0 - (length - 600) / 1000)

    # Signal 3: Sanskrit / citation markers
    has_sanskrit = bool(re.search(r"[ā-ūṛṝḷḹṃḥṅñṭḍṇś]", text))
    has_citation = bool(re.search(r"bg\.?\s*\d+\.\d+|bhagavad.?gita", text.lower()))
    structure_score = (0.5 * has_sanskrit + 0.5 * has_citation)

    return 0.5 * vocab_score + 0.3 * length_score + 0.2 * structure_score


class RewardDataset(Dataset):
    def __init__(self, db_path: str, tokenizer: GPT2Tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load(db_path)

    def _load(self, db_path: str) -> list[dict]:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT text FROM commentaries WHERE acharya = 'PRABHUPADA'")
        rows = cur.fetchall()
        conn.close()

        samples = []
        for (text,) in rows:
            if text and text.strip():
                score = heuristic_score(text)
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                samples.append({
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "score": torch.tensor(score, dtype=torch.float),
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class RewardModel(nn.Module):
    """
    GPT-2 encoder + scalar head.
    Pools the final hidden states (mean pooling over non-padding tokens)
    then projects to a single score via a 2-layer MLP.
    """

    def __init__(self, base_model_name: str = "gpt2"):
        super().__init__()
        self.encoder = GPT2Model.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for gpt2
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )
        # Freeze the encoder — only train the score head
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.score_head(pooled).squeeze(-1)  # (batch,)


def train_reward_model(config: RewardConfig, data_config: DataConfig, device: str) -> RewardModel:
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = RewardDataset(data_config.db_path, tokenizer)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = RewardModel(config.model_name).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )
    loss_fn = nn.MSELoss()

    print(f"Training reward model on {len(dataset)} samples...")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = batch["score"].to(device)

            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.num_epochs} | loss: {total_loss / len(loader):.4f}")

    return model


def rerank_candidates(
    candidates: list[str],
    reward_model: RewardModel,
    tokenizer: GPT2Tokenizer,
    device: str,
) -> tuple[str, list[float]]:
    """Score all candidates and return the best one plus all scores."""
    reward_model.eval()
    scores = []

    with torch.no_grad():
        for text in candidates:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                padding=True,
            )
            score = reward_model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            ).item()
            scores.append(score)

    best_idx = scores.index(max(scores))
    return candidates[best_idx], scores
