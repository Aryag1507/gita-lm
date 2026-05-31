import random
import sqlite3
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from config import DataConfig


@dataclass
class CommentaryRecord:
    chapter: int
    verse_label: str
    sanskrit: str
    translation: str
    commentary: str


def load_records(config: DataConfig) -> list[CommentaryRecord]:
    conn = sqlite3.connect(config.db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT v.chapter, v.verse_label, v.sanskrit, v.translation, c.text
        FROM commentaries c
        JOIN verses v ON c.verse_id = v.id
        WHERE c.acharya = ?
        ORDER BY v.chapter, v.verse
        """,
        (config.acharya,),
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    for chapter, verse_label, sanskrit, translation, commentary in rows:
        if commentary and commentary.strip():
            records.append(
                CommentaryRecord(
                    chapter=chapter,
                    verse_label=verse_label,
                    sanskrit=(sanskrit or "").strip(),
                    translation=(translation or "").strip(),
                    commentary=commentary.strip(),
                )
            )
    return records


def format_prompt(record: CommentaryRecord) -> str:
    return (
        f"### Verse BG {record.chapter}.{record.verse_label}\n"
        f"Sanskrit: {record.sanskrit}\n"
        f"Translation: {record.translation}\n"
        f"### Commentary:\n"
    )


def format_sample(record: CommentaryRecord) -> str:
    return format_prompt(record) + record.commentary + "<|endoftext|>"


def split_records(
    records: list[CommentaryRecord], config: DataConfig
) -> tuple[list, list, list]:
    random.seed(config.seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * config.train_split)
    n_val = int(n * config.val_split)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


class CommentaryDataset(Dataset):
    def __init__(
        self,
        records: list[CommentaryRecord],
        tokenizer: GPT2Tokenizer,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._tokenize(records)

    def _tokenize(self, records: list[CommentaryRecord]) -> list[dict]:
        samples = []
        for record in records:
            text = format_sample(record)
            prompt = format_prompt(record)

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()

            # Build labels: mask prompt tokens with -100 so loss is only
            # computed on the commentary completion, not the input prompt
            prompt_len = len(self.tokenizer(prompt)["input_ids"])
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {k: v for k, v in self.samples[idx].items()}


def build_datasets(
    config: DataConfig,
    tokenizer: GPT2Tokenizer,
) -> tuple[CommentaryDataset, CommentaryDataset, CommentaryDataset]:
    records = load_records(config)
    print(f"Loaded {len(records)} commentary records for acharya: {config.acharya}")

    train_records, val_records, test_records = split_records(records, config)
    print(f"Split → train: {len(train_records)}, val: {len(val_records)}, test: {len(test_records)}")

    train_ds = CommentaryDataset(train_records, tokenizer, config.max_length)
    val_ds = CommentaryDataset(val_records, tokenizer, config.max_length)
    test_ds = CommentaryDataset(test_records, tokenizer, config.max_length)
    return train_ds, val_ds, test_ds
