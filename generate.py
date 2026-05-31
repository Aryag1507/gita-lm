"""Entry point: generate a Prabhupada-style commentary for a given verse."""

from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import get_config
from src.data.dataset import CommentaryRecord, format_prompt
from src.evaluation.evaluate import generate_commentary
from src.lora.lora import inject_lora, load_lora_weights
from src.rag.retriever import CommentaryRetriever
from src.reward.reward_model import RewardModel, rerank_candidates

app = typer.Typer()
console = Console()


def load_model(config):
    tokenizer = GPT2Tokenizer.from_pretrained(config.training.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(config.training.model_name)
    model = inject_lora(model, config.lora)

    lora_path = Path(config.training.output_dir) / "lora_best.pt"
    if lora_path.exists():
        load_lora_weights(model, str(lora_path))
    else:
        console.print("[yellow]Warning: no trained weights found, using base GPT-2[/yellow]")

    model = model.to(config.device)
    return model, tokenizer


@app.command()
def main(
    chapter: int = typer.Option(..., "--chapter", "-c", help="Bhagavad Gita chapter number"),
    verse: str = typer.Option(..., "--verse", "-v", help="Verse number or label (e.g. 47 or 13-14)"),
    use_rag: bool = typer.Option(True, help="Augment with retrieved context"),
    use_reward: bool = typer.Option(True, help="Rerank candidates with reward model"),
    num_candidates: int = typer.Option(4, help="Number of candidates to generate before reranking"),
    max_tokens: int = typer.Option(200, help="Max new tokens to generate"),
):
    cfg = get_config()
    model, tokenizer = load_model(cfg)

    # Fetch the actual verse from DB
    import sqlite3
    conn = sqlite3.connect(cfg.data.db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT chapter, verse_label, sanskrit, translation FROM verses WHERE chapter=? AND verse_label=?",
        (chapter, str(verse)),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        console.print(f"[red]Verse BG {chapter}.{verse} not found in database.[/red]")
        raise typer.Exit(1)

    record = CommentaryRecord(
        chapter=row[0],
        verse_label=row[1],
        sanskrit=(row[2] or ""),
        translation=(row[3] or ""),
        commentary="",
    )

    console.print(Panel(
        f"[bold]BG {record.chapter}.{record.verse_label}[/bold]\n\n"
        f"[italic]{record.translation}[/italic]",
        title="Verse",
    ))

    if use_rag:
        console.print("[dim]Retrieving relevant context...[/dim]")
        retriever = CommentaryRetriever(cfg.rag, cfg.data)
        prompt = retriever.build_rag_prompt(
            record.chapter, record.verse_label, record.translation, record.sanskrit
        )
    else:
        prompt = format_prompt(record)

    console.print(f"[dim]Generating {num_candidates} candidates...[/dim]")
    candidates = generate_commentary(
        model, tokenizer, prompt, cfg.device,
        max_new_tokens=max_tokens,
        num_return_sequences=num_candidates if use_reward else 1,
    )

    if use_reward and len(candidates) > 1:
        console.print("[dim]Reranking with reward model...[/dim]")
        reward_model_path = Path(cfg.reward.output_dir) / "reward_model.pt"
        if reward_model_path.exists():
            reward_model = RewardModel(cfg.reward.model_name).to(cfg.device)
            reward_model.load_state_dict(torch.load(str(reward_model_path), map_location=cfg.device))
            best, scores = rerank_candidates(candidates, reward_model, tokenizer, cfg.device)
            console.print(f"[dim]Scores: {[f'{s:.3f}' for s in scores]}[/dim]")
        else:
            console.print("[yellow]No reward model found, using first candidate.[/yellow]")
            best = candidates[0]
    else:
        best = candidates[0]

    console.print(Panel(best, title="Generated Commentary (Prabhupada style)"))


if __name__ == "__main__":
    app()
