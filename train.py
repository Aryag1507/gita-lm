"""Entry point: fine-tune GPT-2 with LoRA on Prabhupada commentaries."""

import typer
from config import get_config
from src.training.trainer import train

app = typer.Typer()


@app.command()
def main(
    epochs: int = typer.Option(None, help="Override number of training epochs"),
    lr: float = typer.Option(None, help="Override learning rate"),
    rank: int = typer.Option(None, help="Override LoRA rank"),
):
    cfg = get_config()
    if epochs:
        cfg.training.num_epochs = epochs
    if lr:
        cfg.training.learning_rate = lr
    if rank:
        cfg.lora.rank = rank

    train(cfg)


if __name__ == "__main__":
    app()
