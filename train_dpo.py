"""Entry point: DPO-align the LoRA fine-tuned model using reward-scored preferences."""

import typer

from config import get_config, light_mode
from src.dpo.dpo_trainer import train_dpo

app = typer.Typer()


@app.command()
def main(
    beta: float = typer.Option(0.1, help="DPO temperature — how hard to deviate from reference"),
    candidates: int = typer.Option(4, help="Candidates generated per prompt for preference pairs"),
    epochs: int = typer.Option(None, help="Override number of DPO epochs"),
    light: bool = typer.Option(False, "--light", help="Low-resource mode"),
):
    cfg = light_mode() if light else get_config()
    if epochs:
        cfg.training.num_epochs = epochs
    train_dpo(cfg, beta=beta, candidates_per_prompt=candidates)


if __name__ == "__main__":
    app()
