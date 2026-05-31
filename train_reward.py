"""Entry point: train the reward model that reranks generated commentaries."""

from pathlib import Path

import torch
import typer

from config import get_config
from src.reward.reward_model import train_reward_model

app = typer.Typer()


@app.command()
def main(
    epochs: int = typer.Option(None, help="Override number of reward-model epochs"),
):
    cfg = get_config()
    if epochs:
        cfg.reward.num_epochs = epochs

    model = train_reward_model(cfg.reward, cfg.data, cfg.device)

    out_dir = Path(cfg.reward.output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "reward_model.pt"
    torch.save(model.state_dict(), str(out_path))
    print(f"Saved reward model to {out_path}")


if __name__ == "__main__":
    app()
