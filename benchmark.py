"""Entry point: run full benchmark across all model configurations."""

import typer
from config import get_config
from src.evaluation.evaluate import run_benchmark

app = typer.Typer()


@app.command()
def main():
    cfg = get_config()
    run_benchmark(cfg)


if __name__ == "__main__":
    app()
