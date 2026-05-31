"""
Training loop for LoRA fine-tuning on GPT-2.
Logs every run to MLflow: hyperparameters, per-step loss, per-epoch perplexity.
"""

import math
import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

from config import Config
from src.data.dataset import CommentaryDataset, build_datasets
from src.lora.lora import inject_lora, print_param_summary, save_lora_weights


def compute_perplexity(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # outputs.loss is mean loss over non-masked tokens
            # recover total loss by multiplying by number of active tokens
            active_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * active_tokens
            total_tokens += active_tokens

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def train(config: Config) -> None:
    device = config.device

    # --- Tokenizer ---
    tokenizer = GPT2Tokenizer.from_pretrained(config.training.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    train_ds, val_ds, _ = build_datasets(config.data, tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- Model ---
    model = GPT2LMHeadModel.from_pretrained(config.training.model_name)
    model = inject_lora(model, config.lora)
    model = model.to(device)
    print_param_summary(model)

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps,
    )

    # --- MLflow ---
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(exist_ok=True)

    with mlflow.start_run():
        # Log all hyperparameters
        mlflow.log_params({
            "model": config.training.model_name,
            "acharya": config.data.acharya,
            "epochs": config.training.num_epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "lora_rank": config.lora.rank,
            "lora_alpha": config.lora.alpha,
            "lora_dropout": config.lora.dropout,
            "max_length": config.data.max_length,
            "warmup_steps": config.training.warmup_steps,
        })

        global_step = 0
        best_val_perplexity = float("inf")

        for epoch in range(config.training.num_epochs):
            model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / config.training.gradient_accumulation_steps
                loss.backward()
                epoch_loss += outputs.loss.item()

                if (step + 1) % config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                mlflow.log_metric("train_loss_step", outputs.loss.item(), step=global_step)

            avg_train_loss = epoch_loss / len(train_loader)
            train_ppl = math.exp(avg_train_loss)
            val_ppl = compute_perplexity(model, val_loader, device)

            print(
                f"Epoch {epoch + 1}/{config.training.num_epochs} | "
                f"train_loss: {avg_train_loss:.4f} | "
                f"train_ppl: {train_ppl:.2f} | "
                f"val_ppl: {val_ppl:.2f}"
            )
            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "train_perplexity": train_ppl,
                    "val_perplexity": val_ppl,
                },
                step=epoch,
            )

            # Save best checkpoint
            if val_ppl < best_val_perplexity:
                best_val_perplexity = val_ppl
                save_lora_weights(model, str(output_dir / "lora_best.pt"))
                tokenizer.save_pretrained(str(output_dir))
                mlflow.log_metric("best_val_perplexity", best_val_perplexity, step=epoch)
                print(f"  ✓ New best val perplexity: {best_val_perplexity:.2f}")

        mlflow.log_artifact(str(output_dir / "lora_best.pt"))
        print(f"\nTraining complete. Best val perplexity: {best_val_perplexity:.2f}")
