"""
DPO training loop.

Wires together:
  - policy model   : the LoRA fine-tuned GPT-2 (only LoRA params are trained)
  - reference model: a frozen copy of the same fine-tuned model
  - preference data: built on-policy via the reward scorer
  - DPO objective  : src/dpo/dpo.py

Only the LoRA adapter parameters of the policy receive gradients; the reference
model is fully frozen and provides the regularization anchor.
"""
import copy

import mlflow
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import Config
from src.data.dataset import format_prompt, load_records, split_records
from src.dpo.dpo import dpo_step
from src.dpo.preference_data import build_preference_pairs, collate_preference_batch
from src.evaluation.evaluate import generate_commentary
from src.lora.lora import inject_lora, load_lora_weights, print_param_summary
from src.reward.reward_model import heuristic_score


def _load_policy_and_reference(config: Config):
    tokenizer = GPT2Tokenizer.from_pretrained(config.training.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy: fine-tuned model with trainable LoRA adapter
    policy = GPT2LMHeadModel.from_pretrained(config.training.model_name)
    policy = inject_lora(policy, config.lora)
    lora_path = f"{config.training.output_dir}/lora_best.pt"
    try:
        load_lora_weights(policy, lora_path)
    except FileNotFoundError:
        print(f"[dpo] no LoRA checkpoint at {lora_path}; starting from base + fresh adapter")
    policy = policy.to(config.device)

    # Reference: frozen deep copy of the policy at its current (post-SFT) state
    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    return policy, reference, tokenizer


def train_dpo(config: Config, beta: float = 0.1, candidates_per_prompt: int = 4) -> None:
    if config.num_threads and config.num_threads > 0:
        torch.set_num_threads(config.num_threads)

    policy, reference, tokenizer = _load_policy_and_reference(config)
    print_param_summary(policy)

    # Prompts to build preferences over (use the training split's verses)
    records = load_records(config.data)
    train_records, _, _ = split_records(records, config.data)
    prompts = [format_prompt(r) for r in train_records]

    def generate_fn(prompt: str) -> list[str]:
        return generate_commentary(
            policy, tokenizer, prompt, config.device,
            max_new_tokens=config.data.max_length // 2,
            num_return_sequences=candidates_per_prompt,
        )

    print(f"Building preference pairs over {len(prompts)} prompts...")
    pairs = build_preference_pairs(
        prompts, generate_fn, heuristic_score, min_margin=0.02
    )
    print(f"Built {len(pairs)} preference pairs (after margin filter)")

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
    )

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    batch_size = config.training.batch_size
    with mlflow.start_run(run_name="dpo"):
        mlflow.log_params({"beta": beta, "candidates_per_prompt": candidates_per_prompt,
                           "lr": config.training.learning_rate, "batch_size": batch_size})

        step = 0
        for epoch in range(config.training.num_epochs):
            policy.train()
            for i in range(0, len(pairs), batch_size):
                chunk = pairs[i : i + batch_size]
                batch = collate_preference_batch(chunk, tokenizer, config.data.max_length)
                batch = batch.to(config.device)

                metrics = dpo_step(policy, reference, batch, beta=beta)
                loss = metrics["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()

                step += 1
                mlflow.log_metrics(
                    {
                        "dpo_loss": loss.item(),
                        "reward_margin": metrics["reward_margin"],
                        "reward_accuracy": metrics["reward_accuracy"],
                    },
                    step=step,
                )

            print(
                f"Epoch {epoch + 1}/{config.training.num_epochs} | "
                f"loss: {loss.item():.4f} | "
                f"reward_margin: {metrics['reward_margin']:.4f} | "
                f"reward_acc: {metrics['reward_accuracy']:.2f}"
            )

        # Save the DPO-aligned adapter
        from src.lora.lora import save_lora_weights
        save_lora_weights(policy, f"{config.training.output_dir}/lora_dpo.pt")
        print("Saved DPO-aligned adapter to lora_dpo.pt")
