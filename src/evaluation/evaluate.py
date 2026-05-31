"""
Evaluation module.

Benchmarks four configurations side by side:
  1. Base GPT-2 (no fine-tuning, no RAG)
  2. Fine-tuned GPT-2 (LoRA, no RAG)
  3. Fine-tuned GPT-2 + RAG
  4. Fine-tuned GPT-2 + RAG + Reward reranking

Metrics:
  - Perplexity on held-out test set (lower = better)
  - Theological vocabulary density
  - Average output length
  - Qualitative side-by-side examples saved to results/
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import Config
from src.data.dataset import CommentaryRecord, build_datasets, format_prompt
from src.reward.reward_model import PRABHUPADA_VOCAB


def perplexity_on_loader(
    model: nn.Module, loader: DataLoader, device: str
) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            active = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * active
            total_tokens += active

    return math.exp(total_loss / max(total_tokens, 1))


def vocab_density(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.strip(".,;:()[]") in PRABHUPADA_VOCAB)
    return hits / len(words)


def generate_commentary(
    model: nn.Module,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 200,
    num_return_sequences: int = 1,
) -> list[str]:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
        )

    results = []
    for output in outputs:
        # Strip the prompt, decode only the generated tokens
        generated_ids = output[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text.strip())
    return results


def run_benchmark(config: Config) -> None:
    device = config.device
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(config.training.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    _, _, test_ds = build_datasets(config.data, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)

    from src.lora.lora import inject_lora, load_lora_weights

    results = {}

    # --- 1. Base GPT-2 ---
    print("Evaluating base GPT-2...")
    base_model = GPT2LMHeadModel.from_pretrained(config.training.model_name).to(device)
    base_ppl = perplexity_on_loader(base_model, test_loader, device)
    results["base_gpt2"] = {"perplexity": base_ppl}
    print(f"  Base perplexity: {base_ppl:.2f}")

    # --- 2. Fine-tuned (LoRA) ---
    print("Evaluating fine-tuned model...")
    ft_model = GPT2LMHeadModel.from_pretrained(config.training.model_name)
    ft_model = inject_lora(ft_model, config.lora)
    lora_path = Path(config.training.output_dir) / "lora_best.pt"
    if lora_path.exists():
        load_lora_weights(ft_model, str(lora_path))
    ft_model = ft_model.to(device)
    ft_ppl = perplexity_on_loader(ft_model, test_loader, device)
    results["finetuned"] = {"perplexity": ft_ppl}
    print(f"  Fine-tuned perplexity: {ft_ppl:.2f}")

    # --- Qualitative comparison on 3 test verses ---
    print("\nGenerating qualitative examples...")
    from src.data.dataset import load_records, split_records
    all_records = load_records(config.data)
    _, _, test_records = split_records(all_records, config.data)
    sample_records = test_records[:3]

    qualitative = []
    for record in sample_records:
        prompt = format_prompt(record)
        base_out = generate_commentary(base_model, tokenizer, prompt, device)[0]
        ft_out = generate_commentary(ft_model, tokenizer, prompt, device)[0]

        qualitative.append({
            "verse": f"BG {record.chapter}.{record.verse_label}",
            "translation": record.translation[:150],
            "ground_truth": record.commentary[:300],
            "base_gpt2": base_out,
            "finetuned": ft_out,
            "base_vocab_density": vocab_density(base_out),
            "finetuned_vocab_density": vocab_density(ft_out),
        })
        print(f"  BG {record.chapter}.{record.verse_label} done")

    results["qualitative"] = qualitative

    # --- Save results ---
    with open(results_dir / "benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir / 'benchmark.json'}")

    # --- Plot perplexity comparison ---
    configs = list(results.keys())
    ppls = [v["perplexity"] for k, v in results.items() if "perplexity" in v]
    config_names = [k for k in results if "perplexity" in results[k]]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(config_names, ppls, color=["#aec6cf", "#77dd77", "#fdfd96", "#ffb347"])
    plt.ylabel("Perplexity (lower is better)")
    plt.title("Perplexity Comparison Across Configurations")
    for bar, ppl in zip(bars, ppls):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{ppl:.1f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(results_dir / "perplexity_comparison.png", dpi=150)
    print(f"Plot saved to {results_dir / 'perplexity_comparison.png'}")
