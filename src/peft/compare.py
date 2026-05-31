"""
Parameter-efficiency comparison across the PEFT methods implemented here.

This doesn't train anything — it instantiates each adapter on GPT-2 and reports
the trainable-parameter footprint, which is the headline number that justifies
PEFT: how little you have to train to adapt a 124M-param model.

Methods compared:
  - Full fine-tuning (baseline, everything trains)
  - LoRA            (low-rank update matrices)
  - (IA)³           (learned activation rescaling vectors)
  - Prefix tuning   (trainable virtual key/value prefixes)
"""
from transformers import GPT2LMHeadModel

from config import LoRAConfig
from src.lora.lora import inject_lora
from src.peft.ia3 import inject_ia3
from src.peft.prefix_tuning import inject_prefix_tuning


def _count(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def compare_peft_methods(model_name: str = "gpt2", num_virtual_tokens: int = 20) -> list[dict]:
    rows = []

    base = GPT2LMHeadModel.from_pretrained(model_name)
    total = sum(p.numel() for p in base.parameters())
    rows.append({"method": "full_finetune", "trainable": total, "total": total})

    lora = inject_lora(GPT2LMHeadModel.from_pretrained(model_name), LoRAConfig())
    t, tot = _count(lora)
    rows.append({"method": "lora", "trainable": t, "total": tot})

    ia3 = inject_ia3(GPT2LMHeadModel.from_pretrained(model_name))
    t, tot = _count(ia3)
    rows.append({"method": "ia3", "trainable": t, "total": tot})

    prefix = inject_prefix_tuning(
        GPT2LMHeadModel.from_pretrained(model_name), num_virtual_tokens=num_virtual_tokens
    )
    t, tot = _count(prefix)
    rows.append({"method": "prefix_tuning", "trainable": t, "total": tot})

    print(f"\n{'method':<16}{'trainable':>14}{'% of model':>12}")
    print("-" * 42)
    for r in rows:
        pct = 100 * r["trainable"] / r["total"]
        print(f"{r['method']:<16}{r['trainable']:>14,}{pct:>11.3f}%")

    return rows


if __name__ == "__main__":
    compare_peft_methods()
