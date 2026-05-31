from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    db_path: str = "/Users/aryagupta/Desktop/gita-insight-engine/gita.db"
    acharya: str = "PRABHUPADA"          # which acharya to fine-tune on
    max_length: int = 512                # max tokens per training sample
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10
    seed: int = 42


@dataclass
class LoRAConfig:
    rank: int = 8                        # rank of the low-rank matrices A and B
    alpha: int = 16                      # scaling factor: output *= alpha / rank
    dropout: float = 0.1
    target_modules: list = field(        # which attention projections to adapt
        default_factory=lambda: ["attn.c_attn", "attn.c_proj"]
    )


@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    output_dir: str = "checkpoints"
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 4
    eval_steps: int = 50
    save_steps: int = 100
    fp16: bool = False                   # MPS doesn't support fp16


@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_dir: str = "chroma_db"
    collection_name: str = "gita_commentaries"
    top_k: int = 3                       # how many chunks to retrieve


@dataclass
class RewardConfig:
    model_name: str = "gpt2"
    output_dir: str = "reward_checkpoints"
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_candidates: int = 4              # generate N outputs, score, return best


@dataclass
class MLflowConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "gita-lm"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    device: str = "mps"                  # mps | cuda | cpu


def get_config() -> Config:
    return Config()
