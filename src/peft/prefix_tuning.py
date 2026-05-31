"""
Prefix Tuning — from scratch for GPT-2. (Li & Liang, 2021)

Idea: keep the entire pretrained model frozen and instead prepend a small number
of trainable "virtual tokens" to every attention layer. These virtual tokens are
not real tokens in the vocabulary — they are learned key/value vectors injected
directly into the attention mechanism as `past_key_values`. The model attends to
them as if they were prefix context, letting them steer generation.

Why this works (and why it's parameter-efficient):
  - A GPT-2 attention layer computes attention over keys/values of shape
    (batch, n_head, seq, head_dim). Normally these come from projecting the
    actual input tokens.
  - Prefix tuning adds `num_virtual_tokens` extra positions to the K and V of
    *every* layer. The query side is untouched, so real tokens simply get extra
    things to attend to.
  - Only these prefix K/V vectors are trained. For GPT-2 small (12 layers,
    n_embd=768) with 20 virtual tokens that's 12 * 2 * 20 * 768 ≈ 368k params,
    vs 124M for the full model.

Reparameterization trick (Li & Liang): directly optimizing the raw prefix vectors
is unstable, so we instead learn a small embedding table and pass it through an
MLP. The MLP is only used at train time to produce the prefixes; at inference you
can fold it away. We keep it simple and always run the MLP.

The prefix is returned in HuggingFace's `past_key_values` format: a tuple with one
entry per layer, each a (key, value) pair of shape
(batch, n_head, num_virtual_tokens, head_dim).
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

try:  # newer transformers require a Cache object instead of the legacy tuple
    from transformers import DynamicCache
except ImportError:  # pragma: no cover
    DynamicCache = None


class PrefixEncoder(nn.Module):
    """
    Produces per-layer key/value prefixes from a small trainable table.

    Output shape (before reshaping into the past_key_values format):
        (num_virtual_tokens, n_layer * 2 * n_embd)
    The factor 2 is for key and value; n_layer because every layer gets its own.
    """
    def __init__(self, n_layer: int, n_embd: int, n_head: int,
                 num_virtual_tokens: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.num_virtual_tokens = num_virtual_tokens
        self.head_dim = n_embd // n_head

        # Trainable virtual-token indices -> embedding -> MLP reparameterization
        self.prefix_tokens = torch.arange(num_virtual_tokens).long()
        self.embedding = nn.Embedding(num_virtual_tokens, n_embd)
        self.transform = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_layer * 2 * n_embd),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_size: int, device: torch.device) -> tuple:
        tokens = self.prefix_tokens.to(device)              # (num_virtual_tokens,)
        x = self.embedding(tokens)                          # (T, n_embd)
        past = self.transform(x)                            # (T, n_layer*2*n_embd)
        past = self.dropout(past)

        T = self.num_virtual_tokens
        # -> (T, n_layer*2, n_head, head_dim)
        past = past.view(T, self.n_layer * 2, self.n_head, self.head_dim)
        # current dims: (T, n_layer*2, n_head, head_dim)
        # expand to batch -> (batch, T, n_layer*2, n_head, head_dim)
        past = past.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        # reorder to (n_layer*2, batch, n_head, T, head_dim)
        past = past.permute(2, 0, 3, 1, 4)
        # split into per-layer (key, value) pairs
        past = past.split(2)  # n_layer chunks, each (2, batch, n_head, T, head_dim)
        return tuple((chunk[0], chunk[1]) for chunk in past)


class PrefixTuningModel(nn.Module):
    """
    Wraps a frozen GPT2LMHeadModel and injects learned prefixes on every forward.

    The base model's parameters are frozen; only the PrefixEncoder trains. We
    extend the attention mask to cover the virtual tokens so real tokens are
    allowed to attend to the prefix.
    """
    def __init__(self, model: GPT2LMHeadModel, num_virtual_tokens: int = 20,
                 hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.model = model
        cfg = model.config
        self.num_virtual_tokens = num_virtual_tokens

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.prefix_encoder = PrefixEncoder(
            n_layer=cfg.n_layer,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            num_virtual_tokens=num_virtual_tokens,
            hidden=hidden,
            dropout=dropout,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        seq_len = input_ids.shape[1]
        past_key_values = self.prefix_encoder(batch_size, device)

        # Newer transformers wrap past_key_values in a Cache object instead of a
        # tuple. Populate a DynamicCache layer-by-layer from the encoder output.
        if DynamicCache is not None:
            legacy = past_key_values
            cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(legacy):
                cache.update(key, value, layer_idx)
            past_key_values = cache

        # Prepend 1s to the attention mask for the virtual-token positions so
        # the real tokens can attend to the prefix.
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size, self.num_virtual_tokens, device=device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # The prefix occupies cache positions, which would otherwise offset the
        # real tokens' position ids. Pin real tokens to positions [0, seq_len).
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            **kwargs,
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def inject_prefix_tuning(model: GPT2LMHeadModel, num_virtual_tokens: int = 20,
                         hidden: int = 512, dropout: float = 0.0) -> PrefixTuningModel:
    """Freeze GPT-2 and wrap it with a trainable prefix encoder."""
    wrapped = PrefixTuningModel(model, num_virtual_tokens, hidden, dropout)
    trainable = sum(p.numel() for p in wrapped.trainable_parameters())
    total = sum(p.numel() for p in wrapped.parameters())
    print(
        f"Prefix tuning: {num_virtual_tokens} virtual tokens | "
        f"trainable {trainable:,} / {total:,} "
        f"({100 * trainable / total:.3f}%)"
    )
    return wrapped


def prefix_state_dict(model: PrefixTuningModel) -> dict:
    """Extract only the prefix-encoder parameters for saving."""
    return {
        name: param
        for name, param in model.state_dict().items()
        if name.startswith("prefix_encoder.")
    }
