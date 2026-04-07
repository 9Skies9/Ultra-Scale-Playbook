import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff = self.linear2(torch.relu(self.linear1(x)))
        return self.norm2(x + ff)


class MiniTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(config.hidden_dim, config.num_heads, config.ff_dim) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)