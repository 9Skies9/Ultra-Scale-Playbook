import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024


def _tp_is_active(tp_size: int) -> bool:
    return tp_size > 1 and dist.is_available() and dist.is_initialized()


def _gather_last_dim(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    if not _tp_is_active(tp_size):
        return x
    shards = [torch.empty_like(x) for _ in range(tp_size)]
    dist.all_gather(shards, x)
    return torch.cat(shards, dim=-1)


def _reduce_sum(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    if not _tp_is_active(tp_size):
        return x
    dist.all_reduce(x)
    return x


class ColumnParallelLinear(nn.Module):
    def __init__(
        self, in_features, out_features, tp_size, tp_rank, gather_output=False
    ):
        super().__init__()
        assert out_features % tp_size == 0

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.gather_output = gather_output
        self.out_per_rank = out_features // tp_size

        # Each rank stores only its output rows.
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize one logical full weight, then keep this rank's shard.
        master_weight = torch.empty(self.out_features, self.in_features)
        nn.init.uniform_(master_weight, -0.1, 0.1)
        self.weight.data.copy_(master_weight.chunk(self.tp_size, dim=0)[self.tp_rank])

    def forward(self, x):
        y_local = F.linear(x, self.weight)
        if not self.gather_output:
            return y_local
        return _gather_last_dim(y_local, self.tp_size)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size, tp_rank):
        super().__init__()
        assert in_features % tp_size == 0

        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.in_per_rank = in_features // tp_size

        # Each rank stores only its input columns.
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize one logical full weight, then keep this rank's shard.
        master_weight = torch.empty(self.out_features, self.in_features)
        nn.init.uniform_(master_weight, -0.1, 0.1)
        self.weight.data.copy_(master_weight.chunk(self.tp_size, dim=1)[self.tp_rank])

    def forward(self, x):
        y_partial = F.linear(x, self.weight)
        return _reduce_sum(y_partial, self.tp_size)


class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # TP is applied to these four projection layers.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # After TP, each rank may hold fewer heads because q/k/v are sharded.
        local_heads = q.size(-1) // self.head_dim

        q = q.view(batch_size, seq_len, local_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, local_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, local_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, local_heads * self.head_dim)
        )
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = Attention(hidden_dim, num_heads)
        self.mlp = MLP(hidden_dim, ff_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x


class MiniTransformerTPFriendly(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(config.hidden_dim, config.num_heads, config.ff_dim)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.final_proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.final_proj(x)


