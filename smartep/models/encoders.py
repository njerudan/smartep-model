"""Modality encoders for the fusion model (needs the ``research`` extra)."""

from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, embed_dim: int = 16, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), embed_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.lstm.bidirectional else h_n[-1]
        return self.proj(h)


class WatchHistoryAttention(nn.Module):
    """Attention from the educator's tabular profile over their watch-history rows.

    Called ``GraphAttentionEncoder`` in the notebook; renamed because there is no graph.
    """

    def __init__(self, query_dim: int, item_dim: int, embed_dim: int = 16):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(item_dim, embed_dim)
        self.value_proj = nn.Linear(item_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, query, items, mask):
        q = self.query_proj(query).unsqueeze(1)
        k, v = self.key_proj(items), self.value_proj(items)
        scores = (q @ k.transpose(1, 2)).squeeze(1) / self.scale
        scores = scores.masked_fill(mask == 0, float("-inf"))
        empty = mask.sum(dim=1) == 0
        scores = torch.where(empty.unsqueeze(1), torch.zeros_like(scores), scores)
        attn = torch.softmax(scores, dim=1)
        return (attn.unsqueeze(-1) * v).sum(dim=1), attn
