"""Attention-based multimodal fusion (needs the ``research`` extra).

Fixes against the notebook: the model is trained once, with early stopping on the validation split; the
best weights are restored, and the same weights are evaluated and saved.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from smartep.models.encoders import SequenceEncoder, WatchHistoryAttention

INPUT_KEYS = ("tab", "seq", "text", "watch", "watch_mask")


class AttentionFusionModel(nn.Module):
    def __init__(self, tab_dim: int, seq_dim: int, text_dim: int, watch_dim: int, n_labels: int,
                 common_dim: int = 32, seq_hidden: int = 32, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.has_text = text_dim > 0
        self.tab_proj = nn.Linear(tab_dim, common_dim)
        self.seq_encoder = SequenceEncoder(seq_dim, hidden_dim=seq_hidden, embed_dim=common_dim)
        self.text_proj = nn.Linear(text_dim, common_dim) if self.has_text else None
        self.watch_encoder = WatchHistoryAttention(tab_dim, watch_dim, embed_dim=common_dim)
        self.modality_attn = nn.MultiheadAttention(common_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(common_dim)
        self.head = nn.Sequential(nn.Linear(common_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_labels))

    def forward(self, tab, seq, text, watch, watch_mask):
        tokens = [torch.relu(self.tab_proj(tab)), self.seq_encoder(seq)]
        if self.has_text:
            tokens.append(torch.relu(self.text_proj(text)))
        watch_embed, watch_attn = self.watch_encoder(tab, watch, watch_mask)
        tokens.append(watch_embed)
        tokens = torch.stack(tokens, dim=1)
        attended, modality_attn = self.modality_attn(tokens, tokens, tokens)
        logits = self.head(self.norm(attended.mean(dim=1)))
        return logits, modality_attn, watch_attn


@dataclass
class FusionHistory:
    best_epoch: int
    best_val_loss: float
    train_loss: list[float]
    val_loss: list[float]


def _loader(inputs: dict, Y, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensors = [torch.as_tensor(inputs[k], dtype=torch.float32) for k in INPUT_KEYS]
    if Y is not None:
        tensors.append(torch.as_tensor(Y, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=gen)


def train_fusion(train_inputs: dict, Y_train, val_inputs: dict, Y_val, *, n_labels: int, seed: int,
                 lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 32, max_epochs: int = 30,
                 patience: int = 5, device: str = "cpu", **model_kwargs) -> tuple[AttentionFusionModel, FusionHistory]:
    torch.manual_seed(seed)
    model = AttentionFusionModel(
        tab_dim=train_inputs["tab"].shape[1], seq_dim=train_inputs["seq"].shape[2],
        text_dim=train_inputs["text"].shape[1], watch_dim=train_inputs["watch"].shape[2],
        n_labels=n_labels, **model_kwargs,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_dl = _loader(train_inputs, Y_train, batch_size, True, seed)
    val_dl = _loader(val_inputs, Y_val, 256, False, seed)

    best_state, best_loss, best_epoch, waited = None, float("inf"), -1, 0
    history = FusionHistory(-1, float("inf"), [], [])
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        for *xs, y in train_dl:
            xs, y = [x.to(device) for x in xs], y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(*xs)[0], y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        history.train_loss.append(total / len(train_dl.dataset))

        model.eval()
        with torch.no_grad():
            val_total = sum(loss_fn(model(*[x.to(device) for x in xs])[0], y.to(device)).item() * len(y)
                            for *xs, y in val_dl)
        val_loss = val_total / len(val_dl.dataset)
        history.val_loss.append(val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    history.best_epoch, history.best_val_loss = best_epoch, best_loss
    return model, history


@torch.no_grad()
def predict_fusion(model: AttentionFusionModel, inputs: dict, device: str = "cpu") -> np.ndarray:
    model.eval()
    probs = [torch.sigmoid(model(*[x.to(device) for x in xs])[0]).cpu().numpy()
             for xs in _loader(inputs, None, 256, False, 0)]
    return np.concatenate(probs).astype(np.float64)
