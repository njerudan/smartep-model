"""Config loading, hashing and seeding."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(expr: str) -> dict:
    """Turn ``a.b.c=value`` into ``{"a": {"b": {"c": value}}}``; the value is parsed as YAML."""
    if "=" not in expr:
        raise ValueError(f"Override must look like key.path=value, got {expr!r}")
    path, raw = expr.split("=", 1)
    value: Any = yaml.safe_load(raw)
    for key in reversed(path.strip().split(".")):
        value = {key: value}
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for expr in overrides or []:
        cfg = deep_merge(cfg, parse_override(expr))
    return cfg


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - our code uses Generators; this covers libraries on the global RNG
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
