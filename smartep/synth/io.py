"""Write and read a generated dataset.

    <dir>/
      manifest.json              generator version, config, config hash, counts, git commit
      packages.jsonl.gz          one evidence package per line (all subsets)
      index.parquet              packageId, educatorRef, period, course, group attributes, subset
      labels.parquet             packageId + 0/1 per gap area
      severity.parquet           packageId + low/medium/high per gap area ('' = no gap)
      evidence.jsonl.gz          ground-truth evidence per package and gap
      traits.parquet             latent period traits (oracle check only; never a model input)
      classroom_test.jsonl.gz    the packages held out for The Classroom's end-to-end tests
      checks.json                G1 results (written by pipelines.generate_data)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from smartep.synth.generator import SyntheticDataset


def _write_jsonl(path: Path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_dataset(ds: SyntheticDataset, directory: Path, manifest: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "packages.jsonl.gz", ds.packages)
    _write_jsonl(directory / "evidence.jsonl.gz", ds.evidence)
    held_out = set(ds.index.loc[ds.index["subset"] == "classroom_test", "packageId"])
    _write_jsonl(directory / "classroom_test.jsonl.gz", (p for p in ds.packages if p["packageId"] in held_out))
    ds.index.to_parquet(directory / "index.parquet", index=False)
    ds.labels.to_parquet(directory / "labels.parquet", index=False)
    ds.severity.to_parquet(directory / "severity.parquet", index=False)
    ds.traits.to_parquet(directory / "traits.parquet", index=False)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return directory


def read_dataset(directory: Path) -> SyntheticDataset:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return SyntheticDataset(
        packages=read_jsonl(directory / "packages.jsonl.gz"),
        index=pd.read_parquet(directory / "index.parquet"),
        labels=pd.read_parquet(directory / "labels.parquet"),
        severity=pd.read_parquet(directory / "severity.parquet"),
        evidence=read_jsonl(directory / "evidence.jsonl.gz"),
        traits=pd.read_parquet(directory / "traits.parquet"),
        config=manifest["config"],
    )
