"""Copy the registry champion (or a given bundle) to models/gap-model/ for serving, with golden fixtures.

    python scripts/export_model.py                       # champion from artifacts/registry/champion.json
    python scripts/export_model.py --bundle artifacts/<version>

Writes models/gap-model/ (the bundle the API image ships) and tests/fixtures/golden/ (a few holdout
packages with the probabilities this bundle gives them), so a test can check that serving reproduces the
evaluated model exactly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from smartep.serve.predict import GapPredictor
from smartep.synth.io import read_dataset

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "models" / "gap-model"
GOLDEN = REPO / "tests" / "fixtures" / "golden"
KEEP = ("manifest.json", "config.yaml", "metrics.json", "report.md", "preprocess.joblib", "xgb")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--n-golden", type=int, default=8)
    args = parser.parse_args(argv)
    bundle = Path(args.bundle) if args.bundle else Path(json.loads(
        (REPO / "artifacts" / "registry" / "champion.json").read_text(encoding="utf-8"))["bundle_path"])
    if not bundle.is_absolute():
        bundle = REPO / bundle

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in KEEP:
        src = bundle / name
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, OUT / name)
    print(f"exported {bundle.name} -> {OUT}")

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    ds = read_dataset(REPO / manifest["data_dir"])
    hold = [p for p, s in zip(ds.packages, ds.index["subset"], strict=True) if s == "classroom_test"][: args.n_golden]
    predictor = GapPredictor.from_bundle(OUT)
    _, P = predictor.model.predict(predictor._features(hold))
    GOLDEN.mkdir(parents=True, exist_ok=True)
    with gzip.open(GOLDEN / "packages.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in hold:
            f.write(json.dumps(p, separators=(",", ":")) + "\n")
    (GOLDEN / "expected.json").write_text(json.dumps({
        "model_version": manifest["version"],
        "probabilities": {p["packageId"]: np.round(P[i], 8).tolist() for i, p in enumerate(hold)},
    }, indent=2), encoding="utf-8")
    print(f"golden fixtures: {len(hold)} packages -> {GOLDEN}")


if __name__ == "__main__":
    main()
