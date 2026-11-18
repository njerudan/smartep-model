"""Generate a synthetic evidence-package dataset, write it, and run the G1 checks.

    python -m pipelines.generate_data --config configs/generator.yaml
    python -m pipelines.generate_data --config configs/generator.yaml --set n_educators=200

The dataset is written to <output.dir>/<config hash>/, so the same config always lands in the same place
and a model trained on it can record exactly which data it used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from smartep.config import config_hash, load_config
from smartep.contracts.gap_areas import GAP_AREAS
from smartep.registry.bundle import git_commit, library_versions
from smartep.synth.checks import run_checks
from smartep.synth.generator import GENERATOR_VERSION, generate_dataset
from smartep.synth.io import write_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def stats_markdown(ds, checks: dict) -> str:
    sev = ds.severity[list(GAP_AREAS)]
    lines = ["| Gap area | Prevalence | Low | Medium | High | Oracle AUC | Evidence-probe AUC |", "|---|---|---|---|---|---|---|"]
    for area in GAP_AREAS:
        counts = sev[area].value_counts()
        lines.append(f"| {area} | {ds.labels[area].mean():.3f} | {counts.get('low', 0)} | {counts.get('medium', 0)} | "
                     f"{counts.get('high', 0)} | {checks['oracle']['per_area_auc'][area]} | "
                     f"{checks['evidence_probe']['per_area_auc'][area]} |")
    subsets = ds.index["subset"].value_counts().to_dict()
    return "\n".join([
        f"Packages: {len(ds.packages)} ({subsets}); educators: {ds.index['educatorRef'].nunique()}",
        "",
        *lines,
        "",
        f"Macro AUC: oracle {checks['oracle']['macro_auc']} · evidence probe {checks['evidence_probe']['macro_auc']} "
        f"· noise {checks['noise']['macro_auc']}",
        "",
        "Seniority mix: " + json.dumps(ds.index["seniority"].value_counts(normalize=True).round(3).to_dict()),
        "",
        "Institution types: " + json.dumps(ds.index.drop_duplicates("educatorRef")["institution_type"]
                                           .value_counts(normalize=True).round(3).to_dict()),
    ])


def run(cfg: dict, output_root: Path | None = None) -> dict:
    t0 = time.perf_counter()
    cfg_hash = config_hash(cfg)
    out = (output_root or Path(cfg["output"]["dir"])) / cfg_hash
    ds = generate_dataset(cfg)
    t_gen = time.perf_counter() - t0
    print(f"generated {len(ds.packages)} packages in {t_gen:.1f}s", flush=True)

    manifest = {"generator_version": GENERATOR_VERSION, "config_hash": cfg_hash, "config": cfg,
                "n_packages": len(ds.packages), "n_educators": int(ds.index["educatorRef"].nunique()),
                "subsets": ds.index["subset"].value_counts().to_dict(), "git_commit": git_commit(REPO_ROOT),
                "libraries": library_versions()}
    write_dataset(ds, out, manifest)
    checks = run_checks(ds, cfg["checks"], int(cfg["seed"]))
    checks["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
    (out / "stats.md").write_text(stats_markdown(ds, checks) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return {"dir": str(out), "checks": checks}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    result = run(cfg, Path(args.output_dir) if args.output_dir else None)
    c = result["checks"]
    print(json.dumps({k: c[k].get("pass", c[k].get("meets_soft_target")) if isinstance(c[k], dict) else c[k]
                      for k in ("schema", "prevalence", "evidence", "oracle", "noise", "evidence_probe", "G1_pass")}, indent=2))
    print(f"oracle {c['oracle']['macro_auc']} · evidence probe {c['evidence_probe']['macro_auc']} · noise {c['noise']['macro_auc']}")


if __name__ == "__main__":
    main()
