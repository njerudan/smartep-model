# smartep-model

The SMARTEP model: predicts educator gaps from The Classroom's evidence packages and recommends reskilling.
Workplan: `DOCS/SMARTEP Model Development Workplan (Nov 2026 - Jun 2027).docx`. Phase 0 (refactor of the Colab
baseline) and **Phase 1** (synthetic data generator v2) are done; Phase 2 (MVP gap model) is next.

## Setup

Python 3.11 is pinned (3.12+ is not supported yet by every ML dependency). The virtual environment lives
outside OneDrive so sync does not churn on thousands of files.

```bash
python -m pip install --user uv
python -m uv python install 3.11
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/smartep-model"   # PowerShell: $env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\smartep-model"
python -m uv sync --extra dev                      # core + tests
python -m uv sync --extra dev --extra research     # + torch (CPU) for the fusion model
```

Extras: `research` (torch), `text` (MiniLM sentence embeddings), `tracking` (MLflow), `dev` (pytest, ruff).

## Run

```bash
python -m uv run python -m pipelines.generate_data --config configs/generator.yaml   # 5,100 packages + G1 checks (~3 min)
python -m uv run python -m pipelines.train --config configs/train_xgb.yaml
python -m uv run python -m pipelines.train --config configs/train_fusion.yaml
python -m uv run python -m pipelines.train --config configs/train_xgb.yaml --set data.n_educators=500 --set seed=7
```

Each run writes `artifacts/<version>/` (model bundle, `metrics.json`, `report.md`) and records the
promotion decision in `artifacts/registry/` (`champion.json`, `history.jsonl`).

```bash
python -m uv run pytest -q              # all tests (fusion test skipped without torch)
python -m uv run pytest -q -m "not slow"   # skip end-to-end runs
python -m uv run ruff check .
python -m uv run python scripts/export_schemas.py   # after changing a contract
```

## Layout

| Path | What it holds |
|---|---|
| `smartep/contracts/` | Pydantic models of the evidence package (spec §3.2) and AssessmentResult (spec §6); the 9 gap areas |
| `schemas/` | JSON Schema exported from the contracts, to compare with The Classroom's Zod schema |
| `smartep/synth/` | Generator v2: traits, label rules, package builder, evidence rules, G1 checks, dataset I/O. `legacy.py` is the Colab simulation (Phase 0 comparison only) |
| `smartep/data/` | Grouped train/validation/test split; dataset fingerprints |
| `smartep/features/` | Structured package features (`package.py`), plus tabular, text, sequence and watch-history features |
| `smartep/models/` | Multi-label XGBoost, calibration, thresholds and severity, fusion model and encoders |
| `smartep/explain/` | SHAP values (XGBoost TreeSHAP) and per-row explanations |
| `smartep/evaluate/` | Metrics, fairness audit, gates, personas, pilot measures (SUS, competency gain), report |
| `smartep/recommend/` | LinUCB bandit, fairness re-ranking, ranking metrics |
| `smartep/registry/` | Model bundles and champion/challenger promotion |
| `pipelines/train.py` | The one training command |
| `docs/decision-log.md` | Design decisions per phase, and why |
| `docs/data-sheet-synthetic-v2.md` | Every generator rule, label rule, G1 result and limitation |

## Status of results

- `pipelines.train` still uses the Phase 0 data source (`legacy_colab`), whose labels are random, so test
  AUC near 0.5 is the expected result there. Phase 2 switches training to generator v2.
- Generator v2 passes gate G1: oracle AUC 0.916, structured-feature probe 0.827, noise 0.501 (data sheet §4).
  These are synthetic results: they test the pipeline, not usefulness with real educators.
