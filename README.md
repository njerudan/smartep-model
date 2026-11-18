# smartep-model

The SMARTEP model: predicts educator gaps from The Classroom's evidence packages and recommends reskilling.
Workplan: `DOCS/SMARTEP Model Development Workplan (Nov 2026 - Jun 2027).docx`. Phases 0 (refactor), 1 (synthetic
data generator v2) and **2 (MVP gap model v0.1)** are done; Phase 3 (serving and integration) is next.
Model card: `docs/model-card-v0.1.md`.

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
python -m uv sync --extra dev --extra text                                         # optional: MiniLM text features
python -m uv run python -m pipelines.train --config configs/gap_model.yaml    # model v0.1, TF-IDF text (~10 min)
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
| `smartep/training/gap_model.py` | Gap model: CV, calibration, thresholds, severity, evidence and fairness scoring, bundle |
| `smartep/serve/predict.py` | Evidence package in, validated AssessmentResult out |
| `pipelines/train.py` | The one training command (legacy or gap model, by `data.source`) |
| `docs/decision-log.md` | Design decisions per phase, and why |
| `docs/data-sheet-synthetic-v2.md` | Every generator rule, label rule, G1 result and limitation |

## Status of results

- Gap model v0.1 (`configs/gap_model.yaml`, TF-IDF text) passes gate G2: CV macro AUC 0.811 ± 0.001, holdout 0.820,
  worst-area ECE 0.033, evidence hit rate 0.77, no fairness failures. Synthetic and unvalidated.
- The Phase 0 configs (`train_xgb.yaml`, `train_fusion.yaml`) still use the legacy Colab data, whose labels
  are random, so AUC near 0.5 is expected there.
- Generator v2 passes gate G1: oracle AUC 0.916, structured-feature probe 0.827, noise 0.501 (data sheet §4).
  These are synthetic results: they test the pipeline, not usefulness with real educators.
