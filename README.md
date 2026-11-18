# smartep-model

The SMARTEP model: predicts educator gaps from The Classroom's evidence packages and recommends reskilling.
Workplan: `DOCS/SMARTEP Model Development Workplan (Nov 2026 - Jun 2027).docx`. This repo is at **Phase 0**
(foundations and refactor of the Colab baseline).

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
| `smartep/synth/` | `legacy.py`: the Colab simulation (Phase 0 only). Generator v2 arrives in Phase 1 |
| `smartep/data/` | Grouped train/validation/test split; dataset fingerprints |
| `smartep/features/` | Tabular, text, weekly sequence and watch-history features |
| `smartep/models/` | Multi-label XGBoost, calibration, thresholds and severity, fusion model and encoders |
| `smartep/explain/` | SHAP values (XGBoost TreeSHAP) and per-row explanations |
| `smartep/evaluate/` | Metrics, fairness audit, gates, personas, pilot measures (SUS, competency gain), report |
| `smartep/recommend/` | LinUCB bandit, fairness re-ranking, ranking metrics |
| `smartep/registry/` | Model bundles and champion/challenger promotion |
| `pipelines/train.py` | The one training command |
| `docs/decision-log.md` | What changed from the Colab baseline, and why |

## Status of results

The Phase 0 data source (`legacy_colab`) assigns gap labels at random, independently of every feature,
exactly as the notebook did. Test AUC near 0.5 is therefore the expected, correct result. It shows the
pipeline does not leak test information. Meaningful model numbers start with generator v2 in Phase 1.
