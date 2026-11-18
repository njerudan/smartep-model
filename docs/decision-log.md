# Decision log

One entry per design decision: what was decided, and why. Feeds the methodology chapter.

## 2026-11-18 · Phase 1: synthetic data generator v2

Details and every rule: `docs/data-sheet-synthetic-v2.md`.

| Decision | Why |
|---|---|
| Generate evidence packages in schema v1.0, not tables | The same feature code then runs on synthetic and real packages; no training/serving skew |
| Six latent traits drive both the evidence and the labels | Gives a known, documented link to recover; the notebook's labels had none |
| Labels from period traits; package from period traits + observation noise (sd 0.8) | Separates what is true from what the evidence shows; one setting controls task difficulty |
| Observation noise 0.8 | Without it, structured features reached 0.897 vs oracle 0.913, so Phase 2's 0.70 target and Phase 5's ablation would be meaningless. 0.8 gives a probe of 0.827 and leaves room for text features |
| 10% assessor-disagreement flips, prevalence preserved | Real labels from admins will be noisy; perfect labels would overstate achievable AUC |
| Severity from the margin above threshold (tertiles) | Gives the three contract levels a consistent meaning |
| Ground-truth evidence read back from the package by rules | Every gap needs evidence (contract); rules are the reference for Phase 2 evidence mapping |
| Hold out 501 packages (whole educators) for The Classroom | End-to-end tests use data no model was trained on |
| Group attributes not linked to traits by default; optional injected bias | Lets the fairness audit be tested both ways |
| Gate G1 includes an oracle check, a noise check and an evidence probe | Shows the signal exists, the check does not leak, and the signal reaches the observable fields |
| Structured package features (`smartep/features/package.py`) written now | Needed for the evidence probe; Phase 2 extends it with text and evidence mapping |
| Dataset written under `data/synthetic_v2/<config hash>/` | A model can record exactly which data it was trained on |

**G1 passed** (2026-11-18): 5,100/5,100 packages valid; prevalence 0.150–0.251; 9,382 gaps all with evidence;
oracle 0.916; noise 0.501; evidence probe 0.827. 39 tests pass.

## 2026-11-18 · Phase 0: refactor of the Colab baseline

Source: `Source Code.txt` (Colab notebook). Workplan section 3 and Appendix A.

### Kept

- Multi-label framing with one XGBoost model per label (`smartep/models/xgb_multilabel.py`).
- Weekly sequence, text and watch-history modalities and the attention fusion model (`smartep/models/fusion.py`).
- Selection-rate fairness audit, LinUCB bandit with fairness re-ranking, MAP@k and NDCG@k.
- Champion/challenger registry, SUS and competency-gain utilities, KMeans personas.
- The legacy simulation, ported to `smartep/synth/legacy.py` so the refactor can be compared with the notebook.

### Changed

| # | Notebook | Now | Why |
|---|---|---|---|
| 1 | Two different train/test splits; scalers, sequence normalisation and KMeans fitted on all rows | One grouped train/val/test split (70/15/15 by educator); every transform fitted on training rows | Test rows leaked into training, and models were compared on different test sets |
| 2 | No validation set; fixed epochs and trees | Early stopping on validation for XGBoost and fusion; best fusion weights restored | Prevent over-fitting; make results stable |
| 3 | Fusion trained twice; metrics from run 1, checkpoint from run 2 | Trained once; the evaluated weights are the saved weights; every run reloads its bundle and checks it reproduces the test predictions | The registered model had never been evaluated |
| 4 | Feature store and sequence builder defined twice | One version of each | Two versions returned different things |
| 5 | Routing compared risk scores with the AUC gates (0.85, 0.70) | Per-label thresholds tuned for F1 on validation; severity bands; the first "any gap" model and `UnifiedAIEngine` dropped | An AUC measures model quality; it is not a cut-off |
| 6 | Raw probabilities served | Isotonic calibration per label, fitted on validation | Confidence values are shown to admins and must mean what they say |
| 7 | DistilBERT (768 dims) on one template sentence | TF-IDF + SVD (16 dims) fitted on training texts by default; MiniLM optional | The template text carries no signal; 768 dims on 2,000 rows over-fits; DistilBERT is slow on CPU |
| 8 | DistilBERT sentiment on two fixed comments | Dropped; average rating and evaluation count kept | The two comments are a function of the rating, so sentiment duplicated it. Returns in Phase 2 with realistic comments |
| 9 | `GraphAttentionEncoder` | `WatchHistoryAttention` | There is no graph; it attends over an educator's own watch rows. A real graph is planned for Phase 5 |
| 10 | `shap.TreeExplainer` | XGBoost's built-in TreeSHAP (`pred_contribs`) | Same values, no extra dependency, no version clashes between shap and xgboost |
| 11 | Recommender metrics on raw bandit scores; disparity measured before the last re-assignment | Metrics on the served (fairness-constrained) list as well as raw; disparity measured after re-assignment | Report what people actually see |
| 12 | LinUCB inverted each matrix on every score | Sherman-Morrison updates of the inverse | Same result, much faster |
| 13 | Champion compared on macro AUC across runs, whatever the data | Promotion compares only runs on the same dataset (data hash); new data needs `--promote-on-new-data` | Scores on different datasets are not comparable |
| 14 | Salt hard-coded; `timezone` imported after use | Salt from `SMARTEP_ID_SALT`; imports at module top | Secret out of the code; fragile ordering removed |
| 15 | Fairness buckets from test-set quantiles | Bucket edges learned on training rows | Group definitions must not depend on the test data |

### Contracts

- Pydantic models in `smartep/contracts/` written from spec v3 §3.2 (evidence package) and §6 (AssessmentResult).
  The Classroom's Zod schema (`packages/utils/src/validation/assessment/evidence-package.ts`) did not exist yet,
  so the SMARTEP side is the first written copy; `schemas/*.json` are exported from it and must be compared
  with the Zod export once it lands.
- A gap without evidence is rejected by the model (mirrors The Classroom's 422).

### Phase 0 results and gate G0 (2026-11-18, Windows laptop, CPU)

| Run | Runtime | XGBoost macro AUC (test) | Fusion macro AUC (test) | Bundle reload check |
|---|---|---|---|---|
| `configs/train_xgb.yaml` (2,000 educators) | 29 s | 0.515 | — | pass |
| `configs/train_fusion.yaml` (2,000 educators) | 52 s | 0.515 | 0.501 (best epoch 4) | pass |

- AUC near 0.5 is the expected result on random labels and shows no test leakage. The notebook's
  reported AUCs were noise of the same kind.
- All gates other than G0 fail on this data, as expected; G2 targets are only meaningful on generator v2.
- Fairness flags (worst disparity 0.32) and a top-1 recommendation disparity of 0.29 after re-ranking are
  noise on 300 test rows with random labels. The re-ranker's 5-iteration cap (kept from the notebook) is
  not always enough to reach 0.10; revisit in Phase 6.
- About 10 s of each run is the first `import torch` (seeding) when the research extra is installed.
- **G0 passed:** one command runs end to end in under 10 minutes on CPU; ruff clean; 31 tests pass.

### Not changed yet (by design)

- Labels are still the 7 legacy topics and are still random. Phase 1 replaces the data source with
  evidence packages in schema v1.0 and the 9 gap areas.
- Fairness still uses `years_experience`, which the evidence package does not carry. Phase 1 switches to
  seniority, discipline and institution type.
- Single seed and single split. Phase 2 adds 5-fold grouped cross-validation and 5 seeds.
