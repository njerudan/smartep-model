# Decision log

One entry per design decision: what was decided, and why. Feeds the methodology chapter.

## 2026-11-18 · Phase 3: serving (SMARTEP API)

Contract for The Classroom: `docs/api.md`. Operations: `docs/runbook-serving.md`.

**Found at the start:** the M1 ingest API was never built (`api.smartep.co.ke` has no DNS; the Dokploy
Application "smartep.co.ke" is idle), and The Classroom has no assessment code yet (no package builder,
webhook or results endpoint). Phase 3 therefore builds the whole SMARTEP side and tests it end to end
against a stand-in Classroom (`smartep/api/fake_classroom.py`); the real end-to-end check waits for The
Classroom's endpoints.

| Decision | Why |
|---|---|
| Ingest API, queue, worker and callback in this repo, one container | One deployable for a solo developer; the worker shares the loaded model with the web server |
| SQLite job queue on a volume, no Redis | 8 GB server, low volume (a few hundred packages per period); survives restarts |
| Webhook signature: HMAC-SHA256 over `"{timestamp}.{body}"`, ±5 min tolerance, per-org secret; headers `X-SMARTEP-Org/Timestamp/Signature` | The spec did not define signing; this is the common webhook pattern and blocks replays |
| Same 401 for unknown org and bad signature | Does not reveal which organisations exist |
| Per-org allow-lists for download and callback hosts, https only | A package or webhook cannot steer SMARTEP to an arbitrary URL or leak the org API key |
| Verify sha256, byte size, schema v1.0, packageId and orgRef before predicting | Spec §5; refuses tampered or misrouted packages |
| Permanent vs temporary failures; retries 30 s → 6 h until 24 h after acceptance | Spec: results within 24 h; 4xx other than 408/429 will not fix itself |
| Package never stored; result body deleted after delivery; httpx URL logging off | Data minimisation (spec §12); presigned URLs carry access signatures |
| Champion bundle committed to `models/gap-model/` (2.4 MB) and baked into the image | Reproducible deploys without a model store; `scripts/export_model.py` refreshes it |
| Golden fixtures (8 packages + expected probabilities) and a quality check in tests | A change that alters predictions or ships a weaker bundle fails CI |
| CI builds the image and smoke-tests the container | Docker is not installed locally |

**G3 (local part) passed:** 50 holdout packages end to end, all delivered and valid; processing p50 70 ms,
p95 141 ms, max 215 ms (target < 30 s); peak memory 241 MB (target < 1.5 GB); webhook ack 13 ms median.
**Deployed 2026-11-18** (user approved): Dokploy Application "SMARTEP API" at https://api.smartep.co.ke from
commit `40c7ee1`, Dockerfile build, volume `/data`, no organisations configured yet. Checks: `/health` 200
with model `gap-xgb-20261118T130130Z-3f9573`; unsigned webhook 401; status endpoint 401 without token;
HTTP → HTTPS 301. A new application was created rather than reusing the idle "smartep.co.ke" one, which is
left for the product site. Auto-deploy is off: a push does not change production until it is redeployed.
**Still open for G3:** the end-to-end test with The Classroom, once its package webhook and results
endpoint exist.

## 2026-11-18 · MVP drops MiniLM text features

**Decision (Daniel Njeru):** serve the MVP without MiniLM; use TF-IDF + SVD text features instead
(`features.text.encoder: tfidf_svd`). MiniLM stays in the code as an option.

**Why:** on synthetic data MiniLM added no discrimination (no-text 0.810 vs MiniLM 0.811 macro AUC), while it
needs a 90 MB model in memory on the 8 GB CPU server and slowed the first run by ~18 minutes. This matches
item 3 of the MVP cut list.

**Result:** the TF-IDF model is as good or better on every measure (CV macro AUC 0.811 ± 0.001, holdout 0.820,
worst ECE 0.033, evidence hit rate 0.765 vs 0.744, 0 fairness FAIL flags) and passes G2. Bundle
`gap-xgb-20261118T130130Z-3f9573`, promoted with the new `--promote-reason` override (a candidate that does
not score higher is promoted only if every gate passes; the reason is written to the registry history).
TF-IDF's edge on evidence probably comes from templated synthetic text; compare again on pilot data.

## 2026-11-18 · Phase 2: MVP gap model v0.1

Model card: `docs/model-card-v0.1.md`. First bundle `artifacts/gap-xgb-20261118T122125Z-d7dc5b` (MiniLM text);
superseded the same day by the TF-IDF bundle (entry above).

| Decision | Why |
|---|---|
| Grouped 5-fold CV × 5 seeds on the train pool, plus the 501-package `classroom_test` subset as an untouched holdout | Replaces the planned single 70/15/15 split: out-of-fold predictions cover all 4,599 rows (larger groups for fairness), seeds give a spread, and the holdout gives one clean estimate |
| Early stopping, isotonic calibration, thresholds and severity cut-points fitted on a grouped 15% validation split inside each training fold | Nothing tuned on the rows it is scored on |
| Hyperparameters fixed (depth 3, lr 0.05, ≤ 600 trees with early stopping), no grid search | CV sd is 0.001 and early stopping sets tree counts; a grid would add little and, without nesting, bias the estimate |
| Text: MiniLM embeddings of comments, notes and transcripts (16 PCA components each); cached to `~/.cache/smartep/` | Workplan choice; caching cuts re-runs from ~20 min to a few minutes |
| Seniority, discipline and institution type are not model inputs | Kept only for the fairness audit |
| Evidence: SHAP drivers mapped to package records, **area-relevant families first** (e.g. coverage → plan items) | First version ranked by SHAP alone and cited correlated but irrelevant records (coverage hit rate 0.00); relevance follows from the area definitions, not from generator rules |
| Evidence scored against **all** qualifying records, not the 3 stored per gap | The stored list is truncated; a different lesson that also lacks notes is correct evidence |
| Fairness flag from a **95% interval on each pairwise group difference**; FAIL only if the interval lies wholly above 0.10, WARN on the point estimate | The max-minus-min spread grows with noise in small groups (TVET ~70 positives per area, SE ≈ 0.06), so the point rule failed an unbiased generator; a bootstrap of the spread is biased upward too |
| Results built and validated as AssessmentResult in `smartep/serve/predict.py` | Same code path as Phase 3 serving; an invalid result raises instead of being sent |
| Catalogue matcher: skill-tag overlap (0.7) + TF-IDF similarity (0.3), filtered by preferred mode and budget | Simple, explainable, and cannot return a catalogueRef outside the snapshot |

**Results (G2 passed).** CV macro AUC 0.811 ± 0.001 (holdout 0.821); lowest area 0.778; worst-area ECE
0.032; evidence coverage 100%, hit rate 0.744 (holdout 0.776); all 501 holdout results valid; 0 fairness
FAIL flags in all seeds (4–10 WARN, small groups). Severity is weak: 0.424 exact match vs 0.395 baseline.

**Experiments.**

| Run | Macro AUC (CV, seed 42) | Worst ECE | Evidence hit rate | Fairness |
|---|---|---|---|---|
| v0.1 (MiniLM text) | 0.809 | 0.033 | 0.744 | 0 FAIL, 4 WARN |
| No text features | 0.810 | 0.040 | 0.700 | 0 FAIL, 3 WARN |
| Injected bias (`seniority_digital_effect: 1.0`, TF-IDF text) | 0.820 | 0.033 | 0.791 | **3 FAIL**, all seniority: content_currency, resources, video_delivery |

- Text features add no discrimination on this synthetic data and only a little calibration and evidence
  quality. Open question for the MVP: keep MiniLM (90 MB model, slower first run) or serve without it
  (workplan cut list item 3).
- The audit FAILs exactly where bias was injected (digital skill → the three digital-skill areas, for
  seniority only) and stays clean for discipline and institution type, so the interval rule still detects
  a real disparity.

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
