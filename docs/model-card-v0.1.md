# Model card: SMARTEP gap model v0.1

Bundle `artifacts/gap-xgb-20261118T130130Z-3f9573` · config `configs/gap_model.yaml` (TF-IDF text) ·
data `data/synthetic_v2/415ad6b46559` (generator 2.0.0) · trained 2026-11-18 · registry champion
(manual promotion: MVP drops MiniLM). The MiniLM variant (`gap-xgb-20261118T122125Z-d7dc5b`) is kept for
comparison.

> **Unvalidated.** Trained and tested on synthetic data only. Every result says `trainingData: synthetic`,
> `validated: false`, and The Classroom shows the unvalidated-model banner. Do not use it to make decisions
> about real educators until Phase 7 validation is passed.

## Intended use

- **Does:** reads one evidence package (one educator, course and period) and suggests possible gaps in nine
  areas, each with a severity, a calibrated confidence and evidence pointing to plan items, lessons, videos
  (with timestamps) and evaluation questions; suggests courses from the package's reskilling catalogue.
- **For:** an institution admin, who reviews every result before an educator sees it (spec §8).
- **Not for:** appraisal, promotion, discipline or any automated decision; comparing educators with each
  other; packages from contexts unlike the training data (non-technology courses, non-Kenyan calendars).

## Model

| Part | What it is |
|---|---|
| Inputs | 36 structured features from package fields (delivery, coverage, notes, videos, grading, questions, attendance, evaluation ratings) + 48 text features (TF-IDF + truncated SVD of comments, notes and transcripts, 16 components each, fitted on training packages) |
| Not inputs | Seniority, discipline, institution type (kept for the fairness audit) |
| Classifier | One XGBoost model per gap area (depth 3, learning rate 0.05, early stopping on a grouped validation split) |
| Calibration | Isotonic per area, on the validation split |
| Decision | Per-area threshold maximising F1 on validation; severity cut-points tuned on validation |
| Evidence | Top SHAP drivers mapped to package records, area-relevant records first (`smartep/explain/evidence_mapper.py`); text records ranked by TF-IDF similarity to the area description |
| Reskilling | Catalogue matcher: skill-tag overlap + TF-IDF similarity, filtered by preferred mode and budget; only catalogueRefs from the snapshot |
| Dependencies | scikit-learn and XGBoost only; no model download, no torch |

## Evaluation

Grouped 5-fold cross-validation by educator, repeated over 5 seeds, on the 4,599 train-pool packages
(out-of-fold predictions); then the final model scored once on 501 held-out packages from other educators.

| Measure | CV (mean ± sd over seeds) | Holdout | G2 target |
|---|---|---|---|
| Macro AUC | 0.811 ± 0.001 | 0.820 | ≥ 0.70 |
| Lowest area AUC | 0.780 (content coverage) | — | ≥ 0.65 |
| Macro PR-AUC | 0.537 ± 0.002 | — | — |
| Macro F1 | 0.543 ± 0.004 | 0.540 | — |
| Worst-area calibration error (ECE) | 0.033 ± 0.004 | 0.057 | ≤ 0.05 (CV) |
| Gaps with evidence | 100% | 100% | 100% |
| Evidence hit rate (true gaps with ≥ 1 correct record) | 0.765 | 0.787 | reported |
| Results valid against the contract | — | 100% | 100% |
| Severity exact match on true positives | 0.426 ± 0.008 (baseline 0.394) | 0.421 | reported |
| Fairness FAIL flags | 0 in every seed | — | 0 |

Per area, CV AUC ranges from 0.780 (content coverage) to 0.841 (student engagement); evidence hit rates
range from 0.23 (resources) to 0.99 (student engagement). Full tables: the bundle's `report.md`.

**Text encoder comparison** (same data and CV):

| Text features | CV macro AUC | Worst ECE | Evidence hit rate (CV) | Holdout AUC |
|---|---|---|---|---|
| TF-IDF (this model) | 0.811 ± 0.001 | 0.033 | 0.765 | 0.820 |
| MiniLM | 0.811 ± 0.001 | 0.032 | 0.744 | 0.821 |
| None (seed 42 only) | 0.810 | 0.040 | 0.700 | 0.820 |

**Reference points.** On this data an oracle given the hidden traits reaches 0.916, and the structured
fields alone reached 0.827 in the G1 probe (simpler model, one split).

## Fairness

Audit on out-of-fold predictions by seniority, discipline and institution type, for selection rate and
equal opportunity (true-positive rate among real gaps). A pair of groups FAILs when its 95% interval for
the difference lies wholly above 0.10; WARN when only the point estimate does.

- 0 FAIL flags in all 5 seeds. Largest selection-rate disparities stay below 0.07.
- 3–8 WARN flags per seed, equal-opportunity differences in small groups (TVET, professors). The generator
  links no trait to any group, so these are expected noise.
- Injected-bias check: with digital skill tied to seniority in the generator, the audit FAILs for seniority
  on exactly the three digital-skill areas (content currency, resources, video delivery) and nowhere else.

## Limitations and known issues

1. **Synthetic only.** The model has learned the generator's rules (data sheet). Usefulness with real
   educators is unknown until Phase 7.
2. **Severity is weak:** 0.43 exact match against 0.39 for always saying "low", and cut-points tuned for
   accuracy put some confidences near 0.6 in "high". Treat severity as a rough ordering.
3. **Resources evidence is weak** (hit rate 0.23): the model's drivers for this area are mostly ratings and
   slide share, while the ground truth points at specific lessons without notes.
4. **Evidence scores are optimistic:** the ground truth comes from the same kind of rules as the mapper.
5. **Holdout calibration** (worst area ECE 0.057) is looser than CV; 501 packages is a small sample.
6. **Text** is templated in the synthetic data, which favours TF-IDF's exact-phrase matching. Real notes
   and comments vary far more; compare TF-IDF with MiniLM again on pilot data (Phase 7).
7. Text evidence (comments, transcripts) is ranked by word overlap with the area's description and can,
   rarely, quote a positive comment on the same topic.

## Responsible use

Admin review before sharing; educators can respond and disagree (spec §8); feedback is logged for Phase 7
validation. No personal identifiers are needed: packages carry pseudonymous refs only.
