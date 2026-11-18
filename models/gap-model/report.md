# Gap model report: gap-xgb-20261118T130130Z-3f9573

- Data: `data\synthetic_v2\415ad6b46559` (generator v2; 4599 train-pool packages, 501 holdout)
- Evaluation: grouped 5-fold CV by educator × seeds [42, 43, 44, 45, 46]; holdout scored once by the final model
- Config `3f957395f194` · git `9f829d5-dirty` · runtime 619.6 s
- Bundle reload check: **True** · results valid against the contract: **100%**

> Synthetic data: these numbers show the model recovers the generator's planted signal. They are not
> evidence of usefulness with real educators (model card, limitations).

## Gates (G2)

| Gate | Value | Target | Pass |
|---|---|---|---|
| macro_auc | 0.8113 | >= 0.7 | True |
| min_label_auc | 0.7797 | >= 0.65 | True |
| max_ece | 0.0326 | <= 0.05 | True |
| evidence_coverage | 1.0 | >= 1.0 | True |
| results_schema_valid | 1.0 | >= 1.0 | True |
| fairness_fail_flags | 0 | <= 0 | True |

## Cross-validation (out-of-fold, mean ± sd over seeds)

Macro AUC **0.811 ± 0.001** · macro PR-AUC 0.537 ± 0.002 · macro F1 0.543 ± 0.004 · max ECE 0.033 ± 0.004

| Gap area | AUC | PR-AUC | Precision | Recall | F1 | ECE |
|---|---|---|---|---|---|---|
| content_coverage | 0.780 ± 0.003 | 0.467 ± 0.006 | 0.433 ± 0.019 | 0.612 ± 0.030 | 0.506 ± 0.007 | 0.022 ± 0.004 |
| content_currency | 0.816 ± 0.001 | 0.554 ± 0.004 | 0.481 ± 0.021 | 0.628 ± 0.036 | 0.544 ± 0.008 | 0.025 ± 0.006 |
| resources | 0.815 ± 0.002 | 0.609 ± 0.005 | 0.509 ± 0.013 | 0.659 ± 0.037 | 0.574 ± 0.011 | 0.030 ± 0.004 |
| video_delivery | 0.801 ± 0.001 | 0.579 ± 0.006 | 0.533 ± 0.030 | 0.656 ± 0.038 | 0.587 ± 0.008 | 0.029 ± 0.006 |
| session_delivery | 0.820 ± 0.001 | 0.459 ± 0.005 | 0.415 ± 0.011 | 0.554 ± 0.047 | 0.473 ± 0.011 | 0.018 ± 0.003 |
| assessment_feedback | 0.804 ± 0.001 | 0.585 ± 0.010 | 0.498 ± 0.017 | 0.706 ± 0.034 | 0.583 ± 0.006 | 0.025 ± 0.006 |
| student_engagement | 0.841 ± 0.001 | 0.586 ± 0.002 | 0.512 ± 0.021 | 0.652 ± 0.040 | 0.573 ± 0.006 | 0.021 ± 0.003 |
| pedagogy | 0.815 ± 0.002 | 0.546 ± 0.004 | 0.478 ± 0.021 | 0.671 ± 0.030 | 0.558 ± 0.006 | 0.020 ± 0.005 |
| technical_skill | 0.809 ± 0.003 | 0.445 ± 0.009 | 0.441 ± 0.011 | 0.554 ± 0.030 | 0.490 ± 0.006 | 0.021 ± 0.005 |

Severity accuracy on true positives 0.426 ± 0.008 (always-'low' baseline 0.3942).

Evidence (seed 42): coverage 1.0, hit rate 0.765 on 3183 true-positive gaps (share with at least one ground-truth record).

| Gap area | Evidence hit rate |
|---|---|
| content_coverage | 0.5968 |
| content_currency | 0.5921 |
| resources | 0.2285 |
| video_delivery | 0.8978 |
| session_delivery | 0.9748 |
| assessment_feedback | 0.9429 |
| student_engagement | 0.9894 |
| pedagogy | 0.9424 |
| technical_skill | 0.6837 |

## Fairness (out-of-fold, seed 42)

| Group | Worst selection-rate disparity | Worst equal-opportunity difference | FAIL (CI above 0.10) | WARN (point above 0.10) |
|---|---|---|---|---|
| seniority | 0.0568 | 0.1493 | none | pedagogy, technical_skill |
| discipline | 0.0722 | 0.0757 | none | none |
| institution_type | 0.0521 | 0.1782 | none | content_currency, resources, session_delivery, assessment_feedback, technical_skill |

Worst point disparity over seeds: 0.166 ± 0.009; FAIL flags per seed [0, 0, 0, 0, 0]; WARN flags per seed [7, 3, 8, 4, 6].

## Holdout (classroom_test, final model)

Macro AUC **0.8195** · macro F1 0.5397 · max ECE 0.0569 · severity accuracy 0.4211 · evidence hit rate 0.7867 · evidence item precision 0.4504

## Top features (mean |SHAP| on holdout, averaged over areas)

| Feature | mean abs SHAP |
|---|---|
| likert_resources_mean | 0.1152 |
| likert_feedback_mean | 0.1045 |
| attendance_rate | 0.0963 |
| notes_late_share | 0.0953 |
| slides_share | 0.0892 |
| grading_turnaround_hours | 0.0885 |
| evaluation_response_rate | 0.0814 |
| text_notes_9 | 0.0792 |
| text_notes_8 | 0.0753 |
| contact_minutes_ratio | 0.075 |
| likert_organisation_mean | 0.0725 |
| likert_clarity_mean | 0.0664 |

## Registry

Promoted: **True** (manual promotion: MVP drops MiniLM: no AUC gain on synthetic data, lighter on the 8 GB server)
