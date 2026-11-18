# Data sheet: synthetic evidence packages, generator v2

Generator `2.0.0` · config `configs/generator.yaml` · dataset `data/synthetic_v2/415ad6b46559/` (config hash
`415ad6b46559`) · generated 2026-11-18 · rebuild with `python -m pipelines.generate_data --config configs/generator.yaml`.

**Purpose.** Train and test the SMARTEP gap model before real pilot data exists (workplan Phase 1). The data
tests the pipeline and the model's ability to recover a planted signal. It says nothing about real educators:
every rule below is an assumption by the author and must be checked against real packages in Phase 7.

## 1. What is in the dataset

| Item | Value |
|---|---|
| Packages | 5,100 (1,700 educators × 3 periods: Sep–Dec 2025, Jan–Apr 2026, May–Aug 2026) |
| Subsets | 4,599 `train_pool`; 501 `classroom_test` (whole educators, for The Classroom's end-to-end tests) |
| Shape | Evidence package schema v1.0; 12 plan weeks × 2 sessions = 24 lessons per package; ~33 KB per package |
| Courses | 8 technology courses (Python, Databases and SQL, Cloud and Docker, Networks, Web, Cybersecurity, DSA, ML) |
| Institutions | 12 synthetic institutions (university, TVET, college); names and refs are fictional |
| Labels | 9 gap areas (0/1) with severity low / medium / high, and ground-truth evidence per gap |
| Size on disk | 17 MB (gzip JSONL + Parquet) |

Files are listed in `smartep/synth/io.py`. `traits.parquet` holds the latent traits and is used only by the
oracle check; it must never be a model input.

## 2. How a package is generated

Each educator has six latent traits (standard normal; higher is better except workload pressure):

| Trait | Meaning | Correlations |
|---|---|---|
| content_mastery | knows the subject (CK) | +0.3 with pedagogical_skill |
| pedagogical_skill | explains and engages well (PK) | +0.3 with responsiveness |
| digital_skill | uses slides, video and platform tools (TK) | +0.4 with currency |
| workload_pressure | overloaded: cancels, posts late, skips | −0.3 with responsiveness |
| responsiveness | grades and answers questions promptly | |
| currency | keeps materials and examples up to date | |

For each period, the traits drift (sd 0.25). The **package is generated from the period traits plus
observation noise (sd 0.8)**, which stands for everything the evidence does not capture (a hard semester,
class size, a co-teacher). Labels use the period traits without that noise. This one setting controls how
hard the task is (section 4).

Rules that link traits (written t) to package fields; σ is the logistic function:

| Package field | Rule |
|---|---|
| Plan present | P(no plan) = σ(−3 + 1.5·workload); status approved 80%, submitted 12%, draft 8% |
| Plan item covered | P = σ(2 + 1.5·(mastery − difficulty) − 0.8·workload); difficulty rises from −1 (week 1) to +1 (week 12) |
| Session delivered | covered: σ(2.6 − 1.4·workload); not covered: 0.3; 2% status unknown; 4% of lessons co-taught |
| Notes posted | covered and σ(1.2 + 0.9·pedagogy − 0.6·workload); strong phrasing with σ(0.3 + mastery + 0.6·pedagogy) |
| Notes late | σ(−1 + 1.3·workload); late notes appear 1–10 days after the session |
| Slides | covered and σ(0.3 + digital + 0.3·pedagogy) |
| Documents | Poisson(1.2·σ(0.2 + 0.8·pedagogy − 0.4·workload)), at most 2 |
| Outdated examples | in notes, slides and documents with σ(base − 1.6·currency); outdated documents carry a 2009–2015 year |
| Video posted | required video: σ(0.4 + 1.3·digital); extra video: σ(−2 + 0.8·digital); hosted on upload/YouTube with σ(0.5 + digital) |
| Transcript | upload or YouTube, 90%; hesitant segments σ(−1 − 1.4·mastery); slide-reading σ(−1.2 − pedagogy) |
| Video viewing | viewers Binomial(enrolled, σ(0.3 + 0.5·pedagogy)); furthest point Beta with mean σ(−0.2 + 0.7·pedagogy + 0.6·digital) |
| Attendance | Binomial(recorded, σ(1 + 0.6·pedagogy + 0.3·responsiveness − 0.3·workload + noise)) |
| Grading | graded Poisson(1 + 4·σ(responsiveness − 0.3·workload)); turnaround log-normal around 60 h × e^(−0.45·resp + 0.35·workload) |
| Questions answered in 72 h | Binomial(asked, σ(0.6 + 1.3·responsiveness − 0.5·workload)) |
| Evaluations | 2 rounds; response rate σ(−0.4 + 0.5·pedagogy + 0.3·resp); fewer than 5 responses = below threshold, questions withheld |
| Likert answers | ordered logit on dimension quality (clarity, organisation, feedback, engagement, resources, overall) |
| Comments | 30% of respondents; dimension chosen at random; negative with σ(−0.7 − 1.2·quality) |

Code: `smartep/synth/generator.py` (rules), `phrases.py` (text banks), `catalogue.py` (courses, terms,
reskilling catalogue).

## 3. How labels are made

For each gap area, a deficiency score = period traits × unit-length weights + noise (sd 0.35). The top share
of packages (the prevalence below) are gaps. Then 10% of gaps are flipped to no gap, and the same number of
non-gaps are flipped to gap, to stand for assessor disagreement; prevalence is unchanged. Severity is the
tertile of the score's margin above the threshold; flipped-in gaps are low.

| Gap area | Weights (sign: + means more gap) | Prevalence |
|---|---|---|
| content_coverage | mastery −0.7, workload +0.5 | 0.20 |
| content_currency | currency −0.9, digital −0.3 | 0.20 |
| resources | pedagogy −0.5, workload +0.5, digital −0.3 | 0.25 |
| video_delivery | digital −0.9, pedagogy −0.3 | 0.25 |
| session_delivery | workload +0.9, responsiveness −0.2 | 0.15 |
| assessment_feedback | responsiveness −0.9, workload +0.3 | 0.25 |
| student_engagement | responsiveness −0.5, pedagogy −0.6, digital −0.2 | 0.20 |
| pedagogy | pedagogy −1.0, mastery −0.2 | 0.20 |
| technical_skill | mastery −0.9, digital −0.3 | 0.15 |

Ground-truth evidence is read back from the package by fixed rules (`smartep/synth/evidence.py`): uncovered
plan items, undelivered sessions, missing notes or videos, low-completion videos with the drop-off second,
slow grading, low attendance, outdated terms, hesitant transcript segments and matching negative comments.
Every gap has at least one item; when no record qualifies (mostly flipped labels), the item points at the
last evaluation round.

## 4. Gate G1 results

| Check | Result | Target | Pass |
|---|---|---|---|
| Schema | 5,100 of 5,100 packages validate | all | yes |
| Prevalence | 0.150 – 0.251 per area | 0.10 – 0.40 | yes |
| Evidence | 9,382 gaps, 0 without evidence, 0 invalid items | none missing | yes |
| Oracle (XGBoost on latent traits) | macro AUC 0.916 | ≥ 0.80 | yes |
| Noise (XGBoost on random features) | macro AUC 0.501 | 0.50 ± 0.05 | yes |
| Evidence probe (XGBoost on 37 structured package features) | macro AUC 0.827 | ≥ 0.65 (soft) | yes |

Per area (test split of the train pool, grouped by educator):

| Gap area | Oracle AUC | Evidence-probe AUC |
|---|---|---|
| content_coverage | 0.894 | 0.787 |
| content_currency | 0.915 | 0.854 |
| resources | 0.900 | 0.808 |
| video_delivery | 0.918 | 0.832 |
| session_delivery | 0.917 | 0.815 |
| assessment_feedback | 0.902 | 0.809 |
| student_engagement | 0.937 | 0.854 |
| pedagogy | 0.927 | 0.860 |
| technical_skill | 0.930 | 0.820 |

**Choosing the difficulty.** With no observation noise the structured probe reached 0.897, almost the oracle
ceiling, because 24 lessons per package average the noise away. The Phase 2 target (0.70) would then be
trivial and the Phase 5 fusion model would have no room to show a gain. Observation noise was tuned on
900-educator runs (0.5 → 0.857, 0.7 → 0.848, 0.9 → 0.773) and set to 0.8, which leaves a gap of about
0.09 between what the evidence shows and the ceiling. Text features (Phase 2) should close part of it.

## 5. Groups and fairness

Seniority, discipline and institution type are carried for the fairness audit and are **not** linked to any
trait by default, so any difference in gap rates is sampling noise. Largest gap-rate spread across groups:
seniority 0.043, institution type 0.040, discipline 0.060 (discipline follows the course; small groups such as
TVET, 396 packages, are noisy). Setting `bias.seniority_digital_effect` shifts digital skill by seniority, to
test that the audit detects an injected bias.

Mix: lecturer 38.8%, assistant lecturer 28.7%, senior lecturer 21.7%, professor 10.8%; institutions (by
educator) university 72.9%, college 17.2%, TVET 9.9% (12 institutions drawn from 60 / 15 / 25%).

## 6. Known limitations

- **Circularity.** The rules are the author's assumptions. A model that does well here has learned the
  generator. Real-data validation (Phase 7) is the only test of usefulness.
- **Text is templated.** Phrase banks are small, so text features will find exact phrases. Treat text
  results as an upper bound until real notes and comments are available.
- **Evidence rules share the generator's knowledge** (outdated-term list, hesitation markers). Evidence
  matching scores in Phase 2 are therefore optimistic.
- **Independence.** Gaps depend only on the current period; there is no learning across periods, and
  `previousPackages` is a list of refs only.
- **No real calendar effects** (public holidays, strikes, exam weeks) and no multi-educator courses
  beyond a 4% co-taught flag.
- Plan missing in 13% of packages, more than the 5–10% expected; heavy-tailed workload drives it.

## 7. Change log

| Version | Date | Change |
|---|---|---|
| 2.0.0 | 2026-11-18 | First version; observation noise 0.8 chosen as above |
