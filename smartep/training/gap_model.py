"""MVP gap model (model v0.1): multi-label XGBoost on evidence packages.

Evaluation design:
- Only the ``train_pool`` subset is used for development; ``classroom_test`` (whole educators) is an
  untouched holdout, scored once by the final model.
- Grouped 5-fold cross-validation by educator, repeated over 5 seeds. Inside each training fold a grouped
  15% validation split is used for early stopping, isotonic calibration, per-area thresholds and severity
  cut-points. Text features are fitted on the fold's training rows only.
- Reported numbers are computed on out-of-fold predictions (all train-pool rows, each predicted by a
  model that never saw that educator), averaged over seeds with their standard deviation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from smartep.config import config_hash, set_seeds
from smartep.contracts.gap_areas import GAP_AREAS
from smartep.data.splits import grouped_holdout, grouped_kfold
from smartep.evaluate.fairness import fairness_audit
from smartep.evaluate.metrics import label_metrics, macro
from smartep.explain.evidence_mapper import map_evidence
from smartep.features.package import package_feature_frame
from smartep.features.package_text import PackageTextFeaturizer
from smartep.models.calibration import LabelCalibrator
from smartep.models.thresholds import tune_severity_cuts, tune_thresholds
from smartep.models.xgb_multilabel import MultiLabelXGB
from smartep.registry.bundle import git_commit, library_versions, load_bundle, make_version, save_bundle
from smartep.registry.champion import promote_if_better
from smartep.synth.io import read_dataset

LABELS = list(GAP_AREAS)
FAIRNESS_GROUPS = ("seniority", "discipline", "institution_type")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FittedGapModel:
    labels: list[str]
    struct_names: list[str]
    text: PackageTextFeaturizer
    xgb: MultiLabelXGB
    calibrator: LabelCalibrator
    thresholds: np.ndarray
    severity_cuts: np.ndarray            # (n_labels, 2)
    feature_names: list[str] = field(default_factory=list)

    def features(self, struct_X: np.ndarray, packages: list[dict]) -> np.ndarray:
        return np.hstack([struct_X, self.text.transform(packages)]).astype(np.float32)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P_raw = self.xgb.predict_proba(X)
        return P_raw, self.calibrator.transform(P_raw)

    @property
    def ranker(self) -> str:
        """Text-evidence ranking: semantic when the model already uses MiniLM, else TF-IDF."""
        return "minilm" if self.text.encoder == "minilm" else "tfidf"

    def decide(self, P_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """0/1 gap decisions and severity strings ('' = no gap)."""
        pred = (P_cal >= self.thresholds).astype(int)
        c1, c2 = self.severity_cuts[:, 0], self.severity_cuts[:, 1]
        sev = np.where(P_cal >= c2, "high", np.where(P_cal >= c1, "medium", "low"))
        return pred, np.where(pred == 1, sev, "")


def fit_gap_model(struct_X, struct_names, packages, Y, severity, train_idx, val_idx, cfg, seed) -> FittedGapModel:
    tcfg = cfg["features"]["text"]
    text = PackageTextFeaturizer(tcfg["encoder"], int(tcfg["dim"]), seed).fit([packages[i] for i in train_idx])

    def X_of(idx):
        return np.hstack([struct_X[idx], text.transform([packages[i] for i in idx])]).astype(np.float32)

    X_tr, X_val = X_of(train_idx), X_of(val_idx)
    params = dict(cfg["models"]["xgb"])
    esr = params.pop("early_stopping_rounds")
    xgb = MultiLabelXGB(LABELS, params, esr, seed).fit(X_tr, Y[train_idx], X_val, Y[val_idx])
    P_val_raw = xgb.predict_proba(X_val)
    calibrator = LabelCalibrator(cfg["calibration"]["method"]).fit(P_val_raw, Y[val_idx])
    P_val = calibrator.transform(P_val_raw)
    thresholds = tune_thresholds(P_val, Y[val_idx])
    cuts = np.array([tune_severity_cuts(P_val[:, j], severity[val_idx, j], thresholds[j]) for j in range(len(LABELS))])
    return FittedGapModel(LABELS, list(struct_names), text, xgb, calibrator, thresholds, cuts,
                          list(struct_names) + text.feature_names)


# --- evaluation helpers ------------------------------------------------------------------------------------

def severity_scores(true_sev: np.ndarray, pred_sev: np.ndarray) -> dict:
    """Exact-match accuracy of severity on true positives, against always predicting the majority level."""
    mask = (true_sev != "") & (pred_sev != "")
    if not mask.any():
        return {"n": 0, "accuracy": None, "majority_baseline": None}
    truth, pred = true_sev[mask], pred_sev[mask]
    _, counts = np.unique(truth, return_counts=True)
    return {"n": int(mask.sum()), "accuracy": round(float((truth == pred).mean()), 4),
            "majority_baseline": round(float(counts.max() / counts.sum()), 4)}


def evidence_scores(model: FittedGapModel, X: np.ndarray, packages: list[dict], pred: np.ndarray, Y: np.ndarray,
                    truth: dict[str, dict[str, set]], max_per_area: int | None = None, seed: int = 0) -> dict:
    """On true positives: share with at least one ground-truth record (hit rate) and share of items that match.

    ``max_per_area`` scores a fixed random sample of predicted gaps per area (evidence mapping is the slow step).
    """
    hits = matched = items = gaps = 0
    with_evidence = predicted = 0
    per_area: dict[str, list[int]] = {}
    rng = np.random.default_rng(seed)
    for j, area in enumerate(LABELS):
        rows = np.flatnonzero(pred[:, j] == 1)
        if max_per_area is not None and len(rows) > max_per_area:
            rows = np.sort(rng.choice(rows, max_per_area, replace=False))
        if not len(rows):
            continue
        contribs = model.xgb.contributions(X[rows], area)
        area_hits = [0, 0]
        for k, i in enumerate(rows):
            ev, _ = map_evidence(packages[i], area, contribs[k], model.feature_names, ranker=model.ranker)
            predicted += 1
            with_evidence += bool(ev)
            if Y[i, j] != 1:
                continue
            gt = truth[packages[i]["packageId"]].get(area, set())
            keys = {(e["source"], e["ref"]) for e in ev}
            gaps += 1
            hits += bool(keys & gt)
            matched += len(keys & gt)
            items += len(keys)
            area_hits[0] += bool(keys & gt)
            area_hits[1] += 1
        per_area[area] = area_hits
    return {"predicted_gaps": predicted, "with_evidence": with_evidence,
            "coverage": round(with_evidence / predicted, 4) if predicted else None,
            "true_positive_gaps": gaps,
            "hit_rate": round(hits / gaps, 4) if gaps else None,
            "item_precision": round(matched / items, 4) if items else None,
            "hits_per_area": per_area}


def _ground_truth(evidence_rows: list[dict], packages: list[dict]) -> dict[str, dict[str, set]]:
    """Every record that qualifies as evidence for each true gap (not only the 3 stored in the dataset).

    A predicted item is correct if it is any qualifying record; the stored file keeps the first 3 only.
    Uses the generator's evidence rules, for evaluation only (never at prediction time).
    """
    from smartep.synth.evidence import RULES

    by_id = {p["packageId"]: p for p in packages}
    out = {}
    for row in evidence_rows:
        pkg = by_id[row["packageId"]]
        out[row["packageId"]] = {
            g["area"]: {(e["source"], e["ref"]) for e in [*RULES[g["area"]](pkg), *g["evidence"]]} for g in row["gaps"]}
    return out


def _fairness(Y, pred, index: pd.DataFrame, fcfg: dict, seed: int) -> dict:
    out = {}
    cols = ["label", "selection_disparity", "selection_ci", "equal_opportunity_diff", "equal_opportunity_ci", "flag"]
    for attr in FAIRNESS_GROUPS:
        audit = fairness_audit(Y, None, None, index[attr].to_numpy(), LABELS, fcfg["max_disparity"], pred=pred,
                               interval=True)
        out[attr] = {
            "worst_selection_disparity": round(float(audit["selection_disparity"].max()), 4),
            "worst_equal_opportunity_diff": round(float(audit["equal_opportunity_diff"].max()), 4),
            "failed_labels": audit.loc[audit["flag"] == "FAIL", "label"].tolist(),
            "warned_labels": audit.loc[audit["flag"] == "WARN", "label"].tolist(),
            "per_label": audit[cols].to_dict(orient="records"),
        }
    worst = max(max(v["worst_selection_disparity"], v["worst_equal_opportunity_diff"]) for v in out.values())
    n_fail = sum(len(v["failed_labels"]) for v in out.values())
    n_warn = sum(len(v["warned_labels"]) for v in out.values())
    return {"by_group": out, "worst_disparity": round(worst, 4), "n_fail": n_fail, "n_warn": n_warn}


def _metrics(Y, P_raw, P_cal, pred, pred_sev, true_sev) -> dict:
    per = label_metrics(Y, P_raw, P_cal, None, LABELS, pred=pred)
    return {"per_label": per, "macro_auc": macro(per, "auc"), "macro_pr_auc": macro(per, "pr_auc"),
            "macro_f1": macro(per, "f1"), "max_ece": max(m["ece_calibrated"] for m in per.values()),
            "severity": severity_scores(true_sev.ravel(), pred_sev.ravel())}


def _mean_sd(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)}


# --- cross-validation ---------------------------------------------------------------------------------------

def cross_validate(struct_X, struct_names, packages, Y, severity, index, truth, cfg, log) -> dict:
    ccfg = cfg["cv"]
    groups = index["educatorRef"].to_numpy()
    per_seed = []
    for s, seed in enumerate(ccfg["seeds"]):
        n = len(packages)
        P_raw, P_cal = np.zeros((n, len(LABELS))), np.zeros((n, len(LABELS)))
        pred, pred_sev = np.zeros((n, len(LABELS)), dtype=int), np.full((n, len(LABELS)), "", dtype=object)
        evidence = []
        for f, test_idx in enumerate(grouped_kfold(groups, ccfg["folds"], seed)):
            rest = np.setdiff1d(np.arange(n), test_idx)
            tr, va = grouped_holdout(groups, rest, ccfg["val_size"], seed * 100 + f)
            model = fit_gap_model(struct_X, struct_names, packages, Y, severity, tr, va, cfg, seed)
            X_te = model.features(struct_X[test_idx], [packages[i] for i in test_idx])
            P_raw[test_idx], P_cal[test_idx] = model.predict(X_te)
            pred[test_idx], pred_sev[test_idx] = model.decide(P_cal[test_idx])
            if s == 0:  # evidence mapping is costly; score it on the first seed
                evidence.append(evidence_scores(model, X_te, [packages[i] for i in test_idx], pred[test_idx],
                                                Y[test_idx], truth, max_per_area=ccfg.get("evidence_sample_per_area"),
                                                seed=seed + f))
        m = _metrics(Y, P_raw, P_cal, pred, pred_sev, severity)
        m["fairness"] = _fairness(Y, pred, index, cfg["fairness"], seed)
        if evidence:
            tp = sum(e["true_positive_gaps"] for e in evidence)
            by_area = {a: [sum(e["hits_per_area"].get(a, [0, 0])[k] for e in evidence) for k in (0, 1)] for a in LABELS}
            m["evidence"] = {
                "coverage": round(sum(e["with_evidence"] for e in evidence) / max(1, sum(e["predicted_gaps"] for e in evidence)), 4),
                "hit_rate": round(sum((e["hit_rate"] or 0) * e["true_positive_gaps"] for e in evidence) / max(1, tp), 4),
                "true_positive_gaps": tp,
                "hit_rate_per_area": {a: round(h / n, 4) if n else None for a, (h, n) in by_area.items()},
            }
        per_seed.append(m)
        log(f"cv seed {seed}: macro AUC {m['macro_auc']} · macro F1 {m['macro_f1']} · max ECE {m['max_ece']} "
            f"· worst disparity {m['fairness']['worst_disparity']}")

    summary = {
        "folds": ccfg["folds"], "seeds": ccfg["seeds"],
        "macro_auc": _mean_sd(m["macro_auc"] for m in per_seed),
        "macro_pr_auc": _mean_sd(m["macro_pr_auc"] for m in per_seed),
        "macro_f1": _mean_sd(m["macro_f1"] for m in per_seed),
        "max_ece": _mean_sd(m["max_ece"] for m in per_seed),
        "worst_disparity": _mean_sd(m["fairness"]["worst_disparity"] for m in per_seed),
        "fairness_fail_flags": [m["fairness"]["n_fail"] for m in per_seed],
        "fairness_warn_flags": [m["fairness"]["n_warn"] for m in per_seed],
        "severity_accuracy": _mean_sd(m["severity"]["accuracy"] for m in per_seed),
        "severity_majority_baseline": per_seed[0]["severity"]["majority_baseline"],
        "per_label": {lab: {k: _mean_sd(m["per_label"][lab][k] for m in per_seed)
                            for k in ("auc", "pr_auc", "precision", "recall", "f1", "ece_calibrated")}
                      for lab in LABELS},
        "evidence": per_seed[0].get("evidence"),
        "fairness_seed0": per_seed[0]["fairness"],
    }
    return summary


def check_gates(cv: dict, holdout: dict, targets: dict) -> dict:
    label_auc_min = min(v["auc"]["mean"] for v in cv["per_label"].values())
    checks = {
        "macro_auc": (cv["macro_auc"]["mean"], ">=", targets["macro_auc"]),
        "min_label_auc": (label_auc_min, ">=", targets["min_label_auc"]),
        "max_ece": (cv["max_ece"]["mean"], "<=", targets["max_ece"]),
        "evidence_coverage": (holdout["evidence"]["coverage"], ">=", targets["evidence_coverage"]),
        "results_schema_valid": (holdout["results_valid_share"], ">=", 1.0),
        # A label/group pair fails only when its bootstrap interval lies wholly above the limit.
        "fairness_fail_flags": (max(cv["fairness_fail_flags"]), "<=", 0),
    }
    return {k: {"value": v, "target": f"{op} {t}", "pass": bool(v is not None and (v >= t if op == ">=" else v <= t))}
            for k, (v, op, t) in checks.items()}


# --- the run ------------------------------------------------------------------------------------------------

def run_gap_model(cfg: dict, output_dir: Path, promote_on_new_data: bool = False,
                  promote_reason: str | None = None) -> dict:
    from smartep.serve.predict import GapPredictor

    t0 = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    set_seeds(int(cfg["seed"]))
    cfg_hash = config_hash(cfg)
    version = make_version(cfg["run_name"], cfg_hash)
    data_dir = Path(cfg["data"]["dir"])
    ds = read_dataset(data_dir)
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    order = ds.index["packageId"].tolist()
    labels = ds.labels.set_index("packageId").loc[order, LABELS].to_numpy(dtype=np.float32)
    severity = ds.severity.set_index("packageId").loc[order, LABELS].to_numpy(dtype=object)
    pool = (ds.index["subset"] == "train_pool").to_numpy()
    hold = ~pool
    truth = _ground_truth(ds.evidence, ds.packages)
    cache = cfg["features"]["text"].get("cache_path")
    if cache and cfg["features"]["text"]["encoder"] == "minilm":
        from smartep.features.package_text import load_embedding_cache

        log(f"embedding cache: {load_embedding_cache(Path(cache).expanduser())} texts loaded")
    log(f"data: {len(order)} packages ({int(pool.sum())} train pool, {int(hold.sum())} holdout) from {data_dir}")

    struct = package_feature_frame(ds.packages)
    struct_names = list(struct.columns)
    struct_X = struct.to_numpy(dtype=np.float32)
    log(f"structured features: {len(struct_names)}")

    pool_idx, hold_idx = np.flatnonzero(pool), np.flatnonzero(hold)
    pool_pkgs = [ds.packages[i] for i in pool_idx]
    index_pool = ds.index.iloc[pool_idx].reset_index(drop=True)
    cv = cross_validate(struct_X[pool_idx], struct_names, pool_pkgs, labels[pool_idx], severity[pool_idx],
                        index_pool, truth, cfg, log)
    log(f"cv: macro AUC {cv['macro_auc']} · F1 {cv['macro_f1']}")
    if cache and cfg["features"]["text"]["encoder"] == "minilm":
        from smartep.features.package_text import save_embedding_cache

        save_embedding_cache(Path(cache).expanduser())

    # Final model on the whole train pool (grouped validation split for early stopping and calibration).
    groups_pool = index_pool["educatorRef"].to_numpy()
    tr, va = grouped_holdout(groups_pool, np.arange(len(pool_idx)), cfg["cv"]["val_size"], int(cfg["seed"]))
    final = fit_gap_model(struct_X[pool_idx], struct_names, pool_pkgs, labels[pool_idx], severity[pool_idx],
                          tr, va, cfg, int(cfg["seed"]))
    hold_pkgs = [ds.packages[i] for i in hold_idx]
    X_hold = final.features(struct_X[hold_idx], hold_pkgs)
    Ph_raw, Ph_cal = final.predict(X_hold)
    ph_pred, ph_sev = final.decide(Ph_cal)
    holdout = _metrics(labels[hold_idx], Ph_raw, Ph_cal, ph_pred, ph_sev, severity[hold_idx])
    holdout["evidence"] = evidence_scores(final, X_hold, hold_pkgs, ph_pred, labels[hold_idx], truth)
    log(f"holdout: macro AUC {holdout['macro_auc']} · evidence hit rate {holdout['evidence']['hit_rate']}")
    if cache and cfg["features"]["text"]["encoder"] == "minilm":
        save_embedding_cache(Path(cache).expanduser())  # now includes the holdout texts

    # Bundle
    data_hash = data_manifest["config_hash"]
    manifest = {"version": version, "model_name": cfg["model_name"], "config_hash": cfg_hash, "data_hash": data_hash,
                "data_dir": str(data_dir), "generator_version": data_manifest["generator_version"],
                "git_commit": git_commit(REPO_ROOT), "labels": LABELS, "feature_names": final.feature_names,
                "libraries": library_versions(), "training_data": "synthetic", "validated": False}
    bundle_dir = output_dir / version
    preprocess = {"kind": "gap_model_v0", "struct_names": struct_names, "text": final.text,
                  "calibrator": final.calibrator, "thresholds": final.thresholds,
                  "severity_cuts": final.severity_cuts, "feature_names": final.feature_names, "labels": LABELS}
    save_bundle(bundle_dir, manifest=manifest, config=cfg, metrics={}, xgb_model=final.xgb, preprocess=preprocess)
    reloaded = load_bundle(bundle_dir).predict_proba(X_hold)
    reload_ok = bool(np.allclose(reloaded, Ph_cal, atol=1e-6))
    if not reload_ok:
        raise RuntimeError("saved bundle does not reproduce the evaluated predictions")

    # AssessmentResult for every holdout package, validated against the contract; keep a few as examples.
    predictor = GapPredictor.from_bundle(bundle_dir)
    results, valid = predictor.predict_packages(hold_pkgs), 0
    examples_dir = bundle_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    for k, res in enumerate(results):
        valid += res is not None
        if k < cfg["output"].get("n_examples", 5):
            (examples_dir / f"{res.packageId}.result.json").write_text(res.model_dump_json(indent=2, exclude_none=True),
                                                                       encoding="utf-8")
    holdout["results_valid_share"] = round(valid / len(results), 4)
    gates = check_gates(cv, holdout, cfg["gates"])

    promoted, reason = promote_if_better(output_dir / "registry", {
        "version": version, "bundle_path": str(bundle_dir), "macro_auc": cv["macro_auc"]["mean"],
        "fairness_ok": gates["fairness_fail_flags"]["pass"], "data_hash": data_hash,
    }, allow_new_data=promote_on_new_data,
        override_reason=promote_reason if all(g["pass"] for g in gates.values()) else None)
    summary = {"version": version, "config_hash": cfg_hash, "data_hash": data_hash, "manifest": manifest,
               "data": {"dir": str(data_dir), "train_pool": int(pool.sum()), "holdout": int(hold.sum())},
               "cv": cv, "holdout": holdout, "final_model": {"thresholds": dict(zip(LABELS, final.thresholds.round(3).tolist(), strict=True)),
                                                             "severity_cuts": dict(zip(LABELS, final.severity_cuts.round(3).tolist(), strict=True)),
                                                             "best_iterations": {k: int(m.best_iteration) for k, m in final.xgb.models.items()}},
               "top_features": _top_features(final, X_hold), "gates": gates, "bundle_reload_check": reload_ok,
               "registry": {"promoted": promoted, "reason": reason}, "runtime_seconds": round(time.perf_counter() - t0, 1)}
    (bundle_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (bundle_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    log(f"bundle {bundle_dir} · gates {'PASS' if all(g['pass'] for g in gates.values()) else 'FAIL'} · promoted {promoted} ({reason})")
    return summary


def _top_features(model: FittedGapModel, X: np.ndarray, k: int = 12) -> list[list]:
    imp = np.mean([np.abs(model.xgb.contributions(X, lab)).mean(axis=0) for lab in LABELS], axis=0)
    order = np.argsort(-imp)[:k]
    return [[model.feature_names[i], round(float(imp[i]), 4)] for i in order]


def render_report(s: dict) -> str:
    def ms(d):
        return f"{d['mean']:.3f} ± {d['sd']:.3f}"

    cv, ho = s["cv"], s["holdout"]
    lines = [
        f"# Gap model report: {s['version']}", "",
        f"- Data: `{s['data']['dir']}` (generator v2; {s['data']['train_pool']} train-pool packages, {s['data']['holdout']} holdout)",
        f"- Evaluation: grouped {cv['folds']}-fold CV by educator × seeds {cv['seeds']}; holdout scored once by the final model",
        f"- Config `{s['config_hash']}` · git `{s['manifest']['git_commit']}` · runtime {s['runtime_seconds']} s",
        f"- Bundle reload check: **{s['bundle_reload_check']}** · results valid against the contract: **{ho['results_valid_share']:.0%}**",
        "", "> Synthetic data: these numbers show the model recovers the generator's planted signal. They are not",
        "> evidence of usefulness with real educators (model card, limitations).", "",
        "## Gates (G2)", "", "| Gate | Value | Target | Pass |", "|---|---|---|---|",
        *[f"| {k} | {g['value']} | {g['target']} | {g['pass']} |" for k, g in s["gates"].items()], "",
        "## Cross-validation (out-of-fold, mean ± sd over seeds)", "",
        f"Macro AUC **{ms(cv['macro_auc'])}** · macro PR-AUC {ms(cv['macro_pr_auc'])} · macro F1 {ms(cv['macro_f1'])} · "
        f"max ECE {ms(cv['max_ece'])}", "",
        "| Gap area | AUC | PR-AUC | Precision | Recall | F1 | ECE |", "|---|---|---|---|---|---|---|",
        *[f"| {lab} | {ms(v['auc'])} | {ms(v['pr_auc'])} | {ms(v['precision'])} | {ms(v['recall'])} | {ms(v['f1'])} | {ms(v['ece_calibrated'])} |"
          for lab, v in cv["per_label"].items()], "",
        f"Severity accuracy on true positives {ms(cv['severity_accuracy'])} (always-'low' baseline {cv['severity_majority_baseline']}).", "",
        f"Evidence (seed {cv['seeds'][0]}): coverage {cv['evidence']['coverage']}, hit rate {cv['evidence']['hit_rate']} "
        f"on {cv['evidence']['true_positive_gaps']} true-positive gaps (share with at least one ground-truth record).", "",
        "| Gap area | Evidence hit rate |", "|---|---|",
        *[f"| {a} | {h} |" for a, h in cv["evidence"]["hit_rate_per_area"].items()], "",
        "## Fairness (out-of-fold, seed " + str(cv["seeds"][0]) + ")", "",
        "| Group | Worst selection-rate disparity | Worst equal-opportunity difference | FAIL (CI above 0.10) | WARN (point above 0.10) |",
        "|---|---|---|---|---|",
        *[f"| {g} | {v['worst_selection_disparity']} | {v['worst_equal_opportunity_diff']} | {', '.join(v['failed_labels']) or 'none'} | {', '.join(v['warned_labels']) or 'none'} |"
          for g, v in cv["fairness_seed0"]["by_group"].items()], "",
        f"Worst point disparity over seeds: {ms(cv['worst_disparity'])}; FAIL flags per seed {cv['fairness_fail_flags']}; "
        f"WARN flags per seed {cv['fairness_warn_flags']}.", "",
        "## Holdout (classroom_test, final model)", "",
        f"Macro AUC **{ho['macro_auc']}** · macro F1 {ho['macro_f1']} · max ECE {ho['max_ece']} · "
        f"severity accuracy {ho['severity']['accuracy']} · evidence hit rate {ho['evidence']['hit_rate']} · "
        f"evidence item precision {ho['evidence']['item_precision']}", "",
        "## Top features (mean |SHAP| on holdout, averaged over areas)", "", "| Feature | mean abs SHAP |", "|---|---|",
        *[f"| {f} | {v} |" for f, v in s["top_features"]], "",
        "## Registry", "", f"Promoted: **{s['registry']['promoted']}** ({s['registry']['reason']})", "",
    ]
    return "\n".join(lines)
