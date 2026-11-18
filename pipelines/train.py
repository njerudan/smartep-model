"""Train, evaluate, bundle and register one model from a config.

    python -m pipelines.train --config configs/train_xgb.yaml
    python -m pipelines.train --config configs/train_xgb.yaml --set data.n_educators=300 --set seed=7

Every reported number comes from this run: one grouped split, transforms fitted on training rows only,
early stopping and calibration on validation rows, and metrics on the untouched test rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from smartep.config import config_hash, load_config, set_seeds
from smartep.data.splits import frame_hash, grouped_split
from smartep.evaluate.fairness import assign_groups, fairness_audit, fit_bucket_edges
from smartep.evaluate.gates import check_gates
from smartep.evaluate.metrics import hamming_loss, label_metrics, macro
from smartep.evaluate.personas import fit_personas
from smartep.evaluate.report import render_report
from smartep.explain.contributions import explain_row, global_importance
from smartep.features.tabular import TabularFeaturizer, build_legacy_tabular
from smartep.features.text import TextEncoder, aggregate_texts
from smartep.models.calibration import LabelCalibrator
from smartep.models.thresholds import tune_thresholds
from smartep.models.xgb_multilabel import MultiLabelXGB
from smartep.recommend.bandit import LinUCB, train_bandit
from smartep.recommend.fairness_rerank import fairness_constrained_recommend, served_rankings
from smartep.recommend.metrics import map_ndcg_at_k
from smartep.registry.bundle import git_commit, library_versions, load_bundle, make_version, save_bundle
from smartep.registry.champion import promote_if_better
from smartep.synth.legacy import LEGACY_TOPICS, generate_legacy_tables, label_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]


class Timer:
    def __init__(self):
        self.start = self.last = time.perf_counter()

    def step(self, name: str) -> None:
        now = time.perf_counter()
        print(f"[{now - self.start:7.1f}s] {name} ({now - self.last:.1f}s)", flush=True)
        self.last = now

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self.start, 1)


def _decisions(P_val_raw, Y_val, method):
    calibrator = LabelCalibrator(method).fit(P_val_raw, Y_val)
    thresholds = tune_thresholds(calibrator.transform(P_val_raw), Y_val)
    return calibrator, thresholds


def _evaluate(Y, P_raw, P_cal, thresholds, labels) -> dict:
    per_label = label_metrics(Y, P_raw, P_cal, thresholds, labels)
    return {"per_label": per_label, "macro_auc": macro(per_label, "auc"), "macro_pr_auc": macro(per_label, "pr_auc"),
            "macro_f1": macro(per_label, "f1"), "hamming_loss": hamming_loss(Y, P_cal, thresholds)}


def _train_fusion(cfg, tables, ids, split, X_tab, X_text, Y, labels, seed):
    from smartep.features.sequence import SequenceNormalizer, build_weekly_tensor
    from smartep.features.watch_history import build_watch_history
    from smartep.models.fusion import predict_fusion, train_fusion

    fcfg = cfg["features"]
    seq_raw = build_weekly_tensor(tables.lms_logs, ids, fcfg["sequence"]["seq_len_weeks"])
    seq = SequenceNormalizer().fit(seq_raw[split.train]).transform(seq_raw)
    watch, mask = build_watch_history(tables.video_watch_logs, tables.video_catalog, ids, LEGACY_TOPICS,
                                      fcfg["watch_history"]["max_neighbors"])

    def inputs(idx):
        return {"tab": X_tab[idx], "seq": seq[idx], "text": X_text[idx], "watch": watch[idx], "watch_mask": mask[idx]}

    m = dict(cfg["models"]["fusion"])
    m.pop("enabled", None)
    model, history = train_fusion(inputs(split.train), Y[split.train], inputs(split.val), Y[split.val],
                                  n_labels=len(labels), seed=seed, **m)
    P_val = predict_fusion(model, inputs(split.val))
    calibrator, thresholds = _decisions(P_val, Y[split.val], cfg["calibration"]["method"])
    P_test = predict_fusion(model, inputs(split.test))
    result = _evaluate(Y[split.test], P_test, calibrator.transform(P_test), thresholds, labels)
    result.update(best_epoch=history.best_epoch, best_val_loss=round(history.best_val_loss, 4),
                  epochs_run=len(history.val_loss))
    return result, model.state_dict(), {"calibrator": calibrator, "thresholds": thresholds}


def run(cfg: dict, output_dir: Path, promote_on_new_data: bool = False) -> dict:
    if cfg["data"]["source"] == "synthetic_v2":
        from smartep.training.gap_model import run_gap_model

        return run_gap_model(cfg, output_dir, promote_on_new_data)
    return run_legacy(cfg, output_dir, promote_on_new_data)


def run_legacy(cfg: dict, output_dir: Path, promote_on_new_data: bool = False) -> dict:
    """Phase 0 baseline on the Colab simulation (configs/train_xgb.yaml, configs/train_fusion.yaml)."""
    timer = Timer()
    seed = int(cfg["seed"])
    set_seeds(seed)
    cfg_hash = config_hash(cfg)
    version = make_version(cfg["run_name"], cfg_hash)

    # --- Data -------------------------------------------------------------------------------------------
    dcfg = cfg["data"]
    if dcfg["source"] != "legacy_colab":
        raise ValueError(f"unknown data source {dcfg['source']!r} (generator v2 arrives in Phase 1)")
    tables = generate_legacy_tables(int(dcfg["n_educators"]), seed)
    Y_df = label_matrix(tables)
    labels, Y = list(Y_df.columns), Y_df.to_numpy(dtype=np.float32)
    ids = tables.educator_ids
    split = grouped_split(ids, cfg["split"]["val_size"], cfg["split"]["test_size"], seed)
    timer.step(f"data: {len(ids)} rows, {len(labels)} labels, split {split.sizes()}")

    # --- Features (fitted on training rows only) ---------------------------------------------------------
    raw_tab = build_legacy_tabular(tables)
    featurizer = TabularFeaturizer(categorical=["department"]).fit(raw_tab.iloc[split.train])
    X_tab = featurizer.transform(raw_tab)
    tcfg = cfg["features"]["text"]
    texts = aggregate_texts(tables.lectures, ids, "lecture_text", empty="No lecture text available.")
    text_encoder = TextEncoder(tcfg["encoder"], int(tcfg["dim"]), seed).fit([texts[i] for i in split.train])
    X_text = text_encoder.transform(texts)
    X = np.hstack([X_tab, X_text])
    feature_names = featurizer.feature_names + text_encoder.feature_names()
    timer.step(f"features: {X.shape[1]} columns ({X_tab.shape[1]} tabular, {X_text.shape[1]} text)")

    # --- XGBoost -----------------------------------------------------------------------------------------
    xcfg = dict(cfg["models"]["xgb"])
    esr = xcfg.pop("early_stopping_rounds")
    xgb_model = MultiLabelXGB(labels, xcfg, esr, seed).fit(X[split.train], Y[split.train], X[split.val], Y[split.val])
    calibrator, thresholds = _decisions(xgb_model.predict_proba(X[split.val]), Y[split.val], cfg["calibration"]["method"])
    P_test_raw = xgb_model.predict_proba(X[split.test])
    P_test = calibrator.transform(P_test_raw)
    xgb_result = _evaluate(Y[split.test], P_test_raw, P_test, thresholds, labels)
    timer.step(f"xgboost: macro AUC {xgb_result['macro_auc']}")

    # --- Explanations ------------------------------------------------------------------------------------
    importance = global_importance(xgb_model, X[split.test], feature_names)
    overall = importance.mean(axis=1).sort_values(ascending=False)
    top_features = [[f, round(float(v), 4)] for f, v in overall.head(10).items()]
    example_label = labels[int(np.argmax(P_test[0]))]
    example = explain_row(xgb_model.contributions(X[split.test][:1], example_label)[0], feature_names)

    # --- Fairness ----------------------------------------------------------------------------------------
    fcfg = cfg["fairness"]
    group_values = raw_tab[fcfg["group"]].to_numpy()
    edges = fit_bucket_edges(group_values[split.train], fcfg["n_buckets"]) if fcfg.get("numeric", True) else None
    groups = assign_groups(group_values, edges)
    audit = fairness_audit(Y[split.test], P_test, thresholds, groups[split.test], labels, fcfg["max_disparity"])
    fairness = {
        "group": fcfg["group"],
        "worst_disparity": round(float(audit[["selection_disparity", "equal_opportunity_diff"]].to_numpy().max()), 4),
        "failed_labels": audit.loc[audit["flag"] == "FAIL", "label"].tolist(),
        "per_label": audit.to_dict(orient="records"),
    }
    timer.step(f"explain + fairness: worst disparity {fairness['worst_disparity']}")

    # --- Optional fusion model ---------------------------------------------------------------------------
    models = {"xgb": xgb_result}
    fusion_state = fusion_preprocess = None
    if cfg["models"].get("fusion", {}).get("enabled"):
        fusion_result, fusion_state, fusion_preprocess = _train_fusion(cfg, tables, ids, split, X_tab, X_text, Y, labels, seed)
        models["fusion"] = fusion_result
        timer.step(f"fusion: macro AUC {fusion_result['macro_auc']} (best epoch {fusion_result['best_epoch']})")

    # --- Recommender -------------------------------------------------------------------------------------
    recommender = None
    rcfg = cfg.get("recommender", {})
    if rcfg.get("enabled"):
        rng = np.random.default_rng(seed)
        engagement = raw_tab["completion_rate"].to_numpy()
        bandit = LinUCB(len(labels), X_tab.shape[1], alpha=rcfg["alpha"])
        train_bandit(bandit, X_tab[split.train], Y[split.train], engagement[split.train], rcfg["passes"], rng)
        raw_scores = bandit.scores(X_tab[split.test])
        top1, disparity = fairness_constrained_recommend(raw_scores, groups[split.test], fcfg["max_disparity"])
        raw_rank, served_rank = np.argsort(-raw_scores, axis=1), served_rankings(raw_scores, top1)
        recommender = {"top1_disparity": round(disparity, 4)}
        for name, rank in (("raw", raw_rank), ("served", served_rank)):
            m5, n5 = map_ndcg_at_k(Y[split.test], rank, 5)
            recommender[name] = {"MAP@5": m5, "NDCG@5": n5}
        timer.step(f"recommender: served NDCG@5 {recommender['served']['NDCG@5']}")

    # --- Personas (research only) ------------------------------------------------------------------------
    _, persona_ids, silhouette = fit_personas(X_tab[split.train], X_tab, cfg["personas"]["n_clusters"], seed)
    personas = {"n_clusters": cfg["personas"]["n_clusters"], "silhouette_train": silhouette,
                "sizes": np.bincount(persona_ids).tolist()}

    # --- Bundle, reload check, registry ------------------------------------------------------------------
    data_hash = frame_hash(raw_tab.reset_index(), Y_df.reset_index())
    manifest = {"version": version, "config_hash": cfg_hash, "data_hash": data_hash, "git_commit": git_commit(REPO_ROOT),
                "labels": labels, "feature_names": feature_names, "libraries": library_versions(),
                "training_data": "synthetic", "validated": False}
    summary = {
        "version": version, "config_hash": cfg_hash, "data_hash": data_hash, "manifest": manifest,
        "data": {"source": dcfg["source"], "n_rows": len(ids)}, "split": split.sizes(),
        "models": models, "fairness": fairness, "recommender": recommender, "personas": personas,
        "top_features": top_features, "example_explanation": {"label": example_label, "drivers": example},
    }
    summary["gates"] = check_gates(summary, cfg["gates"])

    bundle_dir = output_dir / version
    preprocess = {"featurizer": featurizer, "text_encoder": text_encoder, "calibrator": calibrator,
                  "thresholds": thresholds, "feature_names": feature_names, "labels": labels,
                  "fusion": fusion_preprocess}
    save_bundle(bundle_dir, manifest=manifest, config=cfg, metrics=summary, xgb_model=xgb_model,
                preprocess=preprocess, fusion_state=fusion_state)
    reloaded = load_bundle(bundle_dir).predict_proba(X[split.test])
    summary["bundle_reload_check"] = bool(np.allclose(reloaded, P_test, atol=1e-6))
    if not summary["bundle_reload_check"]:
        raise RuntimeError("saved bundle does not reproduce the evaluated predictions")

    fairness_ok = not fairness["failed_labels"]
    promoted, reason = promote_if_better(output_dir / "registry", {
        "version": version, "bundle_path": str(bundle_dir), "macro_auc": xgb_result["macro_auc"],
        "fairness_ok": fairness_ok, "data_hash": data_hash,
    }, allow_new_data=promote_on_new_data)
    summary["registry"] = {"promoted": promoted, "reason": reason}
    summary["runtime_seconds"] = timer.total

    (bundle_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (bundle_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    _log_mlflow(cfg, summary)
    timer.step(f"bundle saved to {bundle_dir} (promoted: {promoted}, {reason})")
    return summary


def _log_mlflow(cfg: dict, summary: dict) -> None:
    if not cfg.get("tracking", {}).get("mlflow"):
        return
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping tracking (install the 'tracking' extra)")
        return
    mlflow.set_tracking_uri(cfg["tracking"].get("uri", "file:./mlruns"))
    mlflow.set_experiment(cfg["run_name"])
    with mlflow.start_run(run_name=summary["version"]):
        mlflow.log_params({"config_hash": summary["config_hash"], "data_hash": summary["data_hash"], "seed": cfg["seed"]})
        for name, res in summary["models"].items():
            mlflow.log_metric(f"{name}_macro_auc", res["macro_auc"] or 0.0)
            mlflow.log_metric(f"{name}_macro_f1", res["macro_f1"] or 0.0)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    parser.add_argument("--output-dir", default=None, help="defaults to output.dir in the config")
    parser.add_argument("--promote-on-new-data", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    out = Path(args.output_dir or cfg["output"]["dir"])
    summary = run(cfg, out, args.promote_on_new_data)
    print(json.dumps({"version": summary["version"], "gates": {k: g["pass"] for k, g in summary["gates"].items()},
                      "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
