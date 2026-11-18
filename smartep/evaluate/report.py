"""Markdown run report written next to each bundle."""

from __future__ import annotations


def _table(headers: list[str], rows: list[list]) -> str:
    fmt = lambda v: "n/a" if v is None else str(v)  # noqa: E731
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(summary: dict) -> str:
    xgb = summary["models"]["xgb"]
    parts = [
        f"# Run report: {summary['version']}",
        "",
        f"- Data source: `{summary['data']['source']}` ({summary['data']['n_rows']} rows, data hash `{summary['data_hash']}`)",
        f"- Split sizes: {summary['split']}",
        f"- Config hash: `{summary['config_hash']}` · git: `{summary['manifest']['git_commit']}`",
        f"- Runtime: {summary['runtime_seconds']} s",
        f"- Bundle reload check (saved model reproduces evaluated predictions): **{summary['bundle_reload_check']}**",
        "",
    ]
    if summary["data"]["source"] == "legacy_colab":
        parts += [
            "> Legacy Colab data: gap labels are random and independent of every feature, so AUC near 0.5 is",
            "> the expected result. These numbers test the pipeline, not the model. Generator v2 (Phase 1)",
            "> replaces this data.",
            "",
        ]
    parts += ["## XGBoost per label (test split)", ""]
    cols = ["prevalence", "auc", "pr_auc", "threshold", "precision", "recall", "f1", "ece_raw", "ece_calibrated"]
    parts.append(_table(["label", *cols], [[k, *[m[c] for c in cols]] for k, m in xgb["per_label"].items()]))
    parts += ["", f"Macro AUC **{xgb['macro_auc']}** · macro F1 {xgb['macro_f1']} · Hamming loss {xgb['hamming_loss']}", ""]

    if "fusion" in summary["models"]:
        fu = summary["models"]["fusion"]
        parts += ["## Fusion model (test split)", "",
                  f"Macro AUC **{fu['macro_auc']}** · macro F1 {fu['macro_f1']} · best epoch {fu['best_epoch']}", ""]

    gates = summary["gates"]
    parts += ["## Gates", "", _table(["gate", "value", "target", "pass"],
                                     [[k, g["value"], g["target"], g["pass"]] for k, g in gates.items()]), ""]
    fair = summary["fairness"]
    parts += ["## Fairness", "", f"Group attribute `{fair['group']}`; worst disparity {fair['worst_disparity']}; "
              f"failed labels: {fair['failed_labels'] or 'none'}", ""]
    if summary.get("recommender"):
        rec = summary["recommender"]
        parts += ["## Recommender (LinUCB)", "",
                  _table(["list", "MAP@5", "NDCG@5"], [["raw scores", rec["raw"]["MAP@5"], rec["raw"]["NDCG@5"]],
                                                       ["served (fairness-constrained)", rec["served"]["MAP@5"], rec["served"]["NDCG@5"]]]),
                  "", f"Top-1 disparity after constraint: {rec['top1_disparity']}", ""]
    parts += ["## Top features", "", _table(["feature", "mean |SHAP|"], summary["top_features"]), ""]
    parts += ["## Registry", "", f"Promoted: **{summary['registry']['promoted']}** ({summary['registry']['reason']})", ""]
    return "\n".join(parts)
