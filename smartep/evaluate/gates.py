"""Check a run summary against the gate targets in the config."""

from __future__ import annotations


def check_gates(summary: dict, targets: dict) -> dict[str, dict]:
    xgb = summary["models"]["xgb"]
    label_aucs = [m["auc"] for m in xgb["per_label"].values() if m["auc"] is not None]
    max_ece = max((m["ece_calibrated"] for m in xgb["per_label"].values()), default=None)
    fairness_worst = summary["fairness"]["worst_disparity"]
    checks = {
        "macro_auc": (xgb["macro_auc"], targets["macro_auc"], ">="),
        "min_label_auc": (min(label_aucs) if label_aucs else None, targets["min_label_auc"], ">="),
        "max_ece": (max_ece, targets["max_ece"], "<="),
        "max_disparity": (fairness_worst, targets["max_disparity"], "<="),
    }
    out = {}
    for name, (value, target, op) in checks.items():
        ok = value is not None and (value >= target if op == ">=" else value <= target)
        out[name] = {"value": value, "target": f"{op} {target}", "pass": bool(ok)}
    return out
