"""Gate G1 checks for a generated dataset.

1. Schema: every package validates against EvidencePackage v1.0.
2. Prevalence: every gap area between the configured bounds (default 10-40%).
3. Evidence: every gap has at least one evidence item that validates as AssessmentResult evidence.
4. Oracle: XGBoost given the latent period traits reaches high AUC (the signal exists).
5. Noise: XGBoost given random features stays near AUC 0.5 (no leakage in the check itself).
6. Evidence probe (reported, soft target): XGBoost on structured package features lands between noise
   and oracle (the signal reaches what the model can observe).
Checks 4-6 use a grouped split by educator on the train_pool subset only.
"""

from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from smartep.contracts import EvidencePackage
from smartep.contracts.assessment_result import Evidence
from smartep.contracts.gap_areas import GAP_AREAS
from smartep.data.splits import grouped_split
from smartep.evaluate.metrics import label_metrics, macro
from smartep.features.package import package_feature_frame
from smartep.models.xgb_multilabel import MultiLabelXGB
from smartep.synth.generator import SyntheticDataset
from smartep.synth.traits import TRAITS

XGB_PARAMS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8}


def _probe(X, Y, split, seed) -> dict:
    model = MultiLabelXGB(list(GAP_AREAS), XGB_PARAMS, 30, seed).fit(X[split.train], Y[split.train],
                                                                      X[split.val], Y[split.val])
    P = model.predict_proba(X[split.test])
    per = label_metrics(Y[split.test], P, P, np.full(len(GAP_AREAS), 0.5), list(GAP_AREAS))
    return {"macro_auc": macro(per, "auc"), "per_area_auc": {a: m["auc"] for a, m in per.items()}}


def run_checks(ds: SyntheticDataset, targets: dict, seed: int) -> dict:
    results: dict = {}

    invalid = []
    for pkg in ds.packages:
        try:
            EvidencePackage.model_validate(pkg)
        except ValidationError as exc:
            invalid.append({"packageId": pkg["packageId"], "error": str(exc)[:300]})
    results["schema"] = {"valid": len(ds.packages) - len(invalid), "total": len(ds.packages),
                         "pass": not invalid, "examples": invalid[:3]}

    prev = ds.labels[list(GAP_AREAS)].mean().round(4).to_dict()
    lo, hi = targets["prevalence_bounds"]
    results["prevalence"] = {"per_area": prev, "bounds": [lo, hi], "pass": all(lo <= p <= hi for p in prev.values())}

    n_gaps = missing = bad = 0
    for row in ds.evidence:
        for gap in row["gaps"]:
            n_gaps += 1
            missing += not gap["evidence"]
            for item in gap["evidence"]:
                try:
                    Evidence.model_validate(item)
                except ValidationError:
                    bad += 1
    results["evidence"] = {"gaps": n_gaps, "without_evidence": missing, "invalid_items": bad,
                           "pass": missing == 0 and bad == 0}

    pool = (ds.index["subset"] == "train_pool").to_numpy()
    groups = ds.index.loc[pool, "educatorRef"].to_numpy()
    split = grouped_split(groups, 0.15, 0.2, seed)
    Y = ds.labels.loc[pool, list(GAP_AREAS)].to_numpy(dtype=np.float32)

    oracle = _probe(ds.traits.loc[pool, list(TRAITS)].to_numpy(dtype=np.float32), Y, split, seed)
    oracle["pass"] = oracle["macro_auc"] >= targets["oracle_min_auc"]
    results["oracle"] = oracle

    noise_X = np.random.default_rng(seed).normal(size=(int(pool.sum()), len(TRAITS))).astype(np.float32)
    noise = _probe(noise_X, Y, split, seed)
    noise["pass"] = abs(noise["macro_auc"] - 0.5) <= targets["noise_max_deviation"]
    results["noise"] = noise

    features = package_feature_frame([p for p, keep in zip(ds.packages, pool, strict=True) if keep])
    probe = _probe(features.to_numpy(dtype=np.float32), Y, split, seed)
    probe["n_features"] = features.shape[1]
    probe["meets_soft_target"] = probe["macro_auc"] >= targets["evidence_probe_min_auc"]
    results["evidence_probe"] = probe

    results["G1_pass"] = all(results[k]["pass"] for k in ("schema", "prevalence", "evidence", "oracle", "noise"))
    return results
