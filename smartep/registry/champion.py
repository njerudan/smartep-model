"""Champion/challenger promotion.

A candidate is promoted only if it passes the fairness check and either there is no champion, or it
beats the champion's macro AUC on the same dataset (same data_hash). Scores on different datasets are not
comparable, so a candidate trained on new data is promoted only when ``allow_new_data`` is set.
Every decision is appended to history.jsonl.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def decide(candidate: dict, champion: dict | None, allow_new_data: bool) -> tuple[bool, str]:
    if not candidate["fairness_ok"]:
        return False, "fairness check failed"
    if champion is None:
        return True, "no champion yet"
    if candidate["data_hash"] != champion["data_hash"]:
        if allow_new_data:
            return True, "new dataset; promotion allowed explicitly"
        return False, "trained on a different dataset; scores not comparable (use --promote-on-new-data)"
    if candidate["macro_auc"] is not None and candidate["macro_auc"] > (champion.get("macro_auc") or -1):
        return True, "higher macro AUC on the same dataset"
    return False, "macro AUC not higher than champion on the same dataset"


def promote_if_better(registry_dir: Path, candidate: dict, allow_new_data: bool = False) -> tuple[bool, str]:
    """``candidate`` needs version, bundle_path, macro_auc, fairness_ok and data_hash."""
    registry_dir.mkdir(parents=True, exist_ok=True)
    champion_file = registry_dir / "champion.json"
    champion = json.loads(champion_file.read_text(encoding="utf-8")) if champion_file.exists() else None
    promoted, reason = decide(candidate, champion, allow_new_data)
    record = {**candidate, "decided_at": datetime.now(UTC).isoformat(), "promoted": promoted, "reason": reason,
              "previous_champion": champion["version"] if champion else None}
    with open(registry_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    if promoted:
        champion_file.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return promoted, reason
