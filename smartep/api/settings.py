"""Service settings, read from environment variables (all hosts and secrets come from config).

SMARTEP_ORGS holds one entry per organisation (orgRef from the package), as JSON, e.g.

    {"org_daystar": {
        "webhook_secret": "…",                 # HMAC secret The Classroom signs webhooks with
        "results_api_key": "…",                # Classroom org API key, scope assessment:results:write
        "allowed_download_hosts": ["media.theclass.co.ke"],
        "allowed_callback_hosts": ["lms.theclass.co.ke"]
    }}

SMARTEP_ORGS_FILE may point to a JSON file instead (takes precedence).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OrgConfig:
    org_ref: str
    webhook_secret: str
    results_api_key: str
    allowed_download_hosts: tuple[str, ...]
    allowed_callback_hosts: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    db_path: Path
    orgs: dict[str, OrgConfig]
    admin_token: str | None = None
    signature_tolerance_s: int = 300
    max_package_bytes: int = 50 * 1024 * 1024
    http_timeout_s: float = 30.0
    poll_interval_s: float = 1.0
    deliver_within_s: int = 24 * 3600
    retry_delays_s: tuple[int, ...] = (30, 120, 600, 1800, 3600, 3 * 3600, 6 * 3600)
    allow_insecure_http: bool = False   # only for local development and tests
    start_worker: bool = True
    extra: dict = field(default_factory=dict)


def parse_orgs(raw: dict) -> dict[str, OrgConfig]:
    orgs = {}
    for ref, c in raw.items():
        missing = {"webhook_secret", "results_api_key"} - set(c)
        if missing:
            raise ValueError(f"SMARTEP_ORGS[{ref}] is missing {sorted(missing)}")
        orgs[ref] = OrgConfig(ref, c["webhook_secret"], c["results_api_key"],
                              tuple(c.get("allowed_download_hosts", [])), tuple(c.get("allowed_callback_hosts", [])))
    return orgs


def from_env() -> Settings:
    if os.environ.get("SMARTEP_ORGS_FILE"):
        raw = json.loads(Path(os.environ["SMARTEP_ORGS_FILE"]).read_text(encoding="utf-8"))
    else:
        raw = json.loads(os.environ.get("SMARTEP_ORGS", "{}"))
    return Settings(
        model_dir=Path(os.environ.get("SMARTEP_MODEL_DIR", "models/gap-model")),
        db_path=Path(os.environ.get("SMARTEP_DB_PATH", "data/smartep.db")),
        orgs=parse_orgs(raw),
        admin_token=os.environ.get("SMARTEP_ADMIN_TOKEN") or None,
        allow_insecure_http=os.environ.get("SMARTEP_ALLOW_INSECURE_HTTP") == "1",
        start_worker=os.environ.get("SMARTEP_START_WORKER", "1") == "1",
    )
