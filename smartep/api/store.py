"""Job queue in SQLite (one container, no Redis needed on the 8 GB server).

Job states: queued → processing → delivering → delivered, or failed. A job whose step fails for a
temporary reason goes back to ``queued`` (or stays ``delivering``) with ``next_attempt_at`` set. Package
contents are never stored; the result JSON is kept only until it is delivered.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    package_id      TEXT PRIMARY KEY,
    org_ref         TEXT NOT NULL,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL,
    deadline_at     REAL NOT NULL,
    webhook_json    TEXT NOT NULL,
    result_json     TEXT,
    error           TEXT,
    model_version   TEXT,
    latency_ms      REAL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_due ON jobs (status, next_attempt_at);
"""
ACTIVE = ("queued", "processing", "delivering")


class JobStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            # A job left 'processing' by a crash or restart is picked up again.
            self._conn.execute("UPDATE jobs SET status='queued' WHERE status='processing'")

    def enqueue(self, package_id: str, org_ref: str, webhook: dict, deliver_within_s: int) -> tuple[bool, str]:
        """Insert a job; returns (created, status). An existing packageId is not queued twice."""
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT status FROM jobs WHERE package_id=?", (package_id,)).fetchone()
            if row:
                return False, row["status"]
            self._conn.execute(
                "INSERT INTO jobs (package_id, org_ref, status, next_attempt_at, deadline_at, webhook_json, created_at, updated_at)"
                " VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)",
                (package_id, org_ref, now, now + deliver_within_s, json.dumps(webhook), now, now))
            return True, "queued"

    def claim_next(self) -> dict | None:
        """Atomically take the next due job (queued or delivering) and mark it in progress."""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','delivering') AND next_attempt_at<=? "
                "ORDER BY next_attempt_at LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            new_status = "processing" if row["status"] == "queued" else "delivering"
            self._conn.execute("UPDATE jobs SET status=?, attempts=attempts+1, next_attempt_at=?, updated_at=? "
                               "WHERE package_id=?", (new_status, now + 3600, now, row["package_id"]))
            job = dict(row)
            job["status"], job["attempts"] = new_status, row["attempts"] + 1
            return job

    def update(self, package_id: str, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {cols} WHERE package_id=?", (*fields.values(), package_id))

    def get(self, package_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE package_id=?", (package_id,)).fetchone()
        return dict(row) if row else None

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
