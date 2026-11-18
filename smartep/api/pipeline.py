"""One job: download → verify → predict → deliver.

Failures are split into permanent ones (bad checksum, invalid package, wrong organisation, a 4xx from the
callback other than 408/429) which mark the job failed, and temporary ones (network errors, timeouts, 5xx,
408/429) which are retried with growing delays until the 24-hour delivery deadline.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from datetime import UTC, datetime

import httpx
from pydantic import ValidationError

from smartep.api.security import url_allowed
from smartep.api.settings import OrgConfig, Settings
from smartep.api.store import JobStore
from smartep.contracts import EvidencePackage

log = logging.getLogger("smartep.pipeline")


class PermanentError(Exception):
    """Retrying will not help; the job is marked failed."""


class TemporaryError(Exception):
    """Retry later."""


def _retryable_status(code: int) -> bool:
    return code >= 500 or code in (408, 429)


def download_package(client: httpx.Client, url: str, checksum: str, byte_size: int, max_bytes: int) -> dict:
    if byte_size > max_bytes:
        raise PermanentError(f"package is {byte_size} bytes, over the {max_bytes}-byte limit")
    try:
        with client.stream("GET", url) as resp:
            if resp.status_code in (401, 403, 410):
                raise TemporaryError(f"download link refused ({resp.status_code}); may have expired")
            if resp.status_code != 200:
                exc = TemporaryError if _retryable_status(resp.status_code) else PermanentError
                raise exc(f"download returned {resp.status_code}")
            chunks, total = [], 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise PermanentError("package larger than the size limit")
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise TemporaryError(f"download failed: {exc.__class__.__name__}") from exc
    raw = b"".join(chunks)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != checksum:
        raise PermanentError("checksum mismatch")
    if len(raw) != byte_size:
        raise PermanentError(f"byte size {len(raw)} does not match the declared {byte_size}")
    try:
        body = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        return json.loads(body)
    except (OSError, ValueError) as exc:
        raise PermanentError("package is not valid (gzip) JSON") from exc


def refresh_download_url(client: httpx.Client, base_url: str, package_id: str, org: OrgConfig) -> str:
    """Spec §5: ask The Classroom for a fresh presigned link when the old one has expired."""
    resp = client.post(f"{base_url.rstrip('/')}/v1/assessment/packages/{package_id}/refresh-url",
                       headers={"Authorization": f"Bearer {org.results_api_key}"})
    if resp.status_code != 200:
        raise TemporaryError(f"refresh-url returned {resp.status_code}")
    return resp.json()["downloadUrl"]


def validate_package(data: dict, package_id: str, org: OrgConfig, settings: Settings) -> dict:
    try:
        EvidencePackage.model_validate(data)
    except ValidationError as exc:
        raise PermanentError(f"package fails schema v1.0 ({exc.error_count()} errors)") from exc
    if data["packageId"] != package_id:
        raise PermanentError("packageId in the package does not match the webhook")
    if data["classroom"]["orgRef"] != org.org_ref:
        raise PermanentError("package belongs to a different organisation")
    if not url_allowed(data["classroom"]["resultsCallbackUrl"], org.allowed_callback_hosts, settings.allow_insecure_http):
        raise PermanentError("resultsCallbackUrl host is not allowed for this organisation")
    return data


def deliver_result(client: httpx.Client, url: str, result_json: str, org: OrgConfig) -> None:
    try:
        resp = client.post(url, content=result_json, headers={
            "Authorization": f"Bearer {org.results_api_key}", "Content-Type": "application/json"})
    except httpx.HTTPError as exc:
        raise TemporaryError(f"callback failed: {exc.__class__.__name__}") from exc
    if resp.status_code in (200, 201, 202, 204):
        return
    exc_type = TemporaryError if _retryable_status(resp.status_code) else PermanentError
    raise exc_type(f"callback returned {resp.status_code}: {resp.text[:200]}")


class JobRunner:
    def __init__(self, settings: Settings, store: JobStore, predictor, client: httpx.Client):
        self.settings, self.store, self.predictor, self.client = settings, store, predictor, client

    def _retry_or_fail(self, job: dict, status: str, reason: str) -> None:
        delays = self.settings.retry_delays_s
        delay = delays[min(job["attempts"] - 1, len(delays) - 1)]
        next_at = time.time() + delay
        if next_at > job["deadline_at"]:
            self.store.update(job["package_id"], status="failed", error=f"deadline passed: {reason}", result_json=None)
            log.warning(json.dumps({"event": "job_failed", "packageId": job["package_id"], "reason": "deadline", "detail": reason}))
            return
        self.store.update(job["package_id"], status=status, next_attempt_at=next_at, error=reason)
        log.info(json.dumps({"event": "job_retry", "packageId": job["package_id"], "status": status,
                             "attempt": job["attempts"], "in_s": delay, "detail": reason}))

    def run(self, job: dict) -> None:
        pid, org = job["package_id"], self.settings.orgs.get(job["org_ref"])
        if org is None:
            self.store.update(pid, status="failed", error="organisation no longer configured")
            return
        try:
            if job["status"] == "processing":
                self._analyse(job, org)
                job = self.store.get(pid)
            self._deliver(job, org)
        except PermanentError as exc:
            self.store.update(pid, status="failed", error=str(exc), result_json=None)
            log.warning(json.dumps({"event": "job_failed", "packageId": pid, "detail": str(exc)}))
        except TemporaryError as exc:
            current = self.store.get(pid)
            self._retry_or_fail(current, "delivering" if current.get("result_json") else "queued", str(exc))
        except Exception as exc:  # unexpected: keep the service alive, record and retry
            log.exception("unexpected error for %s", pid)
            current = self.store.get(pid)
            self._retry_or_fail(current, "delivering" if current.get("result_json") else "queued",
                                f"internal error: {exc.__class__.__name__}")

    def _analyse(self, job: dict, org: OrgConfig) -> None:
        hook = json.loads(job["webhook_json"])
        url = hook["downloadUrl"]
        expired = datetime.fromisoformat(hook["downloadExpiresAt"].replace("Z", "+00:00")) < datetime.now(UTC)
        if expired and hook.get("classroomBaseUrl"):
            # The refresh call carries the org API key, so its host must be an allowed callback host.
            if not url_allowed(hook["classroomBaseUrl"], org.allowed_callback_hosts, self.settings.allow_insecure_http):
                raise PermanentError("download link expired and classroomBaseUrl host is not allowed")
            url = refresh_download_url(self.client, hook["classroomBaseUrl"], job["package_id"], org)
        if not url_allowed(url, org.allowed_download_hosts, self.settings.allow_insecure_http):
            raise PermanentError("download host is not allowed for this organisation")
        t0 = time.perf_counter()
        data = download_package(self.client, url, hook["checksum"], hook["byteSize"], self.settings.max_package_bytes)
        package = validate_package(data, job["package_id"], org, self.settings)
        result = self.predictor.predict(package)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.store.update(job["package_id"], status="delivering", result_json=result.model_dump_json(exclude_none=True),
                          model_version=result.model.version, latency_ms=latency_ms, error=None,
                          webhook_json=json.dumps({**hook, "resultsCallbackUrl": package["classroom"]["resultsCallbackUrl"]}))
        log.info(json.dumps({"event": "analysed", "packageId": job["package_id"], "org": org.org_ref,
                             "model": result.model.version, "gaps": len(result.gaps), "latency_ms": latency_ms}))

    def _deliver(self, job: dict, org: OrgConfig) -> None:
        hook = json.loads(job["webhook_json"])
        deliver_result(self.client, hook["resultsCallbackUrl"], job["result_json"], org)
        # Delivered: drop the result body; keep metadata only (minimisation).
        self.store.update(job["package_id"], status="delivered", result_json=None, error=None)
        log.info(json.dumps({"event": "delivered", "packageId": job["package_id"], "org": org.org_ref,
                             "attempts": job["attempts"]}))
