"""FastAPI app.

    uvicorn smartep.api.app:app --host 0.0.0.0 --port 8000

Routes:
- ``GET  /health``                       model version and queue counts (no secrets)
- ``POST /v1/packages``                  signed webhook from The Classroom → 202 (docs/api.md)
- ``GET  /v1/packages/{packageId}``      job status, for operators (Bearer SMARTEP_ADMIN_TOKEN)
"""

from __future__ import annotations

import hmac
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from smartep.api.pipeline import JobRunner
from smartep.api.security import url_allowed, verify_signature
from smartep.api.settings import Settings, from_env
from smartep.api.store import JobStore
from smartep.api.worker import Worker

log = logging.getLogger("smartep.api")


class PackageReady(BaseModel):
    """Webhook body (spec §5). ``classroomBaseUrl`` is optional and enables refresh-url on expiry."""

    model_config = ConfigDict(extra="ignore")
    packageId: str = Field(min_length=1, max_length=200)
    schemaVersion: Literal["1.0"]
    educatorRef: str
    trigger: Literal["evaluation_round_closed", "period_end", "manual"]
    downloadUrl: str
    downloadExpiresAt: datetime
    checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    byteSize: int = Field(gt=0)
    classroomBaseUrl: str | None = None


def create_app(settings: Settings | None = None, predictor=None, client: httpx.Client | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or from_env()
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        # httpx logs full request URLs at INFO; presigned download links carry access signatures.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        pred = predictor
        if pred is None:
            from smartep.serve.predict import GapPredictor

            pred = GapPredictor.from_bundle(s.model_dir)
        store = JobStore(s.db_path)
        http = client or httpx.Client(timeout=s.http_timeout_s, follow_redirects=False)
        runner = JobRunner(s, store, pred, http)
        worker = Worker(runner, s.poll_interval_s)
        if s.start_worker:
            worker.start()
        app.state.settings, app.state.store, app.state.worker, app.state.predictor = s, store, worker, pred
        log.info(json.dumps({"event": "started", "model": pred.version, "orgs": sorted(s.orgs)}))
        yield
        worker.stop()
        store.close()
        if client is None:
            http.close()

    app = FastAPI(title="SMARTEP API", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/health")
    def health(request: Request):
        st = request.app.state
        return {"status": "ok", "model": {"name": st.predictor.name, "version": st.predictor.version,
                                          "trainingData": st.predictor.training_data, "validated": st.predictor.validated},
                "jobs": st.store.counts()}

    @app.post("/v1/packages", status_code=202)
    async def package_ready(request: Request,
                            x_smartep_org: str | None = Header(default=None),
                            x_smartep_timestamp: str | None = Header(default=None),
                            x_smartep_signature: str | None = Header(default=None)):
        s: Settings = request.app.state.settings
        org = s.orgs.get(x_smartep_org or "")
        body = await request.body()
        # Same answer for an unknown organisation and a bad signature: do not reveal which orgs exist.
        if org is None or verify_signature(org.webhook_secret, x_smartep_timestamp, body, x_smartep_signature,
                                           s.signature_tolerance_s):
            raise HTTPException(status_code=401, detail="invalid signature")
        try:
            hook = PackageReady.model_validate_json(body)
        except ValidationError as exc:
            return JSONResponse(status_code=422, content={"detail": "invalid webhook body", "errors": exc.error_count()})
        if not url_allowed(hook.downloadUrl, org.allowed_download_hosts, s.allow_insecure_http):
            raise HTTPException(status_code=422, detail="downloadUrl host is not allowed")
        created, status = request.app.state.store.enqueue(hook.packageId, org.org_ref, hook.model_dump(mode="json"),
                                                          s.deliver_within_s)
        log.info(json.dumps({"event": "accepted" if created else "duplicate", "packageId": hook.packageId,
                             "org": org.org_ref, "status": status}))
        return {"packageId": hook.packageId, "status": status, "duplicate": not created}

    @app.get("/v1/packages/{package_id}")
    def package_status(package_id: str, request: Request, authorization: str | None = Header(default=None)):
        token = request.app.state.settings.admin_token
        if not token or not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="unauthorised")
        job = request.app.state.store.get(package_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown package")
        keep = ("package_id", "org_ref", "status", "attempts", "error", "model_version", "latency_ms",
                "created_at", "updated_at", "deadline_at")
        return {k: job[k] for k in keep}

    return app


app = create_app()
