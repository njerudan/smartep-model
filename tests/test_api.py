"""Phase 3: the SMARTEP API against a stand-in Classroom (webhook → download → predict → callback)."""

import gzip
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smartep.api.app import create_app  # noqa: E402
from smartep.api.fake_classroom import APP_HOST, MEDIA_HOST, FakeClassroom  # noqa: E402
from smartep.api.security import sign, url_allowed, verify_signature  # noqa: E402
from smartep.api.settings import Settings, parse_orgs  # noqa: E402
from smartep.contracts import AssessmentResult  # noqa: E402
from smartep.serve.predict import GapPredictor  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "models" / "gap-model"
GOLDEN = REPO / "tests" / "fixtures" / "golden"
SECRET, API_KEY, ADMIN = "whsec-test", "cio-key-test", "admin-test"


def golden_packages() -> list[dict]:
    with gzip.open(GOLDEN / "packages.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def predictor():
    return GapPredictor.from_bundle(MODEL_DIR)


@pytest.fixture
def env(tmp_path, predictor):
    pkgs = golden_packages()
    org = pkgs[0]["classroom"]["orgRef"]
    fake = FakeClassroom(API_KEY)
    settings = Settings(
        model_dir=MODEL_DIR, db_path=tmp_path / "jobs.db", admin_token=ADMIN, start_worker=False,
        retry_delays_s=(0, 0, 0, 0), orgs=parse_orgs({org: {
            "webhook_secret": SECRET, "results_api_key": API_KEY,
            "allowed_download_hosts": [MEDIA_HOST], "allowed_callback_hosts": [APP_HOST]}}))
    client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    app = create_app(settings, predictor=predictor, client=client)
    with TestClient(app) as tc:
        yield tc, fake, org, [p for p in pkgs if p["classroom"]["orgRef"] == org] or pkgs[:1]


def post_webhook(tc, org, body: dict, secret=SECRET, ts=None):
    raw = json.dumps(body).encode()
    ts = ts if ts is not None else int(time.time())
    return tc.post("/v1/packages", content=raw, headers={
        "X-SMARTEP-Org": org, "X-SMARTEP-Timestamp": str(ts), "X-SMARTEP-Signature": sign(secret, ts, raw),
        "Content-Type": "application/json"})


def drain(tc, limit=20):
    worker = tc.app.state.worker
    for _ in range(limit):
        if not worker.run_once():
            return


def test_signatures_and_hosts():
    body = b'{"a":1}'
    sig = sign("s", 1000, body)
    assert verify_signature("s", "1000", body, sig, 300, now=1100) is None
    assert verify_signature("s", "1000", body, sig, 300, now=2000) == "timestamp outside tolerance"
    assert verify_signature("other", "1000", body, sig, 300, now=1100) == "signature mismatch"
    assert verify_signature("s", "1000", b'{"a":2}', sig, 300, now=1100) == "signature mismatch"
    assert url_allowed("https://media.example.test/x", ("media.example.test",))
    assert not url_allowed("http://media.example.test/x", ("media.example.test",))
    assert not url_allowed("https://evil.test/x", ("media.example.test",))


def test_golden_predictions_reproduce_the_evaluated_model(predictor):
    expected = json.loads((GOLDEN / "expected.json").read_text())
    assert predictor.version == expected["model_version"]
    pkgs = golden_packages()
    _, P = predictor.model.predict(predictor._features(pkgs))
    for i, p in enumerate(pkgs):
        np.testing.assert_allclose(P[i], expected["probabilities"][p["packageId"]], atol=1e-6)


def test_bundle_quality_has_not_regressed():
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    assert metrics["cv"]["macro_auc"]["mean"] >= 0.80 and all(g["pass"] for g in metrics["gates"].values())


def test_health(env):
    tc, *_ = env
    body = tc.get("/health").json()
    assert body["status"] == "ok" and body["model"]["validated"] is False


def test_end_to_end_delivery(env):
    tc, fake, org, pkgs = env
    hook = fake.add_package(pkgs[0])
    r = post_webhook(tc, org, hook)
    assert r.status_code == 202 and r.json()["status"] == "queued"
    drain(tc)
    assert len(fake.results) == 1
    result = AssessmentResult.model_validate(fake.results[0])
    assert result.packageId == pkgs[0]["packageId"] and result.model.trainingData == "synthetic"
    job = tc.app.state.store.get(pkgs[0]["packageId"])
    assert job["status"] == "delivered" and job["result_json"] is None and job["latency_ms"] < 30_000
    status = tc.get(f"/v1/packages/{pkgs[0]['packageId']}", headers={"Authorization": f"Bearer {ADMIN}"}).json()
    assert status["status"] == "delivered" and "webhook_json" not in status


def test_rejects_bad_requests(env):
    tc, fake, org, pkgs = env
    hook = fake.add_package(pkgs[0])
    assert post_webhook(tc, org, hook, secret="wrong").status_code == 401
    assert post_webhook(tc, "org_unknown", hook).status_code == 401
    assert post_webhook(tc, org, hook, ts=int(time.time()) - 3600).status_code == 401
    assert post_webhook(tc, org, {**hook, "downloadUrl": "https://evil.test/p.json.gz"}).status_code == 422
    assert post_webhook(tc, org, {**hook, "checksum": "md5:abc"}).status_code == 422
    assert tc.get(f"/v1/packages/{hook['packageId']}").status_code == 401


def test_duplicate_webhook_is_not_processed_twice(env):
    tc, fake, org, pkgs = env
    hook = fake.add_package(pkgs[0])
    assert post_webhook(tc, org, hook).json()["duplicate"] is False
    assert post_webhook(tc, org, hook).json()["duplicate"] is True
    drain(tc)
    assert len(fake.results) == 1


def test_checksum_mismatch_fails_permanently(env):
    tc, fake, org, pkgs = env
    hook = fake.add_package(pkgs[0])
    fake.files[pkgs[0]["packageId"]] = gzip.compress(b'{"tampered": true}')
    post_webhook(tc, org, hook)
    drain(tc)
    job = tc.app.state.store.get(pkgs[0]["packageId"])
    assert job["status"] == "failed" and job["error"] == "checksum mismatch" and not fake.results


def test_callback_retries_then_delivers(env):
    tc, fake, org, pkgs = env
    fake.callback_failures = [503, 429]
    post_webhook(tc, org, fake.add_package(pkgs[0]))
    drain(tc)
    job = tc.app.state.store.get(pkgs[0]["packageId"])
    assert job["status"] == "delivered" and job["attempts"] == 3 and len(fake.results) == 1


def test_callback_client_error_fails(env):
    tc, fake, org, pkgs = env
    fake.callback_failures = [422]
    post_webhook(tc, org, fake.add_package(pkgs[0]))
    drain(tc)
    job = tc.app.state.store.get(pkgs[0]["packageId"])
    assert job["status"] == "failed" and "422" in job["error"] and job["result_json"] is None


def test_package_from_another_organisation_fails(env):
    tc, fake, org, pkgs = env
    other = json.loads(json.dumps(pkgs[0]))
    other["packageId"], other["classroom"]["orgRef"] = "pkg_other", "org_someone_else"
    post_webhook(tc, org, fake.add_package(other))
    drain(tc)
    job = tc.app.state.store.get("pkg_other")
    assert job["status"] == "failed" and "different organisation" in job["error"]


def test_expired_link_is_refreshed(env):
    tc, fake, org, pkgs = env
    hook = fake.add_package(pkgs[0], expires_in_h=-1)
    fake.download_status = 403
    post_webhook(tc, org, hook)
    drain(tc)
    assert fake.refresh_calls == 1 and len(fake.results) == 1
