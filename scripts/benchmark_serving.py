"""Gate G3 measurements: latency and memory of the API over N packages, end to end.

    python scripts/benchmark_serving.py --n 50

Runs the real app (FastAPI + worker + model bundle) in-process against the stand-in Classroom: signed
webhook → 202 → download → checksum → schema check → predict → callback. Reports webhook acknowledgement
time, per-package processing latency (p50, p95, max) and process memory (RSS) after start-up and at peak.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import psutil
from fastapi.testclient import TestClient

from smartep.api.app import create_app
from smartep.api.fake_classroom import APP_HOST, MEDIA_HOST, FakeClassroom
from smartep.api.security import sign
from smartep.api.settings import Settings, parse_orgs
from smartep.contracts import AssessmentResult
from smartep.synth.io import read_dataset

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model-dir", default=str(REPO / "models" / "gap-model"))
    args = parser.parse_args(argv)

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    manifest = json.loads((Path(args.model_dir) / "manifest.json").read_text(encoding="utf-8"))
    ds = read_dataset(REPO / manifest["data_dir"])
    hold = [p for p, s in zip(ds.packages, ds.index["subset"], strict=True) if s == "classroom_test"][: args.n]
    del ds

    fake = FakeClassroom("key")
    orgs = {o: {"webhook_secret": "s", "results_api_key": "key", "allowed_download_hosts": [MEDIA_HOST],
                "allowed_callback_hosts": [APP_HOST]} for o in {p["classroom"]["orgRef"] for p in hold}}
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(model_dir=Path(args.model_dir), db_path=Path(tmp) / "jobs.db", orgs=parse_orgs(orgs),
                            start_worker=False)
        app = create_app(settings, client=httpx.Client(transport=httpx.MockTransport(fake.handler)))
        peak = [0]
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.02)

        threading.Thread(target=sample, daemon=True).start()
        t_start = time.perf_counter()
        with TestClient(app) as tc:
            startup_s = time.perf_counter() - t_start
            rss_ready = proc.memory_info().rss
            acks = []
            for p in hold:
                hook = fake.add_package(p)
                raw = json.dumps(hook).encode()
                ts = int(time.time())
                t0 = time.perf_counter()
                r = tc.post("/v1/packages", content=raw, headers={
                    "X-SMARTEP-Org": p["classroom"]["orgRef"], "X-SMARTEP-Timestamp": str(ts),
                    "X-SMARTEP-Signature": sign("s", ts, raw), "Content-Type": "application/json"})
                acks.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 202, r.text
            t_work = time.perf_counter()
            while tc.app.state.worker.run_once():
                pass
            work_s = time.perf_counter() - t_work
            latencies = [tc.app.state.store.get(p["packageId"])["latency_ms"] for p in hold]
            counts = tc.app.state.store.counts()
        stop.set()

    for r in fake.results:
        AssessmentResult.model_validate(r)
    mb = 1024 * 1024
    out = {
        "packages": len(hold), "delivered": counts.get("delivered", 0), "failed": counts.get("failed", 0),
        "results_valid": len(fake.results),
        "startup_s": round(startup_s, 2),
        "webhook_ack_ms": {"p50": round(float(np.median(acks)), 1), "max": round(max(acks), 1)},
        "processing_ms": {"p50": round(float(np.median(latencies)), 1),
                          "p95": round(float(np.percentile(latencies, 95)), 1), "max": round(max(latencies), 1)},
        "throughput_per_min": round(len(hold) / work_s * 60, 1),
        "rss_mb": {"before_start": round(rss0 / mb), "ready": round(rss_ready / mb), "peak": round(peak[0] / mb)},
        "g3_targets": {"processing_under_30s": max(latencies) < 30_000, "memory_under_1500mb": peak[0] < 1500 * mb},
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
