"""A stand-in for The Classroom, for tests, local demos and the serving benchmark.

It serves gzip packages at ``https://media.example.test/files/<packageId>.json.gz``, answers the
refresh-url call, and accepts results at the package's ``resultsCallbackUrl``. Use it as an httpx
transport: ``httpx.Client(transport=httpx.MockTransport(fake.handler))``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta

import httpx

MEDIA_HOST = "media.example.test"
APP_HOST = "app.example.test"


class FakeClassroom:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.files: dict[str, bytes] = {}
        self.results: list[dict] = []
        self.callback_failures: list[int] = []   # status codes to return before accepting
        self.download_status = 200
        self.refresh_calls = 0

    def add_package(self, pkg: dict, *, expires_in_h: float = 24) -> dict:
        """Store the package and return the webhook body The Classroom would send for it."""
        raw = gzip.compress(json.dumps(pkg).encode())
        self.files[pkg["packageId"]] = raw
        return {
            "packageId": pkg["packageId"], "schemaVersion": "1.0", "educatorRef": pkg["educator"]["educatorRef"],
            "trigger": pkg["trigger"], "downloadUrl": f"https://{MEDIA_HOST}/files/{pkg['packageId']}.json.gz",
            "downloadExpiresAt": (datetime.now(UTC) + timedelta(hours=expires_in_h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "checksum": "sha256:" + hashlib.sha256(raw).hexdigest(), "byteSize": len(raw),
            "classroomBaseUrl": f"https://{APP_HOST}",
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == MEDIA_HOST and request.method == "GET" and path.startswith("/files/"):
            if self.download_status != 200:
                return httpx.Response(self.download_status)
            pid = path.removeprefix("/files/").removesuffix(".json.gz")
            return httpx.Response(200, content=self.files[pid]) if pid in self.files else httpx.Response(404)
        if host == APP_HOST and request.headers.get("authorization") != f"Bearer {self.api_key}":
            return httpx.Response(401, json={"error": "bad api key"})
        if host == APP_HOST and path.endswith("/refresh-url"):
            self.refresh_calls += 1
            self.download_status = 200
            pid = path.split("/")[-2]
            return httpx.Response(200, json={"downloadUrl": f"https://{MEDIA_HOST}/files/{pid}.json.gz"})
        if host == APP_HOST and path == "/v1/assessment/results" and request.method == "POST":
            if self.callback_failures:
                return httpx.Response(self.callback_failures.pop(0), json={"error": "simulated"})
            self.results.append(json.loads(request.content))
            return httpx.Response(201, json={"ok": True})
        return httpx.Response(404)
