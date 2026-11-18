"""Webhook signatures and URL allow-listing.

Signature: ``X-SMARTEP-Signature: sha256=<hex>`` where hex = HMAC-SHA256(secret, f"{timestamp}.{raw body}"),
and ``X-SMARTEP-Timestamp`` is Unix seconds. Requests older or newer than the tolerance are rejected, so a
captured request cannot be replayed later.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlparse


def sign(secret: str, timestamp: str | int, body: bytes) -> str:
    mac = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def verify_signature(secret: str, timestamp: str | None, body: bytes, signature: str | None,
                     tolerance_s: int, now: float | None = None) -> str | None:
    """Return None when valid, otherwise the reason it is not."""
    if not timestamp or not signature:
        return "missing signature headers"
    try:
        ts = int(timestamp)
    except ValueError:
        return "bad timestamp"
    if abs((now if now is not None else time.time()) - ts) > tolerance_s:
        return "timestamp outside tolerance"
    if not hmac.compare_digest(sign(secret, ts, body), signature):
        return "signature mismatch"
    return None


def url_allowed(url: str, allowed_hosts: tuple[str, ...], allow_insecure_http: bool = False) -> bool:
    """https only (http allowed for local development), and the host must be on the organisation's list."""
    parsed = urlparse(url)
    if parsed.scheme != "https" and not (allow_insecure_http and parsed.scheme == "http"):
        return False
    return (parsed.hostname or "").lower() in {h.lower() for h in allowed_hosts}
