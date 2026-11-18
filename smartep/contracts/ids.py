"""Pseudonymisation of raw identifiers.

Evidence packages already carry pseudonymous refs. Use this only when SMARTEP itself has to derive an
identifier from a raw value. The salt comes from the environment, never from source code.
"""

from __future__ import annotations

import hashlib
import os

SALT_ENV = "SMARTEP_ID_SALT"


def get_salt() -> str:
    salt = os.environ.get(SALT_ENV)
    if not salt:
        raise RuntimeError(f"{SALT_ENV} is not set; see .env.example")
    return salt


def pseudonymize(raw_id: str, salt: str | None = None, length: int = 16) -> str:
    salt = salt if salt is not None else get_salt()
    return hashlib.sha256(f"{salt}:{raw_id}".encode()).hexdigest()[:length]
