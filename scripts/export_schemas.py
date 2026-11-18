"""Write JSON Schema files for the contracts into schemas/.

The pydantic models in smartep/contracts are the source of truth on the SMARTEP side; these files are
what The Classroom's Zod schema is compared against. Run after changing a contract:

    python scripts/export_schemas.py
"""

from __future__ import annotations

import json
from pathlib import Path

from smartep.contracts import AssessmentResult, EvidencePackage

SCHEMAS = {
    "evidence-package.v1.json": EvidencePackage,
    "assessment-result.v1.json": AssessmentResult,
}
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def render(model) -> str:
    schema = model.model_json_schema()
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> None:
    SCHEMA_DIR.mkdir(exist_ok=True)
    for name, model in SCHEMAS.items():
        (SCHEMA_DIR / name).write_text(render(model), encoding="utf-8")
        print("wrote", SCHEMA_DIR / name)


if __name__ == "__main__":
    main()
