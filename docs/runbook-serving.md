# Runbook: SMARTEP API

## What runs

One container from this repo's `Dockerfile`: uvicorn serving `smartep.api.app:app` on port 8000, with a worker
thread in the same process (one copy of the model in memory). SQLite job queue at `/data/smartep.db` (mount a
volume). The model is `models/gap-model/` inside the image. Measured on a laptop CPU (50 packages):
processing p95 141 ms, peak memory 241 MB, start-up 7 s.

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `SMARTEP_ORGS` (or `SMARTEP_ORGS_FILE`) | yes | per-organisation secrets and allowed hosts (docs/api.md §5) |
| `SMARTEP_ADMIN_TOKEN` | recommended | Bearer token for `GET /v1/packages/{id}` |
| `SMARTEP_DB_PATH` | no (`/data/smartep.db`) | job queue file; keep it on a volume |
| `SMARTEP_MODEL_DIR` | no (`/app/models/gap-model`) | model bundle |
| `SMARTEP_ALLOW_INSECURE_HTTP` | no | `1` allows http URLs; local development only |

Secrets live only in Dokploy's environment settings, never in the repo.

## Deploy (Dokploy, cloud.smartep.co.ke)

1. Application from GitHub `njerudan/smartep-model`, branch `main`, build type **Dockerfile**, port 8000.
2. Volume: mount `/data` (named volume, e.g. `smartep-data`).
3. Environment: `SMARTEP_ORGS`, `SMARTEP_ADMIN_TOKEN`.
4. Domain `api.smartep.co.ke` → port 8000, HTTPS (Let's Encrypt). Cloudflare: A record `api` → server IP.
5. Deploy; check `https://api.smartep.co.ke/health` shows the expected model version.

## Release a new model

1. Train: `python -m pipelines.train --config configs/gap_model.yaml` (promotion recorded in `artifacts/registry/`).
2. Export the champion: `python scripts/export_model.py` (writes `models/gap-model/` and the golden fixtures).
3. `pytest` (golden predictions and the quality check must pass), update the model card, commit, push.
4. Redeploy in Dokploy; `/health` shows the new version.

## Roll back

Redeploy the previous commit in Dokploy (Deployments → previous → redeploy), or revert the model commit and push.
Jobs in the queue survive restarts; a job interrupted mid-analysis is picked up again on start.

## Re-run or inspect a package

- Status: `curl -H "Authorization: Bearer $SMARTEP_ADMIN_TOKEN" https://api.smartep.co.ke/v1/packages/<id>`.
- A failed package is not retried automatically. The Classroom re-sends it as a new package (spec §3.5:
  packages are immutable); SMARTEP treats a repeated `packageId` as a duplicate.

## Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| 401 on every webhook | secret or orgRef mismatch, or server clock skew > 5 min | compare `SMARTEP_ORGS` with The Classroom's org setting; check NTP |
| 422 `downloadUrl host is not allowed` | object-storage host not in `allowed_download_hosts` | add the host |
| job `failed: checksum mismatch` | package changed after the webhook was signed | The Classroom re-sends |
| job `failed: resultsCallbackUrl host is not allowed` | Classroom domain changed (spec §11) | add the new host |
| jobs stuck in `delivering` | Classroom results endpoint down or returning 5xx | retries continue for 24 h; fix the endpoint |
| `/health` fails at start | model bundle missing or library mismatch | rebuild the image from a commit with `models/gap-model/` |
