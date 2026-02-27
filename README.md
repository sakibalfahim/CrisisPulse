# Disaster Tweet Classifier & API

[![CI](https://github.com/sakibalfahim/disaster-tweets-api/actions/workflows/ci.yml/badge.svg)](https://github.com/sakibalfahim/CrisisPulse/actions)
[Model (HF)](https://huggingface.co/sakibalfahim/CrisisPulse) · [Demo (Space)](https://huggingface.co/spaces/sakibalfahim/CrisisPulse)

## One-line
BERT-based binary classifier (Disaster vs Not Disaster) with a production-ready FastAPI service, Docker image, CI, and deployment notes.

---

## Table of contents
- [Quick local smoke test (Docker)](#quick-local-smoke-test-docker)
- [Google Cloud Run (recommended production flow)](#google-cloud-run-recommended-production-flow)
- [GitHub Actions / CI](#github-actions--ci)
- [API spec](#api-spec)
- [Environment variables & secrets](#environment-variables--secrets)
- [Testing](#testing)
- [Production sizing & ops](#production-sizing--ops)
- [Observability [and miscellaneous]](#observability-and-miscellaneous)
- [License & contact](#license--contact)

---

## Quick local smoke test (Docker)
Build and run locally to verify the API:
```bash
# build
docker build -t disaster-tweets-api:local .

# run (replace HF_TOKEN)
docker run --rm -p 8080:8080 \
  -e HF_TOKEN="hf_xxx" -e HF_REPO_ID="sakibalfahim/disaster-tweets-bert" \
  disaster-tweets-api:local

# test
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Huge explosion reported near the financial district — multiple injuries."}'
```

If you have GPU access in your environment and want GPU in Docker, add `--gpus all` and ensure appropriate base image.

---

## Google Cloud Run (recommended production flow)
Build and push with Cloud Build and deploy with Cloud Run. Replace `PROJECT_ID`, `REGION`, and `hf_xxx`.
```bash
gcloud auth login
gcloud config set project PROJECT_ID

# Build container in GCP
gcloud builds submit --tag gcr.io/PROJECT_ID/disaster-tweets-api:latest

# Deploy (recommended sizing)
gcloud run deploy disaster-tweets-api \
  --image=gcr.io/PROJECT_ID/disaster-tweets-api:latest \
  --platform=managed \
  --region=REGION \
  --allow-unauthenticated \
  --set-env-vars HF_REPO_ID="sakibalfahim/disaster-tweets-bert" \
  --memory=4Gi --cpu=2 --concurrency=1
```
**Secrets:** store `HF_TOKEN` as a Cloud Run secret and mount it into the service; do not put the token directly in CLI history.

---

## GitHub Actions / CI
Current workflow `.github/workflows/ci.yml`:
- Builds Docker image using `docker/build-push-action`
- Pushes image to GHCR (configured)  
  Next CI improvement: run `pytest` inside a lightweight test container to validate quick smoke tests (tests included in `tests/test_basic.py`).

---

## API spec
**Base:** `GET /health` and `POST /predict`  
- `GET /health`

Returns service readiness and device info.
```bash
{"status":"ok","device":"cuda"|"cpu"}
```
- `POST /predict`

Request:
```bash
{
  "text": "Single tweet string"
}
```
or
```bash
{
  "text": ["tweet1", "tweet2"]
}
```
Response:
```bash
{
  "predictions": ["Disaster"],
  "confidences": [{"Disaster":0.987,"Not Disaster":0.013}],
  "latency_ms": 123.45
}
```

---

## Environment variables & secrets
- `HF_TOKEN` — Hugging Face token (READ permission). **Store as secret**.
- `HF_REPO_ID` — model repo id, default `sakibalfahim/disaster-tweets-bert`.

Do not commit tokens. Use cloud secret managers or GitHub Secrets for CI.

---

## Testing
A lightweight test file `tests/test_basic.py` is included that fake-injects a minimal `transformers` shim so CI can run fast without downloading the real model.
Run locally:
```bash
pip install -r requirements.txt
pip install pytest httpx
pytest -q
```
CI will run container build; next step is to run tests inside the image before pushing.

---

## Production sizing & ops
- Model artifact ~400+MB (safetensors). Memory usage depends on batch size and device.
- Start with: `--memory=4Gi`, `--cpu=2`, `--concurrency=1`. Increase memory to `8Gi` if using larger batches.
- Use concurrency=1 to avoid memory competition inside a single instance.
- If you expect sustained high QPS, use autoscaling with a reasonable min instance count to mitigate cold starts.

---

## Observability [and miscellaneous]
- `/metrics` exposes basic in-memory metrics (requests, avg latency). Add Prometheus exporter or forward logs to Cloud Monitoring.
- Add health-checks & readiness for autoscalers.
- Implement request rate limiting + authentication for production.

---

## Security
- Keep `HF_TOKEN` secret and rotate periodically.
- Consider private network / API gateway for production.
- Validate and sanitize inputs; set a max length for text to avoid resource abuse.

---

## Next enhancements (roadmap)
- CI: run `pytest` inside built image; fail fast for regressions.
- Add async batching & queueing to maximize GPU throughput.
- Add model warmup & quantization options (ORT, bitsandbytes) to reduce memory and improve latency.
- Add monitoring (Prometheus), tracing, and alerting.

---

## Files of interest
- `app/main.py` — FastAPI app and inference logic
- `Dockerfile` — production image
- `.github/workflows/ci.yml` — CI build & push
- `tests/test_basic.py` — smoke tests
- `README.md` — this file

---

## License & contact
MIT License

Author: `sakibalfahim` — contact via mail, LinkedIn, GitHub, or Hugging Face profile.
