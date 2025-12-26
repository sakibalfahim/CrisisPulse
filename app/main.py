# app/main.py
import os
import time
import logging
from typing import List, Union, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- logging ----
LOG = logging.getLogger("disaster_api")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
LOG.addHandler(handler)

# ---- config ----
REPO_ID = os.environ.get("HF_REPO_ID", "sakibalfahim/disaster-tweets-bert")
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # may be None if model is public
LABEL_MAP = {0: "Not Disaster", 1: "Disaster"}

app = FastAPI(title="Disaster-Tweets-API", version="1.0")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lazy-loaded globals
_tokenizer = None
_model = None

# simple runtime metrics
app.state.requests = 0
app.state.total_latency = 0.0

# ---- models / helpers ----
def is_model_loaded() -> bool:
    return (_tokenizer is not None) and (_model is not None)

def load_model():
    global _tokenizer, _model
    if is_model_loaded():
        return
    LOG.info("Loading tokenizer and model from %s (token provided: %s)", REPO_ID, bool(HF_TOKEN))
    token_arg = {"token": HF_TOKEN} if HF_TOKEN else {}
    _tokenizer = AutoTokenizer.from_pretrained(REPO_ID, use_fast=True, **token_arg)
    _model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, **token_arg)
    _model.to(_device)
    _model.eval()
    LOG.info("Model loaded, device=%s", _device)

def predict_texts(texts: List[str]) -> List[Dict[str, Any]]:
    if not is_model_loaded():
        load_model()
    inputs = _tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    results = []
    for p in probs:
        idx = int(p.argmax())
        results.append({"label": LABEL_MAP.get(idx, str(idx)), "score": float(p[idx])})
    return results

# ---- request / response models ----
class PredictRequest(BaseModel):
    text: Union[str, List[str]]

class PredictResponse(BaseModel):
    predictions: List[str]
    confidences: List[Dict[str, float]]
    latency_ms: float

# ---- endpoints ----
@app.get("/health")
def health():
    """Always-available lightweight health check (does NOT load model)."""
    return {"status": "ok", "device": str(_device)}

@app.get("/ready")
def ready():
    """Reports whether the model is loaded and service is ready for real inference."""
    return {"ready": is_model_loaded()}

@app.get("/metrics")
def metrics():
    reqs = getattr(app.state, "requests", 0)
    avg_latency = (app.state.total_latency / reqs) if reqs > 0 else 0.0
    return {"requests": reqs, "avg_latency_ms": avg_latency*1000.0, "model_loaded": is_model_loaded()}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    texts = req.text if isinstance(req.text, list) else [req.text]
    if not texts or any((t is None or str(t).strip() == "") for t in texts):
        raise HTTPException(status_code=400, detail="text must be a non-empty string or list of strings")

    start = time.time()
    try:
        results = predict_texts(texts)
    except Exception as e:
        LOG.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.time() - start
    app.state.requests += 1
    app.state.total_latency += latency

    preds = [r["label"] for r in results]
    confidences = [{LABEL_MAP[0]: float(1 - r["score"]), LABEL_MAP[1]: float(r["score"])} for r in results]

    return PredictResponse(predictions=preds, confidences=confidences, latency_ms=latency*1000.0)
