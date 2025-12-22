# app/main.py
import os
import time
from typing import List, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO_ID = os.environ.get("HF_REPO_ID", "sakibalfahim/disaster-tweets-bert")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
LABEL_MAP = {0: "Not Disaster", 1: "Disaster"}

app = FastAPI(title="Disaster-Tweets-API", version="1.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    if HF_TOKEN is None:
        raise RuntimeError("HF_TOKEN environment variable is required")
    # load tokenizer + model (auth with token)
    app.state.tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=HF_TOKEN)
    app.state.model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, token=HF_TOKEN)
    app.state.model.to(device)
    app.state.model.eval()
    app.state.requests = 0
    app.state.total_latency = 0.0

class PredictRequest(BaseModel):
    text: Union[str, List[str]]

class PredictResponse(BaseModel):
    predictions: List[str]
    confidences: List[dict]
    latency_ms: float

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "device": str(device)}

@app.get("/metrics", tags=["meta"])
def metrics():
    reqs = app.state.requests if hasattr(app.state, "requests") else 0
    avg_latency = (app.state.total_latency / reqs) if reqs>0 else 0.0
    return {"requests": reqs, "avg_latency_ms": avg_latency*1000.0}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest, request: Request):
    texts = req.text if isinstance(req.text, list) else [req.text]
    if not texts or any((t is None or str(t).strip()=="") for t in texts):
        raise HTTPException(status_code=400, detail="text must be a non-empty string or list of strings")

    tokenizer = app.state.tokenizer
    model = app.state.model

    start = time.time()
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    latency = time.time() - start

    preds = [LABEL_MAP[int(p.argmax())] for p in probs]
    confidences = [{ LABEL_MAP[i]: float(p[i]) for i in range(len(p)) } for p in probs]

    # update simple metrics
    app.state.requests += 1
    app.state.total_latency += latency

    return PredictResponse(predictions=preds, confidences=confidences, latency_ms=latency*1000.0)
