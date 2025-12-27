# tests/test_basic.py
import os, sys, types, importlib
from fastapi.testclient import TestClient

# Prepare a safe test environment: provide a fake transformers module
# and ensure a token env var is set so app startup doesn't abort.
os.environ.setdefault("HF_TOKEN", "hf_test_token")
os.environ.setdefault("HF_REPO_ID", "sakibalfahim/disaster-tweets-bert")

# Create a tiny fake transformers module that returns minimal objects.
class DummyTokenizer:
    def __init__(self, *a, **k): pass
    def __call__(self, texts, truncation=True, padding=True, return_tensors="pt"):
        # minimal structure expected by app: keys exist (values will not be used)
        return {"input_ids": None, "attention_mask": None}
    def save_pretrained(self, path): pass

class DummyModel:
    def __init__(self, *a, **k): pass
    def to(self, device): pass
    def eval(self): pass
    def __call__(self, **kwargs):
        # Provide logits in the shape the app expects. Use plain Python lists;
        # the app converts logits via torch.nn.functional.softmax when running
        # real model. For our tests we only check shape and response structure.
        import types
        return types.SimpleNamespace(logits=[[0.2, 0.8]])

# Inject the fake module before the app is imported
mod = types.ModuleType("transformers")
mod.AutoTokenizer = lambda *a, **k: DummyTokenizer()
mod.AutoModelForSequenceClassification = lambda *a, **k: DummyModel()
sys.modules["transformers"] = mod

# Now import the FastAPI app and run tests
app = importlib.import_module("app.main").app
client = TestClient(app)

def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json().get("status") == "ok"

def test_predict_single_text():
    payload = {"text": "Huge fire reported near the forest edge."}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "predictions" in body and isinstance(body["predictions"], list)
    assert "confidences" in body and isinstance(body["confidences"], list)
    assert body["predictions"][0] in ["Not Disaster", "Disaster"]
    assert isinstance(body["latency_ms"], (int, float))
