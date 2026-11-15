"""
Vercel Python function: runs the registered model inside the device.

The model (461 MB) exceeds Vercel's 100 MB deploy limit, so it is fetched to
/tmp on first use and cached for the life of the instance.

Peak memory stays under the Hobby cap (2048 MB) because ONNX runs with the CPU
memory arena and memory pattern disabled, INT4 weights, and sequential execution.
"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler

# Vercel's deployment filesystem is read-only. All caches must live in /tmp.
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

MODEL_REPO = os.environ.get("MODEL_ID", "onnx-community/Qwen2.5-0.5B-Instruct")
MODEL_DIR = "/tmp/models"
DEVICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "device")

_model = None


def ensure_model():
    if os.path.exists(os.path.join(MODEL_DIR, "onnx", "model_q4f16.onnx")):
        return
    from huggingface_hub import hf_hub_download
    for f in ("onnx/model_q4f16.onnx", "tokenizer.json", "config.json", "generation_config.json"):
        hf_hub_download(repo_id=MODEL_REPO, filename=f, local_dir=MODEL_DIR)


def get_model():
    global _model
    if _model is None:
        ensure_model()
        sys.path.insert(0, DEVICE_DIR)
        from infer import Model
        _model = Model(MODEL_DIR)
    return _model


class handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._json(200, {"ok": True})

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        prompt = (q.get("prompt") or [None])[0]
        max_tokens = int((q.get("max_tokens") or ["64"])[0])
        if not prompt:
            return self._json(200, {
                "device": "infer-device-python",
                "model": MODEL_REPO,
                "usage": "GET or POST with prompt and max_tokens",
                "region": os.environ.get("VERCEL_REGION", "unknown"),
            })
        self._run(prompt, max_tokens)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            body = {}
        prompt = body.get("prompt")
        max_tokens = int(body.get("max_tokens") or 64)
        if not prompt:
            return self._json(400, {"error": "prompt required"})
        self._run(prompt, max_tokens)

    def _run(self, prompt, max_tokens):
        try:
            t0 = time.time()
            m = get_model()
            r = m.chat(prompt, min(max(max_tokens, 1), 256))
            r["device"] = "infer-device-python"
            r["runtime"] = "Vercel Python function"
            r["region"] = os.environ.get("VERCEL_REGION", "unknown")
            r["wall_ms"] = int((time.time() - t0) * 1000)
            self._json(200, r)
        except Exception as e:
            self._json(500, {"device": "infer-device-python", "error": str(e)[:500]})

    def log_message(self, *a):
        pass
