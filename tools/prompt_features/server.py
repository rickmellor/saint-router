"""saint-features: tiny local sidecar serving NVIDIA prompt-task-and-complexity-classifier probabilities.

SAINT itself stays torch-free; this runs in any torch env and answers
    POST /features       {"input": ["prompt", ...]}  ->  {"features": [[32 floats], ...], "dim": 32}
    POST /v1/embeddings  OpenAI-compatible; nomic-embed-text-v1.5 (mean-pooled, L2-normalized) — the same model the
                         CPU `embed` seat serves, ~10-20 ms on a GPU instead of seconds
    GET  /health, /v1/models
Usage:  python server.py [--port 8005] [--device cuda|cpu] [--model DIR]
"""
from __future__ import annotations

import argparse, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classifier_eval"))
import nvidia_features as nv  # noqa: E402

LOCK = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8005); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--model", default=None)
    ap.add_argument("--embed-model", default="/mnt/data/models/nomic-ai/nomic-embed-text-v1.5"); ap.add_argument("--embed-max-tokens", type=int, default=2048); a = ap.parse_args()
    if a.model: nv.MODEL = Path(a.model)
    _, tok, net = nv.load(a.device)

    def features(texts: list[str]) -> list[list[float]]:
        with LOCK, torch.no_grad():
            enc = tok([t[:6000] for t in texts], return_tensors="pt", max_length=512, padding=True, truncation=True).to(a.device)
            return np.concatenate([p.float().cpu().numpy() for p in net(enc["input_ids"], enc["attention_mask"])], 1).round(5).tolist()

    from transformers import AutoModel, AutoTokenizer
    etok = AutoTokenizer.from_pretrained(a.embed_model)
    enet = AutoModel.from_pretrained(a.embed_model, trust_remote_code=True, torch_dtype=torch.float16 if a.device == "cuda" else torch.float32).to(a.device).eval()

    def embed(texts: list[str]) -> tuple[list[list[float]], int]:
        out, ntok = [], 0
        with LOCK, torch.no_grad():
            for i in range(0, len(texts), 8):
                enc = etok(texts[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=a.embed_max_tokens).to(a.device)
                m = enc["attention_mask"].unsqueeze(-1); h = enet(**enc)[0].float(); v = (h * m).sum(1) / m.sum(1)
                out += torch.nn.functional.normalize(v, dim=-1).cpu().tolist(); ntok += int(enc["attention_mask"].sum())
        return out, ntok

    features(["warm up"]); embed(["warm up"])

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode(); self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/v1/models"):
                return self._send(200, {"object": "list", "data": [{"id": "nomic-embed", "object": "model"}]})
            self._send(200, {"status": "ok", "device": a.device, "dim": 32})

        def do_POST(self):  # noqa: N802
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                if self.path.startswith("/v1/embeddings"):
                    inp = req["input"]; vecs, ntok = embed([inp] if isinstance(inp, str) else list(inp))
                    return self._send(200, {"object": "list", "model": "nomic-embed", "usage": {"prompt_tokens": ntok, "total_tokens": ntok},
                                            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]})
                f = features(list(req["input"]))
                self._send(200, {"features": f, "dim": len(f[0])})
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": str(e)[:200]})

        def log_message(self, *_):
            pass

    print(f"saint-features on {a.host}:{a.port} ({a.device})", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
