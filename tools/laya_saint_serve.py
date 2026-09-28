"""laya-serve pinned to SAINT's fine-tuned checkpoint: Jev-compatible POST /v1/systemone.
env: LAYA_SAINT_MODEL (checkpoint dir), LAYA_DEVICE (cuda|cpu|xpu), LAYA_PORT, LAYA_HOST, LAYA_API_KEY, LAYA_THREADS."""
import os, uvicorn
from laya.router import Router
from laya.serve import create_app, _apply_thread_limit
_apply_thread_limit()
path = os.environ.get("LAYA_SAINT_MODEL", os.path.expanduser("~/.config/saint/classifier_eval/laya-ft/laya-saint-v3"))
router = Router(models={"english": path}, device=os.environ.get("LAYA_DEVICE") or None, max_loaded=1, lang_guess="en")
router.preload(["english"])
uvicorn.run(create_app(router), host=os.environ.get("LAYA_HOST", "127.0.0.1"), port=int(os.environ.get("LAYA_PORT", "8130")), log_level=os.environ.get("LAYA_LOG_LEVEL", "warning"))
