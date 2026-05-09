from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI

from goorouter.config import Config
from goorouter.storage import open_db


def build_app(cfg: Config, *, db_path: Path) -> FastAPI:
    app = FastAPI(title="goorouter", version="0.1.0")
    app.state.cfg = cfg
    app.state.db = open_db(db_path)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ids = ["goo-auto", "goo-explain", *(f"goo-{name}" for name in cfg.backends)]
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "owned_by": "goorouter", "created": 0}
                for mid in ids
            ],
        }

    return app
