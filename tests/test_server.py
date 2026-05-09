from fastapi.testclient import TestClient

from goorouter.server import build_app
from tests.test_router import _cfg


def test_v1_models_lists_virtual_and_backends(tmp_path):
    cfg = _cfg()
    app = build_app(cfg, db_path=tmp_path / "log.sqlite")
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    ids = {m["id"] for m in data["data"]}
    expected = {"goo-auto", "goo-explain"} | {f"goo-{name}" for name in cfg.backends}
    assert ids == expected
