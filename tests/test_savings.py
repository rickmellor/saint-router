"""`saint savings`: baseline selection, per-seat energy, bench tok/s overrides, renamed backends."""
from pathlib import Path

import pytest

from saint import savings as S
from saint.config import load_config
from saint.storage import LogRow, log_request, open_db

TOML = """
[server]
host = "127.0.0.1"
port = 4000

[backends.local-coder]
provider = "openai"
base_url = "http://localhost:8002/v1"
model = "qwen"
api_key = "x"
aliases = ["coder"]
timeout_s = 60
gpus = 2
tok_s = 80

[backends.local-worker]
provider = "openai"
base_url = "http://localhost:8006/v1"
model = "qwen"
api_key = "x"
aliases = []
timeout_s = 60
gpus = 2

[backends.cloud-large]
provider = "anthropic"
model = "claude-opus-5-5"
api_key_env = "ANTHROPIC_API_KEY"
aliases = ["opus"]
timeout_s = 120
price_in = 4.0
price_out = 20.0

[backends.cloud-exquisite]
provider = "anthropic"
model = "claude-fable-5"
api_key_env = "ANTHROPIC_API_KEY"
aliases = ["fable", "flagship", "exquisite"]
timeout_s = 120
price_in = 10.0
price_out = 50.0

[classifier]
backend = "local-coder"
max_input_chars = 8000
timeout_s = 5

[routing]
default_urgency = "normal"
default_on_failure = "local-coder"
[routing.policy.normal]
"code,trivial"    = "local-coder"
"code,medium"     = "local-coder"
"code,hard"       = "cloud-large"
"general,trivial" = "local-coder"
"general,medium"  = "local-coder"
"general,hard"    = "cloud-large"
[routing.policy.urgent]
"code,trivial"    = "local-coder"
"code,medium"     = "cloud-large"
"code,hard"       = "cloud-large"
"general,trivial" = "cloud-large"
"general,medium"  = "cloud-large"
"general,hard"    = "cloud-large"
[routing.policy.patient]
"code,trivial"    = "local-coder"
"code,medium"     = "local-coder"
"code,hard"       = "local-coder"
"general,trivial" = "local-coder"
"general,medium"  = "local-coder"
"general,hard"    = "local-coder"

[logging]
db_path = "DBPATH"
prompt_storage = "full"

[energy]
price_kwh  = 0.36
host_watts = 1500
gpu_watts  = 300
total_gpus = 6
"""


def _row(backend, tin, tout, lat, i):
    return LogRow(request_id=f"r-{backend}-{i}", model_field="saint-auto", prefixes_raw=None, pinned_backend=None,
                  urgency_used="normal", classifier_used=None, classifier_fallback_reason=None,
                  classifier_input_chars=10, classifier_input_truncated_from=None, classifier_latency_ms=1,
                  classifier_domain="code", classifier_complexity="medium", classifier_reason=None,
                  backend_chosen=backend, backend_latency_ms=lat, tokens_in=tin, tokens_out=tout,
                  success=True, error_kind=None, prompt_content="p", prompt_storage_mode="full")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("SAINT_TEST_LOCAL", "1")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(TOML.replace("DBPATH", str(tmp_path / "log.sqlite")))
    cfg = load_config(cfg_path)
    conn = open_db(Path(cfg.logging.db_path))
    for i in range(6):
        log_request(conn, _row("local-coder", 200, 1000, 10_000, i))    # 100 tok/s in the log, 80 in config
        log_request(conn, _row("local-worker", 200, 500, 10_000, i))    # 50 tok/s, log-measured only
    log_request(conn, _row("cloud-large", 1000, 100, 3000, 0))
    log_request(conn, _row("cloud-flagship", 1000, 100, 3000, 0))       # renamed backend, priced via alias
    return cfg, conn


def test_baseline_never_a_local_seat(env):
    cfg, conn = env
    rep = S.compute(conn, cfg, period="day")
    assert cfg.routing.default_on_failure == "local-coder"
    assert (rep["baseline"], rep["baseline_how"]) == ("cloud-large", "policy hard tier")
    assert rep["total_if_all_cloud"] > 0 and rep["savings"] > 0
    assert S.pick_baseline(cfg, "cloud-exquisite") == ("cloud-exquisite", "flag")
    with pytest.raises(ValueError):
        S.pick_baseline(cfg, "cloud-nope")


def test_local_energy_is_per_seat_and_prefers_bench_tok_s(env):
    cfg, conn = env
    rows = {r.backend: r for r in S.compute(conn, cfg, period="day")["rows"]}
    coder, worker = rows["local-coder"], rows["local-worker"]
    # 2 GPUs x 300 W; host base = max(0, 1500 - 6x300) = 0 -> 600 W, not the whole 1500 W host
    assert coder.watts == worker.watts == 600
    assert (coder.tok_s, coder.source) == (80, "bench")
    assert (worker.tok_s, worker.source) == (50, "log")
    assert coder.elec_per_mtok == pytest.approx(600 * 0.36 / (80 * 3.6))
    assert worker.elec_per_mtok == pytest.approx(600 * 0.36 / (50 * 3.6))
    assert coder.tok_per_w == pytest.approx(80 / 600)
    assert coder.actual_cost == pytest.approx(6000 / 1e6 * coder.elec_per_mtok)


def test_renamed_backend_is_priced_through_its_alias(env):
    cfg, conn = env
    rep = S.compute(conn, cfg, period="day")
    fl = next(r for r in rep["rows"] if r.backend == "cloud-flagship")
    assert fl.kind == "cloud" and fl.priced_as == "cloud-exquisite"
    assert fl.actual_cost == pytest.approx((1000 * 10.0 + 100 * 50.0) / 1e6)
    assert any("cloud-flagship priced as cloud-exquisite" in n for n in rep["notes"])


def test_render_shows_baseline_reason_and_energy_lines(env):
    cfg, conn = env
    text = S.render(S.compute(conn, cfg, period="day"), color=False)
    assert "baseline cloud-large (policy hard tier)" in text
    assert "tok/W" in text and "[bench]" in text and "[log]" in text
    assert "OVERSPEND" not in text
