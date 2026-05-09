import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch


SAMPLE_CFG = """
[server]
host = "127.0.0.1"
port = 4000

[backends.cloud-large]
provider = "anthropic"
model = "claude-opus-4-7"
api_key_env = "ANTHROPIC_API_KEY"
aliases = ["opus"]
timeout_s = 120

[backends.local-small]
provider = "openai"
base_url = "http://localhost:1234/v1"
model = "qwen2.5-3b-instruct"
api_key = "lm-studio"
timeout_s = 60

[classifier]
backend = "local-small"
max_input_chars = 8000
timeout_s = 5

[routing]
default_urgency = "normal"
default_on_failure = "cloud-large"

[routing.policy.normal]
"code,trivial"    = "local-small"
"code,medium"     = "local-small"
"code,hard"       = "cloud-large"
"general,trivial" = "local-small"
"general,medium"  = "local-small"
"general,hard"    = "cloud-large"

[routing.policy.urgent]
"code,trivial"    = "local-small"
"code,medium"     = "cloud-large"
"code,hard"       = "cloud-large"
"general,trivial" = "cloud-large"
"general,medium"  = "cloud-large"
"general,hard"    = "cloud-large"

[routing.policy.patient]
"code,trivial"    = "local-small"
"code,medium"     = "local-small"
"code,hard"       = "local-small"
"general,trivial" = "local-small"
"general,medium"  = "local-small"
"general,hard"    = "local-small"

[logging]
db_path = "{db}"
prompt_storage = "full"
"""


def test_cli_help_lists_commands():
    out = subprocess.run([sys.executable, "-m", "goorouter", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for cmd in ("serve", "explain", "policy", "config", "log", "relabel"):
        assert cmd in out.stdout


def test_cli_serve_help():
    out = subprocess.run([sys.executable, "-m", "goorouter", "serve", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    out_low = out.stdout.lower()
    assert "host" in out_low or "port" in out_low


def test_cli_explain_prints_decision(tmp_path):
    cfg_path = tmp_path / "config.toml"
    db_path = tmp_path / "log.sqlite"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    payload = json.dumps({"domain": "code", "complexity": "hard", "reason": "novel refactor"})
    response = {"choices": [{"message": {"content": payload}}]}

    from typer.testing import CliRunner

    from goorouter import cli as cli_mod
    runner = CliRunner()
    with patch("goorouter.classifier.call_backend", AsyncMock(return_value=response)):
        result = runner.invoke(cli_mod.app, ["explain", "rewrite my code", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Routing decision" in result.output
    assert "code" in result.output and "hard" in result.output


def test_cli_policy_show(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=(tmp_path / "log.sqlite").as_posix()))
    from typer.testing import CliRunner

    from goorouter import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["policy", "show", "--config", str(cfg_path)])
    assert result.exit_code == 0
    for urgency in ("normal", "urgent", "patient"):
        assert urgency in result.output
    assert "code,hard" in result.output


def test_cli_config_show_masks_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key-do-not-print")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=(tmp_path / "log.sqlite").as_posix()))
    from typer.testing import CliRunner

    from goorouter import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.app, ["config", "show", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "secret-key-do-not-print" not in result.output
    assert "Cloud backends present" in result.output


def test_cli_log_show_and_id(tmp_path):
    from goorouter.storage import log_request, open_db
    from tests.test_storage import _row

    db_path = tmp_path / "log.sqlite"
    conn = open_db(db_path)
    log_request(conn, _row(request_id="row-A", backend_chosen="local-coder"))
    log_request(conn, _row(request_id="row-B", backend_chosen="cloud-large"))
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    from typer.testing import CliRunner

    from goorouter import cli as cli_mod
    runner = CliRunner()

    r1 = runner.invoke(cli_mod.app, ["log", "show", "--config", str(cfg_path)])
    assert r1.exit_code == 0, r1.output
    assert "row-A" in r1.output and "row-B" in r1.output

    r2 = runner.invoke(cli_mod.app, ["log", "show", "--backend", "local-coder",
                                      "--config", str(cfg_path)])
    assert r2.exit_code == 0
    assert "row-A" in r2.output and "row-B" not in r2.output

    r3 = runner.invoke(cli_mod.app, ["log", "id", "1", "--config", str(cfg_path)])
    assert r3.exit_code == 0
    assert "row-A" in r3.output


def test_cli_relabel_last_and_by_id(tmp_path):
    from goorouter.storage import get_by_id, log_request, open_db
    from tests.test_storage import _row

    db_path = tmp_path / "log.sqlite"
    conn = open_db(db_path)
    log_request(conn, _row(request_id="r1"))
    log_request(conn, _row(request_id="r2"))
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    from typer.testing import CliRunner

    from goorouter import cli as cli_mod
    runner = CliRunner()

    r = runner.invoke(cli_mod.app, ["relabel", "last", "cloud-large",
                                     "--note", "wrong",
                                     "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output

    conn = open_db(db_path)
    last = get_by_id(conn, 2)
    assert last is not None and last["relabel_backend"] == "cloud-large"

    r = runner.invoke(cli_mod.app, ["relabel", "by-id", "1", "local-small",
                                     "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output
    first = get_by_id(conn, 1)
    assert first is not None and first["relabel_backend"] == "local-small"


def test_cli_relabel_undefined_backend_rejected(tmp_path):
    from goorouter.storage import log_request, open_db
    from tests.test_storage import _row

    db_path = tmp_path / "log.sqlite"
    conn = open_db(db_path)
    log_request(conn, _row())
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SAMPLE_CFG.format(db=db_path.as_posix()))

    from typer.testing import CliRunner

    from goorouter import cli as cli_mod
    runner = CliRunner()
    r = runner.invoke(cli_mod.app, ["relabel", "last", "phantom",
                                     "--config", str(cfg_path)])
    assert r.exit_code != 0
    assert "phantom" in r.output
