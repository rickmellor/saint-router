from pathlib import Path

import pytest

from saint.config import (
    BackendConfig,
    ClassifierConfig,
    Config,
    LoggingConfig,
    RoutingConfig,
    ServerConfig,
    load_config,
)

EXAMPLE_TOML = """
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
aliases = []
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
db_path = "${TEST_HOME}/log.sqlite"
prompt_storage = "full"
"""


def test_config_dataclass_construction():
    cfg = Config(
        server=ServerConfig(host="127.0.0.1", port=4000),
        backends={
            "cloud-large": BackendConfig(
                name="cloud-large",
                provider="anthropic",
                model="claude-opus-4-7",
                api_key_env="ANTHROPIC_API_KEY",
                api_key=None,
                base_url=None,
                aliases=("opus", "claude"),
                timeout_s=120,
            )
        },
        classifier=ClassifierConfig(
            backend="cloud-large",
            fallback_backend=None,
            max_input_chars=8000,
            timeout_s=5,
            prompt_template_path=None,
        ),
        routing=RoutingConfig(
            default_urgency="normal",
            default_on_failure="cloud-large",
            policy={
                "normal": {"code,trivial": "cloud-large"},
                "urgent": {"code,trivial": "cloud-large"},
                "patient": {"code,trivial": "cloud-large"},
            },
        ),
        logging=LoggingConfig(
            db_path="~/.config/saint/log.sqlite",
            prompt_storage="full",
        ),
    )
    assert cfg.backends["cloud-large"].aliases == ("opus", "claude")
    assert cfg.routing.default_on_failure == "cloud-large"


def test_load_config_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(EXAMPLE_TOML)
    cfg = load_config(cfg_path)
    assert cfg.server.port == 4000
    assert "cloud-large" in cfg.backends
    assert cfg.backends["cloud-large"].aliases == ("opus",)
    assert Path(cfg.logging.db_path) == Path(tmp_path) / "log.sqlite"


def test_load_config_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    toml = EXAMPLE_TOML.replace('db_path = "${TEST_HOME}/log.sqlite"', 'db_path = "~/log.sqlite"')
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml)
    cfg = load_config(cfg_path)
    assert Path(cfg.logging.db_path) == Path(tmp_path) / "log.sqlite"


def test_validate_missing_classifier_backend(tmp_path):
    toml = EXAMPLE_TOML.replace('backend = "local-small"', 'backend = "phantom"')
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "phantom" in str(e.value)


def test_validate_missing_policy_cell(tmp_path):
    # Drop one cell from policy.normal: change "general,hard" line into a different urgency block opening
    toml = EXAMPLE_TOML.replace(
        '"general,hard"    = "cloud-large"\n\n[routing.policy.urgent]',
        "[routing.policy.urgent]",
        1,
    )
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "general,hard" in str(e.value)


def test_validate_undefined_backend_in_policy(tmp_path):
    toml = EXAMPLE_TOML.replace(
        '"code,hard"       = "cloud-large"',
        '"code,hard"       = "phantom"',
        1,
    )
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "phantom" in str(e.value)


def test_validate_default_on_failure_undefined(tmp_path):
    toml = EXAMPLE_TOML.replace(
        'default_on_failure = "cloud-large"',
        'default_on_failure = "phantom"',
    )
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "phantom" in str(e.value)


def test_validate_collects_all_errors(tmp_path):
    toml = EXAMPLE_TOML.replace('backend = "local-small"', 'backend = "phantom1"')
    toml = toml.replace('default_on_failure = "cloud-large"', 'default_on_failure = "phantom2"')
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    msg = str(e.value)
    assert "phantom1" in msg and "phantom2" in msg


def test_validate_invalid_default_urgency(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    toml = EXAMPLE_TOML.replace('default_urgency = "normal"', 'default_urgency = "wat"')
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "default_urgency" in str(e.value) and "wat" in str(e.value)


def test_validate_invalid_prompt_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    toml = EXAMPLE_TOML.replace('prompt_storage = "full"', 'prompt_storage = "loud"')
    p = tmp_path / "c.toml"; p.write_text(toml)
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "prompt_storage" in str(e.value) and "loud" in str(e.value)


def test_load_config_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(ValueError) as e:
        load_config(missing)
    assert "does-not-exist.toml" in str(e.value) or str(missing) in str(e.value)


def test_load_config_malformed_toml(tmp_path):
    p = tmp_path / "broken.toml"
    p.write_text("this is = = not = valid = toml [[[")
    with pytest.raises(ValueError) as e:
        load_config(p)
    assert "broken.toml" in str(e.value) or str(p) in str(e.value)


def test_resolve_backend_by_name_or_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    p = tmp_path / "c.toml"; p.write_text(EXAMPLE_TOML)
    cfg = load_config(p)
    from saint.config import resolve_backend

    assert resolve_backend(cfg, "cloud-large").name == "cloud-large"
    assert resolve_backend(cfg, "opus").name == "cloud-large"
    assert resolve_backend(cfg, "nope") is None


def _write_cfg_with_cache(tmp_path, cache_block: str):
    import tests.test_cli as tc
    p = tmp_path / "config.toml"
    p.write_text(tc.SAMPLE_CFG.format(db=(tmp_path / "log.sqlite").as_posix()) + "\n" + cache_block)
    return p


def test_cache_config_defaults(tmp_path):
    from saint.config import load_config
    cfg = load_config(_write_cfg_with_cache(tmp_path, ""))
    assert cfg.cache.turn_cache is True
    assert cfg.cache.turn_ttl_s == 300.0
    assert cfg.cache.conversation_ttl_s == 900.0
    assert cfg.cache.short_follow_up_max_chars == 40
    assert cfg.cache.sticky_conversations is False
    assert cfg.cache.anthropic_prompt_caching is True
    assert cfg.cache.anthropic_cache_ttl is None


def test_cache_config_parses_and_normalizes(tmp_path):
    from saint.config import load_config
    cfg = load_config(_write_cfg_with_cache(tmp_path, """
[cache]
turn_ttl_s = 120
conversation_affinity = false
sticky_conversations = true
anthropic_cache_ttl = "5m"
prompt_cache_min_chars = 1000
"""))
    assert cfg.cache.turn_ttl_s == 120.0
    assert cfg.cache.conversation_affinity is False
    assert cfg.cache.sticky_conversations is True
    assert cfg.cache.anthropic_cache_ttl is None  # "5m" normalizes to default
    assert cfg.cache.prompt_cache_min_chars == 1000


def test_cache_config_validation_errors(tmp_path):
    import pytest
    from saint.config import load_config
    with pytest.raises(ValueError, match="turn_ttl_s"):
        load_config(_write_cfg_with_cache(tmp_path, "[cache]\nturn_ttl_s = 0\n"))
    with pytest.raises(ValueError, match="anthropic_cache_ttl"):
        load_config(_write_cfg_with_cache(tmp_path, '[cache]\nanthropic_cache_ttl = "2h"\n'))
    with pytest.raises(ValueError, match="short_follow_up_max_chars"):
        load_config(_write_cfg_with_cache(tmp_path, "[cache]\nshort_follow_up_max_chars = -1\n"))


def test_hardening_knobs_parse_and_validate(tmp_path):
    import pytest
    from saint.config import load_config
    cfg = load_config(_write_cfg_with_cache(tmp_path, ""))
    assert cfg.routing.retry_same_backend is True
    assert cfg.routing.breaker_failures == 3
    assert cfg.routing.multimodal_backend is None
    # unknown fallback / self-referential on_error / bad prices / unknown embeddings backend
    base = _write_cfg_with_cache(tmp_path, "").read_text()
    bad1 = base.replace('[backends.local-small]', '[backends.local-small]\non_error = "nope"')
    p = tmp_path / "bad1.toml"; p.write_text(bad1)
    with pytest.raises(ValueError, match="on_error 'nope'"):
        load_config(p)
    bad2 = base.replace('[backends.local-small]', '[backends.local-small]\nprice_in = -1')
    p2 = tmp_path / "bad2.toml"; p2.write_text(bad2)
    with pytest.raises(ValueError, match="price_in"):
        load_config(p2)
    bad3 = base + '\n'  # append to [logging]-terminated file is fine for TOML? use routing key
    bad3 = base.replace('[routing]', '[routing]\nembeddings_backend = "ghost"')
    p3 = tmp_path / "bad3.toml"; p3.write_text(bad3)
    with pytest.raises(ValueError, match="embeddings_backend 'ghost'"):
        load_config(p3)


def _bedrock_block(extra: str = "") -> str:
    return f"""
[backends.bedrock-sonnet]
provider = "bedrock"
model = "global.anthropic.claude-sonnet-5"
aws_region = "us-east-1"
aws_profile = "ClaudeCode"
drop_params = ["temperature"]
default_max_tokens = 8192
{extra}
[bedrock]
credential_process = "~/cc-bedrock/credential-process"
"""


def test_bedrock_backend_parses(tmp_path):
    from saint.config import load_config
    cfg = load_config(_write_cfg_with_cache(tmp_path, _bedrock_block()))
    b = cfg.backends["bedrock-sonnet"]
    assert b.provider == "bedrock" and b.aws_region == "us-east-1"
    assert b.aws_profile == "ClaudeCode"
    assert b.drop_params == ("temperature",)
    assert b.default_max_tokens == 8192
    assert cfg.has_bedrock is True
    assert cfg.bedrock.credential_process.endswith("cc-bedrock/credential-process")
    assert cfg.bedrock.spawn_sso_login is True
    assert cfg.bedrock.auth_cooldown_s == 300.0


def test_bedrock_validation_matrix(tmp_path):
    import pytest
    from saint.config import load_config

    def _try(block, match):
        p = tmp_path / f"c{abs(hash(match))}.toml"
        p.write_text(_write_cfg_with_cache(tmp_path, "").read_text() + block)
        with pytest.raises(ValueError, match=match):
            load_config(p)

    _try('\n[backends.b1]\nprovider = "bedrock"\nmodel = "m"\n', "requires aws_region")
    _try('\n[backends.b2]\nprovider = "bedrock"\nmodel = "m"\naws_region = "us-east-1"\n'
         'api_key = "x"\n', "must not set api_key")
    _try('\n[backends.b3]\nprovider = "bedrock"\nmodel = "m"\naws_region = "us-east-1"\n'
         'base_url = "http://x"\n', "must not set base_url")
    # aws fields on a non-bedrock backend
    _try('\n[backends.b5]\nprovider = "openai"\nmodel = "m"\nbase_url = "http://x"\n'
         'api_key = "k"\naws_region = "us-east-1"\n', "bedrock only")
    _try(_bedrock_block().replace('credential_process = "~/cc-bedrock/credential-process"',
                                  "auth_cooldown_s = 0"), "auth_cooldown_s")


def test_no_bedrock_stays_inert(tmp_path):
    from saint.config import load_config
    cfg = load_config(_write_cfg_with_cache(tmp_path, ""))
    assert cfg.has_bedrock is False and cfg.bedrock is None
