from goorouter.config import BackendConfig, Config, ClassifierConfig, RoutingConfig, ServerConfig, LoggingConfig


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
            db_path="~/.goorouter/log.sqlite",
            prompt_storage="full",
        ),
    )
    assert cfg.backends["cloud-large"].aliases == ("opus", "claude")
    assert cfg.routing.default_on_failure == "cloud-large"
