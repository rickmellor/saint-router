from saint.policy import resolve_policy

POLICY = {
    "normal": {
        "code,trivial": "local-coder",
        "code,medium": "local-coder",
        "code,hard": "cloud-large",
        "general,trivial": "local-small",
        "general,medium": "local-large",
        "general,hard": "cloud-large",
    },
    "urgent": {
        "code,trivial": "local-coder",
        "code,medium": "cloud-small",
        "code,hard": "cloud-large",
        "general,trivial": "cloud-small",
        "general,medium": "cloud-small",
        "general,hard": "cloud-large",
    },
    "patient": {
        "code,trivial": "local-coder",
        "code,medium": "local-coder",
        "code,hard": "local-large",
        "general,trivial": "local-small",
        "general,medium": "local-small",
        "general,hard": "local-large",
    },
}


def test_resolve_normal_code_medium():
    assert resolve_policy(POLICY, "normal", "code", "medium") == "local-coder"


def test_resolve_urgent_general_medium():
    assert resolve_policy(POLICY, "urgent", "general", "medium") == "cloud-small"


def test_resolve_patient_code_hard():
    assert resolve_policy(POLICY, "patient", "code", "hard") == "local-large"
