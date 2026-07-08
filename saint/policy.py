from __future__ import annotations


def resolve_policy(
    policy: dict[str, dict[str, str]],
    urgency: str,
    domain: str,
    complexity: str,
) -> str:
    """Look up the destination backend name for (urgency, domain, complexity).

    Caller is responsible for having validated the policy table at config-load time.
    """
    return policy[urgency][f"{domain},{complexity}"]
