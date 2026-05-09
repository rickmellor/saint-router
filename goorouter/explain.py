from __future__ import annotations

from goorouter.config import Config
from goorouter.router import RoutingDecision


def format_decision(decision: RoutingDecision, cfg: Config) -> str:
    lines: list[str] = []
    lines.append(f"Routing decision: {decision.backend}")
    backend = cfg.backends.get(decision.backend)
    if backend:
        lines.append(f"  Target: {backend.provider} {backend.model}")
    lines.append("")

    lines.append(f"Prefixes parsed:    {decision.parsed.raw or '(none)'}")
    lines.append(f"Effective urgency:  {decision.urgency}")

    if decision.pinned_backend:
        lines.append(f"Pinned backend:     {decision.pinned_backend} (classifier skipped)")
    elif decision.multimodal:
        lines.append(
            f"Multimodal content: routed to default_on_failure ({cfg.routing.default_on_failure})"
        )
    elif decision.classifier_outcome is None:
        lines.append("Classifier:         skipped (empty or no user message)")
    else:
        out = decision.classifier_outcome
        lines.append(f"Classifier used:    {out.classifier_used or '(both failed)'}")
        if out.fallback_reason:
            lines.append(f"Fallback reason:    {out.fallback_reason}")
        truncation_note = (
            f" (truncated from {out.input_truncated_from})"
            if out.input_truncated_from else ""
        )
        lines.append(f"Classifier input:   {out.input_chars} chars{truncation_note}")
        if out.result:
            lines.append(
                f"Classifier output:  domain={out.result.domain} "
                f"complexity={out.result.complexity}"
            )
            lines.append(f"Classifier reason:  {out.result.reason}")
            lines.append(f"Classifier latency: {out.result.latency_ms}ms")
            lines.append(
                f"Policy lookup:      "
                f'policy.{decision.urgency}["{out.result.domain},{out.result.complexity}"]'
                f" = {decision.backend}"
            )
        else:
            lines.append(
                f"All classifiers failed; using default_on_failure → {decision.backend}"
            )

    lines.append("")
    lines.append("(No tokens consumed at the destination backend.)")
    return "\n".join(lines)
