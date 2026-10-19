from __future__ import annotations

from saint.capabilities import missing
from saint.config import Config
from saint.router import RoutingDecision


def format_decision(decision: RoutingDecision, cfg: Config, resolver=None) -> str:
    lines: list[str] = []
    lines.append(f"Routing decision: {decision.backend}")
    backend = cfg.backends.get(decision.backend)
    if backend and (backend.provider or backend.model):
        lines.append(f"  Target: {backend.provider} {backend.model}")
    lines.append("")

    lines.append(f"Prefixes parsed:    {decision.parsed.raw or '(none)'}")
    lines.append(f"Effective urgency:  {decision.urgency}")
    lines.append(f"Required caps:      {', '.join(sorted(decision.required_capabilities)) or '(none)'}")

    if getattr(decision, "client_rule", None):
        lines.append(f"Client rule:        {decision.client_rule} → {decision.backend} (classifier skipped)")

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

    # johnny binding (liveness-aware; reflects reality, never triggers a load)
    if resolver is not None:
        from saint.binding import describe_for_explain

        d = describe_for_explain(cfg, decision.backend, resolver)
        if d and d.get("state") == "pool":
            lines.append("")
            for m in d.get("members", []):
                seat = f" [seat {m['seat']}]" if m.get("seat") else ""
                caps = (" caps " + ",".join(sorted(m.get("capabilities") or []))
                        if m.get("capabilities") else "")
                lines.append(f"  member {m['name']}: {m['state']}{seat}{caps}")
            b = cfg.backends.get(decision.backend)
            if b is not None:
                mode = "sticky" if b.pool_sticky else "per-request"
                lines.append(f"  next pick:          {d.get('next') or '(none)'} "
                             f"({b.pool_strategy}, {mode})")
        elif d:
            lines.append("")
            lines.append(f"johnny seat:        {d.get('seat') or '(none)'} [{d['state']}]")
            caps = frozenset(d.get("capabilities") or ())
            if caps:
                lines.append(f"Seat capabilities:  {', '.join(sorted(caps))}")
            lacking = missing(decision.required_capabilities, caps)
            if lacking:
                lines.append(f"  !! this backend lacks: {','.join(sorted(lacking))} "
                             f"— dispatch will skip it")
            if d.get("eta_s") is not None:
                lines.append(f"  eta_s:            {d['eta_s']}")
            if d["state"] == "ready":
                lines.append(f"  override:         {d.get('endpoint')} / {d.get('model')}")
                lines.append(f"  static baseline:  {d.get('static_baseline')}")
                lines.append("  would serve via:  johnny override (johnny_ready)")
            else:
                lines.append(f"  would serve via:  {d['served_by']} ({d['state_at_dispatch']})")

    lines.append("")
    lines.append("(No tokens consumed at the destination backend.)")
    return "\n".join(lines)
