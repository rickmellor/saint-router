# Jev (TypeSafe AI) as a SAINT classifier — evaluation and how to switch

Status 2026-09-21: **implemented behind `classifier.mode = "jev"`, not yet validated live** — TypeSafe is in early
access (waitlist) and there is no API key on specul8. Everything below the "What Jev is" section that concerns
accuracy is a plan, not a result.

## Why look at it
Live numbers for the current two-tier classifier (embedding head → Haiku labeller), 2026-09-21:

| path | share of fresh classifications | mean latency |
|---|---|---|
| turn / conversation cache | (210 of 383 logged) | 0 ms |
| embedding head, confident | 48 % | 343 ms (docs say ~30 ms; cold seat + johnny resolve) |
| deferred to Haiku | 52 % | 1 418 ms |
| 1B CPU fallback | rare | 7.8 s |

Drift check: routing agreement **80.3 %** (threshold 90 %), domain 91.8 %, **complexity 55.7 %**. The head is close
to blind on the medium/hard boundary — the axis that decides local vs cloud — and half of requests pay ~1.4 s.
Note: it is not a cosine/centroid classifier; it is two multinomial logistic-regression heads on L2-normalised
768-d nomic embeddings, confidence = min of the two softmax maxima, gate 0.6.

## What Jev is (sources: typesafe.ai launch post, docs.typesafe.ai, 2026-09-15 →)
- A non-autoregressive "System One" model: `POST https://api.typesafe.ai/v1/systemone` with `state` (text) and a map
  of typed `questions` (`choice` ≤ 255 options, `score`, `noul`); all answered in one parallel pass. Each answer
  carries `choice`, a full `probabilities` map and a `confidence`. Trained with "RLCD" for calibrated probabilities;
  no calibration metrics (ECE/Brier) are published.
- Claims: 70–500 ms end to end, $0.042 per million input tokens, output free, 64K context (32K state), 1 200 req/min.
  No public accuracy benchmarks; vendor comparisons are against frontier LLMs wrapped for structured output.
- **API only.** No open weights, no self-hosting, no fine-tuning ("the same weights serve every account"); behaviour is
  steered only by question wording and option descriptions. `jev-latest` can change without notice → we pin `jev-1.13.0`.
- Vendor-documented weaknesses (Jev 1.13 "jaggedness"): literal reading, no arithmetic/date reasoning, accuracy decays
  with irrelevant context ("context rot"), follows instructions embedded in the data, weaker outside English.
- Privacy: states it does not train on user data; zero-retention is an enterprise option. Detailed DPA not reviewed.

## Fit for SAINT
For: one call answers both axes with real probabilities, so the gate is principled; if the latency claim holds it
replaces a 1.4 s Haiku deferral with a sub-500 ms call; cost is negligible (a 2K-token prompt ≈ $0.00008);
`ignore_after` + `max_input_chars` already trim the input, which is exactly what its context-rot weakness wants.

Against: it adds a WAN dependency and sends the last user message off the LAN on every uncached request (the
embedding head is fully local; Haiku deferrals already leave the LAN, but only for the uncertain half);
70–500 ms is not better than the head's confident path; no way to train it on Rick's domains (the head has 901
domain-specific seed prompts); complexity ("would a 30B local model handle this?") is a judgment about *our fleet*,
which Jev can only learn from option descriptions; prompt-injection in user text can steer the label.

## Design
`mode = "jev"` → `saint/jev_classifier.py` asks two `choice` questions in one request, confidence = min of the two,
gate = `jev_min_confidence` or `min_confidence`. `None` (unsure) or any `JevError` (no key, timeout, non-200, bad
JSON, unexpected label) falls through to the embedding head when `embedding_backend` is configured, then the LLM
labeller; `fallback_reason` records the path (`jev_defer`, `jev_error`, `jev_defer+embedding_defer`, …) and
`classifier_used = "typesafe-jev"` marks Jev-labelled rows, which are excluded from head training and drift checks
so the existing self-improvement loop keeps measuring head vs LLM labels.

```toml
[classifier]
mode = "jev"                 # was "embedding"; switch back any time
jev_model = "jev-1.13.0"     # pinned
jev_timeout_s = 3.0
# jev_min_confidence = 0.6   # defaults to min_confidence
# everything else (embedding_backend, head_path, backend, fallback_backend) stays as it is
```
Add `TYPESAFE_API_KEY=…` to `~/.config/saint/env`, then `systemctl --user restart saint.service`.

## Validation plan (needs a key)
1. Offline replay before any live traffic: run the logged prompts that have Haiku labels (the drift-check rows)
   through Jev; report agreement per axis, routing agreement, coverage at gates 0.5/0.6/0.7, latency p50/p99, and the
   confusion on medium↔hard. Compare with the head's 80.3 / 91.8 / 55.7 on the same rows.
2. Check calibration on those rows (reliability bins) — the product's central claim.
3. Tune the option descriptions once against the misses (they are the only lever), re-run.
4. Only then flip `mode` for a day and compare `saint log stats` + `x-saint-*` headers.
Decision rule suggestion: adopt if routing agreement ≥ 90 % with ≥ 85 % coverage and p99 < 800 ms; otherwise keep
`embedding` and consider using Jev labels as a cheaper *teacher* for the head instead of Haiku.
