# Fully local classifier (kind-of-work rubric)

SAINT's classifier can run with no cloud dependency at all: a local LLM seat is the labeller,
and the millisecond head is trained from that labeller's output.

## The rubric: who should do the work

`tools/classifier_eval/classifier.prompt.v3` defines `complexity` by the **kind** of work, not
its difficulty:

| complexity | meaning | typical policy target |
|---|---|---|
| `hard` | thinking work: **ideation, architecture, validation, orchestration** | cloud |
| `medium` | manual work (the default): implement, debug, refactor, run, summarize, extract, explain | local |
| `trivial` | greetings, acknowledgements, one-line lookups | local |

Difficulty alone never makes a prompt `hard`; machine-generated utility prompts (summaries,
extraction, compaction) are `medium`. Install it with `classifier.prompt_template_path`.

## Pieces

| piece | what | config |
|---|---|---|
| labeller | any local chat seat (a 26B-class model agrees with a much larger judge on 96% of cloud/local calls) | `classifier.backend` |
| head | logistic head over the prompt embedding **+ optional extra features** | `classifier.mode = "embedding"` |
| `saint-features` sidecar | `tools/prompt_features/server.py`: NVIDIA prompt-task-and-complexity-classifier probabilities (`/features`) and an OpenAI-compatible nomic-embed `/v1/embeddings`, both on one GPU (~1.1 GB, ~10–25 ms) | `classifier.feature_url`, an `[backends.*]` entry for the embeddings |
| oversize handling | `oversize = "truncate"` clips long prompts to head+tail and classifies them normally | `classifier.oversize` |
| training data | curated label files + logged labels newer than a cut-off | `classifier.label_files`, `classifier.labels_since` |

If the sidecar is down the head abstains and the LLM labeller answers; if the labeller is
down too the request goes to `routing.default_on_failure` — point that at a local backend.

## Building labels

Real prompts are personal data: keep label files **outside the repo**
(`~/.config/saint/classifier_eval/`); the eval folder's `.gitignore` refuses them.

    cd tools/classifier_eval
    # 1. two local labellers vote on every distinct logged prompt (thinking off, resumable)
    python relabel.py --template classifier.prompt.v3 --out relabel-v3.jsonl
    python relabel.py --template classifier.prompt.v3 --out relabel-v3-gold.jsonl --source gold
    # 2. hand-adjudicate only the prompts where the votes disagree on hard vs not-hard
    # 3. A/B the candidates and pick the gate (needs a torch env + the sidecar)
    python train_ab.py labels.jsonl --bert
    python gate_sweep.py labels.jsonl
    # 4. train the production head
    saint classifier train

`eval_labeller.py BACKEND --template … --sets …` scores any backend as a labeller.

## Measured (2026-09, 2,584 prompts, 5-fold CV)

| classifier | cloud/local routing correct |
|---|---|
| nomic embedding + head | 94.0 % |
| + lexical cues | 94.4 % |
| + NVIDIA features + lexical (deployed) | 94.6 % |
| NVIDIA features alone | 85.5 % |
| ModernBERT-base fine-tune | 88.8 % |

With `min_confidence = 0.8` the head answers ~83 % of requests at 96.5 % cloud/local accuracy
(~60 ms); the rest defer to the labeller (~400 ms). Most of the gain over the previous setup came
from the rubric and the labels, not from the extra model.
