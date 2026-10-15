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

## Laya (2026-09-26) — the classifier since 2026-09-26

`classifier.mode = "laya"`: Convai's open **Laya** decision model (ModernBERT-large + typed decision head, Apache 2.0)
fine-tuned on the rubric-v3 labels **with its own decision head and request-time question kept** (`saint/laya_classifier.py`,
transport `saint/jev_classifier.py`). Served locally by `laya-serve` pinned to the checkpoint (`tools/laya_saint_serve.py`,
systemd --user `laya-saint.service`, `:8130`, Jev `/v1/systemone` wire protocol). Zero-shot Laya was useless (60–75 %);
fine-tuned it replaced the nomic head and the NVIDIA feature sidecar (retired the same day — `tools/prompt_features/server.py`
now serves nomic-embed only unless started with `--features`).

| classifier (2,584 rubric-v3 prompts) | cloud/local routing | coverage @ gate 0.8 | acc on covered | latency |
|---|---|---|---|---|
| nomic head + NVIDIA + lexical (was deployed) | 94.6 % (5-fold) | ~83 % | 96.5 % | ~60 ms + 400 ms gemma deferrals |
| **Laya full fine-tune, domain + complexity in one pass (deployed)** | **97.1 %** (3-fold) | **96.4 %** | **97.6 %** | ~22 ms |

- **The question text is part of the model's input.** `saint-training.json` next to the checkpoint holds the exact
  instructions + option descriptions it was trained with; `classifier.laya_questions_path` points at it and SAINT sends it
  verbatim (re-read on mtime change).
- Chain: laya → LLM labeller (`laya_fallthrough = "llm"`), or → embedding head → LLM with `"embedding"`. Laya rows never
  train the head (`storage.fetch_training_rows`); the LLM labeller stays the reference. `saint classifier status --drift`
  replays Laya over the LLM-labelled rows since `labels_since` and reports domain / complexity / **cloud-local** / routing
  agreement; the retrain flag + `x-saint-retrain` carry Laya's fix.
- Train / export / deploy (~6–10 min on the 4080, labels outside the repo; needs `pip install "laya[serve]"` in a torch env and
  ~12 GB free VRAM — stop the seats on that card first; `--bs 8` hit OOM at 14.9 GB on 2026-10-14, `--bs 4` trains in 11.6 GB).
  When the drift flag names disagreeing prompts: dump them, adjudicate against the rubric, append the rows (with the agreeing ones)
  to the traffic label file, retrain, then bump `classifier.labels_since` past them — the drift check treats the LLM labeller as the
  reference, so adjudications that overrule it keep showing as "disagreements" until they leave the window:

      python tools/classifier_eval/laya_finetune.py --variant full --questions both --epochs 4 --head-lr 1e-4 --no-cv --final --tag v4
      python tools/classifier_eval/laya_export_checkpoint.py ~/.config/saint/classifier_eval/laya-ft/laya_saint_v4.pt \
             ~/.config/saint/classifier_eval/laya-ft/laya-saint-v4 ~/.config/saint/classifier_eval/laya-ft/laya_saint_v4.json
      # point laya-saint.service (LAYA_SAINT_MODEL) and classifier.laya_questions_path at the new dir, then
      systemctl --user restart laya-saint.service && systemctl --user restart saint.service

  `laya_finetune.py` without `--no-cv` runs the 3-fold A/B first (`--variant head|full|both`, `--split gold2traffic`,
  `--max-len`, `--shuffle-options`); `laya_zeroshot_ab.py` is the zero-shot / frozen-encoder study, `laya_wording_sensitivity.py`
  shows how much the fine-tuned model still reads the question, `nomic_gold2traffic.py` is the head's baseline on the
  distribution-shift split. Train with fp32 weights under bf16 autocast — pure-bf16 weight training underfits (that recipe
  produced the pessimistic ModernBERT-base number above).
- What it does **not** do: reduce retraining. Trained on the gold seeds only and tested on traffic, Laya (88.3 %) generalises
  no better than the head (90.4 %); it wins once it has seen real traffic. Rollback: `mode = "embedding"` (the nomic-only
  head retrained 2026-09-26 without NVIDIA features).
