---
pretty_name: BTZSC Leaderboard Results
license: mit
tags:
  - leaderboard
  - text-classification
  - zero-shot
  - evaluation
language:
  - en
---

# BTZSC Results

This repository stores model submissions for the **BTZSC** leaderboard.

**BTZSC: A Benchmark for Zero-Shot Text Classification across Cross-Encoders, Embedding Models, Rerankers and LLMs**.

- Paper: https://openreview.net/forum?id=IxMryAz2p3
- Eval harness: https://github.com/IliasAarab/btzsc
- Leaderboard Space: https://huggingface.co/spaces/btzsc/btzsc-leaderboard

Benchmark summary:

- 22 English single-label datasets
- 4 task families: sentiment, topic, intent, emotion
- Strict zero-shot protocol (no BTZSC-label training/tuning)
- Primary metric: macro-F1

## What this repo contains

- One JSON file per model evaluation run in `results/<model_type>/<model-name>.json`
- Reproducibility metadata (BTZSC version, commit, precision, batch size)
- Full per-dataset metrics for all 22 BTZSC datasets

## Schema

Each submission follows schema version `1.0` with:

- `model`: model id, type, parameter count, revision
- `evaluation`: harness versioning and runtime metadata
- `results.overall`: averaged macro F1 / accuracy / macro precision / macro recall
- `results.by_task`: sentiment/topic/intent/emotion aggregates
- `results.by_dataset`: per-dataset metric blocks

## Contributing results

Destination path format:

- `results/<model_type>/<model-name>.json`

Recommended flow:

1. Export with the official harness (`btzsc evaluate ... --output-json ...`).
2. Validate locally (`python validate.py results/<model_type>/<model-name>.json`).
3. Add your file at the required path.
4. Submit by one of these methods:
  - Web UI upload on Hugging Face (no clone required)
  - Git workflow (direct push if you have write access, otherwise fork + PR)
  - API workflow via `huggingface_hub` with `create_pr=True` (PR-based)

In short: **add** means placing the JSON at the correct path; **submit** means publishing that change to this remote repo.

See [SUBMISSION.md](SUBMISSION.md) for full requirements and review checks.

PRs adding result files are validated in CI with `validate.py`.
