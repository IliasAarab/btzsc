---
pretty_name: BTZSC Leaderboard Results
license: apache-2.0
tags:
  - leaderboard
  - text-classification
  - zero-shot
  - evaluation
language:
  - en
---

# BTZSC Results

This repository stores model submissions for the **BTZSC** leaderboard:

**BTZSC: A Benchmark for Zero-Shot Text Classification across Cross-Encoders, Embedding Models, Rerankers and LLMs**.

- Paper: https://openreview.net/pdf?id=IxMryAz2p3
- Eval harness: https://github.com/btzsc/btzsc
- Leaderboard Space: https://huggingface.co/spaces/btzsc/btzsc-leaderboard

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
- `results.by_dataset`: per-dataset metric blocks (ground truth)

## Contributing results

See [SUBMISSION.md](SUBMISSION.md) for exact instructions.

PRs adding result files are validated in CI with `validate.py`.
