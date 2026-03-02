---
title: BTZSC Leaderboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.8.0
python_version: 3.12
app_file: app.py
pinned: false
---

# BTZSC Leaderboard Space

This Space renders the BTZSC leaderboard from result JSON files stored in the dataset repo.

BTZSC evaluates strict zero-shot text classification across:

- 22 English single-label datasets
- 4 task families: sentiment, topic, intent, emotion
- 4 major model families: embedding models, NLI cross-encoders, rerankers, and instruction-tuned LLMs

Ranking policy:

- Primary ranking metric: macro-F1
- Additional displayed metrics: accuracy, macro-precision, macro-recall

Data source configuration:

- Default dataset repo: `btzsc/btzsc-results`
- Override with env var: `BTZSC_RESULTS_REPO`

Useful links:

- Paper (OpenReview): https://openreview.net/forum?id=IxMryAz2p3
- Eval harness (GitHub): https://github.com/IliasAarab/btzsc
- Results dataset: https://huggingface.co/datasets/btzsc/btzsc-results

## How to Add a New Leaderboard Entry

This Space is read-only: it displays whatever is present in the results dataset.

To add a new entry, submit a valid JSON result file to:

- https://huggingface.co/datasets/btzsc/btzsc-results
- Required path: `results/<model_type>/<model-name>.json`

Submission options:

- Web UI upload on the dataset page
- Git workflow (direct push or fork + PR)
- `huggingface_hub` API upload with `create_pr=True` (PR-based)

Full submission requirements are documented in:

- https://huggingface.co/datasets/btzsc/btzsc-results/blob/main/SUBMISSION.md

Once merged, the new entry appears in this Space on refresh/update.

For local testing, set:

- `BTZSC_RESULTS_LOCAL_DIR=/path/to/results_repo`

Then run:

```bash
python app.py
```
