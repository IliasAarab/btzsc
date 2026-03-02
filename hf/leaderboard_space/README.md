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

This Space renders the BTZSC leaderboard from result JSON files stored in the dataset repo:

- Default dataset repo: `btzsc/btzsc-results`
- Override with env var: `BTZSC_RESULTS_REPO`

For local testing, set:

- `BTZSC_RESULTS_LOCAL_DIR=/path/to/results_repo`

Then run:

```bash
python app.py
```
