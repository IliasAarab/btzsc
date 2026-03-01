<p align="center">
	<img src="./docs/" align="center" width="30%" alt="BTZSC banner">
</p>
<h1 align="center">BTZSC</h1>
<p align="center">
	<em>A unified benchmark for zero-shot text classification across modern model families.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/last-commit/IliasAarab/btzsc?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/IliasAarab/btzsc?style=default&color=0080ff" alt="repo-top-language">
</p>
<br>

<details><summary>Table of Contents</summary>

- [Overview](#overview)
- [Dataset](#dataset)
- [Install](#install)
- [Quick start (Python)](#quick-start-python)
- [Quick start (CLI)](#quick-start-cli)
- [Custom models](#custom-models)
- [Packaging & release](#packaging--release)
- [Notes](#notes)

</details>
<hr>

## Overview

BTZSC is a benchmark package for zero-shot text classification across embedding models, NLI cross-encoders, rerankers, and instruction-tuned LLMs.

## Dataset

https://huggingface.co/datasets/btzsc/btzsc

### Install

```bash
uv sync
```

### Quick start (Python)

```python
from btzsc import BTZSCBenchmark

benchmark = BTZSCBenchmark(tasks=["sentiment", "topic"])
results = benchmark.evaluate(
	model="intfloat/e5-base-v2",
	model_type="embedding",
	batch_size=64,
)

print(results.summary())
print(results.per_dataset())
```

### Quick start (CLI)

```bash
btzsc evaluate --model intfloat/e5-base-v2 --type embedding --tasks sentiment,topic
btzsc baselines --metric f1 --top 10
btzsc list-datasets
```

### Custom models

Subclass `BaseModel` and implement:
- `predict_scores(texts, labels, batch_size)`
- `predict(texts, labels, batch_size)`

Then pass your instance to `BTZSCBenchmark.evaluate()`.

### Packaging & release

Build locally:

```bash
uv build
```

Publish from GitHub using tags (workflow: `.github/workflows/publish.yml`):

1. Bump `version` in `pyproject.toml`.
2. Commit and push to `master`.
3. Create and push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions will build with `uv` and publish to PyPI via trusted publishing.

For first-time setup on PyPI:

- Create the project on PyPI.
- Add a trusted publisher for this repository.
- Set workflow path to `.github/workflows/publish.yml`.

### Notes

- The BTZSC dataset stores rows as `(text, hypothesis, labels)` where `labels` is binary entailment.
- The package reconstructs grouped multiclass samples internally.
- Baseline tables are bundled from the published BTZSC paper runs.
