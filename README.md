<p align="center">
  <img src="https://raw.githubusercontent.com/IliasAarab/btzsc/main/docs/images/btzsc_benchmark.png" align="center" width="40%" alt="BTZSC banner">
</p>
<h1 align="center">BTZSC</h1>
<p align="center">
  <em>A unified benchmark for zero-shot text classification across embedding models, cross-encoders, rerankers, and LLMs.</em>
</p>
<p align="center">
  <a href="https://github.com/IliasAarab/btzsc/tags"><img src="https://img.shields.io/github/v/tag/IliasAarab/btzsc?style=default&color=0080ff&label=version" alt="version"></a>
  <a href="https://pypi.org/project/btzsc/"><img src="https://img.shields.io/pypi/pyversions/btzsc?style=default&color=0080ff" alt="python-versions"></a>
  <a href="https://github.com/IliasAarab/btzsc/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/IliasAarab/btzsc/ci.yml?branch=main&style=default&label=ci&color=0080ff" alt="ci-status"></a>
  <a href="https://github.com/IliasAarab/btzsc/actions/workflows/publish.yml"><img src="https://img.shields.io/github/actions/workflow/status/IliasAarab/btzsc/publish.yml?style=default&label=publish&color=0080ff" alt="publish-status"></a>
  <a href="https://github.com/IliasAarab/btzsc/blob/main/LICENSE"><img src="https://img.shields.io/github/license/IliasAarab/btzsc?style=default&color=0080ff" alt="license"></a>
</p>
<br>

<details><summary>Table of Contents</summary>

- [Overview](#overview)
- [Paper and Resources](#paper-and-resources)
- [For Users](#for-users)
	- [Installation](#installation)
	- [Quick Start (Python API)](#quick-start-python-api)
	- [Quick Start (CLI)](#quick-start-cli)
	- [Supported Model Types](#supported-model-types)
	- [Extending with Custom Models](#extending-with-custom-models)
	- [Submitting to the Leaderboard](#submitting-to-the-leaderboard)
	- [Benchmark Protocol](#benchmark-protocol)
	- [Dataset](#dataset)
- [Citing](#citing)
- [License](#license)
- [For Developers](#for-developers)
	- [Developer Setup](#developer-setup)
	- [Project Structure](#project-structure)
	- [Quality Checks](#quality-checks)
	- [Packaging and Release](#packaging-and-release)

</details>
<hr>

## Overview

BTZSC is a benchmark package for evaluating zero-shot text classification models under a unified interface.
It helps you compare very different model families using the same datasets, task groupings, and metrics.

It is also the evaluation harness behind the [BTZSC Hugging Face leaderboard](https://huggingface.co/spaces/btzsc/btzsc-leaderboard): you can run the benchmark locally, export a leaderboard-ready JSON artifact, and submit new entries to keep the public results up to date.

The package includes:

- Dataset loaders for BTZSC benchmark tasks.
- A shared benchmark runner across model adapters.
- Built-in adapters for embedding, NLI, reranker, and LLM-style models.
- Baseline comparison utilities and a CLI for reproducible evaluation.

## Paper and Resources

- Paper (OpenReview): https://openreview.net/forum?id=IxMryAz2p3
- PDF: https://openreview.net/pdf?id=IxMryAz2p3
- Eval harness (GitHub): https://github.com/IliasAarab/btzsc
- Leaderboard results dataset: https://huggingface.co/datasets/btzsc/btzsc-results
- Leaderboard Space: https://huggingface.co/spaces/btzsc/btzsc-leaderboard

## For Users

### Installation

Install with `pip`:

```bash
pip install btzsc
```

Install with `uv` in an existing project:

```bash
uv add btzsc
```

Run as a standalone CLI tool with `uvx` (no project install needed):

```bash
uvx btzsc list-datasets
```

### Quick Start (Python API)

Use this as a recommended first workflow:

1. Start with one or two task groups to validate your setup.
2. Inspect summary and per-dataset outputs.
3. Compare against bundled baselines.
4. Export a leaderboard-ready JSON artifact.

API notes:

- `BTZSCBenchmark(tasks=...)` accepts either task groups (`"sentiment"`, `"topic"`, `"intent"`, `"emotion"`) or explicit dataset names. Leave empty to run all datasets.
- `evaluate(model=..., model_type=...)` returns a `BTZSCResults` object.
- `model_type` is required when `model` is a string model ID (if you pass a `BaseModel` instance, you can omit it). Choose from `embedding`, `nli`, `reranker`, `llm`.
- Use `max_samples` for quick smoke tests; increase `batch_size` for throughput if your hardware allows it.

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

# Compare against bundled baselines
print(results.compare_baselines(metric="f1"))

# Export leaderboard-ready JSON
results.to_json("results/embedding/e5-base-v2.json")
```


### Quick Start (CLI)

Equivalent end-to-end CLI flow:

Note: when `--model` is a model ID string, you must also provide `--type`.

```bash
# 1) Explore benchmark metadata
btzsc list-datasets
btzsc list-model-types

# 2) Run an initial benchmark
btzsc evaluate --model intfloat/e5-base-v2 --type embedding --tasks sentiment,topic

# 3) Compare with packaged baselines
btzsc baselines --metric f1 --top 10

# 4) Export JSON for leaderboard submission
btzsc evaluate \
	--model intfloat/e5-base-v2 \
	--type embedding \
	--output-json results/embedding/e5-base-v2.json

# 5) Validate the JSON locally
btzsc validate-result results/embedding/e5-base-v2.json
```

Tip: run a small pilot first, then repeat with your full task scope for final reporting.

### Supported Model Types

BTZSC currently supports these adapter families:

- `embedding`
- `nli`
- `reranker`
- `llm`

Pass the model type explicitly (`model_type` in Python or `--type` in CLI).

### Extending with Custom Models

To make a custom model compatible with BTZSC, implement an adapter that subclasses `BaseModel`.

Contract requirements:

- `predict_scores(texts, labels, batch_size)` must return a score matrix with shape `(len(texts), len(labels))` where higher means more likely.
- `predict(texts, labels, batch_size)` must return predicted label indices with shape `(len(texts),)`.
- Set `model_type` on your class. Use `embedding`, `nli`, `reranker`, or `llm` when applicable.

```python
import numpy as np

from btzsc.models.base import BaseModel


class MyCustomAdapter(BaseModel):
	model_type = "embedding"

	def __init__(self, model_name: str = "my-org/my-model"):
		self.model_name = model_name

	def predict_scores(
		self,
		texts: list[str],
		labels: list[str],
		batch_size: int = 32,
	) -> np.ndarray:
		# Replace this with your real scoring implementation.
		return np.zeros((len(texts), len(labels)), dtype=float)

	def predict(
		self,
		texts: list[str],
		labels: list[str],
		batch_size: int = 32,
	) -> np.ndarray:
		scores = self.predict_scores(texts, labels, batch_size=batch_size)
		return scores.argmax(axis=1)
```

Run it in the benchmark:

```python
from btzsc import BTZSCBenchmark

benchmark = BTZSCBenchmark(tasks=["sentiment", "topic"])
custom_model = MyCustomAdapter("my-org/my-model")

results = benchmark.evaluate(
	model=custom_model,
	batch_size=32,
	max_samples=200,
)

print(results.summary())
results.to_json("results/custom/my-model.json")
```

When you pass a `BaseModel` instance to `evaluate()`, you do not need `model_type=...` in the call.

### Submitting to the Leaderboard

After exporting your JSON (`results.to_json(...)` or `--output-json`), first validate it:

```bash
btzsc validate-result results/<model_type>/<model-name>.json
```

Then publish it to the results dataset repo:

https://huggingface.co/datasets/btzsc/btzsc-results

Required destination path format:

`results/<model_type>/<model-name>.json`

Example:

`results/embedding/e5-base-v2.json`

You can submit using any of these workflows:

1. Web UI (no clone required)

   - Open the results repo page: https://huggingface.co/datasets/btzsc/btzsc-results
   - Go to **Files and versions** and upload your JSON at the required path.
   - If you do not have write access, fork the repo and open a PR.

2. Git workflow (clone/fork + push)

   - Clone (or fork) `btzsc/btzsc-results`, add your JSON at the required path, then push.
   - If you pushed to a fork, open a PR to `btzsc/btzsc-results`.

```bash
git lfs install
git clone https://huggingface.co/datasets/btzsc/btzsc-results
cd btzsc-results

# Copy your exported JSON into the correct folder
mkdir -p results/reranker
cp /path/to/my_result.json results/reranker/my-model.json

git add results/reranker/my-model.json
git commit -m "Add BTZSC results for my-model"
git push
```

3. API workflow (`huggingface_hub`, PR-based)

   - Authenticate first (`huggingface-cli login` or `HF_TOKEN`).
   - `create_pr=True` creates a PR branch instead of pushing directly to `main`.

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="results/reranker/my-model.json",
    path_in_repo="results/reranker/my-model.json",
    repo_id="btzsc/btzsc-results",
    repo_type="dataset",
    commit_message="Add BTZSC results for my-model",
    create_pr=True,
)
```

The leaderboard Space reads from this results dataset and updates as new valid entries are added.

For full submission requirements, see `hf/results_repo/SUBMISSION.md`.

### Benchmark Protocol

BTZSC follows a strict zero-shot protocol:

- 22 English single-label datasets
- 4 task families: sentiment, topic, intent, emotion
- No BTZSC-label training or tuning on evaluation datasets
- Primary leaderboard metric: macro-F1
- Secondary metrics: accuracy, macro-precision, macro-recall

The leaderboard is continuously updated as new submissions are added.

### Dataset

BTZSC benchmark data is available on Hugging Face:

https://huggingface.co/datasets/btzsc/btzsc

To load the raw paired-format rows with `datasets`:

```python
from datasets import get_dataset_config_names, load_dataset

repo_id = "btzsc/btzsc"

# Each dataset is a config name (e.g. "agnews", "imdb", ...)
print(get_dataset_config_names(repo_id)[:5])

# Load one dataset's test split
ds = load_dataset(repo_id, "agnews", split="test")
print(ds.column_names)
print(ds[0])
```

The dataset stores rows as `(text, hypothesis, labels)` where `labels` is binary entailment.
The package reconstructs grouped multiclass samples internally for evaluation.

## Citing

```bibtex
@inproceedings{aarab2026btzsc,
	title     = {BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, and Rerankers},
	author    = {Aarab, Ilias},
	booktitle = {International Conference on Learning Representations (ICLR) 2026},
	year      = {2026},
	note      = {OpenReview PDF: https://openreview.net/pdf?id=IxMryAz2p3},
	url       = {https://openreview.net/forum?id=IxMryAz2p3}
}
```

## License

Released under the MIT license.

----

## For Developers

### Developer Setup

```bash
git clone https://github.com/IliasAarab/btzsc.git
cd btzsc
uv sync --dev
```

### Project Structure

High-level layout:

- `src/btzsc/benchmark.py`: benchmark orchestration and result objects.
- `src/btzsc/data.py`: dataset loading and task grouping.
- `src/btzsc/metrics.py`: metric computation and summaries.
- `src/btzsc/baselines.py`: baseline loading and comparison table creation.
- `src/btzsc/models/`: model adapters (`embedding`, `nli`, `reranker`, `llm`).
- `src/btzsc/cli.py`: command-line interface.

### Quality Checks

Run formatting, linting, and typing checks before opening a PR:

```bash
uv run ruff format
uv run ruff check
uv run pyright
```

### Packaging and Release

Build locally:

```bash
uv build
```

Release process:

1. Bump `version` in `pyproject.toml`.
2. Commit and push to `main`.
3. Create and push a version tag, for example:

```bash
git tag v0.1.1
git push origin v0.1.1
```

GitHub Actions builds and publishes tagged releases to PyPI via trusted publishing.


