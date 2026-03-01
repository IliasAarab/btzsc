## BTZSC

BTZSC is a benchmark package for zero-shot text classification across embedding models, NLI cross-encoders, rerankers, and instruction-tuned LLMs.

Dataset: https://huggingface.co/datasets/btzsc/btzsc

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

### Notes

- The BTZSC dataset stores rows as `(text, hypothesis, labels)` where `labels` is binary entailment.
- The package reconstructs grouped multiclass samples internally.
- Baseline tables are bundled from the published BTZSC paper runs.
