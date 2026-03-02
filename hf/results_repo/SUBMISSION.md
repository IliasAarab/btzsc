# Submitting to the BTZSC Leaderboard

## Requirements

- Your model must be publicly available on the Hugging Face Hub.
- Results must be generated using the official `btzsc` harness.
- All 22 datasets must be evaluated (partial submissions are not accepted).

## Steps

### 1) Install the harness

```bash
pip install btzsc
```

### 2) Run the evaluation

```bash
btzsc evaluate \
  --model <your-hf-model-id> \
  --type <embedding|nli|reranker|llm> \
  --output-json results/<model_type>/<model-name>.json
```

Example:

```bash
btzsc evaluate \
  --model intfloat/e5-base-v2 \
  --type embedding \
  --output-json results/embedding/e5-base-v2.json
```

### 3) Validate locally

```bash
python validate.py results/<model_type>/<model-name>.json
```

### 4) Open a Pull Request

1. Fork this repository.
2. Place your JSON file in `results/<model_type>/<model-name>.json`.
3. Open a PR with:
   - The result JSON file.
   - A short model description (architecture/training notes).
   - Confirmation that model weights are public on Hugging Face Hub.

## Merge checks

- JSON passes schema validation (CI).
- Model is public on Hugging Face Hub.
- Scores are plausible (spot checks on difficult datasets).
- `evaluation.btzsc_version` matches a released version.

## Data contamination disclosure

If any BTZSC dataset was explicitly used in your model's training data, disclose it in your PR description. We track this for transparency.
