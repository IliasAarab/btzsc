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

All submission actions in this step target the results dataset repo `btzsc/btzsc-results`: https://huggingface.co/datasets/btzsc/btzsc-results.

Required destination path:

- `results/<model_type>/<model-name>.json`

Example:

- `results/embedding/e5-base-v2.json`

Choose one submission workflow:

1. Web UI (no clone required)
  - Open https://huggingface.co/datasets/btzsc/btzsc-results and upload the JSON in **Files and versions** at the required path.
  - If you do not have write access, fork `btzsc/btzsc-results` and open a PR to `btzsc/btzsc-results`.

2. Git workflow (clone/fork + push)
  - Clone or fork `https://huggingface.co/datasets/btzsc/btzsc-results`.
  - Add your JSON at the required path.
  - Push directly (if you have write access) or push to your fork and open a PR to `btzsc/btzsc-results`.

3. API workflow (`huggingface_hub`, PR-based)
  - Authenticate first (`huggingface-cli login` or `HF_TOKEN`).
  - Use `create_pr=True` against repo_id `btzsc/btzsc-results` to open a PR branch programmatically.
  - If PR creation is restricted for your account, upload to your fork and open a PR to `btzsc/btzsc-results` manually.

For every PR, include:

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
