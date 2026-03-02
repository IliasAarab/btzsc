"""Validate BTZSC leaderboard result JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_DATASETS = [
    "amazonpolarity",
    "imdb",
    "appreviews",
    "yelpreviews",
    "rottentomatoes",
    "financialphrasebank",
    "emotiondair",
    "empathetic",
    "banking77",
    "biasframes_intent",
    "massive",
    "agnews",
    "yahootopics",
    "trueteacher",
    "manifesto",
    "capsotu",
    "biasframes_offensive",
    "biasframes_sex",
    "wikitoxic_toxicaggregated",
    "wikitoxic_obscene",
    "wikitoxic_threat",
    "wikitoxic_insult",
]

REQUIRED_TASKS = ["sentiment", "topic", "intent", "emotion"]
REQUIRED_METRICS = ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
VALID_MODEL_TYPES = ["embedding", "nli", "reranker", "llm"]
EXPECTED_ARGC = 2


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_metric_block(errors: list[str], block: dict[str, Any], prefix: str) -> None:
    for metric in REQUIRED_METRICS:
        value = block.get(metric)
        if value is None:
            errors.append(f"Missing {prefix}.{metric}")
            continue
        if not _is_number(value):
            errors.append(f"{prefix}.{metric} must be numeric")
            continue
        if not (0.0 <= float(value) <= 1.0):
            errors.append(f"{prefix}.{metric}={value} is out of range [0, 1]")


def validate(path: str | Path) -> list[str]:
    errors: list[str] = []

    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_version") != "1.0":
        errors.append(f"Unsupported schema_version: {data.get('schema_version')}")

    model = data.get("model", {})
    if not model.get("name"):
        errors.append("Missing model.name")
    if model.get("model_type") not in VALID_MODEL_TYPES:
        errors.append(f"Invalid model.model_type: {model.get('model_type')}")

    evaluation = data.get("evaluation", {})
    if not evaluation.get("btzsc_version"):
        errors.append("Missing evaluation.btzsc_version")
    if not evaluation.get("btzsc_commit"):
        errors.append("Missing evaluation.btzsc_commit")

    results = data.get("results", {})

    overall = results.get("overall", {})
    _validate_metric_block(errors, overall, "results.overall")

    by_task = results.get("by_task", {})
    for task in REQUIRED_TASKS:
        if task not in by_task:
            errors.append(f"Missing results.by_task.{task}")
            continue
        block = by_task.get(task)
        if not isinstance(block, dict):
            errors.append(f"results.by_task.{task} must be an object")
            continue
        _validate_metric_block(errors, block, f"results.by_task.{task}")

    by_dataset = results.get("by_dataset", {})
    for dataset_name in REQUIRED_DATASETS:
        if dataset_name not in by_dataset:
            errors.append(f"Missing results.by_dataset.{dataset_name}")
            continue
        block = by_dataset.get(dataset_name)
        if not isinstance(block, dict):
            errors.append(f"results.by_dataset.{dataset_name} must be an object")
            continue
        _validate_metric_block(errors, block, f"results.by_dataset.{dataset_name}")

    return errors


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGC:
        print("Usage: python validate.py <result.json>")
        sys.exit(1)

    errors = validate(sys.argv[1])
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("Validation passed.")
