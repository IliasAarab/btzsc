"""Validation utilities for BTZSC leaderboard result JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from btzsc.data import BTZSC_DATASETS, TASK_GROUPS

VALID_MODEL_TYPES = {"embedding", "nli", "reranker", "llm"}
REQUIRED_METRICS = {"macro_f1", "accuracy", "macro_precision", "macro_recall"}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_metric_block(errors: list[str], block: dict[str, Any], location: str) -> None:
    for metric in REQUIRED_METRICS:
        value = block.get(metric)
        if value is None:
            errors.append(f"Missing {location}.{metric}")
            continue
        if not _is_number(value):
            errors.append(f"{location}.{metric} must be numeric")
            continue
        if not (0.0 <= float(value) <= 1.0):
            errors.append(f"{location}.{metric}={value} is out of range [0, 1]")


def validate_result_payload(data: dict[str, Any]) -> list[str]:
    """Validate a BTZSC leaderboard JSON payload.

    Args:
        data: Parsed JSON payload.

    Returns:
        List of validation errors. Empty list means valid.
    """
    errors: list[str] = []

    if data.get("schema_version") != "1.0":
        errors.append(f"Unsupported schema_version: {data.get('schema_version')}")

    model = data.get("model", {})
    if not model.get("name"):
        errors.append("Missing model.name")

    model_type = model.get("model_type")
    if model_type not in VALID_MODEL_TYPES:
        errors.append(f"Invalid model.model_type: {model_type}")

    evaluation = data.get("evaluation", {})
    if not evaluation.get("btzsc_version"):
        errors.append("Missing evaluation.btzsc_version")
    if not evaluation.get("btzsc_commit"):
        errors.append("Missing evaluation.btzsc_commit")

    results = data.get("results", {})

    overall = results.get("overall", {})
    _check_metric_block(errors, overall, "results.overall")

    by_task = results.get("by_task", {})
    for task in TASK_GROUPS:
        if task not in by_task:
            errors.append(f"Missing results.by_task.{task}")
            continue
        task_metrics = by_task.get(task)
        if not isinstance(task_metrics, dict):
            errors.append(f"results.by_task.{task} must be an object")
            continue
        _check_metric_block(errors, task_metrics, f"results.by_task.{task}")

    by_dataset = results.get("by_dataset", {})
    for dataset_name in BTZSC_DATASETS:
        if dataset_name not in by_dataset:
            errors.append(f"Missing results.by_dataset.{dataset_name}")
            continue
        ds_metrics = by_dataset.get(dataset_name)
        if not isinstance(ds_metrics, dict):
            errors.append(f"results.by_dataset.{dataset_name} must be an object")
            continue
        _check_metric_block(errors, ds_metrics, f"results.by_dataset.{dataset_name}")

    return errors


def validate_result_file(path: str | Path) -> list[str]:
    """Validate a BTZSC leaderboard JSON file."""
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    return validate_result_payload(data)
