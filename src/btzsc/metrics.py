"""Evaluation metrics for BTZSC benchmark."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(
    predictions: np.ndarray,
    references: np.ndarray,
    *,
    zero_division: float = 0.0,
) -> dict[str, float]:
    """Compute classification metrics for a single dataset.

    Args:
        predictions: Predicted label indices, shape ``(n_samples,)``.
        references:  True label indices, shape ``(n_samples,)``.
        zero_division: Value for metrics when there is division by zero.

    Returns:
        Dict with keys ``macro_f1``, ``accuracy``, ``macro_precision``, ``macro_recall``.
    """
    zero_division_any = cast(Any, zero_division)

    return {
        "macro_f1": float(f1_score(references, predictions, average="macro", zero_division=zero_division_any)),
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_precision": float(
            precision_score(references, predictions, average="macro", zero_division=zero_division_any)
        ),
        "macro_recall": float(recall_score(references, predictions, average="macro", zero_division=zero_division_any)),
    }


def compute_task_summary(
    per_dataset_results: dict[str, dict[str, float]],
    task_groups: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-dataset metrics into per-task-group averages.

    Mirrors the summary format used in the BTZSC paper (Table 3).

    Args:
        per_dataset_results: Mapping ``{dataset_name: {metric_name: value}}``.
        task_groups: Mapping ``{task_name: [dataset_names]}``.

    Returns:
        Dict ``{task_name: {metric_name: mean_value}, "overall": {...}}``.
    """
    metrics_keys = ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
    summary: dict[str, dict[str, float]] = {}

    all_values: dict[str, list[float]] = {k: [] for k in metrics_keys}

    for task_name, ds_names in task_groups.items():
        task_values: dict[str, list[float]] = {k: [] for k in metrics_keys}

        for ds_name in ds_names:
            if ds_name not in per_dataset_results:
                continue
            result = per_dataset_results[ds_name]
            for k in metrics_keys:
                if k in result:
                    task_values[k].append(result[k])
                    all_values[k].append(result[k])

        summary[task_name] = {k: float(np.mean(v)) if v else 0.0 for k, v in task_values.items()}

    summary["overall"] = {k: float(np.mean(v)) if v else 0.0 for k, v in all_values.items()}

    return summary
