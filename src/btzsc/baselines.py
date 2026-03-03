from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

_PKG_DIR = Path(__file__).resolve().parent
_BASELINES_DIR = _PKG_DIR / "_baselines"
_META_DIR = _PKG_DIR / "_meta"

_METRIC_TO_FILE = {
    "f1": "f1_scores.csv",
    "accuracy": "acc_scores.csv",
    "precision": "precision_scores.csv",
    "recall": "recall_scores.csv",
    "roc": "roc_scores.csv",
}

_METRIC_TO_USER_COLUMN = {
    "f1": "macro_f1",
    "accuracy": "accuracy",
    "precision": "macro_precision",
    "recall": "macro_recall",
    "roc": "roc",
}


def get_baselines(metric: str = "f1") -> pd.DataFrame:
    """Load packaged baseline scores for a metric.

    Args:
        metric: Metric key. One of `"f1"`, `"accuracy"`, `"precision"`, `"recall"`, or `"roc"`.

    Returns:
        Baseline score table loaded from CSV.

    Raises:
        ValueError: If `metric` is not recognized.
    """
    key = metric.lower()
    if key not in _METRIC_TO_FILE:
        msg = f"Unknown metric {metric!r}. Choose from {list(_METRIC_TO_FILE)}"
        raise ValueError(msg)
    path = _BASELINES_DIR / _METRIC_TO_FILE[key]
    return pd.read_csv(path)


def get_model_info() -> dict:
    """Load baseline model metadata from YAML.

    Returns:
        Parsed model metadata mapping.
    """
    path = _META_DIR / "models.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def compare(user_results: pd.DataFrame, metric: str = "f1", model_name: str = "__your_model__") -> pd.DataFrame:
    """Compare user results against baseline models.

    Args:
        user_results: Per-dataset metrics for the evaluated model.
        metric: Metric key used for loading baseline scores and ranking.
        model_name: Label used for the user row in the comparison table.

    Returns:
        Ranking table with baseline models and the user model row.

    Raises:
        ValueError: If required columns are missing or there are no overlapping datasets.
    """
    key = metric.lower()
    baselines = get_baselines(metric=key)

    if "dataset" not in user_results.columns:
        msg = "user_results must include a 'dataset' column"
        raise ValueError(msg)

    metric_col = _METRIC_TO_USER_COLUMN.get(key)
    if metric_col is None or metric_col not in user_results.columns:
        msg = (
            f"user_results missing required metric column {metric_col!r} for metric {metric!r}. "
            f"Available columns: {list(user_results.columns)}"
        )
        raise ValueError(msg)

    baseline_dataset_cols = [col for col in baselines.columns if col != "mdl"]
    user_dataset_scores = user_results.set_index("dataset")[metric_col]
    matched_datasets = [name for name in baseline_dataset_cols if name in user_dataset_scores.index]
    if not matched_datasets:
        msg = "No overlapping datasets between baseline table and user_results"
        raise ValueError(msg)

    user_mean = float(user_dataset_scores.loc[matched_datasets].mean())
    baseline_means = baselines[matched_datasets].mean(axis=1)
    table = pd.DataFrame({"model": baselines["mdl"], metric: baseline_means})
    table = pd.concat([table, pd.DataFrame([{"model": model_name, metric: user_mean}])], ignore_index=True)
    return table.sort_values(metric, ascending=False).reset_index(drop=True)
