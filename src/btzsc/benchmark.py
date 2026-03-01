from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm

from btzsc.models.base import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


@dataclass
class BTZSCResults:
    """Container for benchmark outputs.

    Attributes:
        per_dataset_results: Mapping of dataset name to metric dictionary.
        task_summary: Mapping of task group to aggregated metric dictionary.
        model_name: Label used when showing this run in baseline comparisons.
    """

    per_dataset_results: dict[str, dict[str, float]]
    task_summary: dict[str, dict[str, float]]
    model_name: str = "__your_model__"

    def per_dataset(self) -> pd.DataFrame:
        """Return per-dataset metrics as a DataFrame.

        Returns:
            A table with one row per dataset and metric columns.
        """
        import pandas as pd

        rows: list[dict[str, str | float]] = []
        for name, vals in self.per_dataset_results.items():
            row: dict[str, str | float] = {"dataset": name}
            row.update(vals)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Return task-level summary metrics as a DataFrame.

        Returns:
            A table with one row per task group.
        """
        import pandas as pd

        rows: list[dict[str, str | float]] = []
        for name, vals in self.task_summary.items():
            row: dict[str, str | float] = {"task": name}
            row.update(vals)
            rows.append(row)
        return pd.DataFrame(rows)

    def compare_baselines(self, metric: str = "f1") -> pd.DataFrame:
        """Compare this run to packaged baseline models.

        Args:
            metric: Baseline metric key to rank by.

        Returns:
            A ranking table containing baseline models and this run.
        """
        from btzsc.baselines import compare as compare_baselines_df

        return compare_baselines_df(self.per_dataset(), metric=metric, model_name=self.model_name)

    def to_dict(self) -> dict:
        """Serialize benchmark results to a dictionary.

        Returns:
            A JSON-serializable dictionary representation.
        """
        return {
            "per_dataset_results": self.per_dataset_results,
            "task_summary": self.task_summary,
            "model_name": self.model_name,
        }

    def to_csv(self, path: str | Path) -> None:
        """Write per-dataset metrics to CSV.

        Args:
            path: Output CSV path.
        """
        self.per_dataset().to_csv(path, index=False)


class BTZSCBenchmark:
    """Benchmark orchestrator for zero-shot text classification models."""

    def __init__(self, tasks: list[str] | None = None, *, cache_dir: str | None = None) -> None:
        """Initialize benchmark configuration.

        Args:
            tasks: Optional task-group names or dataset names to evaluate.
            cache_dir: Optional HuggingFace cache directory.
        """
        self.tasks = tasks
        self.cache_dir = cache_dir

    def _resolve_model(self, model: str | BaseModel, model_type: str | None = None) -> BaseModel:
        """Instantiate or validate a model adapter.

        Args:
            model: Model name string or pre-built adapter instance.
            model_type: Optional explicit adapter type.

        Returns:
            A concrete `BaseModel` implementation.

        Raises:
            ValueError: If `model_type` is unsupported.
        """
        if isinstance(model, BaseModel):
            return model
        resolved = (model_type or self._auto_detect_model_type(model)).lower()
        if resolved == "embedding":
            from btzsc.models.embedding import EmbeddingModel

            return EmbeddingModel(model)
        if resolved == "nli":
            from btzsc.models.nli import NLIModel

            return NLIModel(model)
        if resolved == "reranker":
            from btzsc.models.reranker import RerankerModel

            return RerankerModel(model)
        if resolved == "llm":
            from btzsc.models.llm import LLMModel

            return LLMModel(model)
        msg = f"Unsupported model_type: {model_type!r}"
        raise ValueError(msg)

    @staticmethod
    def _auto_detect_model_type(model_name: str) -> str:
        """Infer adapter type from a model name.

        Args:
            model_name: Model identifier to inspect.

        Returns:
            One of `"embedding"`, `"nli"`, `"reranker"`, or `"llm"`.
        """
        low = model_name.lower()
        if "reranker" in low or "ms-marco" in low:
            return "reranker"
        if "mnli" in low or "nli" in low:
            return "nli"
        if any(k in low for k in ["e5", "bge", "gte", "embedding", "minilm"]):
            return "embedding"
        return "llm"

    def evaluate(
        self,
        model: str | BaseModel,
        *,
        model_type: str | None = None,
        batch_size: int = 32,
        max_samples: int | None = None,
        show_progress: bool = True,
    ) -> BTZSCResults:
        """Run benchmark evaluation for a model.

        Args:
            model: Model name string or pre-built adapter instance.
            model_type: Optional explicit adapter type.
            batch_size: Inference batch size used by adapters.
            max_samples: Optional per-dataset sample cap.
            show_progress: Whether to render a progress bar.

        Returns:
            A `BTZSCResults` object containing per-dataset and summary metrics.
        """
        from btzsc.data import BTZSC_DATASETS, TASK_GROUPS, load_all_datasets
        from btzsc.metrics import compute_metrics, compute_task_summary

        eval_model = self._resolve_model(model, model_type=model_type)

        if self.tasks is None:
            dataset_names = BTZSC_DATASETS
            load_tasks = None
        else:
            task_names = set(TASK_GROUPS)
            if all(t in task_names for t in self.tasks):
                load_tasks = self.tasks
                dataset_names = [d for t in self.tasks for d in TASK_GROUPS[t]]
            else:
                load_tasks = None
                dataset_names = self.tasks

        datasets = load_all_datasets(tasks=load_tasks, max_samples=max_samples, cache_dir=self.cache_dir)
        if load_tasks is None and self.tasks is not None:
            datasets = {k: v for k, v in datasets.items() if k in set(dataset_names)}

        per_dataset: dict[str, dict[str, float]] = {}
        iterator = datasets.items()
        if show_progress:
            iterator = tqdm(list(iterator), desc="Evaluating", unit="dataset")

        for ds_name, ds in iterator:
            preds = eval_model.predict(ds.texts, ds.labels, batch_size=batch_size)
            per_dataset[ds_name] = compute_metrics(predictions=preds, references=ds.references)

        summary = compute_task_summary(per_dataset, TASK_GROUPS)
        model_name = (
            model if isinstance(model, str) else getattr(eval_model, "model_name", eval_model.__class__.__name__)
        )
        return BTZSCResults(per_dataset_results=per_dataset, task_summary=summary, model_name=model_name)
