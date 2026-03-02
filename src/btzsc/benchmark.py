from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from tqdm import tqdm

from btzsc.models.base import BaseModel

if TYPE_CHECKING:
    import pandas as pd


ONE_THOUSAND = 1_000
ONE_MILLION = 1_000_000
ONE_BILLION = 1_000_000_000


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
    model_type: str = "unknown"
    model_params: int | None = None
    model_revision: str = "unknown"
    precision: str = "unknown"
    batch_size: int = 32
    max_samples: int | None = None
    device: str = "cpu"

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
            "model_type": self.model_type,
            "model_params": self.model_params,
            "model_revision": self.model_revision,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "device": self.device,
        }

    @staticmethod
    def _get_btzsc_version() -> str:
        try:
            return version("btzsc")
        except PackageNotFoundError:
            return "unknown"

    @staticmethod
    def _get_btzsc_commit() -> str:
        git_executable = shutil.which("git")
        if git_executable is None:
            return "unknown"

        try:
            package_dir = resources.files("btzsc")
            with resources.as_file(package_dir) as package_path:
                repo_dir = package_path.parents[1]
                result = subprocess.run(
                    [git_executable, "rev-parse", "HEAD"],
                    cwd=repo_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    commit = result.stdout.strip()
                    return commit[:12] if commit else "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return "unknown"

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        mapping = {
            "embedding": "embedding",
            "nli": "nli",
            "reranker": "reranker",
            "llm": "llm",
            "base_encoder": "base_encoder",
        }
        return mapping.get(model_type.lower(), model_type.lower())

    @staticmethod
    def _format_params(params: int | None) -> str:
        if params is None:
            return "unknown"
        if params >= ONE_BILLION:
            value = params / ONE_BILLION
            return f"{value:.1f}B".replace(".0B", "B")
        if params >= ONE_MILLION:
            value = params / ONE_MILLION
            return f"{value:.0f}M"
        if params >= ONE_THOUSAND:
            value = params / ONE_THOUSAND
            return f"{value:.0f}K"
        return str(params)

    def to_json(self, path: str | Path | None = None) -> dict:
        """Serialize benchmark results to the BTZSC leaderboard JSON schema.

        Args:
            path: Optional output path. If provided, JSON is written to disk.

        Returns:
            JSON-serializable payload matching leaderboard schema v1.0.
        """
        by_task: dict[str, dict[str, float]] = {}
        for task_name in ["sentiment", "topic", "intent", "emotion"]:
            vals = self.task_summary.get(task_name, {})
            by_task[task_name] = {
                "macro_f1": float(vals.get("macro_f1", 0.0)),
                "accuracy": float(vals.get("accuracy", 0.0)),
                "macro_precision": float(vals.get("macro_precision", 0.0)),
                "macro_recall": float(vals.get("macro_recall", 0.0)),
            }

        overall_vals = self.task_summary.get("overall", {})
        payload = {
            "schema_version": "1.0",
            "model": {
                "name": self.model_name,
                "model_type": self._normalize_model_type(self.model_type),
                "params": self._format_params(self.model_params),
                "revision": self.model_revision,
                "url": f"https://huggingface.co/{self.model_name}",
            },
            "evaluation": {
                "btzsc_version": self._get_btzsc_version(),
                "btzsc_commit": self._get_btzsc_commit(),
                "timestamp": datetime.now(UTC).isoformat(),
                "device": self.device or platform.processor() or "cpu",
                "precision": self.precision,
                "batch_size": self.batch_size,
                "max_samples": self.max_samples,
            },
            "results": {
                "overall": {
                    "macro_f1": float(overall_vals.get("macro_f1", 0.0)),
                    "accuracy": float(overall_vals.get("accuracy", 0.0)),
                    "macro_precision": float(overall_vals.get("macro_precision", 0.0)),
                    "macro_recall": float(overall_vals.get("macro_recall", 0.0)),
                },
                "by_task": by_task,
                "by_dataset": self.per_dataset_results,
            },
        }

        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return payload

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

    @staticmethod
    def _resolve_model(model: str | BaseModel, model_type: str | None = None) -> BaseModel:
        """Instantiate or validate a model adapter.

        Args:
            model: Model name string or pre-built adapter instance.
            model_type: Explicit adapter type when `model` is a string.

        Returns:
            A concrete `BaseModel` implementation.

        Raises:
            ValueError: If `model_type` is missing for a model name string or unsupported.
        """
        if isinstance(model, BaseModel):
            return model
        if model_type is None:
            msg = "model_type is required when model is a string. Choose from: embedding, nli, reranker, llm"
            raise ValueError(msg)
        resolved = model_type.lower()
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
    def _resolve_model_params(model_name: str) -> int | None:
        short_name = model_name.split("/")[-1]
        meta_dir = resources.files("btzsc") / "_meta"
        with resources.as_file(meta_dir / "models.yml") as p:
            model_info = yaml.safe_load(p.read_text(encoding="utf-8"))
        return model_info.get("model_to_params", {}).get(short_name)

    @staticmethod
    def _resolve_model_revision(eval_model: BaseModel) -> str:
        model_obj = getattr(eval_model, "model", None)
        config = getattr(model_obj, "config", None)
        commit_hash = getattr(config, "_commit_hash", None)
        if isinstance(commit_hash, str) and commit_hash:
            return commit_hash
        return "unknown"

    @staticmethod
    def _resolve_precision(eval_model: BaseModel) -> str:
        model_obj = getattr(eval_model, "model", None)
        dtype = getattr(model_obj, "dtype", None)
        if dtype is None:
            return "unknown"
        text = str(dtype)
        return text.replace("torch.", "")

    @staticmethod
    def _resolve_device(eval_model: BaseModel) -> str:
        device = getattr(eval_model, "device", None)
        if device is not None:
            return str(device)
        model_obj = getattr(eval_model, "model", None)
        model_device = getattr(model_obj, "device", None)
        if model_device is not None:
            return str(model_device)
        return "cpu"

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
            model_type: Explicit adapter type when `model` is a string.
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
        return BTZSCResults(
            per_dataset_results=per_dataset,
            task_summary=summary,
            model_name=model_name,
            model_type=getattr(eval_model, "model_type", model_type or "unknown"),
            model_params=self._resolve_model_params(model_name),
            model_revision=self._resolve_model_revision(eval_model),
            precision=self._resolve_precision(eval_model),
            batch_size=batch_size,
            max_samples=max_samples,
            device=self._resolve_device(eval_model),
        )
