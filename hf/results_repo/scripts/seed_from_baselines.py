from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src" / "btzsc"
BASELINES_DIR = SRC_DIR / "_baselines"
META_MODELS = SRC_DIR / "_meta" / "models.yml"
RESULTS_ROOT = ROOT / "hf" / "results_repo" / "results"

DATASETS = [
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

TASK_GROUPS = {
    "sentiment": [
        "amazonpolarity",
        "imdb",
        "appreviews",
        "yelpreviews",
        "rottentomatoes",
        "financialphrasebank",
    ],
    "emotion": ["emotiondair", "empathetic"],
    "intent": ["banking77", "biasframes_intent", "massive"],
    "topic": [
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
    ],
}

TYPE_MAP = {
    "embedding_model": "embedding",
    "nli_cross_encoder": "nli",
    "reranker": "reranker",
    "llm": "llm",
}

CSV_BY_METRIC = {
    "macro_f1": BASELINES_DIR / "f1_scores.csv",
    "accuracy": BASELINES_DIR / "acc_scores.csv",
    "macro_precision": BASELINES_DIR / "precision_scores.csv",
    "macro_recall": BASELINES_DIR / "recall_scores.csv",
}

ONE_THOUSAND = 1_000
ONE_MILLION = 1_000_000
ONE_BILLION = 1_000_000_000


def read_metric_table(path: Path) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["mdl"]
            table[model] = {}
            for ds in DATASETS:
                if ds not in row or row[ds] in {None, ""}:
                    raise KeyError(ds)
                table[model][ds] = float(row[ds])
    return table


def short_params(n: int | None) -> str:
    if n is None:
        return "unknown"
    if n >= ONE_BILLION:
        value = n / ONE_BILLION
        return f"{value:.1f}B".replace(".0B", "B")
    if n >= ONE_MILLION:
        return f"{n // ONE_MILLION}M"
    if n >= ONE_THOUSAND:
        return f"{n // ONE_THOUSAND}K"
    return str(n)


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6)


def get_git_commit() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        return "unknown"

    try:
        out = subprocess.check_output([git_executable, "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5)
        return out.strip()[:12]
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main() -> None:
    metric_tables = {metric: read_metric_table(path) for metric, path in CSV_BY_METRIC.items()}

    models_meta = yaml.safe_load(META_MODELS.read_text(encoding="utf-8"))
    model_to_type: dict[str, str] = models_meta["model_to_type"]
    model_to_params: dict[str, int] = models_meta["model_to_params"]

    commit = get_git_commit()
    timestamp = datetime.now(UTC).isoformat()

    written = 0
    for model_name, raw_type in model_to_type.items():
        if raw_type == "base_encoder":
            continue

        model_type = TYPE_MAP[raw_type]
        by_dataset: dict[str, dict[str, float]] = {}
        for ds in DATASETS:
            by_dataset[ds] = {
                metric: metric_tables[metric][model_name][ds]
                for metric in ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
            }

        by_task: dict[str, dict[str, float]] = {}
        for task_name, ds_names in TASK_GROUPS.items():
            by_task[task_name] = {
                metric: avg([by_dataset[d][metric] for d in ds_names])
                for metric in ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
            }

        overall = {
            metric: avg([by_dataset[d][metric] for d in DATASETS])
            for metric in ["macro_f1", "accuracy", "macro_precision", "macro_recall"]
        }

        payload = {
            "schema_version": "1.0",
            "model": {
                "name": model_name,
                "model_type": model_type,
                "params": short_params(model_to_params.get(model_name)),
                "revision": "unknown",
                "url": f"https://huggingface.co/{model_name}",
            },
            "evaluation": {
                "btzsc_version": "0.1.1",
                "btzsc_commit": commit,
                "timestamp": timestamp,
                "device": "unknown",
                "precision": "unknown",
                "batch_size": 32,
                "max_samples": None,
            },
            "results": {
                "overall": overall,
                "by_task": by_task,
                "by_dataset": by_dataset,
            },
        }

        model_dir = RESULTS_ROOT / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path = model_dir / f"{model_name}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written += 1

    print(f"Wrote {written} seed JSON files to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
