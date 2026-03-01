"""Dataset loading and grouping utilities for BTZSC benchmark."""

from __future__ import annotations

from importlib import resources

import numpy as np
import yaml
from datasets import load_dataset

REPO_ID = "btzsc/btzsc"

# ──────────────────────────────────────────────────────────────────────
# Dataset metadata
# ──────────────────────────────────────────────────────────────────────


def _load_meta() -> dict[str, dict[str, str]]:
    meta_dir = resources.files("btzsc") / "_meta"
    with resources.as_file(meta_dir / "datasets.yml") as p:
        return yaml.safe_load(p.read_text(encoding="utf-8"))["btzsc_info"]


_META: dict[str, dict[str, str]] | None = None


def _get_meta() -> dict[str, dict[str, str]]:
    global _META  # noqa: PLW0603
    if _META is None:
        _META = _load_meta()
    return _META


BTZSC_DATASETS: list[str] = [
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

TASK_GROUPS: dict[str, list[str]] = {
    "sentiment": [
        "amazonpolarity",
        "imdb",
        "appreviews",
        "yelpreviews",
        "rottentomatoes",
        "financialphrasebank",
    ],
    "emotion": [
        "emotiondair",
        "empathetic",
    ],
    "intent": [
        "banking77",
        "biasframes_intent",
        "massive",
    ],
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


def get_dataset_info(name: str) -> dict[str, str]:
    """Return task/domain metadata for a dataset."""
    meta = _get_meta()
    if name not in meta:
        msg = f"Unknown dataset: {name!r}. Choose from: {list(meta)}"
        raise KeyError(msg)
    return meta[name]


# ──────────────────────────────────────────────────────────────────────
# Dataset loading and grouping
# ──────────────────────────────────────────────────────────────────────


class BTZSCDataset:
    """A loaded BTZSC dataset, grouped into multi-class samples.

    Attributes:
        name:       Dataset identifier (e.g. ``"agnews"``).
        texts:      List of input texts (length = n_samples).
        labels:     List of verbalized label descriptions per sample
                    (same order for every sample; length = n_classes).
        references: Array of true label indices (shape ``(n_samples,)``).
        task:       Task family (``"sentiment"``, ``"topic"``, etc.).
        domain:     Domain tag (``"news"``, ``"social-media"``, etc.).
    """

    def __init__(
        self,
        name: str,
        texts: list[str],
        labels: list[str],
        references: np.ndarray,
        task: str,
        domain: str,
    ) -> None:
        self.name = name
        self.texts = texts
        self.labels = labels
        self.references = references
        self.task = task
        self.domain = domain

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    @property
    def n_samples(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        return (
            f"BTZSCDataset(name={self.name!r}, n_samples={self.n_samples}, "
            f"n_classes={self.n_classes}, task={self.task!r})"
        )


def _get_n_classes(labels_col: list[int]) -> int:
    """Count consecutive elements before the binary label pattern repeats."""
    first = labels_col[0]
    for i in range(1, len(labels_col)):
        if labels_col[i] == first:
            return i
    return len(labels_col)


def load_btzsc_dataset(
    name: str,
    *,
    max_samples: int | None = None,
    cache_dir: str | None = None,
) -> BTZSCDataset:
    """Load a single BTZSC dataset and decode the paired format into multi-class samples.

    The HuggingFace dataset stores each (text, hypothesis) pair as a row with
    a binary ``labels`` column (1 = entailment). This function groups consecutive
    rows to reconstruct (text, label_descriptions, ground_truth_idx) triples.

    Args:
        name: Dataset identifier (e.g. ``"agnews"``).
        max_samples: If set, limit to this many *grouped* samples.
        cache_dir: HuggingFace cache directory.

    Returns:
        A :class:`BTZSCDataset` instance.
    """
    if name not in BTZSC_DATASETS:
        msg = f"Unknown dataset: {name!r}. Choose from: {BTZSC_DATASETS}"
        raise ValueError(msg)

    ds = load_dataset(REPO_ID, name=name, split="test", cache_dir=cache_dir)

    # Determine number of classes from the binary label pattern
    labels_col: list[int] = ds["labels"]  # type: ignore[assignment]
    n_classes = _get_n_classes(labels_col)

    n_rows = len(ds)
    n_samples_total = n_rows // n_classes

    if max_samples is not None:
        n_samples_total = min(n_samples_total, max_samples)

    # Extract columns
    all_texts: list[str] = ds["text"]  # type: ignore[assignment]
    all_hypotheses: list[str] = ds["hypothesis"]  # type: ignore[assignment]
    all_labels: list[int] = labels_col

    # Build multi-class grouped data
    texts: list[str] = []
    references: list[int] = []
    label_names: list[str] | None = None

    for i in range(n_samples_total):
        offset = i * n_classes
        texts.append(all_texts[offset])

        # Collect hypotheses for this sample (ordered by class)
        sample_labels = [all_hypotheses[offset + j] for j in range(n_classes)]
        if label_names is None:
            label_names = sample_labels

        # Ground truth = index where labels==1
        sample_binary = [all_labels[offset + j] for j in range(n_classes)]
        true_idx = sample_binary.index(1) if 1 in sample_binary else 0
        references.append(true_idx)

    info = get_dataset_info(name)

    return BTZSCDataset(
        name=name,
        texts=texts,
        labels=label_names or [],
        references=np.array(references, dtype=np.int64),
        task=info["task"],
        domain=info["domain"],
    )


def load_all_datasets(
    tasks: list[str] | None = None,
    max_samples: int | None = None,
    cache_dir: str | None = None,
) -> dict[str, BTZSCDataset]:
    """Load multiple BTZSC datasets.

    Args:
        tasks: If given, only load datasets belonging to these task groups
               (e.g. ``["sentiment", "topic"]``).
        max_samples: Limit per dataset.
        cache_dir: HuggingFace cache directory.

    Returns:
        Dict mapping dataset names to :class:`BTZSCDataset` instances.
    """
    if tasks is not None:
        dataset_names = [name for task in tasks for name in TASK_GROUPS.get(task, [])]
    else:
        dataset_names = BTZSC_DATASETS

    return {name: load_btzsc_dataset(name, max_samples=max_samples, cache_dir=cache_dir) for name in dataset_names}
