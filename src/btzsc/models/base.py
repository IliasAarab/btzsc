"""Abstract base class for all BTZSC model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BaseModel(ABC):
    """Abstract interface that every BTZSC model adapter must implement.

    Subclass this to plug in any model — the benchmark orchestrator only
    calls :meth:`predict` and :meth:`predict_scores`.

    Example::

        class MyModel(BaseModel):
            model_type = "custom"

            def predict_scores(
                self, texts, labels, batch_size=32
            ): ...  # return (n_texts, n_labels) score matrix

            def predict(self, texts, labels, batch_size=32):
                return self.predict_scores(texts, labels, batch_size).argmax(axis=1)
    """

    model_type: str = "custom"
    """One of ``"embedding"``, ``"nli"``, ``"reranker"``, ``"llm"``, or ``"custom"``."""

    @abstractmethod
    def predict(
        self,
        texts: list[str],
        labels: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Predict the best label index for each text.

        Args:
            texts: Input texts to classify.
            labels: Verbalized candidate label descriptions.
            batch_size: Batch size for inference.

        Returns:
            Array of predicted label indices, shape ``(len(texts),)``.
        """

    @abstractmethod
    def predict_scores(
        self,
        texts: list[str],
        labels: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return a score matrix over all labels for each text.

        Scores need not be probabilities but must be comparable across
        the label axis (higher = more likely).

        Args:
            texts: Input texts to classify.
            labels: Verbalized candidate label descriptions.
            batch_size: Batch size for inference.

        Returns:
            Score matrix of shape ``(len(texts), len(labels))``.
        """
