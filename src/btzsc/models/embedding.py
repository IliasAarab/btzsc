from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from btzsc.models.base import BaseModel


class EmbeddingModel(BaseModel):
    """Zero-shot classifier using embedding similarity."""

    model_type = "embedding"

    def __init__(self, model_name: str, *, device: str | None = None, normalize: bool = True) -> None:
        """Initialize an embedding model adapter.

        Args:
            model_name: SentenceTransformer model identifier.
            device: Optional device override.
            normalize: Whether to L2-normalize embeddings before scoring.
        """
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    def _format_query(self, text: str) -> str:
        """Apply model-specific query prompt formatting.

        Args:
            text: Input text to classify.

        Returns:
            Formatted query string for the embedding model.
        """
        name = self.model_name.lower()
        if "e5-" in name:
            return f"query: {text}"
        if "qwen3-embedding" in self.model_name or "e5-mistral" in name:
            inst = "Given a piece of text, retrieve relevant label descriptions that best match the text"
            return f"Instruct: {inst}\nQuery: {text}"
        return text

    def _format_label(self, label: str) -> str:
        """Apply model-specific label formatting.

        Args:
            label: Candidate label description.

        Returns:
            Formatted label string for the embedding model.
        """
        if "e5-" in self.model_name.lower():
            return f"passage: {label}"
        return label

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 32) -> np.ndarray:
        """Compute similarity scores between texts and labels.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Encoding batch size.

        Returns:
            Score matrix of shape `(len(texts), len(labels))`.
        """
        q = [self._format_query(t) for t in texts]
        y = [self._format_label(label) for label in labels]
        q_emb = self.model.encode(
            q, batch_size=batch_size, normalize_embeddings=self.normalize, show_progress_bar=False
        )
        y_emb = self.model.encode(
            y, batch_size=batch_size, normalize_embeddings=self.normalize, show_progress_bar=False
        )
        return np.asarray(q_emb) @ np.asarray(y_emb).T

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 32) -> np.ndarray:
        """Predict best label index for each text.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Encoding batch size.

        Returns:
            Predicted label indices with shape `(len(texts),)`.
        """
        scores = self.predict_scores(texts, labels, batch_size=batch_size)
        return scores.argmax(axis=1)
