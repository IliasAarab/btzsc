from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from btzsc.models.base import BaseModel


class NLIModel(BaseModel):
    """Zero-shot classifier based on NLI entailment scoring."""

    model_type = "nli"

    def __init__(self, model_name: str, *, device: str | None = None, torch_dtype: torch.dtype | None = None) -> None:
        """Initialize an NLI model adapter.

        Args:
            model_name: Sequence-classification model identifier.
            device: Optional device override.
            torch_dtype: Optional model loading dtype.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.entailment_idx = self._find_entailment_idx()

    def _find_entailment_idx(self) -> int:
        """Infer entailment class index from model label mapping.

        Returns:
            Index corresponding to entailment-like output.
        """
        mapping = getattr(self.model.config, "label2id", {}) or {}
        lower = {str(k).lower(): int(v) for k, v in mapping.items()}
        for key in ["entailment", "label_2", "true", "yes"]:
            if key in lower:
                return lower[key]
        if mapping:
            return int(max(mapping.values()))
        return 0

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 16) -> np.ndarray:
        """Compute entailment scores for text-label pairs.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Pairwise inference batch size.

        Returns:
            Score matrix of shape `(len(texts), len(labels))`.
        """
        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for text in texts:
                pairs = [(text, label) for label in labels]
                row_scores: list[float] = []
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i : i + batch_size]
                    a = [x[0] for x in batch_pairs]
                    b = [x[1] for x in batch_pairs]
                    enc = self.tokenizer(a, b, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    logits = self.model(**enc).logits
                    if logits.ndim == 1:
                        vals = logits
                    elif logits.shape[-1] == 1:
                        vals = logits[:, 0]
                    else:
                        vals = logits[:, self.entailment_idx]
                    row_scores.extend(vals.detach().float().cpu().tolist())
                all_scores.append(np.array(row_scores, dtype=np.float32))
        return np.stack(all_scores, axis=0)

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 16) -> np.ndarray:
        """Predict best label index for each text.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Pairwise inference batch size.

        Returns:
            Predicted label indices with shape `(len(texts),)`.
        """
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)
