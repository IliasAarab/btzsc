from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from btzsc.models.base import BaseModel


class RerankerModel(BaseModel):
    """Zero-shot classifier based on cross-encoder reranking."""

    model_type = "reranker"

    def __init__(self, model_name: str, *, device: str | None = None, torch_dtype: torch.dtype | None = None) -> None:
        """Initialize a reranker adapter.

        Args:
            model_name: Sequence-classification reranker identifier.
            device: Optional device override.
            torch_dtype: Optional model loading dtype.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.is_qwen_reranker = "Qwen3-Reranker" in model_name
        self.token_true_id, self.token_false_id = 9693, 2152

    def _score_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract a relevance score vector from model logits.

        Args:
            logits: Raw model output logits.

        Returns:
            A one-dimensional relevance score tensor.
        """
        if logits.ndim == 1:
            return logits
        if logits.shape[-1] == 1:
            return logits[:, 0]
        mapping = getattr(self.model.config, "label2id", {}) or {}
        lower = {str(k).lower(): int(v) for k, v in mapping.items()}
        for key in ["relevant", "entailment", "true", "yes"]:
            if key in lower:
                return logits[:, lower[key]]
        return logits[:, -1]

    @staticmethod
    def _qwen_prompt(text: str, label: str) -> str:
        """Build the instruction prompt for Qwen reranker variants.

        Args:
            text: Input text.
            label: Candidate label description.

        Returns:
            Fully formatted chat-style prompt.
        """
        inst = "Given a piece of text, retrieve relevant label descriptions that best match the text"
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return f"{prefix}<Instruct>: {inst}\n<Query>: {text}\n<Document>: {label}{suffix}"

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 8) -> np.ndarray:
        """Compute relevance scores for all text-label combinations.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Pairwise inference batch size.

        Returns:
            Score matrix of shape `(len(texts), len(labels))`.
        """
        output: list[np.ndarray] = []
        with torch.no_grad():
            for text in texts:
                row: list[float] = []
                for i in range(0, len(labels), batch_size):
                    chunk = labels[i : i + batch_size]
                    if self.is_qwen_reranker:
                        prompts = [self._qwen_prompt(text, lbl) for lbl in chunk]
                        enc = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(
                            self.device
                        )
                        logits = self.model(**enc).logits[:, -1, :]
                        true_vec = logits[:, self.token_true_id]
                        false_vec = logits[:, self.token_false_id]
                        scores = torch.stack([false_vec, true_vec], dim=1).softmax(dim=1)[:, 1]
                    else:
                        enc = self.tokenizer(
                            [text] * len(chunk), chunk, padding=True, truncation=True, return_tensors="pt"
                        ).to(self.device)
                        logits = self.model(**enc).logits
                        scores = self._score_from_logits(logits)
                    row.extend(scores.detach().float().cpu().tolist())
                output.append(np.array(row, dtype=np.float32))
        return np.stack(output, axis=0)

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 8) -> np.ndarray:
        """Predict best label index for each text.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Pairwise inference batch size.

        Returns:
            Predicted label indices with shape `(len(texts),)`.
        """
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)
