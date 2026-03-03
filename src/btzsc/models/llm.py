from __future__ import annotations

import string
from typing import Any, cast

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from btzsc.models.base import BaseModel


class LLMModel(BaseModel):
    """Zero-shot classifier based on causal language model next-token scoring."""

    model_type = "llm"
    _SYMBOLS = (
        list(string.ascii_uppercase)
        + list(string.ascii_lowercase)
        + list("αβγδεζηθικλμνξοπρστυφχψω")
        + list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
    )
    _DEFAULT_PROMPT = (
        "You are a text classifier.\n"
        "You will be given a text and several mutually exclusive options.\n"
        "Each option is prefixed by a single letter (e.g. A, b, λ, ...).\n"
        "Your task is to choose the single best option.\n\n"
        "IMPORTANT:\n"
        "- Answer with EXACTLY ONE LETTER used to prefix the options.\n"
        "- Do NOT output any words, punctuation, or explanation.\n\n"
        "TEXT:\n"
        "{text}\n\n"
        "OPTIONS:\n"
        "{options}\n\n"
        "Answer: The correct option is letter "
    )

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a causal-LM adapter.

        Args:
            model_name: Causal language model identifier.
            device: Optional device override.
            torch_dtype: Optional model loading dtype.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = cast(Any, AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, text: str, labels: list[str]) -> str:
        """Construct a multiple-choice classification prompt.

        Args:
            text: Input text.
            labels: Candidate label descriptions.

        Returns:
            Prompt instructing the model to output one option letter.
        """
        opts = []
        for i, label in enumerate(labels):
            opts.append(f"{self._SYMBOLS[i]}) {label}")
        options = "\n".join(opts)
        return self._DEFAULT_PROMPT.format(text=text, options=options)

    def _get_letter_ids(self, n_labels: int) -> torch.Tensor:
        """Resolve token IDs for option letters.

        Args:
            n_labels: Number of candidate labels.

        Returns:
            Tensor of token IDs for option letters.

        Raises:
            ValueError: If an option symbol is not a single tokenizer token.
        """
        ids: list[int] = []
        for sym in self._SYMBOLS[:n_labels]:
            tok = self.tokenizer(sym, add_special_tokens=False).input_ids
            if len(tok) != 1:
                msg = f"Option symbol {sym!r} is not single-token for this tokenizer"
                raise ValueError(msg)
            ids.append(tok[0])
        return torch.tensor(ids, device=self.device, dtype=torch.long)

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 8) -> np.ndarray:
        """Compute label probabilities from next-token distributions.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Prompt inference batch size.

        Returns:
            Probability matrix of shape `(len(texts), len(labels))`.

        Raises:
            ValueError: If number of labels exceeds available option symbols.
        """
        if len(labels) > len(self._SYMBOLS):
            msg = f"Too many labels ({len(labels)}), max supported is {len(self._SYMBOLS)}"
            raise ValueError(msg)
        letter_ids = self._get_letter_ids(len(labels))
        prompts = [self._build_prompt(text, labels) for text in texts]

        rows: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i : i + batch_size]
                enc = self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits[:, -1, :]
                probs = logits.softmax(dim=-1)[:, letter_ids]
                probs /= probs.sum(dim=-1, keepdim=True)
                rows.append(probs.detach().float().cpu().numpy())
        return np.concatenate(rows, axis=0)

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 8) -> np.ndarray:
        """Predict best label index for each text.

        Args:
            texts: Input texts.
            labels: Candidate label descriptions.
            batch_size: Prompt inference batch size.

        Returns:
            Predicted label indices with shape `(len(texts),)`.
        """
        return self.predict_scores(texts, labels, batch_size=batch_size).argmax(axis=1)
