"""Local, discriminative NLI verifier with explicit label and sentence mappings.

The model sees a premise and a hypothesis only. It never reads claim annotations,
reference document IDs, or evaluation predictions at inference time.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.data.verifier_context import format_premise

LABEL_MAP = {"entailment": "SUPPORT", "contradiction": "CONTRADICT", "neutral": "NEI"}


class NLIVerifier:
    def __init__(self, path: str | Path, *, threads: int = 4, max_length: int = 512, device: str = "cpu", include_title: bool = False, window_overlap: int = 1):
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, dtype=torch.float32,
        ).to(device).eval()
        self.device = device
        self.labels = [LABEL_MAP[self.model.config.id2label[i].lower()]
                       for i in range(self.model.config.num_labels)]
        self.max_length = max_length
        self.include_title, self.window_overlap = include_title, window_overlap

    def format_premise(self, title: str, text: str) -> str:
        return format_premise(title, text, self.include_title)

    def score(self, premises: list[str], hypotheses: list[str], batch_size: int = 4) -> list[dict]:
        if len(premises) != len(hypotheses):
            raise ValueError("Each premise needs one hypothesis")
        rows = []
        for start in range(0, len(premises), batch_size):
            batch = self.tokenizer(
                premises[start:start + batch_size], hypotheses[start:start + batch_size],
                truncation="only_first", padding=True, max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.inference_mode():
                probabilities = self.model(**batch).logits.softmax(-1).tolist()
            rows.extend(dict(zip(self.labels, p, strict=True)) for p in probabilities)
        return rows

    def windows(self, claim: str, sentences: list[str], *, overlap: int | None = None, title: str = "") -> list[list[int]]:
        """Cover every sentence within the model context; keep hypotheses untruncated."""
        budget = self.max_length - len(self.tokenizer.encode(claim, add_special_tokens=False)) - 8
        overlap = self.window_overlap if overlap is None else overlap
        if self.include_title:
            budget -= len(self.tokenizer.encode(self.format_premise(title, ""), add_special_tokens=False))
        lengths = [len(self.tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
        windows = []
        start = 0
        while start < len(sentences):
            end, used = start, 0
            while end < len(sentences) and (used + lengths[end] <= budget or end == start):
                used += lengths[end]
                end += 1
            windows.append(list(range(start, end)))
            if end == len(sentences):
                break
            start = max(start + 1, end - overlap)
        return windows
