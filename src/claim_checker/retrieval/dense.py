"""BGE encoding, tokenizer-aware sentence windows, exact inner-product search, max-window scoring."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from claim_checker.retrieval.bm25 import rank_scores

WINDOW_FORMAT = "title-newline-sentences-v1"


def window_text(title: str, sentences: Sequence[str]) -> str:
    body = " ".join(s for s in sentences if s)
    return f"{title}\n{body}".strip() if title else body


def make_windows(
    title: str,
    sentences: Sequence[str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap: int = 1,
) -> list[dict]:
    """Greedy sentence-aware windows whose token count (as measured by ``count_tokens``) fits.

    Every window repeats the title and records its zero-based ``sentence_ids``. A single
    sentence that alone exceeds the budget becomes its own window flagged ``truncated=True``
    (the model will truncate it; the flag makes this explicit rather than silent).
    Consecutive windows share ``overlap`` sentences so evidence near a boundary is not split.
    """
    n = len(sentences)
    if n == 0:
        text = window_text(title, [])
        return [{"sentence_ids": [], "text": text, "tokens": count_tokens(text), "truncated": False}]
    windows: list[dict] = []
    start = 0
    while start < n:
        end = start + 1
        text = window_text(title, sentences[start:end])
        tokens = count_tokens(text)
        truncated = tokens > max_tokens
        if not truncated:
            while end < n:
                cand = window_text(title, sentences[start : end + 1])
                cand_tokens = count_tokens(cand)
                if cand_tokens > max_tokens:
                    break
                end, text, tokens = end + 1, cand, cand_tokens
        windows.append(
            {"sentence_ids": list(range(start, end)), "text": text, "tokens": tokens, "truncated": truncated}
        )
        if end >= n:
            break
        start = end - overlap if end - overlap > start else end
    return windows


def load_cached_first(factory: Callable, model_name: str, **kwargs):
    """Load a Sentence Transformers model from the local Hugging Face cache without any network
    request; only if it is not cached yet, fall back to a normal (downloading) load. This keeps a
    prepared installation working offline and avoids Hub round-trips on every start."""
    try:
        return factory(model_name, local_files_only=True, **kwargs)
    except Exception:
        return factory(model_name, **kwargs)


class SentenceEncoder:
    """Lazy wrapper around a SentenceTransformer bi-encoder (CPU by default)."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._light_tokenizer = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()  # long-text token counting is intentional
            self._model = load_cached_first(SentenceTransformer, self.model_name, device=self.device)
        return self._model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        """Token count including special tokens ([CLS], [SEP])."""
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def _load_light_tokenizer(self):
        """``tokenizer.json`` of the same Hugging Face repository through the ``tokenizers``
        library: no model weights and no torch/transformers import (about 0.5 s instead of ~10 s).
        Falls back to ``transformers.AutoTokenizer`` if the repository has no ``tokenizer.json``."""
        try:
            from tokenizers import Tokenizer

            local = Path(self.model_name) / "tokenizer.json"
            if local.is_file():
                path = str(local)
            else:
                from huggingface_hub import hf_hub_download

                path = load_cached_first(lambda name, **kw: hf_hub_download(name, "tokenizer.json", **kw),
                                         self.model_name)
            tok = Tokenizer.from_file(path)
            return lambda text: len(tok.encode(text, add_special_tokens=True).ids)
        except Exception:
            from transformers import AutoTokenizer

            hf = load_cached_first(AutoTokenizer.from_pretrained, self.model_name)
            return lambda text: len(hf(text, add_special_tokens=True, truncation=False)["input_ids"])

    def count_tokens_light(self, text: str) -> int:
        """Same count as :meth:`count_tokens` (special tokens included) but loads only the
        tokenizer, so the claim token budget can be enforced before heavy resources are loaded.
        Once the full model is loaded its own tokenizer is used."""
        if self._model is not None:
            return self.count_tokens(text)
        if self._light_tokenizer is None:
            self._light_tokenizer = self._load_light_tokenizer()
        return int(self._light_tokenizer(text))

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)


class DenseIndex:
    """Exact inner-product search over normalized window embeddings, collapsed to documents."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        window_doc_ids: Sequence[str],
        embeddings: np.ndarray,
        encode_query: Callable[[str], np.ndarray],
    ):
        if len(window_doc_ids) != embeddings.shape[0]:
            raise ValueError("window metadata and embeddings are misaligned")
        self.doc_ids = list(doc_ids)
        pos = {d: i for i, d in enumerate(self.doc_ids)}
        missing = {d for d in window_doc_ids if d not in pos}
        if missing:
            raise ValueError(f"windows reference unknown documents: {sorted(missing)[:5]}")
        self.window_doc_idx = np.array([pos[d] for d in window_doc_ids], dtype=np.int64)
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.encode_query = encode_query

    def window_scores(self, query: str) -> np.ndarray:
        q = np.asarray(self.encode_query(query), dtype=np.float32).reshape(-1)
        return self.embeddings @ q

    def doc_scores(self, query: str) -> np.ndarray:
        sims = self.window_scores(query)
        scores = np.full(len(self.doc_ids), -np.inf, dtype=np.float64)
        np.maximum.at(scores, self.window_doc_idx, sims.astype(np.float64))
        return scores

    def search(self, query: str, top_n: int = 30) -> list[tuple[str, float]]:
        return rank_scores(self.doc_scores(query), self.doc_ids, top_n)
