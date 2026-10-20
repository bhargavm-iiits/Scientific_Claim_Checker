"""Claim-window cross-encoder re-ranking. Long abstracts are windowed and max-aggregated.

Re-ranking scores are relevance scores, not probabilities that a claim is true.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from claim_checker.retrieval.dense import load_cached_first, make_windows
from claim_checker.schemas import CorpusDocument

# [CLS] claim [SEP] passage [SEP]
PAIR_SPECIAL_TOKENS = 3


def rerank_documents(
    claim: str,
    docs: Sequence[CorpusDocument],
    score_pairs: Callable[[list[tuple[str, str]]], Sequence[float]],
    count_tokens: Callable[[str], int],
    max_tokens: int = 512,
) -> list[tuple[str, float, int]]:
    """Return ``[(doc_id, max_window_score, n_windows)]`` sorted by score, ties by doc id.

    ``count_tokens`` counts tokens *without* special tokens.
    """
    if not docs:
        return []
    budget = max_tokens - count_tokens(claim) - PAIR_SPECIAL_TOKENS
    if budget < 16:
        raise ValueError("claim is too long for the re-ranker input budget")
    pairs: list[tuple[str, str]] = []
    owner: list[int] = []
    n_windows: list[int] = []
    for i, doc in enumerate(docs):
        windows = make_windows(doc.title, doc.sentences, count_tokens, budget, overlap=1)
        n_windows.append(len(windows))
        for w in windows:
            pairs.append((claim, w["text"]))
            owner.append(i)
    scores = list(score_pairs(pairs))
    best = [float("-inf")] * len(docs)
    for i, s in zip(owner, scores, strict=True):
        best[i] = max(best[i], float(s))
    ranked = [(docs[i].doc_id, best[i], n_windows[i]) for i in range(len(docs))]
    return sorted(ranked, key=lambda t: (-t[1], t[0]))


class CrossEncoderScorer:
    """Lazy CPU cross-encoder."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512, batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            self._model = load_cached_first(CrossEncoder, self.model_name, device=self.device,
                                            max_length=self.max_length)
        return self._model

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        out = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in out]
