"""Deterministic scientific tokenizer and BM25 (Okapi) search with stable tie-breaking."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import numpy as np

TOKENIZER_VERSION = "sci-tok-v1"

# Words joined by hyphens/periods/slashes stay together ("covid-19", "il-6", "3.5", "tnf-alpha")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does for from had has have
    in into is it its may might of on or such than that the their them then there these they
    this those to was were which while will with would""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, NFKC-normalize and split; keep numbers and hyphenated terms.

    A compound such as ``il-6`` yields the compound plus its parts (``il-6``, ``il``, ``6``) so
    that queries written with spaces still match. Stopwords are removed; numbers are never
    removed. Output order is deterministic.
    """
    text = unicodedata.normalize("NFKC", text or "").casefold()
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        tok = match.group(0)
        if any(sep in tok for sep in "-./"):
            tokens.append(tok)
            for part in re.split(r"[-./]", tok):
                if part and part not in _STOPWORDS:
                    tokens.append(part)
        elif tok not in _STOPWORDS:
            tokens.append(tok)
    return tokens


def rank_scores(scores: np.ndarray, doc_ids: Sequence[str], top_n: int) -> list[tuple[str, float]]:
    """Sort by descending score, ties broken by ascending document id (stable, reproducible)."""
    order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), doc_ids[i]))
    return [(doc_ids[i], float(scores[i])) for i in order[:top_n]]


class BM25Index:
    """BM25Okapi over one representation (title + abstract) per document."""

    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str]):
        from rank_bm25 import BM25Okapi

        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must align")
        self.doc_ids = list(doc_ids)
        tokenized = [tokenize(t) for t in texts]
        # rank_bm25 fails on documents with no tokens; give them a placeholder token.
        tokenized = [t if t else ["<empty>"] for t in tokenized]
        self._bm25 = BM25Okapi(tokenized)

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        if not q:
            return np.zeros(len(self.doc_ids), dtype=np.float64)
        return np.asarray(self._bm25.get_scores(q), dtype=np.float64)

    def search(self, query: str, top_n: int = 30) -> list[tuple[str, float]]:
        return rank_scores(self.scores(query), self.doc_ids, top_n)
