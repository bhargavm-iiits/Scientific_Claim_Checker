"""Reciprocal rank fusion with duplicate removal and stable tie-breaking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def dedupe(ids: Iterable[str]) -> list[str]:
    """Keep the first occurrence of every id, preserving order."""
    seen: set[str] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def rrf(rankings: Sequence[Sequence[str]], k: int = 60, top_n: int | None = None) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Duplicates inside one list count once (their best rank). Ties are broken by document id.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(dedupe(ranking), start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_n] if top_n is not None else fused
