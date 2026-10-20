import json

import numpy as np
import pytest

from claim_checker.errors import IndexNotReadyError, StaleIndexError
from claim_checker.retrieval.bm25 import BM25Index, rank_scores, tokenize
from claim_checker.retrieval.dense import DenseIndex, make_windows
from claim_checker.retrieval.hybrid import dedupe, rrf
from claim_checker.retrieval.index import check_index, expected_identity, identity_fingerprint
from claim_checker.retrieval.rerank import rerank_documents


def words(text: str) -> int:
    return len(text.split())


def test_tokenizer_preserves_numbers_and_hyphenated_terms():
    toks = tokenize("IL-6 and COVID-19 rose by 3.5% in the cohort")
    assert "il-6" in toks and "covid-19" in toks and "3.5" in toks
    assert "6" in toks and "19" in toks  # parts kept for matching
    assert "the" not in toks and "and" not in toks
    assert tokenize("IL-6 and COVID-19 rose by 3.5%") == tokenize("IL-6 and COVID-19 rose by 3.5%")


def test_bm25_ranks_relevant_document_first_with_stable_ties(corpus):
    ids = list(corpus)
    idx = BM25Index(ids, [corpus[d].text for d in ids])
    assert idx.search("IL-6 sepsis", 2)[0][0] == "303"
    twins = BM25Index(["b", "a", "c", "d", "e"],
                      ["same text here", "same text here", "other words", "more words", "unrelated stuff"])
    top = twins.search("same text", 3)
    assert [d for d, _ in top[:2]] == ["a", "b"]  # tie broken by document id
    assert rank_scores(np.array([1.0, 1.0]), ["z", "y"], 2) == [("y", 1.0), ("z", 1.0)]


def test_rrf_scores_and_duplicate_removal():
    fused = rrf([["a", "b", "a"], ["b", "c"]], k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert [d for d, _ in fused] == ["b", "a", "c"]
    assert dedupe(["x", "y", "x"]) == ["x", "y"]


def test_windows_fit_budget_and_cover_every_sentence():
    sents = [f"sentence number {i} has five words" for i in range(20)]
    wins = make_windows("Title", sents, words, max_tokens=20, overlap=1)
    assert all(w["tokens"] <= 20 for w in wins)
    covered = sorted({i for w in wins for i in w["sentence_ids"]})
    assert covered == list(range(20))
    assert wins[1]["sentence_ids"][0] == wins[0]["sentence_ids"][-1]  # overlap of one sentence
    assert all(w["text"].startswith("Title") for w in wins)


def test_oversized_single_sentence_is_flagged_not_dropped():
    wins = make_windows("T", ["short one", "x " * 50, "tail sentence"], words, max_tokens=10, overlap=0)
    assert any(w["truncated"] and w["sentence_ids"] == [1] for w in wins)
    assert sorted({i for w in wins for i in w["sentence_ids"]}) == [0, 1, 2]


def test_dense_collapses_windows_to_document_max():
    emb = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.0, -1.0]], dtype=np.float32)
    dense = DenseIndex(["d1", "d2"], ["d1", "d1", "d2", "d2"], emb, lambda q: np.array([0.0, 1.0]))
    res = dense.search("q", 2)
    assert res[0] == ("d1", pytest.approx(1.0))  # second window of d1 wins (evidence near the end)
    assert res[1] == ("d2", pytest.approx(0.8))
    with pytest.raises(ValueError):
        DenseIndex(["d1"], ["d9"], emb[:1], lambda q: q)


def test_rerank_windows_long_abstracts_and_takes_max(corpus):
    long_doc = corpus["101"].model_copy(update={"sentences": ["filler words here"] * 30 + ["KEY finding sentence"]})
    docs = [long_doc, corpus["303"]]
    seen = []

    def score(pairs):
        seen.extend(pairs)
        return [5.0 if "KEY" in p[1] else 0.0 for p in pairs]

    ranked = rerank_documents("claim", docs, score, words, max_tokens=30)
    assert ranked[0][0] == "101" and ranked[0][1] == 5.0 and ranked[0][2] > 1
    assert any("KEY" in p[1] for p in seen)  # the end of the long abstract was scored


def test_index_staleness_is_detected(cfg, corpus, tmp_path):
    processed = cfg.path("processed_dir")
    processed.mkdir(parents=True)
    (processed / "corpus.jsonl").write_text(
        "".join(json.dumps(d.model_dump()) + "\n" for d in corpus.values()), encoding="utf-8")
    with pytest.raises(IndexNotReadyError):
        check_index(cfg)
    index_dir = cfg.path("index_dir")
    index_dir.mkdir(parents=True)
    ident = expected_identity(cfg)
    for name in ("documents.jsonl", "windows.jsonl"):
        (index_dir / name).write_text("", encoding="utf-8")
    np.save(index_dir / "embeddings.npy", np.zeros((1, 2), dtype=np.float32))
    (index_dir / "manifest.json").write_text(json.dumps({"identity": ident, "fingerprint": identity_fingerprint(ident)}),
                                             encoding="utf-8")
    assert check_index(cfg)["fingerprint"] == identity_fingerprint(ident)
    cfg.retrieval.max_window_tokens = 256  # embedding configuration changed
    with pytest.raises(StaleIndexError):
        check_index(cfg)
    cfg.retrieval.max_window_tokens = 512
    (processed / "corpus.jsonl").write_text("changed\n", encoding="utf-8")  # corpus changed
    with pytest.raises(StaleIndexError):
        check_index(cfg)
