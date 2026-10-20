"""Index construction, fingerprints, stale-index detection and the composed Retriever."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from claim_checker.config import AppConfig, canonical_json, sha256_text
from claim_checker.data.sources import pkg_version
from claim_checker.data.store import (
    CORPUS_FILE,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from claim_checker.errors import INDEX_SETUP_HINT, IndexNotReadyError, StaleIndexError
from claim_checker.retrieval.bm25 import TOKENIZER_VERSION, BM25Index
from claim_checker.retrieval.dense import WINDOW_FORMAT, DenseIndex, SentenceEncoder, make_windows
from claim_checker.retrieval.hybrid import rrf
from claim_checker.retrieval.rerank import CrossEncoderScorer, rerank_documents
from claim_checker.schemas import CorpusDocument, RetrievalHit, utc_now

MANIFEST = "manifest.json"
DOCUMENTS = "documents.jsonl"
WINDOWS = "windows.jsonl"
EMBEDDINGS = "embeddings.npy"


def expected_identity(cfg: AppConfig, corpus_sha256: str | None = None) -> dict:
    """Everything that, if changed, makes an existing index stale."""
    if corpus_sha256 is None:
        corpus_path = cfg.path("processed_dir") / CORPUS_FILE
        if not corpus_path.exists():
            raise IndexNotReadyError(
                f"Processed corpus missing: {corpus_path}", stage="index",
                hint="Run `claim-checker prepare-data` first.",
            )
        corpus_sha256 = sha256_file(corpus_path)
    r = cfg.retrieval
    return {
        "corpus_sha256": corpus_sha256,
        "bm25_tokenizer": TOKENIZER_VERSION,
        "bm25_text": ("title + newline + source passage" if cfg.knowledge.kind == "general_science"
                      else "title + newline + MTEB abstract text"),
        "embedding_model": r.embedding_model,
        "query_instruction": r.query_instruction,
        "max_window_tokens": r.max_window_tokens,
        "window_overlap_sentences": r.window_overlap_sentences,
        "window_format": WINDOW_FORMAT,
        "embeddings_normalized": True,
    }


def identity_fingerprint(identity: dict) -> str:
    return sha256_text(canonical_json(identity))[:16]


def _model_revision(repo_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception:
        return None


def build_index(
    cfg: AppConfig,
    corpus: dict[str, CorpusDocument],
    *,
    encoder: SentenceEncoder | None = None,
    log=print,
) -> dict:
    index_dir = cfg.path("index_dir")
    index_dir.mkdir(parents=True, exist_ok=True)
    identity = expected_identity(cfg)
    r = cfg.retrieval
    encoder = encoder or SentenceEncoder(r.embedding_model, r.embedding_device, r.embedding_batch_size)

    t0 = time.perf_counter()
    docs = list(corpus.values())
    max_tokens = min(r.max_window_tokens, encoder.max_seq_length)
    windows = []
    for doc in docs:
        for w_i, w in enumerate(
            make_windows(doc.title, doc.sentences, encoder.count_tokens, max_tokens, r.window_overlap_sentences)
        ):
            windows.append({"window_id": len(windows), "doc_id": doc.doc_id, "window_index": w_i, **w})
    t_windows = time.perf_counter() - t0
    log(f"Built {len(windows)} windows for {len(docs)} documents ({t_windows:.1f}s); encoding on {r.embedding_device}")

    t1 = time.perf_counter()
    emb = encoder.encode([w["text"] for w in windows], show_progress=True)
    t_encode = time.perf_counter() - t1

    np.save(index_dir / EMBEDDINGS, emb)
    write_jsonl(index_dir / DOCUMENTS, [{"doc_id": d.doc_id, "title": d.title, "text": d.text} for d in docs])
    write_jsonl(index_dir / WINDOWS, windows)
    multi = sum(1 for d in {w["doc_id"] for w in windows if w["window_index"] > 0})
    manifest = {
        "created_at": utc_now(),
        "identity": identity,
        "fingerprint": identity_fingerprint(identity),
        "documents": len(docs),
        "windows": len(windows),
        "documents_with_multiple_windows": multi,
        "truncated_single_sentence_windows": sum(1 for w in windows if w["truncated"]),
        "effective_max_window_tokens": max_tokens,
        "embedding_dim": int(emb.shape[1]),
        "embeddings_sha256": sha256_file(index_dir / EMBEDDINGS),
        "embedding_model_revision": _model_revision(r.embedding_model),
        "timings_seconds": {"windows": round(t_windows, 2), "encode": round(t_encode, 2)},
        "environment": {
            "sentence_transformers": pkg_version("sentence-transformers"),
            "torch": pkg_version("torch"),
            "rank_bm25": pkg_version("rank-bm25"),
            "device": r.embedding_device,
        },
    }
    write_json(index_dir / MANIFEST, manifest)
    return manifest


def check_index(cfg: AppConfig) -> dict:
    """Return the manifest if the index exists and matches the current corpus/config."""
    index_dir = cfg.path("index_dir")
    missing = [n for n in (MANIFEST, DOCUMENTS, WINDOWS, EMBEDDINGS) if not (index_dir / n).exists()]
    if missing:
        raise IndexNotReadyError(f"Index files missing in {index_dir}: {missing}", stage="index", hint=INDEX_SETUP_HINT)
    manifest = read_json(index_dir / MANIFEST)
    expected = expected_identity(cfg)
    actual = manifest.get("identity", {})
    if actual != expected:
        changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise StaleIndexError(
            f"Index at {index_dir} is stale: {', '.join(changed)} changed since it was built.",
            stage="index",
            hint="Rebuild it with `claim-checker build-index`.",
        )
    return manifest


@dataclass
class RetrievalOutput:
    hits: list[RetrievalHit]
    timings: dict[str, float] = field(default_factory=dict)
    candidates: dict[str, list[tuple[str, float]]] = field(default_factory=dict)


class Retriever:
    """BM25 + dense + hybrid RRF, optional cross-encoder re-ranking."""

    def __init__(
        self,
        cfg: AppConfig,
        corpus: dict[str, CorpusDocument],
        bm25: BM25Index,
        dense: DenseIndex,
        fingerprint: str,
        reranker: CrossEncoderScorer | None = None,
        rerank_fn: Callable[[list[tuple[str, str]]], Sequence[float]] | None = None,
        rerank_count_tokens: Callable[[str], int] | None = None,
    ):
        self.cfg = cfg
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.fingerprint = fingerprint
        r = cfg.retrieval
        self.reranker = reranker or CrossEncoderScorer(
            r.reranker_model, r.reranker_device, r.reranker_max_tokens, r.reranker_batch_size
        )
        self._rerank_fn = rerank_fn
        self._rerank_count = rerank_count_tokens

    @classmethod
    def load(cls, cfg: AppConfig, corpus: dict[str, CorpusDocument], encoder: SentenceEncoder | None = None):
        manifest = check_index(cfg)
        index_dir = cfg.path("index_dir")
        documents = read_jsonl(index_dir / DOCUMENTS)
        windows = read_jsonl(index_dir / WINDOWS)
        emb = np.load(index_dir / EMBEDDINGS)
        doc_ids = [d["doc_id"] for d in documents]
        if set(doc_ids) != set(corpus):
            raise StaleIndexError("Index documents do not match the prepared corpus", stage="index",
                                  hint="Rebuild it with `claim-checker build-index`.")
        r = cfg.retrieval
        encoder = encoder or SentenceEncoder(r.embedding_model, r.embedding_device, r.embedding_batch_size)

        def encode_query(q: str) -> np.ndarray:
            return encoder.encode([r.query_instruction + q])[0]

        bm25 = BM25Index(doc_ids, [d["text"] for d in documents])
        dense = DenseIndex(doc_ids, [w["doc_id"] for w in windows], emb, encode_query)
        retriever = cls(cfg, corpus, bm25, dense, manifest["fingerprint"])
        retriever.encoder = encoder
        return retriever

    # -- components -------------------------------------------------------
    def candidates(self, claim: str, methods: Sequence[str] = ("bm25", "dense", "hybrid")) -> RetrievalOutput:
        n = self.cfg.retrieval.candidates_per_method
        out = RetrievalOutput(hits=[])
        need_bm25 = "bm25" in methods or "hybrid" in methods
        need_dense = "dense" in methods or "hybrid" in methods
        if need_bm25:
            t = time.perf_counter()
            out.candidates["bm25"] = self.bm25.search(claim, n)
            out.timings["bm25"] = time.perf_counter() - t
        if need_dense:
            t = time.perf_counter()
            out.candidates["dense"] = self.dense.search(claim, n)
            out.timings["dense"] = time.perf_counter() - t
        if "hybrid" in methods:
            t = time.perf_counter()
            out.candidates["hybrid"] = rrf(
                [[d for d, _ in out.candidates["bm25"]], [d for d, _ in out.candidates["dense"]]],
                k=self.cfg.retrieval.rrf_k,
            )
            out.timings["fuse"] = time.perf_counter() - t
        return out

    def rerank(self, claim: str, doc_ids: Sequence[str]) -> tuple[list[tuple[str, float, int]], float]:
        t = time.perf_counter()
        fn = self._rerank_fn or self.reranker.score
        count = self._rerank_count or self.reranker.count_tokens
        ranked = rerank_documents(
            claim, [self.corpus[d] for d in doc_ids], fn, count, self.cfg.retrieval.reranker_max_tokens
        )
        return ranked, time.perf_counter() - t

    # -- main entry --------------------------------------------------------
    def retrieve(self, claim: str, method: str = "hybrid", rerank: bool = True, keep: int | None = None) -> RetrievalOutput:
        r = self.cfg.retrieval
        keep = keep or r.keep_top
        out = self.candidates(claim, (method,))
        base = out.candidates[method]
        if self.cfg.knowledge.kind == "general_science":
            # Textbook editions can repeat the same passage. Keep the highest
            # ranked copy within the same subject/section context before reranking.
            unique, seen = [], set()
            for doc_id, score in base:
                doc = self.corpus[doc_id]
                key = (doc.source.get("subject"), doc.title.partition(" — ")[2],
                       " ".join(" ".join(doc.sentences).split()))
                if key not in seen:
                    seen.add(key)
                    unique.append((doc_id, score))
            base = unique
        label = method
        if rerank:
            shortlist = [d for d, _ in base[: r.rerank_top_n]]
            ranked, t_rr = self.rerank(claim, shortlist)
            out.timings["rerank"] = t_rr
            base_scores = dict(base)
            scored = [(d, s, {method: base_scores[d], "rerank_windows": nw}) for d, s, nw in ranked]
            label = f"{method}+rerank"
        else:
            scored = [(d, s, {method: s}) for d, s in base]
        out.hits = [
            RetrievalHit(
                doc_id=d, rank=i, method=label, score=float(s), index_fingerprint=self.fingerprint,
                title=self.corpus[d].title, stage_scores={k: float(v) for k, v in extra.items()},
            )
            for i, (d, s, extra) in enumerate(scored[:keep], start=1)
        ]
        return out


def index_dir_files(cfg: AppConfig) -> list[Path]:
    d = cfg.path("index_dir")
    return [d / n for n in (MANIFEST, DOCUMENTS, WINDOWS, EMBEDDINGS)]
