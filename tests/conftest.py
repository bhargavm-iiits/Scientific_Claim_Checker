"""Synthetic fixtures: a tiny corpus, scripted fake Ollama transport and fake retriever.

No downloads, no model weights, no running Ollama server.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from claim_checker.config import load_config, load_experiments
from claim_checker.errors import ModelUnavailableError
from claim_checker.models.ollama_client import OllamaClient
from claim_checker.retrieval.index import RetrievalOutput
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import CorpusDocument, RetrievalHit

ROOT = Path(__file__).resolve().parents[1]


def make_doc(doc_id: str, title: str, sentences: list[str]) -> CorpusDocument:
    return CorpusDocument(doc_id=doc_id, title=title, text=f"{title}\n{' '.join(sentences)}", sentences=sentences)


@pytest.fixture
def corpus() -> dict[str, CorpusDocument]:
    docs = [
        make_doc("101", "Vitamin D and bone density", [
            "Vitamin D is a fat-soluble vitamin.",
            "We randomized 300 adults to vitamin D or placebo.",
            "Vitamin D supplementation increased bone mineral density by 3.5%.",
            "No serious adverse events occurred.",
        ]),
        make_doc("202", "Vitamin D trial in older women", [
            "Older women were enrolled in a cohort.",
            "Vitamin D supplementation did not change bone mineral density.",
            "Falls were recorded for two years.",
        ]),
        make_doc("303", "IL-6 signalling in sepsis", [
            "IL-6 is a cytokine.",
            "IL-6 levels were elevated in septic patients.",
        ]),
        make_doc("404", "Unrelated astronomy abstract", [
            "Stars form in molecular clouds.",
            "This abstract is about astronomy.",
        ]),
    ]
    return {d.doc_id: d for d in docs}


@pytest.fixture
def cfg(tmp_path):
    overrides = {"paths": {k: str(tmp_path / k) for k in (
        "raw_dir", "processed_dir", "splits_dir", "index_dir", "runs_dir", "retrieval_dir", "cache_dir")}}
    overrides["paths"]["prompts_dir"] = str(ROOT / "prompts")
    # Pin the product default to v1 so tests written against the v1 judge stay independent of the
    # product default in configs/default.yaml; v2 tests choose v2 explicitly.
    overrides["product"] = {"prompt_set": "v1", "judge_think": False}
    return load_config(root=ROOT, overrides=overrides)


@pytest.fixture
def experiments():
    return load_experiments(root=ROOT)


class FakeTransport:
    """Scripted Ollama transport. ``responder(payload) -> str`` produces message content."""

    def __init__(self, responder: Callable[[dict], str] | None = None, models=None, fail: Exception | None = None):
        self.responder = responder or (lambda p: "{}")
        self.models = models if models is not None else [{"name": "qwen3:4b", "digest": "sha256:fake", "size": 1}]
        self.fail = fail
        self.calls: list[dict] = []

    def chat(self, payload: dict) -> str:
        self.calls.append(payload)
        if self.fail:
            raise self.fail
        return self.responder(payload)

    def list_models(self):
        if self.fail:
            raise self.fail
        return self.models

    def version(self):
        return "0.0-fake"


def doc_id_of(payload: dict) -> str | None:
    m = re.search(r"Document (\d+)", payload["messages"][1]["content"]) or re.search(
        r"document (\d+)", payload["messages"][1]["content"])
    return m.group(1) if m else None


def purpose_of(payload: dict) -> str:
    system = payload["messages"][0]["content"]
    if "Evidence agent" in system:
        return "evidence"
    if "Judge agent" in system:
        return "judge"
    if "WITHOUT access" in system:
        return "judge_only"
    return "retrieve_judge"


def is_v2_judge(payload: dict) -> bool:
    """True for a v2 judge request (its schema has the checklist fields)."""
    return "evidence_direction" in ((payload.get("format") or {}).get("properties") or {})


V2_DIRECTION = {"SUPPORT": "same_as_claim", "CONTRADICT": "opposite_to_claim", "NEI": "not_stated"}


def v2_answer(verdict: str, cited: list[int], *, same_subject: str | None = None, direction: str | None = None,
              reason: str | None = None) -> dict:
    """A v2 judge answer; by default its checklist is consistent with ``verdict``."""
    return {"claim_assertion": "scripted claim", "evidence_finding": "scripted finding",
            "same_subject": same_subject or ("yes" if verdict != "NEI" else "partly"),
            "evidence_direction": direction or V2_DIRECTION[verdict],
            "reason": reason or f"scripted {verdict}", "verdict": verdict, "cited_sentence_ids": cited,
            "confidence": 0.8}


def scripted(evidence: dict[str, list[int]], judge: dict[str, tuple[str, list[int]]]):
    """Responder: evidence ids and (verdict, cited ids) per document id."""

    def respond(payload: dict) -> str:
        p = purpose_of(payload)
        d = doc_id_of(payload)
        if p == "evidence":
            return json.dumps({"sentence_ids": evidence.get(d, []), "note": "scripted"})
        if p == "judge":
            verdict, cited = judge.get(d, ("NEI", []))
            if is_v2_judge(payload):
                return json.dumps(v2_answer(verdict, cited))
            return json.dumps({"verdict": verdict, "cited_sentence_ids": cited, "reason": f"scripted {verdict}",
                               "confidence": 0.8})
        if p == "judge_only":
            return json.dumps({"verdict": "SUPPORT", "reason": "general knowledge", "confidence": 0.6})
        return json.dumps({"verdict": "SUPPORT", "citations": [{"doc_id": "101", "sentence_id": 2}],
                           "reason": "doc 101 shows an increase", "confidence": 0.7})

    return respond


class FakeRetriever:
    fingerprint = "fakefp"

    def __init__(self, corpus, order=("101", "202", "303", "404")):
        self.corpus = corpus
        self.order = list(order)
        self.calls = 0

    def retrieve(self, claim, method="hybrid", rerank=True, keep=10):
        self.calls += 1
        hits = [RetrievalHit(doc_id=d, rank=i, method=f"{method}{'+rerank' if rerank else ''}",
                             score=1.0 / i, index_fingerprint=self.fingerprint, title=self.corpus[d].title)
                for i, d in enumerate(self.order[:keep], start=1)]
        return RetrievalOutput(hits=hits, timings={"bm25": 0.001})


@pytest.fixture
def make_checker(cfg, corpus, experiments):
    def _make(responder=None, transport=None, retriever=None, **kw):
        transport = transport or FakeTransport(responder)
        llm = OllamaClient(cfg.model, cache_dir=None, transport=transport)
        retriever = retriever or FakeRetriever(corpus)
        checker = ClaimChecker(cfg, experiments=experiments, corpus=corpus, retriever=retriever, llm=llm,
                               token_counter=lambda s: len(s.split()), **kw)
        checker.transport = transport
        return checker

    return _make


@pytest.fixture
def unavailable():
    return ModelUnavailableError("Cannot reach Ollama at http://127.0.0.1:11434", stage="model",
                                 hint="Install Ollama and run `ollama serve`")
