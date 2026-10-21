"""Experimental joint reader: per-paper evidence comparisons, deterministic aggregation.

Only a live claim and genuinely retrieved source documents enter the model.
Native Ollama thinking is used without a hand-written chat-template continuation.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from claim_checker.errors import ModelOutputError
from claim_checker.schemas import CorpusDocument, DocumentJudgement


class PaperComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    comparison: str = Field(description="At most two short sentences comparing the claim with this paper's actual findings.")
    same_relationship: bool = Field(description="Whether the claimed entities, outcome and explicit scope match this paper's evidence.")
    relation: Literal["supports", "opposes", "insufficient"]
    sentence_ids: list[StrictInt] = Field(description="Zero-based source sentence IDs that establish the comparison.")
    confidence: float = Field(ge=0, le=1, description="Uncalibrated model-reported confidence.")


class ReaderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    papers: list[PaperComparison]


def reader_messages(claim: str, docs: list[CorpusDocument], instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps({
            "claim": claim,
            "papers": [{"doc_id": d.doc_id, "title": d.title,
                        "sentences": [{"id": i, "text": s} for i, s in enumerate(d.sentences)]}
                       for d in docs],
        }, ensure_ascii=False)},
    ]


def comparison_problems(output: ReaderOutput, docs: list[CorpusDocument]) -> list[str]:
    expected = {d.doc_id: d for d in docs}
    ids = [p.doc_id for p in output.papers]
    problems = []
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        problems.append("Return exactly one comparison for each supplied doc_id; no omitted, invented or duplicate papers.")
    for paper in output.papers:
        doc = expected.get(paper.doc_id)
        if doc is None:
            continue
        if any(i < 0 or i >= len(doc.sentences) for i in paper.sentence_ids):
            problems.append(f"Use only supplied sentence IDs for {paper.doc_id}.")
        if paper.relation != "insufficient":
            if not paper.same_relationship:
                problems.append(f"A different or unestablished relationship in {paper.doc_id} requires insufficient.")
            if not paper.sentence_ids:
                problems.append(f"A decisive comparison for {paper.doc_id} must cite source sentence IDs.")
    return problems


def read_papers(client, claim: str, docs: list[CorpusDocument], instruction: str) -> tuple[list[DocumentJudgement], dict]:
    if len({d.doc_id for d in docs}) != len(docs):
        raise ValueError("Duplicate source documents")
    if not docs:
        return [], {"calls": 0, "native_thinking": True}
    response = client.structured(
        reader_messages(claim, docs, instruction), ReaderOutput,
        validator=lambda out: comparison_problems(out, docs), purpose="reasoned_reader_v1",
        think=True, think_budget=None,
    )
    remaining = comparison_problems(response.parsed, docs)
    if remaining:
        raise ModelOutputError("Reader output remains invalid after bounded repair: " + "; ".join(remaining), stage="judge")
    by_id = {p.doc_id: p for p in response.parsed.papers}
    judgements = []
    for doc in docs:
        paper = by_id[doc.doc_id]
        label = {"supports": "SUPPORT", "opposes": "CONTRADICT", "insufficient": "NEI"}[paper.relation]
        judgements.append(DocumentJudgement(
            doc_id=doc.doc_id, verdict=label,
            cited_sentence_ids=sorted(set(paper.sentence_ids)) if label != "NEI" else [],
            context_sentence_ids=list(range(len(doc.sentences))), mode="full_abstract",
            reason=paper.comparison, confidence=paper.confidence, cached=response.cached,
            prompt_set="reasoned_reader_v1",
            checklist={"same_relationship": str(paper.same_relationship), "relation": paper.relation},
        ))
    return judgements, {"native_thinking": True, "thinking_chars": response.thinking_chars,
                        "eval_tokens": response.eval_tokens, "thinking_fallback": response.think_fallback,
                        "cached": response.cached, "attempts": response.attempts}
