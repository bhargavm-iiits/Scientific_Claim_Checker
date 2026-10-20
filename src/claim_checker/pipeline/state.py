"""Typed LangGraph state and pipeline options."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from claim_checker.schemas import (
    DocumentJudgement,
    ErrorInfo,
    EvidenceSelection,
    RetrievalHit,
    StageEvent,
)


class PipelineOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k: int = 3
    retrieval_method: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True
    evidence_mode: Literal["evidence", "full_abstract"] = "evidence"
    oracle_doc_ids: list[str] | None = None
    variant: str | None = None
    # v2 (added): judge prompt set and qwen3 thinking for the judge step.
    prompt_set: Literal["v1", "v2", "v3"] = "v1"
    judge_think: bool = False
    judge_backend: Literal["ollama", "nli", "lora"] = "ollama"

    @model_validator(mode="after")
    def _lora_contract(self):
        if self.judge_backend == "lora" and (self.evidence_mode != "full_abstract" or self.prompt_set != "v1" or self.judge_think):
            raise ValueError("The LoRA judge requires full_abstract, prompt_set=v1 and judge_think=false")
        return self


def merge_timings(left: dict[str, float] | None, right: dict[str, float] | None) -> dict[str, float]:
    out = dict(left or {})
    for k, v in (right or {}).items():
        out[k] = out.get(k, 0.0) + float(v)
    return out


class GraphState(TypedDict, total=False):
    claim: str
    claim_id: str | None
    options: PipelineOptions
    status: Literal["running", "ok", "error"]
    error: ErrorInfo | None
    hits: list[RetrievalHit]
    doc_ids: list[str]
    selections: dict[str, EvidenceSelection]
    judgements: dict[str, DocumentJudgement]
    verdict: str | None
    conflict: bool
    citation_repairs: int
    pending_repair: list[str]
    # v2 only: per-document supervisor feedback for the repair round, and the consistency flags that
    # triggered a re-ask (kept on the repaired judgement as ``before_repair:<flag>``).
    repair_feedback: dict[str, str]
    consistency_pending: dict[str, list[str]]
    explanation: str
    events: Annotated[list[StageEvent], operator.add]
    timings: Annotated[dict[str, float], merge_timings]
    retrieval_cache: dict[str, Any]  # retrieval_cache_hit (+ retrieve_original_compute_s on a hit)
    extra: dict[str, Any]


STAGES = ("validate_input", "retrieve", "select_evidence", "judge", "aggregate", "supervise", "explain")


class StageStatus(BaseModel):
    stage: str
    status: Literal["pending", "ok", "error", "skipped"] = "pending"
    seconds: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
