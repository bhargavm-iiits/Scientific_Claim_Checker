"""Shared Pydantic records: documents, claims, evidence, judgements, results and LLM outputs.

Sentence positions are zero-based everywhere internally. Interfaces may display them
one-based, but only when rendering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Verdict = Literal["SUPPORT", "CONTRADICT", "NEI"]
VERDICTS: tuple[str, ...] = ("SUPPORT", "CONTRADICT", "NEI")
DECISIVE: tuple[str, ...] = ("SUPPORT", "CONTRADICT")


def normalize_doc_id(value: Any) -> str:
    """Normalize an integer or string document identifier to one canonical string.

    ``13734012``, ``"13734012"`` and ``" 013734012 "`` all map to ``"13734012"``.
    Non-numeric identifiers are returned stripped.
    """
    if value is None:
        raise ValueError("document id is None")
    if isinstance(value, bool):
        raise ValueError("document id cannot be a boolean")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"non-integer numeric document id: {value!r}")
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("empty document id")
    if text.isdigit():
        return str(int(text))
    return text


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


class CorpusDocument(_Record):
    doc_id: str
    title: str
    text: str  # retrieval text (title + abstract)
    sentences: list[str]  # original sentence list, zero-based positions
    source: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> str:
        return normalize_doc_id(v)


class ClaimPair(_Record):
    claim_id: str
    claim: str
    doc_id: str
    title: str
    verdict: Verdict
    evidence: list[int]
    split: str

    @field_validator("doc_id", "claim_id", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> str:
        return normalize_doc_id(v)


class ClaimGold(_Record):
    claim_id: str
    claim: str
    split: str
    cited_doc_ids: list[str]
    reference_doc_ids: list[str]  # cited docs present in the corpus
    evidence_by_doc: dict[str, list[int]]
    label_by_doc: dict[str, Verdict]
    label: Verdict | None  # None when the annotation is conflicting
    conflict: bool = False

    @property
    def evidence_doc_ids(self) -> list[str]:
        return sorted(d for d, s in self.evidence_by_doc.items() if s)


# ---------------------------------------------------------------------------
# Pipeline records
# ---------------------------------------------------------------------------


class RetrievalHit(_Record):
    doc_id: str
    rank: int  # 1-based rank
    method: str
    score: float
    index_fingerprint: str = ""
    title: str = ""
    stage_scores: dict[str, float] = Field(default_factory=dict)


class EvidenceSelection(_Record):
    doc_id: str
    sentence_ids: list[int]
    note: str = ""
    windowed: bool = False
    dropped_invalid_ids: list[int] = Field(default_factory=list)
    cached: bool = False


class DocumentJudgement(_Record):
    doc_id: str
    verdict: Verdict
    cited_sentence_ids: list[int] = Field(default_factory=list)
    reason: str = ""
    confidence: float | None = None  # model-reported, uncalibrated
    mode: str = "evidence"
    context_sentence_ids: list[int] = Field(default_factory=list)
    skipped_model: bool = False
    repaired: bool = False
    downgraded: bool = False
    dropped_invalid_ids: list[int] = Field(default_factory=list)
    cached: bool = False
    # ---- v2 judge only (None / empty for v1 judgements) ------------------------------------------
    prompt_set: str | None = None
    # The v2 checklist the model filled in before its verdict (claim_assertion, evidence_finding,
    # same_subject, evidence_direction).
    checklist: dict[str, str] | None = None
    # Thinking bookkeeping: {"think": bool, "thinking_chars": int, "eval_tokens": int | None,
    # "fallback": bool}. Only the length of the thinking text is kept, never the text.
    thinking: dict[str, Any] | None = None
    # Supervisor consistency findings (e.g. "checklist_opposite", "reason_negates_support").
    consistency_flags: list[str] = Field(default_factory=list)
    # v2: ``reason`` with sentence references rewritten one-based for display ("Sentence [0]" ->
    # "Sentence 1"); ``reason`` itself is the model's text (zero-based, as in the prompt).
    reason_display: str | None = None


class EvidenceQuote(_Record):
    doc_id: str
    title: str
    sentence_id: int  # zero-based
    quote: str
    doc_verdict: Verdict | None = None


class StageEvent(_Record):
    stage: str
    status: Literal["ok", "warning", "error", "info"] = "ok"
    message: str = ""
    timestamp: str = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)


class DocumentReport(_Record):
    doc_id: str
    title: str
    rank: int
    retrieval_score: float | None = None
    retrieval_method: str = ""
    selected_sentence_ids: list[int] | None = None
    judgement: DocumentJudgement | None = None
    # Full stored sentence list of the abstract (zero-based positions), so a saved result can render
    # the abstract without the corpus. Filled by the pipeline; None in older results / baselines.
    sentences: list[str] | None = None
    source: dict[str, Any] = Field(default_factory=dict)


class ErrorInfo(_Record):
    error_type: str
    message: str
    stage: str | None = None
    hint: str | None = None


class ClaimResult(_Record):
    """Shared validated result returned to the UI, the notebook and the CLI."""

    status: Literal["ok", "error"]
    claim: str
    claim_id: str | None = None
    verdict: Verdict | None = None  # None whenever status == "error"
    conflict: bool = False
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    documents: list[DocumentReport] = Field(default_factory=list)
    explanation: str = ""
    timings: dict[str, float] = Field(default_factory=dict)  # seconds per stage + "total"
    metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[StageEvent] = Field(default_factory=list)
    error: ErrorInfo | None = None

    @property
    def judgements(self) -> list[DocumentJudgement]:
        return [d.judgement for d in self.documents if d.judgement is not None]

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)


# ---------------------------------------------------------------------------
# Structured LLM outputs (submitted to Ollama as JSON schemas)
# ---------------------------------------------------------------------------


class EvidenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence_ids: list[int] = Field(
        description="Zero-based ids of sentences that bear on the claim; empty if none."
    )
    note: str = Field(description="One short sentence describing what the evidence is about.")


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Verdict
    cited_sentence_ids: list[int] = Field(
        description="Ids of the supplied sentences that justify the verdict."
    )
    reason: str = Field(description="One or two short sentences explaining the verdict.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence 0-1.")


SameSubject = Literal["yes", "partly", "no"]
EvidenceDirection = Literal["same_as_claim", "opposite_to_claim", "no_effect_found", "not_stated"]


class JudgeOutputV2(BaseModel):
    """v2 judge answer. Field ORDER is the generation order (Ollama's grammar follows the schema's
    property order): the checklist and the reason are written before the verdict, so the label is
    decoded after the model has stated what the claim asserts and what the evidence reports."""

    model_config = ConfigDict(extra="forbid")
    claim_assertion: str = Field(description="What the claim asserts: subject, outcome, population, direction.")
    evidence_finding: str = Field(description="What the supplied sentences report about that relation.")
    same_subject: SameSubject = Field(description="Do the sentences concern the same subject, outcome and population?")
    evidence_direction: EvidenceDirection = Field(description="How the reported finding relates to the claim.")
    reason: str = Field(description="One or two short sentences explaining the verdict.")
    verdict: Verdict
    cited_sentence_ids: list[int] = Field(description="Ids of the supplied sentences that justify the verdict.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence 0-1.")


class VerdictAnswer(BaseModel):
    """v3 evidence comparison before the verdict; shared with development experiments."""

    model_config = ConfigDict(extra="forbid")
    reason: str = Field(description="A concise comparison of the claim with the reported result.")
    verdict: Verdict
    cited_sentence_ids: list[int]
    confidence: float = Field(ge=0, le=1)


class JudgeOnlyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Verdict
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    sentence_id: int


class RetrieveJudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Verdict
    citations: list[Citation]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class DoctorProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Literal["SUPPORT", "CONTRADICT", "NEI"]
    sentence_ids: list[int]
    reason: str
