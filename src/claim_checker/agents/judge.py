"""Judge agent. Modes:

* ``evidence``      - judge sees only the sentences chosen by the evidence agent (main system);
* ``full_abstract`` - judge sees every sentence (evidence-agent-off ablation, same template/schema);
* ``judge_only``    - claim only, no documents (baseline, separate template);
* single prompt     - :func:`retrieve_and_judge`, the "retrieve + judge in one prompt" baseline.

Prompt sets: ``v1`` (frozen ``prompts/judge.jinja2`` + :class:`JudgeOutput`) and ``v2``
(``prompts/v2/judge.jinja2`` + :class:`JudgeOutputV2`: checklist and reason generated before the
verdict, optional qwen3 thinking). The v1 request is byte-identical to before.

Reference labels and annotated evidence are never placed in model inputs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from claim_checker.agents.evidence import numbered, validate_sentence_ids
from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import JudgeConfig, JudgeV2Config
from claim_checker.models.ollama_client import OllamaClient, cache_warning_event
from claim_checker.schemas import (
    DECISIVE,
    CorpusDocument,
    DocumentJudgement,
    JudgeOnlyOutput,
    JudgeOutput,
    JudgeOutputV2,
    RetrieveJudgeOutput,
    normalize_doc_id,
)

JudgeMode = str  # "evidence" | "full_abstract"


def judge_document(
    client: OllamaClient,
    prompts: PromptLibrary,
    claim: str,
    doc: CorpusDocument,
    mode: JudgeMode,
    selected_ids: Sequence[int] | None,
    cfg: JudgeConfig,
    feedback: str | None = None,
    *,
    prompt_set: str = "v1",
    think: bool = False,
    v2cfg: JudgeV2Config | None = None,
) -> tuple[DocumentJudgement, list[dict]]:
    if prompt_set not in ("v1", "v2"):
        raise ValueError(f"unknown prompt set {prompt_set!r}")
    v2 = prompt_set == "v2"
    v2cfg = v2cfg or JudgeV2Config()
    events: list[dict] = []
    if mode == "full_abstract":
        context_ids = list(range(len(doc.sentences)))
    elif mode == "evidence":
        context_ids = sorted(set(selected_ids or []))
    else:
        raise ValueError(f"unknown judge mode {mode!r}")

    if mode == "evidence" and not context_ids and cfg.skip_when_no_evidence:
        events.append({"status": "info", "message": f"no evidence selected in {doc.doc_id}; NEI without a model call"})
        return DocumentJudgement(
            doc_id=doc.doc_id, verdict="NEI", reason="No sentence in this abstract was selected as evidence.",
            confidence=None, mode=mode, context_sentence_ids=[], skipped_model=True,
            prompt_set="v2" if v2 else None,
        ), events

    messages = prompts.render(
        prompts.v2_name("judge") if v2 else "judge", claim=claim, doc_id=doc.doc_id, title=doc.title, mode=mode,
        sentences=numbered(doc.sentences, context_ids), feedback=feedback,
    )

    def validator(out: JudgeOutput | JudgeOutputV2) -> list[str]:
        _, bad = validate_sentence_ids(out.cited_sentence_ids, context_ids)
        return [f"cited ids {bad} are not among the supplied sentence ids {context_ids}"] if bad else []

    if v2:
        resp = client.structured(messages, JudgeOutputV2, validator=validator, purpose="judge",
                                 think=bool(think),
                                 think_budget=v2cfg.think_budget_tokens if think else None)
        if resp.think_fallback:
            events.append({"status": "warning", "message": f"judge thinking for {doc.doc_id} used the whole output "
                           "budget without answering; answered again without thinking",
                           "data": {"flag": "think_fallback", "doc_id": doc.doc_id}})
    else:
        resp = client.structured(messages, JudgeOutput, validator=validator, purpose="judge")
    out = resp.parsed
    valid, bad = validate_sentence_ids(out.cited_sentence_ids, context_ids)
    if bad:
        events.append({"status": "warning", "message": f"dropped invalid cited ids {bad} for {doc.doc_id}"})
    if (w := cache_warning_event(resp, f"judge/{doc.doc_id}")) is not None:
        events.append(w)
    if resp.attempts > 1:
        events.append({"status": "warning", "message": f"judge output for {doc.doc_id} needed a repair attempt"})
    if resp.repair_fallback:
        events.append({"status": "warning", "message": f"judge repair for {doc.doc_id} was not valid JSON; kept the "
                       "first schema-valid answer and filtered its invalid ids",
                       "data": {"flag": "repair_fallback", "doc_id": doc.doc_id, "problems": resp.problems}})
    return DocumentJudgement(
        doc_id=doc.doc_id,
        verdict=out.verdict,
        cited_sentence_ids=valid,
        reason=out.reason.strip()[:600],
        confidence=float(out.confidence),
        mode=mode,
        context_sentence_ids=context_ids,
        repaired=feedback is not None,
        dropped_invalid_ids=bad,
        cached=resp.cached,
        **(_v2_fields(out, resp, think) if v2 else {}),
    ), events


def _v2_fields(out: JudgeOutputV2, resp, think: bool) -> dict:
    return {
        "prompt_set": "v2",
        "checklist": {
            "claim_assertion": out.claim_assertion.strip()[:400],
            "evidence_finding": out.evidence_finding.strip()[:400],
            "same_subject": out.same_subject,
            "evidence_direction": out.evidence_direction,
        },
        "thinking": {"think": bool(think), "thinking_chars": int(resp.thinking_chars or 0),
                     "eval_tokens": resp.eval_tokens, "budget_forced": int(resp.budget_forced or 0),
                     "fallback": bool(resp.think_fallback)},
    }


def judge_claim_only(client: OllamaClient, prompts: PromptLibrary, claim: str) -> tuple[JudgeOnlyOutput, bool]:
    resp = client.structured(prompts.render("judge_only", claim=claim), JudgeOnlyOutput, purpose="judge_only")
    return resp.parsed, resp.cached  # type: ignore[return-value]


# The single-prompt template introduces each abstract as ``Document {{ d.doc_id }}: <title>``, and the
# model sometimes echoes that header in its citation (``"doc_id": "Document 16927286"``). Strip such
# a label before normalizing so a valid citation is not rejected (and a decisive verdict downgraded).
_DOC_LABEL = re.compile(r"^\s*(?:document|doc)\s*[:#]?\s*", flags=re.IGNORECASE)


def citation_doc_id(value) -> str:
    """Canonical document id of a model citation: an echoed ``Document``/``Doc`` label (optionally
    followed by ``:`` or ``#``) is removed, then the id is normalized like corpus ids."""
    if isinstance(value, str):
        stripped = _DOC_LABEL.sub("", value, count=1)
        value = stripped if stripped.strip() else value
    try:
        return normalize_doc_id(value)
    except ValueError:
        return str(value).strip()


def _citation_key(c) -> tuple[str, int]:
    """Normalize a model citation's document id the same way corpus ids are normalized."""
    return citation_doc_id(c.doc_id), c.sentence_id


def retrieve_and_judge(
    client: OllamaClient,
    prompts: PromptLibrary,
    claim: str,
    docs: Sequence[CorpusDocument],
) -> tuple[dict, list[dict]]:
    """One model call over all supplied abstracts producing a verdict and (doc_id, sentence_id) citations.

    Invalid citations are dropped. A decisive verdict left without any valid citation after the
    repair attempt is downgraded to NEI and flagged (same grounding policy as the pipeline).
    """
    allowed = {(d.doc_id, i) for d in docs for i in range(len(d.sentences))}
    messages = prompts.render(
        "retrieve_judge", claim=claim,
        documents=[{"doc_id": d.doc_id, "title": d.title, "sentences": numbered(d.sentences)} for d in docs],
    )

    def validator(out: RetrieveJudgeOutput) -> list[str]:
        problems = []
        bad = [(c.doc_id, c.sentence_id) for c in out.citations if _citation_key(c) not in allowed]
        if bad:
            problems.append(f"citations {bad} do not refer to supplied (doc_id, sentence_id) pairs")
        if out.verdict in DECISIVE and len(out.citations) - len(bad) == 0:
            problems.append(f"verdict {out.verdict} requires at least one valid citation")
        return problems

    resp = client.structured(messages, RetrieveJudgeOutput, validator=validator, purpose="retrieve_judge")
    out: RetrieveJudgeOutput = resp.parsed  # type: ignore[assignment]
    events: list[dict] = []
    if (w := cache_warning_event(resp, "retrieve_judge")) is not None:
        events.append(w)
    if resp.repair_fallback:
        events.append({"status": "warning", "message": "single-prompt repair was not valid JSON; kept the first "
                       "schema-valid answer and filtered its invalid citations",
                       "data": {"flag": "repair_fallback", "problems": resp.problems}})
    citations, dropped = [], []
    for c in out.citations:
        key = _citation_key(c)
        (citations if key in allowed else dropped).append(key)
    citations = sorted(set(citations))
    verdict = out.verdict
    downgraded = False
    if dropped:
        events.append({"status": "warning", "message": f"dropped invalid citations {dropped}"})
    if verdict in DECISIVE and not citations:
        events.append({"status": "warning", "message": f"decisive verdict {verdict} without valid citation downgraded to NEI",
                       "data": {"flag": "downgraded_no_citation"}})
        verdict, downgraded = "NEI", True
    return {
        "verdict": verdict,
        "model_verdict": out.verdict,
        "citations": citations,
        "dropped": dropped,
        "reason": out.reason.strip()[:600],
        "confidence": float(out.confidence),
        "downgraded": downgraded,
        "cached": resp.cached,
        "attempts": resp.attempts,
    }, events
