"""Supervisor: input/readiness checks, output validation and bounded repair routing.

Policy for a decisive verdict (SUPPORT/CONTRADICT) that cites no valid sentence: the supervisor
routes the document back to the judge once with explicit feedback (bounded by
``judge.max_citation_repairs``); if it still cites nothing, the document judgement is downgraded
to NEI with ``downgraded=True`` and a ``downgraded_no_citation`` event. This guarantees every
successful decisive verdict carries valid source evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from claim_checker.errors import InputValidationError
from claim_checker.schemas import DECISIVE, CorpusDocument, DocumentJudgement

CITATION_FEEDBACK = (
    "Your previous verdict was {verdict} but it cited no valid sentence id. Either cite the supplied "
    "sentence ids that justify a decisive verdict, or answer NEI."
)


def validate_claim_text(
    claim: str | None, max_chars: int, max_tokens: int, count_tokens: Callable[[str], int] | None
) -> str:
    """Reject blank, oversized or over-budget claims before any expensive work."""
    if claim is None or not str(claim).strip():
        raise InputValidationError("The claim is empty. Enter a scientific claim to check.", stage="validate_input")
    text = " ".join(str(claim).split())
    if len(text) > max_chars:
        raise InputValidationError(
            f"The claim is {len(text)} characters long; the limit is {max_chars}.", stage="validate_input",
            hint="Shorten the claim to a single scientific statement.",
        )
    if count_tokens is not None:
        n = count_tokens(text)
        if n > max_tokens:
            raise InputValidationError(
                f"The claim uses {n} tokens; the limit is {max_tokens}.", stage="validate_input",
                hint="Shorten the claim to a single scientific statement.",
            )
    return text


def uncited_decisive(judgements: Iterable[DocumentJudgement]) -> list[str]:
    return [j.doc_id for j in judgements if j.verdict in DECISIVE and not j.cited_sentence_ids]


def downgrade(j: DocumentJudgement) -> DocumentJudgement:
    return j.model_copy(update={
        "verdict": "NEI",
        "downgraded": True,
        "reason": f"[downgraded from {j.verdict}: no valid cited sentence] {j.reason}".strip(),
    })


def quote_problems(judgements: Iterable[DocumentJudgement], corpus: dict[str, CorpusDocument]) -> list[str]:
    """Every cited id must be a real sentence of a supplied document and inside the judge's context."""
    problems = []
    for j in judgements:
        doc = corpus.get(j.doc_id)
        if doc is None:
            problems.append(f"judgement for unknown document {j.doc_id}")
            continue
        for i in j.cited_sentence_ids:
            if not 0 <= i < len(doc.sentences):
                problems.append(f"{j.doc_id}: cited sentence {i} out of range")
            elif j.context_sentence_ids and i not in j.context_sentence_ids:
                problems.append(f"{j.doc_id}: cited sentence {i} was not supplied to the judge")
    return problems
