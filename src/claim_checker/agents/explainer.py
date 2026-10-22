"""Explainer: deterministic plain-English rendering of validated decisions (no model call)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from claim_checker.schemas import DocumentJudgement, ErrorInfo

ContextSource = Literal["retrieved", "supplied"]

VERDICT_WORDS = {
    "SUPPORT": "SUPPORTED",
    "CONTRADICT": "CONTRADICTED",
    "NEI": "NOT ENOUGH INFORMATION",
}


# How the examined abstracts reached the judge: found by retrieval, or supplied up front (oracle
# mode supplies the reference abstracts, so they must not be described as "retrieved").
SOURCE_PHRASES = {
    "retrieved": ("the retrieved abstracts", "abstract"),
    "supplied": ("the supplied reference abstracts", "reference abstract"),
}


def _source(context: str, document_kind: str = "abstract") -> tuple[str, str]:
    if context not in SOURCE_PHRASES:
        raise ValueError(f"unknown context source {context!r}; expected one of {sorted(SOURCE_PHRASES)}")
    if document_kind == "passage":
        return ("the retrieved source passages", "source passage") if context == "retrieved" else ("the supplied source passages", "source passage")
    return SOURCE_PHRASES[context]


def _plural(n: int, noun: str) -> str:
    return f"{noun}{'s' if n != 1 else ''}"


def _insufficient(context: str) -> str:
    return ("the available collection is insufficient" if context == "retrieved"
            else "the supplied abstracts are insufficient")


def _doc_ref(doc_id: str, titles: Mapping[str, str]) -> str:
    title = titles.get(doc_id, "")
    return f'"{title}" (document {doc_id})' if title else f"document {doc_id}"


def _doc_phrase(j: DocumentJudgement, titles: Mapping[str, str]) -> str:
    return _doc_ref(j.doc_id, titles)


def _sentences(ids: Sequence[int]) -> str:
    # Display one-based sentence numbers (conversion happens only when rendering).
    nums = [str(i + 1) for i in ids]
    if not nums:
        return ""
    return f" (sentence{'s' if len(nums) > 1 else ''} {', '.join(nums)})"


def explain(
    verdict: str | None,
    conflict: bool,
    judgements: Sequence[DocumentJudgement],
    titles: Mapping[str, str],
    error: ErrorInfo | None = None,
    n_documents: int | None = None,
    context: ContextSource = "retrieved",
    document_kind: Literal["abstract", "passage"] = "abstract",
) -> str:
    """Explain a per-document pipeline decision. ``context`` says whether the examined abstracts
    were ``"retrieved"`` from the collection or ``"supplied"`` (oracle reference abstracts)."""
    if error is not None or verdict is None:
        return _operational(error)
    source, source_noun = _source(context, document_kind)
    n = n_documents if n_documents is not None else len(judgements)
    supp = [j for j in judgements if j.verdict == "SUPPORT"]
    contra = [j for j in judgements if j.verdict == "CONTRADICT"]
    parts: list[str] = []
    window_conflicts = [j for j in judgements if "window_conflict" in j.consistency_flags]
    if conflict and window_conflicts and not (supp and contra):
        parts.append("The examined passages produced inconsistent decisions within "
                     + ", ".join(_doc_phrase(j, titles) for j in window_conflicts)
                     + f". The result is NOT ENOUGH INFORMATION; read the quoted sentences in the full {document_kind} context.")
    elif conflict:
        parts.append(
            ("The sources produced conflicting evidence: " if document_kind == "passage" else "The papers produced conflicting evidence: ")
            + "; ".join(f"{_doc_phrase(j, titles)} supports the claim" for j in supp)
            + "; while "
            + "; ".join(f"{_doc_phrase(j, titles)} contradicts it" for j in contra)
            + ". Under this prototype's policy, conflicting evidence is reported as NOT ENOUGH INFORMATION "
            "with a conflict flag, so you should read the quoted sentences from both sides."
        )
    elif verdict in ("SUPPORT", "CONTRADICT"):
        deciders = supp if verdict == "SUPPORT" else contra
        verb = "support" if verdict == "SUPPORT" else "contradict"
        if len(deciders) == 1:
            verb += "s"
        parts.append(
            f"Verdict: the claim is {VERDICT_WORDS[verdict]} by {source}. "
            f"{len(deciders)} of {n} examined {_plural(n, source_noun)} {verb} it."
        )
        for j in deciders:
            parts.append(f"{_doc_phrase(j, titles)}{_sentences(j.cited_sentence_ids)}: {display_reason(j)}")
        others = n - len(deciders)
        if others:
            parts.append(f"The other {others} {document_kind}{'s' if others != 1 else ''} did not give decisive evidence.")
    else:
        parts.append(
            f"Verdict: NOT ENOUGH INFORMATION. None of the {n} examined {_plural(n, source_noun)} "
            f"gave decisive evidence for or against the claim. This means {_insufficient(context)} - it "
            "does not mean the claim is false."
        )
    inconsistent = [j for j in judgements if j.downgraded and j.consistency_flags
                    and not all(f.startswith("before_repair:") for f in j.consistency_flags)]
    if any(j.downgraded and j not in inconsistent for j in judgements):
        parts.append(
            "Note: a decisive judgement without a valid cited sentence was downgraded to NOT ENOUGH INFORMATION."
        )
    if inconsistent:  # v2 supervisor consistency check
        parts.append(
            "Note: a decisive judgement that contradicted the judge's own checklist or reason (even after one "
            "re-check) was downgraded to NOT ENOUGH INFORMATION."
        )
    confs = [j.confidence for j in judgements if j.confidence is not None and j.verdict == verdict]
    if confs and not conflict:
        parts.append(f"Model-reported confidence (uncalibrated): {max(confs):.2f}.")
    return " ".join(p.strip() for p in parts if p.strip())


def _sentence_end(text: str) -> str:
    """``text`` ending with exactly one sentence-final mark (no ``..`` when it already ends with one)."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _operational(error: ErrorInfo | None) -> str:
    msg = error.message if error else "unknown error"
    hint = f" Suggested fix: {_sentence_end(error.hint)}" if error and error.hint else ""
    return (
        "The check could not be completed because of an operational problem, so no verdict is "
        f"given. This is not a scientific judgement. Details: {_sentence_end(msg)}{hint}"
    )


# Model reasons refer to the zero-based ids shown in the prompt ("Sentence [0] states ...", "[3]").
# For display they are rewritten one-based, matching the quotes and the UI ("sentence 1", "[S4]").
# Only numbers that were actually supplied to the judge are rewritten; stored ids stay zero-based.
_SENT_REF = re.compile(
    r"\b(?P<word>[Ss]entences?)\s*\[(?P<a>\d+)\]"
    r"|\[(?P<b>\d+)\]"
    r"|\b(?P<word2>[Ss]entences?)\s+(?P<c>\d+)\b"
)


def one_based_reason(reason: str, supplied_ids) -> str:
    """Rewrite zero-based sentence references in a v2 judge reason to one-based display numbers."""
    allowed = set(supplied_ids or [])

    def sub(m: re.Match) -> str:
        if m.group("a") is not None:
            n = int(m.group("a"))
            return f"{m.group('word')} {n + 1}" if n in allowed else m.group(0)
        if m.group("b") is not None:
            n = int(m.group("b"))
            return f"[S{n + 1}]" if n in allowed else m.group(0)
        n = int(m.group("c"))
        return f"{m.group('word2')} {n + 1}" if n in allowed else m.group(0)

    return _SENT_REF.sub(sub, reason or "")


def display_reason(j: DocumentJudgement) -> str:
    """The reason as shown to people: v2 reasons get one-based sentence numbers; v1 text is unchanged."""
    if j.prompt_set == "v2":
        return one_based_reason(j.reason, j.context_sentence_ids)
    return j.reason


def explain_single_prompt(
    verdict: str | None,
    *,
    reason: str,
    citations: Mapping[str, Sequence[int]],
    titles: Mapping[str, str],
    n_documents: int,
    confidence: float | None = None,
    downgraded: bool = False,
    model_verdict: str | None = None,
    context: ContextSource = "retrieved",
    error: ErrorInfo | None = None,
) -> str:
    """Explain the single-prompt baseline: one *claim-level* verdict from one model call over all
    examined abstracts, grounded by ``citations`` (``{doc_id: [zero-based sentence ids]}``, already
    validated against the supplied documents). The baseline produces no per-document verdicts, so
    none are invented here."""
    if error is not None or verdict is None:
        return _operational(error)
    source, source_noun = _source(context)
    n = n_documents
    parts = [f"Single-prompt baseline: one model call read all {n} {_plural(n, source_noun)} together "
             "and gave one claim-level verdict (abstracts are not judged one by one)."]
    cited = [(d, sorted(set(ids))) for d, ids in citations.items() if ids]
    refs = "; ".join(f"{_doc_ref(d, titles)}{_sentences(ids)}" for d, ids in cited)
    if verdict in ("SUPPORT", "CONTRADICT"):
        verb = "supports" if verdict == "SUPPORT" else "contradicts"
        parts.append(f"Verdict: the claim is {VERDICT_WORDS[verdict]} by {source}.")
        parts.append(f"Cited evidence that {verb} it: {refs}.")
        if reason.strip():
            parts.append(f"Model reason: {reason.strip()}")
    else:
        if downgraded:
            parts.append(
                f"Verdict: NOT ENOUGH INFORMATION. The model answered {model_verdict or 'decisively'} but cited "
                "no valid sentence of the examined abstracts, so the answer was downgraded to NOT ENOUGH "
                "INFORMATION."
            )
        else:
            parts.append(
                "Verdict: NOT ENOUGH INFORMATION. The model found no decisive evidence for or against the claim "
                f"in the {n} examined {_plural(n, source_noun)}. This means {_insufficient(context)} - it does "
                "not mean the claim is false."
            )
        if reason.strip():
            parts.append(f"Model reason: {reason.strip()}")
        if refs:
            parts.append(f"Sentences the model pointed to: {refs}.")
    if confidence is not None and not downgraded:
        parts.append(f"Model-reported confidence (uncalibrated): {confidence:.2f}.")
    return " ".join(p.strip() for p in parts if p.strip())
