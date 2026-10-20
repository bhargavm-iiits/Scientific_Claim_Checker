"""Verdict consistency check for the v2 judge (deterministic; no model call).

The v2 judge fills a short checklist (``same_subject``, ``evidence_direction``) and a ``reason``
before its verdict. The supervisor compares the verdict with them:

| verdict    | inconsistent when                                                                    |
|------------|--------------------------------------------------------------------------------------|
| SUPPORT    | direction is opposite_to_claim / no_effect_found / not_stated, or same_subject is   |
|            | not "yes", or the reason literally argues against support ("contradicts", ...)      |
| CONTRADICT | direction is same_as_claim / not_stated, or same_subject is "no", or the reason      |
|            | literally says the evidence supports the claim                                       |
| NEI        | never (an NEI that disagrees with its checklist is only reported, not repaired)      |

An inconsistent judgement is re-asked once (sharing the supervisor's single repair round with the
missing-citation re-ask); if it is still inconsistent it is downgraded to NEI. The rules mirror the
decision rules written in ``prompts/v2/judge.jinja2``. v1 judgements carry no checklist and are
never checked here (v1 behaviour is unchanged).
"""

from __future__ import annotations

import re

from claim_checker.schemas import DocumentJudgement

# Reason wording that argues against a SUPPORT verdict.
_NEGATES_SUPPORT = re.compile(
    r"\b(contradict\w*|refut\w*|do(?:es)?\s+not\s+support\w*|doesn't\s+support\w*|not\s+supported|"
    r"opposite\s+(?:of|to)\s+(?:the|what\s+the)\s+claim|not\s+enough\s+(?:information|evidence)|"
    r"insufficient\s+(?:evidence|information))\b",
    flags=re.IGNORECASE,
)
# Reason wording that argues for support (rejected for a CONTRADICT verdict), unless negated.
_AFFIRMS_SUPPORT = re.compile(
    r"(?<!not )(?<!n't )\b(?:directly\s+)?(?:supports|confirms|agrees\s+with)\s+the\s+claim\b",
    flags=re.IGNORECASE,
)

FLAG_TEXT = {
    "checklist_opposite": "your checklist says the evidence goes in the opposite direction to the claim",
    "checklist_no_effect": "your checklist says the evidence found no effect where the claim asserts one",
    "checklist_not_stated": "your checklist says the sentences do not state the claimed result",
    "checklist_same_direction": "your checklist says the evidence states the same finding as the claim",
    "checklist_different_subject": "your checklist says the sentences are about a different subject",
    "checklist_partial_subject": "your checklist says the sentences only partly match the claim's subject, "
                                 "outcome or population",
    "reason_negates_support": "your reason argues that the evidence does not support the claim",
    "reason_affirms_support": "your reason says the evidence supports the claim",
}

CONSISTENCY_FEEDBACK = (
    "Your previous verdict was {verdict}, but {problems}. Re-read the sentences, fill in the checklist "
    "again and give the verdict that follows from it: SUPPORT only if the sentences directly state the "
    "claim's relation for the same subject and population in the same direction; CONTRADICT if they "
    "state the opposite direction or no effect where the claim asserts one; otherwise NEI."
)


def consistency_flags(j: DocumentJudgement) -> list[str]:
    """Problems between a v2 judgement's verdict and its own checklist / reason (empty = consistent)."""
    if j.checklist is None or j.skipped_model or j.verdict == "NEI":
        return []
    subject = j.checklist.get("same_subject")
    direction = j.checklist.get("evidence_direction")
    flags: list[str] = []
    if j.verdict == "SUPPORT":
        if direction == "opposite_to_claim":
            flags.append("checklist_opposite")
        elif direction == "no_effect_found":
            flags.append("checklist_no_effect")
        elif direction == "not_stated":
            flags.append("checklist_not_stated")
        if subject == "no":
            flags.append("checklist_different_subject")
        elif subject == "partly":
            flags.append("checklist_partial_subject")
        if _NEGATES_SUPPORT.search(j.reason or ""):
            flags.append("reason_negates_support")
    elif j.verdict == "CONTRADICT":
        if direction == "same_as_claim":
            flags.append("checklist_same_direction")
        elif direction == "not_stated":
            flags.append("checklist_not_stated")
        if subject == "no":
            flags.append("checklist_different_subject")
        if _AFFIRMS_SUPPORT.search(j.reason or ""):
            flags.append("reason_affirms_support")
    return flags


def nei_disagrees_with_checklist(j: DocumentJudgement) -> bool:
    """An NEI whose checklist reads like a decisive verdict (reported only, never repaired)."""
    if j.checklist is None or j.skipped_model or j.verdict != "NEI":
        return False
    return (j.checklist.get("same_subject") == "yes"
            and j.checklist.get("evidence_direction") in ("same_as_claim", "opposite_to_claim", "no_effect_found"))


def consistency_feedback(j: DocumentJudgement, flags: list[str]) -> str:
    return CONSISTENCY_FEEDBACK.format(verdict=j.verdict,
                                       problems="; ".join(FLAG_TEXT.get(f, f) for f in flags))


def downgrade_inconsistent(j: DocumentJudgement, flags: list[str]) -> DocumentJudgement:
    return j.model_copy(update={
        "verdict": "NEI",
        "downgraded": True,
        "consistency_flags": list(flags),
        "reason": f"[downgraded from {j.verdict}: verdict inconsistent with the judge's own checklist/reason "
                  f"({', '.join(flags)})] {j.reason}".strip(),
    })
