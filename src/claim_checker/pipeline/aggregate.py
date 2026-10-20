"""Deterministic document-to-claim aggregation.

| Document decisions                         | Final                       |
|--------------------------------------------|-----------------------------|
| >= 1 SUPPORT, no CONTRADICT                | SUPPORT                     |
| >= 1 CONTRADICT, no SUPPORT                | CONTRADICT                  |
| all usable decisions NEI (or none)         | NEI                         |
| both SUPPORT and CONTRADICT                | NEI with conflict=True      |

Operational failures never reach this function: they end the graph with status=error.
"""

from __future__ import annotations

from collections.abc import Iterable

from claim_checker.schemas import DocumentJudgement


def aggregate_verdicts(verdicts: Iterable[str]) -> tuple[str, bool]:
    labels = set(verdicts)
    unknown = labels - {"SUPPORT", "CONTRADICT", "NEI"}
    if unknown:
        raise ValueError(f"unsupported verdict labels {sorted(unknown)}")
    has_s, has_c = "SUPPORT" in labels, "CONTRADICT" in labels
    if has_s and has_c:
        return "NEI", True
    if has_s:
        return "SUPPORT", False
    if has_c:
        return "CONTRADICT", False
    return "NEI", False


def aggregate(judgements: Iterable[DocumentJudgement]) -> tuple[str, bool]:
    judgements = list(judgements)
    if any("window_conflict" in j.consistency_flags for j in judgements):
        return "NEI", True
    return aggregate_verdicts(j.verdict for j in judgements)
