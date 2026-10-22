"""Adapt the trained Qwen judge's cited decisions to the shared workflow records."""
from __future__ import annotations

import json

from claim_checker.models.lora_judge import parse_answer
from claim_checker.pipeline.aggregate import aggregate_verdicts
from claim_checker.schemas import CorpusDocument, DocumentJudgement


def judge_lora(judge, claim: str, doc: CorpusDocument) -> DocumentJudgement:
    windows = judge.judge(claim, doc)
    answers = []
    for window in windows:
        ids = window["sentence_ids"]
        if any(type(i) is not int or i < 0 or i >= len(doc.sentences) for i in ids):
            raise ValueError("LoRA context includes a sentence outside the source abstract")
        answers.append(parse_answer(json.dumps(window["answer"]), ids))
    verdict, conflict = aggregate_verdicts(a.verdict for a in answers)
    cited = sorted({i for a in answers if a.verdict == verdict or conflict for i in a.cited_sentence_ids})
    if conflict:
        reason = "Different passages in this abstract yielded inconsistent decisions; read the quoted source sentences in context."
    elif verdict == "NEI":
        reason = "The supplied abstract does not establish either direction of this claim."
    else:
        reason = "The abstract reports: " + " ".join(doc.sentences[i] for i in cited)
    return DocumentJudgement(doc_id=doc.doc_id, verdict=verdict, cited_sentence_ids=cited,
                             reason=reason, context_sentence_ids=list(range(len(doc.sentences))),
                             mode="full_abstract", prompt_set="lora", skipped_model=not windows,
                             consistency_flags=["window_conflict"] if conflict else [])
