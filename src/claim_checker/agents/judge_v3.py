"""v3 judge: full-context relationship comparison and training demonstrations."""
from __future__ import annotations

from collections.abc import Sequence

from claim_checker.agents.evidence import numbered, validate_sentence_ids
from claim_checker.agents.examples import ExampleLibrary, prepend_examples
from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import JudgeV3Config
from claim_checker.models.ollama_client import OllamaClient
from claim_checker.pipeline.aggregate import aggregate_verdicts
from claim_checker.schemas import CorpusDocument, DocumentJudgement, VerdictAnswer


def judge_v3(client: OllamaClient, prompts: PromptLibrary, claim: str, doc: CorpusDocument,
             context_ids: Sequence[int], cfg: JudgeV3Config, examples: ExampleLibrary | None = None,
             feedback: str | None = None) -> tuple[DocumentJudgement, list[dict]]:
    ids, bad_context = validate_sentence_ids(context_ids, range(len(doc.sentences)))
    if bad_context:
        raise ValueError(f"Invalid context sentence ids: {bad_context}")
    windows, window, chars = [], [], 0
    for sid in ids:
        if window and chars + len(doc.sentences[sid]) > cfg.max_context_chars:
            windows.append(window)
            window, chars = [], 0
        window.append(sid)
        chars += len(doc.sentences[sid])
    if window:
        windows.append(window)
    answers, events = [], []
    for window in windows:
        messages = prompts.render("v3/judge", claim=claim, doc_id=doc.doc_id, title=doc.title,
                                  sentences=numbered(doc.sentences, window))
        if cfg.use_examples and examples is not None:
            messages = prepend_examples(messages, examples.retrieve(claim, doc_id=doc.doc_id))
        if feedback:
            messages[1]["content"] += "\nSupervisor feedback: " + feedback

        def validator(answer, window=window):
            _, bad = validate_sentence_ids(answer.cited_sentence_ids, window)
            return [f"Only sentence ids {window} may be cited; remove {bad}"] if bad else []

        response = client.structured(messages, VerdictAnswer, validator=validator, purpose="judge_v3")
        answer = response.parsed
        cited, bad = validate_sentence_ids(answer.cited_sentence_ids, window)
        answers.append((answer, cited, bad, response.cached))
    verdict, conflict = aggregate_verdicts(a.verdict for a, *_ in answers)
    contributing = [a for a in answers if a[0].verdict == verdict] if not conflict else answers
    cited = sorted({sid for _, ids, *_ in contributing for sid in ids})
    return DocumentJudgement(
        doc_id=doc.doc_id, verdict=verdict, cited_sentence_ids=cited,
        reason=" ".join(a.reason for a, *_ in contributing)[:1200] if answers else "No evidence sentences supplied.",
        confidence=max((a.confidence for a, *_ in contributing), default=None),
        mode="full_abstract" if ids == list(range(len(doc.sentences))) else "evidence",
        context_sentence_ids=ids, prompt_set="v3", repaired=bool(feedback),
        cached=bool(answers) and all(a[3] for a in answers), skipped_model=not answers,
        dropped_invalid_ids=sorted({sid for _, _, bad, _ in answers for sid in bad}),
        consistency_flags=["window_conflict"] if conflict else [],
    ), events
