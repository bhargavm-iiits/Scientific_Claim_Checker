"""Optional scientific NLI judge, using genuine retrieved evidence only."""
from __future__ import annotations

from claim_checker.pipeline.aggregate import aggregate_verdicts
from claim_checker.schemas import CorpusDocument, DocumentJudgement


def choose_label(scores: dict[str, float], threshold: float | dict[str, float]) -> str:
    label = max(scores, key=scores.get)
    cutoff = threshold.get(label, 0.0) if isinstance(threshold, dict) else threshold
    return label if label == "NEI" or scores[label] >= cutoff else "NEI"


def judge_nli(verifier, claim: str, doc: CorpusDocument, context_ids: list[int], *, threshold: float | dict[str, float]) -> DocumentJudgement:
    if any(i < 0 or i >= len(doc.sentences) for i in context_ids):
        raise ValueError("NLI context contains a sentence outside the document")
    ids = sorted(set(context_ids))
    if not ids:
        return DocumentJudgement(doc_id=doc.doc_id, verdict="NEI", mode="evidence", prompt_set="nli",
                                 skipped_model=True, reason="No evidence sentences were supplied.")
    sentences = [doc.sentences[n] for n in ids]
    def render(text):
        return verifier.format_premise(doc.title, text) if hasattr(verifier, "format_premise") else text

    windows = verifier.windows(claim, sentences, title=doc.title) if getattr(verifier, "include_title", False) else verifier.windows(claim, sentences)
    premises = [render(" ".join(sentences[n] for n in window)) for window in windows]
    scores = verifier.score(premises, [claim] * len(premises))
    labels = [choose_label(s, threshold) for s in scores]
    verdict, conflict = aggregate_verdicts(labels)
    cited, confidence = [], None
    if verdict != "NEI":
        best = max((i for i, label in enumerate(labels) if label == verdict), key=lambda i: scores[i][verdict])
        candidates = [ids[n] for n in windows[best]]
        sentence_scores = verifier.score([render(doc.sentences[n]) for n in candidates], [claim] * len(candidates))
        order = sorted(range(len(candidates)), key=lambda i: (-sentence_scores[i][verdict], candidates[i]))
        cited = sorted(candidates[n] for n in order[:2] if sentence_scores[n][verdict] >= 0.5)
        if not cited:
            cited = [candidates[order[0]]]
        confidence = scores[best][verdict]
        kind = "source passage" if doc.source.get("collection") == "general_science" else "abstract"
        reason = f"The {kind} reports: " + " ".join(doc.sentences[n] for n in cited)
    elif conflict:
        for label in ("SUPPORT", "CONTRADICT"):
            best = max((i for i, value in enumerate(labels) if value == label), key=lambda i: scores[i][label])
            candidates = [ids[n] for n in windows[best]]
            sentence_scores = verifier.score([render(doc.sentences[n]) for n in candidates], [claim] * len(candidates))
            chosen = max(range(len(candidates)), key=lambda i: sentence_scores[i][label])
            cited.append(candidates[chosen])
        cited = sorted(set(cited))
        reason = "Different passages in this abstract yielded inconsistent decisions; read the quoted sentences in their full context."
    else:
        reason = "The retrieved abstract does not provide sufficiently strong evidence for either direction of this claim."
    return DocumentJudgement(
        doc_id=doc.doc_id, verdict=verdict, cited_sentence_ids=cited, context_sentence_ids=ids,
        reason=reason, confidence=confidence, prompt_set="nli",
        mode="full_abstract" if ids == list(range(len(doc.sentences))) else "evidence",
        consistency_flags=["window_conflict"] if conflict else [],
    )
