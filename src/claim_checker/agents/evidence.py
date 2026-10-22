"""Evidence agent: the model selects sentence ids; application code builds the quotations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import EvidenceConfig
from claim_checker.models.ollama_client import OllamaClient, cache_warning_event
from claim_checker.schemas import CorpusDocument, EvidenceOutput, EvidenceQuote, EvidenceSelection


def numbered(sentences: Sequence[str], ids: Iterable[int] | None = None) -> list[dict]:
    """Sentence records for prompts; ids are zero-based source positions."""
    ids = list(range(len(sentences))) if ids is None else list(ids)
    return [{"id": i, "text": sentences[i]} for i in ids]


def validate_sentence_ids(ids: Iterable[int], allowed: Iterable[int]) -> tuple[list[int], list[int]]:
    """Split ids into (valid unique sorted ids, invalid ids). Duplicates are removed."""
    allowed_set = set(allowed)
    valid, invalid = set(), []
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int):
            invalid.append(i)
        elif i in allowed_set:
            valid.add(i)
        else:
            invalid.append(i)
    return sorted(valid), invalid


def build_quotes(doc: CorpusDocument, sentence_ids: Iterable[int], doc_verdict=None) -> list[EvidenceQuote]:
    """Quotations come from stored source sentences only - never from model text."""
    quotes = []
    for i in sorted(set(sentence_ids)):
        if not 0 <= i < len(doc.sentences):
            raise ValueError(f"sentence id {i} out of range for document {doc.doc_id}")
        quotes.append(
            EvidenceQuote(doc_id=doc.doc_id, title=doc.title, sentence_id=i, quote=doc.sentences[i],
                          doc_verdict=doc_verdict)
        )
    return quotes


def sentence_windows(sentences: Sequence[str], max_sentences: int, max_chars: int) -> list[tuple[int, int]]:
    """Bounded [start, end) windows over the sentence list for oversized abstracts."""
    n = len(sentences)
    if n == 0:
        return []
    out, start = [], 0
    while start < n:
        end, chars = start, 0
        while end < n and end - start < max_sentences and (chars + len(sentences[end]) <= max_chars or end == start):
            chars += len(sentences[end])
            end += 1
        out.append((start, end))
        start = end
    return out


def select_evidence(
    client: OllamaClient,
    prompts: PromptLibrary,
    claim: str,
    doc: CorpusDocument,
    cfg: EvidenceConfig,
) -> tuple[EvidenceSelection, list[dict]]:
    """Return the validated selection and stage-event dicts."""
    events: list[dict] = []
    windows = sentence_windows(doc.sentences, cfg.max_sentences_per_call, cfg.max_chars_per_call)
    if not windows:
        events.append({"status": "warning", "message": f"document {doc.doc_id} has no sentences"})
        return EvidenceSelection(doc_id=doc.doc_id, sentence_ids=[], note="abstract has no sentences"), events
    windowed = len(windows) > 1
    if windowed:
        events.append({"status": "info", "message": f"document {doc.doc_id} split into {len(windows)} evidence windows",
                       "data": {"windows": windows}})
    selected: set[int] = set()
    dropped: list[int] = []
    notes, cached = [], True
    for start, end in windows:
        allowed = range(start, end)
        messages = prompts.render(
            "evidence", claim=claim, doc_id=doc.doc_id, title=doc.title,
            sentences=numbered(doc.sentences, allowed), windowed=windowed,
        )

        def validator(out: EvidenceOutput, allowed=allowed) -> list[str]:
            _, bad = validate_sentence_ids(out.sentence_ids, allowed)
            return [f"sentence ids {bad} are not in the numbered list ({allowed.start}-{allowed.stop - 1})"] if bad else []

        resp = client.structured(messages, EvidenceOutput, validator=validator, purpose="evidence")
        cached = cached and resp.cached
        valid, bad = validate_sentence_ids(resp.parsed.sentence_ids, allowed)
        selected.update(valid)
        dropped.extend(bad)
        notes.append(resp.parsed.note.strip())
        if (w := cache_warning_event(resp, f"evidence/{doc.doc_id}")) is not None:
            events.append(w)
        if resp.attempts > 1:
            events.append({"status": "warning", "message": f"evidence output for {doc.doc_id} needed a repair attempt"})
        if resp.repair_fallback:
            events.append({"status": "warning", "message": f"evidence repair for {doc.doc_id} was not valid JSON; kept "
                           "the first schema-valid answer and filtered its invalid ids",
                           "data": {"flag": "repair_fallback", "doc_id": doc.doc_id, "problems": resp.problems}})
    if dropped:
        events.append({"status": "warning", "message": f"dropped invalid evidence ids {dropped} for {doc.doc_id}"})
    selection = EvidenceSelection(
        doc_id=doc.doc_id,
        sentence_ids=sorted(selected),
        note=" ".join(n for n in notes if n)[:500],
        windowed=windowed,
        dropped_invalid_ids=dropped,
        cached=cached,
    )
    return selection, events
