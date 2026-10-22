"""Source-preserving positive windows for supervised verifier training."""
from __future__ import annotations


def format_premise(title: str, text: str, include_title: bool = False) -> str:
    return f"Title: {title}\nAbstract: {text}" if include_title else text


def positive_window(tokenizer, claim, title, sentences, alternatives, *, max_length=512, include_title=False):
    """Retain one complete annotated rationale; add neighbouring source context if it fits.

    Gold indices are used exclusively by the training builder. Runtime windowing
    does not call this function and never receives annotations.
    """
    def render(ids):
        return format_premise(title, " ".join(sentences[i] for i in ids), include_title)

    def fits(ids):
        return len(tokenizer(render(ids), claim, add_special_tokens=True)["input_ids"]) <= max_length

    for rationale in sorted((sorted(set(a)) for a in alternatives if a), key=lambda a: (len(a), a[-1] - a[0], a)):
        if any(i < 0 or i >= len(sentences) for i in rationale):
            raise ValueError("Training rationale contains an invalid source index")
        if not fits(rationale):
            continue
        start, end = rationale[0], rationale[-1] + 1
        ids = list(range(start, end))
        if not fits(ids):
            return {"context_sentence_ids": rationale, "rationale_sentence_ids": rationale, "premise": render(rationale)}
        left_first = True
        while True:
            proposals = []
            if start:
                proposals.append((start - 1, end))
            if end < len(sentences):
                proposals.append((start, end + 1))
            if not left_first:
                proposals.reverse()
            picked = next(((a, b) for a, b in proposals if fits(list(range(a, b)))), None)
            if picked is None:
                break
            start, end = picked
            left_first = not left_first
        ids = list(range(start, end))
        return {"context_sentence_ids": ids, "rationale_sentence_ids": rationale, "premise": render(ids)}
    return None
