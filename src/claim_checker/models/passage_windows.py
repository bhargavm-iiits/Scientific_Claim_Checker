"""Annotation-free short source passages with complete sentence coverage."""
from claim_checker.data.verifier_context import format_premise


def passage_windows(tokenizer, claim, title, sentences, *, width, max_length=512, include_title=True):
    if width not in (1, 2):
        raise ValueError("Short passage width must be one or two")

    def fits(ids):
        premise = format_premise(title, " ".join(sentences[i] for i in ids), include_title)
        return len(tokenizer(premise, claim, add_special_tokens=True)["input_ids"]) <= max_length

    windows = []
    covered = set()
    for start in range(max(1, len(sentences) - width + 1)) if sentences else []:
        ids = list(range(start, min(start + width, len(sentences))))
        if not fits(ids):
            ids = [start]
        if not fits(ids):
            raise ValueError("A source sentence and its claim exceed the short-passage token budget")
        windows.append(ids)
        covered.update(ids)
    for i in range(len(sentences)):
        if i not in covered:
            if not fits([i]):
                raise ValueError("A source sentence and its claim exceed the short-passage token budget")
            windows.append([i])
    return windows


def best_passage(scores, *, support_threshold, contradict_threshold, margin):
    """One source passage determines a document; close opposing scores abstain."""
    if not scores:
        return "NEI", None
    best = {label: max(range(len(scores)), key=lambda i: scores[i][label]) for label in ("SUPPORT", "CONTRADICT")}
    support = scores[best["SUPPORT"]]["SUPPORT"]
    contradict = scores[best["CONTRADICT"]]["CONTRADICT"]
    if support == contradict or abs(support - contradict) < margin:
        return "NEI", None
    label = "SUPPORT" if support > contradict else "CONTRADICT"
    cutoff = support_threshold if label == "SUPPORT" else contradict_threshold
    if max(support, contradict) < cutoff:
        return "NEI", None
    return label, best[label]
