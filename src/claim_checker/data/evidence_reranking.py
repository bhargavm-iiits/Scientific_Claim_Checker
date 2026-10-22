"""Source-preserving abstract-window supervision for evidence-aware ranking."""
from claim_checker.retrieval.dense import make_windows


def labeled_windows(tokenizer, claim, doc, *, reference, alternatives, max_length=512):
    """Keep complete rationales; mark nonreference negatives as weak supervision."""
    count = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])  # noqa: E731
    budget = max_length - count(claim) - 3
    if budget < 16:
        raise ValueError("Claim exceeds the ranking input budget")
    rationales = [set(ids) for ids in alternatives if ids]
    if any(any(i < 0 or i >= len(doc.sentences) for i in ids) for ids in rationales):
        raise ValueError("Annotated rationale is outside the source")
    union = set().union(*rationales) if rationales else set()
    rows, omissions = [], []
    for window in make_windows(doc.title, doc.sentences, count, budget, overlap=1):
        ids = set(window["sentence_ids"])
        if window["truncated"]:
            omissions.append({"reason": "oversized_sentence", "sentence_ids": sorted(ids)})
            continue
        positive = any(rationale <= ids for rationale in rationales)
        if rationales and not positive and ids & union:
            omissions.append({"reason": "partial_rationale", "sentence_ids": sorted(ids)})
            continue
        tokens = tokenizer(claim, window["text"], add_special_tokens=True)["input_ids"]
        if len(tokens) > max_length:
            raise ValueError("Ranking window exceeds the exact pair budget")
        rows.append({"doc_id": doc.doc_id, "claim": claim, "premise": window["text"],
                     "sentence_ids": window["sentence_ids"], "target": int(positive),
                     "supervision": "annotated_rationale" if positive else "weak_nonreference" if not reference
                     else "weak_nonrationale" if rationales else "annotated_NEI"})
    return rows, omissions
