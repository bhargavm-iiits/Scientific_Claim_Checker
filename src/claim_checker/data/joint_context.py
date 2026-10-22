"""Align original source sentences with tokens for joint label/rationale learning."""
from claim_checker.data.verifier_context import format_premise


def source_context(doc, ids):
    if not ids or ids != sorted(set(ids)) or any(i < 0 or i >= len(doc.sentences) for i in ids):
        raise ValueError("Joint context requires ordered, unique original sentence IDs")
    prefix = format_premise(doc.title, "", True)
    text, spans = prefix, []
    for index, sid in enumerate(ids):
        if index:
            text += " "
        start = len(text)
        text += doc.sentences[sid]
        spans.append((sid, start, len(text)))
    return text, spans


def encode_context(tokenizer, claim, doc, ids, *, positives=None, max_length=512):
    premise, spans = source_context(doc, ids)
    encoded = tokenizer(premise, claim, return_offsets_mapping=True,
                        truncation="only_first", max_length=max_length)
    sequences = encoded.sequence_ids()
    offsets = encoded.pop("offset_mapping")
    source_ids = []
    for sequence, (left, right) in zip(sequences, offsets, strict=True):
        matches = [(min(right, end) - max(left, start), sid) for sid, start, end in spans
                   if sequence == 0 and right > left and min(right, end) > max(left, start)]
        source_ids.append(max(matches)[1] if matches else -1)
    if not any(sid >= 0 for sid in source_ids):
        raise ValueError("No source token fits the joint reader context")
    encoded["source_mask"] = [int(sid >= 0) for sid in source_ids]
    if positives is not None:
        encoded["rationale_targets"] = [int(sid in positives) if sid >= 0 else -100 for sid in source_ids]
    return {"inputs": dict(encoded), "token_source_ids": source_ids, "premise": premise}


def supervised_context(tokenizer, row, doc, gold, *, max_length=512):
    ids = row.get("context_sentence_ids", list(range(len(doc.sentences))))
    premise, _ = source_context(doc, ids)
    if premise != row["premise"]:
        raise ValueError("Training premise does not reconstruct exactly from source sentences")
    if row["label"] != "NEI":
        if len(tokenizer(premise, row["claim"])["input_ids"]) > max_length:
            raise ValueError("Positive joint-training context would truncate annotated evidence")
        if not set(row["rationale_sentence_ids"]) <= set(ids):
            raise ValueError("Joint training lost an annotated rationale")
    positives = set(gold.evidence_by_doc.get(doc.doc_id, []))
    encoded = encode_context(tokenizer, row["claim"], doc, ids, positives=positives, max_length=max_length)
    return {"claim_id": row["claim_id"], "doc_id": doc.doc_id, "label": row["label"],
            "view": row["view"], "class_weight": .25 if row["view"] == "hard_negative" else 1., **encoded}
