"""Training-only in-context examples; evaluation labels never enter the runtime.

The builder explicitly removes all development and validation claim texts. The
runtime reads only the resulting training artifact and excludes the live claim.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from claim_checker.data.splits import normalize_claim_text
from claim_checker.data.store import read_jsonl, write_jsonl
from claim_checker.retrieval.bm25 import BM25Index
from claim_checker.schemas import ClaimGold, CorpusDocument


def build_examples(
    golds: list[ClaimGold],
    corpus: dict[str, CorpusDocument],
    excluded_texts: set[str],
) -> list[dict]:
    """Create demonstrations from training annotations, excluding whole claim groups."""
    excluded = {normalize_claim_text(t) for t in excluded_texts}
    examples = []
    for gold in golds:
        if gold.split != "train":
            raise ValueError("Only the training split may supply demonstrations")
        if gold.conflict or normalize_claim_text(gold.claim) in excluded:
            continue
        for doc_id in gold.reference_doc_ids:
            doc = corpus[doc_id]
            ids = gold.evidence_by_doc.get(doc_id, [])
            label = gold.label_by_doc.get(doc_id, "NEI")
            if label != "NEI" and not ids:
                continue
            # For decisive examples, quote annotated evidence, never an invented rationale.
            # NEI examples have no annotated rationale; show the stored abstract instead.
            text = " ".join(doc.sentences[i] for i in ids) if ids else " ".join(doc.sentences)
            if len(text) > 2800:
                continue
            examples.append({"key": f"{gold.claim_id}/{doc_id}", "claim_id": gold.claim_id,
                             "claim": gold.claim, "doc_id": doc_id, "title": doc.title,
                             "evidence": text, "verdict": label, "source_split": "train"})
    return examples


class ExampleLibrary:
    """Select one lexical neighbour per class from a frozen training artifact."""

    def __init__(self, examples: list[dict]):
        self.examples = {e["key"]: e for e in examples}
        if len(self.examples) != len(examples):
            raise ValueError("Duplicate example identifiers")
        if any(e.get("source_split") != "train" for e in examples):
            raise ValueError("Demonstrations must be training-only")
        by_label = defaultdict(list)
        for e in examples:
            by_label[e["verdict"]].append(e)
        self.indexes = {
            label: BM25Index([e["key"] for e in items], [e["claim"] for e in items])
            for label, items in by_label.items() if items
        }

    @classmethod
    def load(cls, path: str | Path):
        return cls(read_jsonl(path))

    def save(self, path: str | Path):
        write_jsonl(path, self.examples.values())

    def retrieve(self, claim: str, *, doc_id: str | None = None) -> list[dict]:
        text = normalize_claim_text(claim)
        picked = []
        for label in ("SUPPORT", "CONTRADICT", "NEI"):
            index = self.indexes.get(label)
            if index is None:
                continue
            for key, _ in index.search(claim, top_n=len(index.doc_ids)):
                e = self.examples[key]
                if normalize_claim_text(e["claim"]) == text or e["doc_id"] == doc_id:
                    continue
                picked.append(e)
                break
        return picked


def prepend_examples(messages: list[dict], examples: list[dict]) -> list[dict]:
    """Keep target evidence separate from labelled, training-only demonstrations."""
    text = "Training examples from OTHER claims and papers (not evidence for the target):\n\n"
    for example in examples:
        text += (f"Example claim: {example['claim']}\nPassage: {example['evidence']}\n"
                 f"Reference label for that example: {example['verdict']}\n\n")
    return [messages[0], {**messages[1], "content": text + "Now check ONLY the target below:\n\n" + messages[1]["content"]}]
