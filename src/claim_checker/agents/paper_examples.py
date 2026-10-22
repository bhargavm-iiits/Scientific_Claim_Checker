"""Optional TRAIN demonstrations aligned to a genuinely retrieved paper.

This is annotation-assisted prompting, explicitly distinct from corpus-only RAG.
Evaluation claim texts must be excluded when building the source artifact.
"""
from collections import defaultdict

from claim_checker.agents.examples import ExampleLibrary
from claim_checker.data.splits import normalize_claim_text
from claim_checker.retrieval.bm25 import BM25Index


class PaperExampleLibrary(ExampleLibrary):
    def __init__(self, examples):
        super().__init__(examples)
        groups = defaultdict(list)
        for example in examples:
            groups[example["doc_id"]].append(example)
        self.paper_indexes = {doc_id: BM25Index([e["key"] for e in group], [e["claim"] for e in group])
                              for doc_id, group in groups.items()}

    def retrieve_for_paper(self, claim, doc_id, limit=3):
        index = self.paper_indexes.get(doc_id)
        if index is None:
            return []
        seen, selected = set(), []
        current = normalize_claim_text(claim)
        for key, _ in index.search(claim, top_n=len(index.doc_ids)):
            example = self.examples[key]
            if normalize_claim_text(example["claim"]) == current or example["verdict"] in seen:
                continue
            selected.append(example)
            seen.add(example["verdict"])
            if len(selected) >= limit:
                break
        return selected


def prepend_paper_examples(messages, examples):
    if not examples:
        return messages
    prefix = ("Labeled TRAINING demonstrations for OTHER claims about this same retrieved abstract. "
              "Each label belongs ONLY to its example claim. Do not copy an example label to the current claim. "
              "Changed quantities, negation, direction, subjects or outcomes can change the verdict. "
              "Re-evaluate the current claim using the numbered abstract below.\n\n")
    for example in examples:
        prefix += (f"TRAINING claim: {example['claim']}\nSource passage: {example['evidence']}\n"
                   f"Label for this TRAINING claim only: {example['verdict']}\n\n")
    return [messages[0], {**messages[1], "content": prefix + "CURRENT TARGET:\n\n" + messages[1]["content"]}]
