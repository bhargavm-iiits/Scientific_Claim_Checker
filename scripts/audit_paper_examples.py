"""Verify the exact TRAIN artifact used by annotation-assisted judge experiments."""
from collections import Counter
from pathlib import Path

from claim_checker.agents.paper_examples import PaperExampleLibrary
from claim_checker.config import load_config
from claim_checker.data.splits import load_split, normalize_claim_text
from claim_checker.data.store import load_corpus, load_gold, read_jsonl, sha256_file, write_json


def main():
    cfg = load_config()
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    corpus = load_corpus(cfg.path("processed_dir"))
    excluded = {normalize_claim_text(g.claim) for g in validation.values()}
    for name in ("dev_tune", "smoke"):
        excluded.update(normalize_claim_text(train[c].claim) for c in load_split(cfg, name)["claim_ids"])
    source = Path(cfg.root) / "artifacts/examples/train.jsonl"
    examples = read_jsonl(source)
    for example in examples:
        gold = train[example["claim_id"]]
        doc_id = example["doc_id"]
        if (example["source_split"] != "train" or example["claim"] != gold.claim or
                normalize_claim_text(example["claim"]) in excluded):
            raise ValueError("Demonstration contains an excluded or mismatched claim")
        if doc_id not in gold.reference_doc_ids or example["verdict"] != gold.label_by_doc.get(doc_id, "NEI"):
            raise ValueError("Demonstration label/document does not match TRAIN annotations")
        ids = gold.evidence_by_doc.get(doc_id, [])
        expected = " ".join(corpus[doc_id].sentences[i] for i in ids) if ids else " ".join(corpus[doc_id].sentences)
        if example["evidence"] != expected:
            raise ValueError("Demonstration passage is not the original TRAIN source")
    library = PaperExampleLibrary(examples)
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(Path(cfg.root) / "artifacts/runs/dev_v1/retrieval.jsonl")}
    coverage = []
    for claim_id in load_split(cfg, "dev_tune")["claim_ids"]:
        keys = [e["key"] for hit in retrieval[claim_id][:3]
                for e in library.retrieve_for_paper(train[claim_id].claim, hit["doc_id"])]
        coverage.append({"claim_id": claim_id, "training_example_keys": keys})
    report = {"purpose": "TRAIN-annotation-assisted prompting; not corpus-only RAG", "examples": len(examples),
              "labels": dict(Counter(e["verdict"] for e in examples)), "source_sha256": sha256_file(source),
              "excluded_text_overlap": 0, "source_or_label_mismatches": 0,
              "development_claims_with_any_retrieved_paper_demo": sum(bool(r["training_example_keys"]) for r in coverage),
              "development_population": len(coverage), "coverage": coverage}
    write_json(source.parent / "paper_memory_audit.json", report)
    print({key: value for key, value in report.items() if key != "coverage"})


if __name__ == "__main__":
    main()
