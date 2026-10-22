"""Audit the actual saved training rows against the excluded evaluation populations."""
import argparse
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import load_split, normalize_claim_text
from claim_checker.data.store import (
    load_corpus,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from claim_checker.data.verifier_context import format_premise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    cfg = load_config()
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    manifest = read_json(args.folder / "training_manifest.json")
    final_fit = manifest.get("settings", {}).get("final_fit", False)
    if not final_fit:
        for name in ("dev_tune", "smoke"):
            forbidden.update(normalize_claim_text(train[c].claim) for c in load_split(cfg, name)["claim_ids"])
    rows = read_jsonl(args.folder / "training_pairs.jsonl")
    corpus = load_corpus(cfg.path("processed_dir"))
    for row in rows:
        gold = train[row["claim_id"]]
        assert gold.split == "train" and row["claim"] == gold.claim
        assert normalize_claim_text(row["claim"]) not in forbidden
        if row["view"] == "hard_negative":
            assert row["label"] == "NEI" and row["doc_id"] not in gold.reference_doc_ids
        else:
            assert row["doc_id"] in gold.reference_doc_ids
            assert row["label"] == gold.label_by_doc.get(row["doc_id"], "NEI")
        doc = corpus[row["doc_id"]]
        ids = row.get("context_sentence_ids")
        if ids is None:
            ids = gold.evidence_by_doc[row["doc_id"]] if row["view"] == "rationale" else list(range(len(doc.sentences)))
        assert all(0 <= i < len(doc.sentences) for i in ids)
        assert row["premise"] == format_premise(doc.title, " ".join(doc.sentences[i] for i in ids), row.get("include_title", False))
        if row.get("rationale_sentence_ids"):
            assert set(row["rationale_sentence_ids"]).issubset(ids)
            assert set(row["rationale_sentence_ids"]).issubset(gold.evidence_by_doc[row["doc_id"]])
    result = {"rows": len(rows), "claims": len({r["claim_id"] for r in rows}),
              "final_fit": final_fit,
              "annotated_pairs": sum(r["view"] != "hard_negative" for r in rows),
              "weak_negative_pairs": sum(r["view"] == "hard_negative" for r in rows),
              "evaluation_text_overlap": 0, "annotation_mismatches": 0,
              "sha256": sha256_file(args.folder / "training_pairs.jsonl")}
    write_json(args.folder / "training_integrity.json", result)
    print(result)


if __name__ == "__main__":
    main()
