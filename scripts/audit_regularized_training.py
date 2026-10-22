"""Audit the actual grouped fit/stop rows against source annotations and folds."""
import argparse
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import normalize_claim_text
from claim_checker.data.store import (
    load_corpus,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from claim_checker.data.verifier_context import format_premise

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config()
    protocol_path = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    protocol = read_json(protocol_path)
    split = protocol["folds"][args.fold]
    folder = ROOT / f"artifacts/models/regularized_grouped_v1_fold{args.fold}"
    identity, integrity = read_json(folder / "identity.json"), read_json(folder / "integrity.json")
    assert identity["protocol_sha256"] == sha256_file(protocol_path)
    assert protocol["train_gold_sha256"] == sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl")
    assert identity["base_weights_sha256"] == sha256_file(ROOT / protocol["recipe"]["base_dir"] / "model.safetensors")
    train = load_gold(cfg.path("processed_dir"), "train")
    forbidden = {normalize_claim_text(g.claim) for g in load_gold(cfg.path("processed_dir"), "validation").values()}
    corpus = load_corpus(cfg.path("processed_dir"))
    texts = {key: {normalize_claim_text(train[c].claim) for c in split[key]} for key in ("train", "stop", "audit")}
    papers = {key: {d for c in split[key] for d in train[c].reference_doc_ids} for key in ("train", "stop", "audit")}
    assert all(not texts[key] & forbidden for key in texts)
    for sets in (texts, papers):
        assert not (sets["train"] & sets["stop"] or sets["train"] & sets["audit"] or sets["stop"] & sets["audit"])
    counts = {}
    for key, filename, hash_key in (("train", "training_pairs.jsonl", "training_pairs_sha256"),
                                     ("stop", "stop_pairs.jsonl", "stop_pairs_sha256")):
        path = folder / filename
        assert integrity[hash_key] == sha256_file(path)
        rows = read_jsonl(path)
        for row in rows:
            assert row["claim_id"] in split[key]
            gold, doc = train[row["claim_id"]], corpus[row["doc_id"]]
            assert row["claim"] == gold.claim and gold.split == "train"
            if row["view"] == "hard_negative":
                assert key == "train" and row["label"] == "NEI" and row["doc_id"] not in gold.reference_doc_ids
            else:
                assert row["doc_id"] in gold.reference_doc_ids
                assert row["label"] == gold.label_by_doc.get(row["doc_id"], "NEI")
            if key == "train":
                assert row["doc_id"] not in papers["stop"] | papers["audit"]
            ids = row.get("context_sentence_ids", list(range(len(doc.sentences))))
            assert all(0 <= i < len(doc.sentences) for i in ids)
            assert row["premise"] == format_premise(doc.title, " ".join(doc.sentences[i] for i in ids), True)
            if row.get("rationale_sentence_ids"):
                assert set(row["rationale_sentence_ids"]) <= set(gold.evidence_by_doc[row["doc_id"]]) & set(ids)
        counts[key] = {"claims": len({r["claim_id"] for r in rows}), "pairs": len(rows)}
    result = {"counts": counts, "source_or_annotation_mismatches": 0, "train_stop_audit_reference_paper_overlap": 0,
              "train_stop_audit_normalized_text_overlap": 0, "validation_text_overlap": 0,
              "source_hashes_match": True, "protocol_sha256": sha256_file(protocol_path)}
    write_json(folder / "postfit_data_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()
