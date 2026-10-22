"""Audit the actual saved LoRA supervision against original train annotations."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

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
from claim_checker.models.lora_judge import parse_answer, prompt_ids

ROOT = Path(__file__).resolve().parents[1]


def audit(folder):
    cfg = load_config()
    manifest = read_json(folder / "training_manifest.json")
    rows = read_jsonl(folder / "training_rows.jsonl")
    if sha256_file(folder / "training_rows.jsonl") != manifest["training_sha256"]:
        raise ValueError("Training-data hash mismatch")
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    excluded = {normalize_claim_text(g.claim) for g in validation.values()}
    if not manifest["settings"]["final_fit"]:
        for name in ("dev_tune", "smoke"):
            excluded.update(normalize_claim_text(train[c].claim) for c in load_split(cfg, name)["claim_ids"])
    corpus = load_corpus(cfg.path("processed_dir"))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts/models/qwen25_15b_base", local_files_only=True)
    counts = Counter()
    for row in rows:
        gold, doc = train[row["claim_id"]], corpus[row["doc_id"]]
        if row["claim"] != gold.claim or normalize_claim_text(row["claim"]) in excluded:
            raise ValueError("Excluded/mismatched training text")
        if row["provenance"] == "weak_negative":
            if row["doc_id"] in gold.reference_doc_ids or row["label"] != "NEI":
                raise ValueError("Invalid hard negative")
        else:
            if row["doc_id"] not in gold.reference_doc_ids or row["label"] != gold.label_by_doc.get(row["doc_id"], "NEI"):
                raise ValueError("Reference label mismatch")
            if not set(row["cited_sentence_ids"]).issubset(gold.evidence_by_doc.get(row["doc_id"], [])):
                raise ValueError("Training citation not in original rationale")
        expected = prompt_ids(tokenizer, row["claim"], doc.title, doc.sentences, row["sentence_ids"])
        if row["prefix_ids"] != expected:
            raise ValueError("Tokenized prompt is not the original numbered source")
        target = tokenizer.decode(row["target_ids"], skip_special_tokens=True)
        answer = parse_answer(target, row["sentence_ids"])
        if answer.verdict != row["label"] or answer.cited_sentence_ids != row["cited_sentence_ids"]:
            raise ValueError("Target-token supervision mismatch")
        counts[row["provenance"]] += 1
    result = {"rows": len(rows), "unique_claims": len({r["claim_id"] for r in rows}),
              "final_fit": manifest["settings"]["final_fit"], "excluded_text_overlap": 0,
              "source_or_target_mismatches": 0, "provenance": dict(counts),
              "training_sha256": manifest["training_sha256"]}
    write_json(folder / "training_integrity.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    print(audit(parser.parse_args().folder))
