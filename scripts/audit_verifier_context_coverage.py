"""Check whether truncated positive training inputs retain an annotated rationale."""
import argparse
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from claim_checker.config import load_config
from claim_checker.data.store import load_corpus, load_gold, read_jsonl, sha256_file, write_json
from claim_checker.data.verifier_context import format_premise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--base", type=Path, default=Path("artifacts/models/nli_large_base"))
    args = p.parse_args()
    cfg = load_config()
    corpus = load_corpus(cfg.path("processed_dir"))
    gold = load_gold(cfg.path("processed_dir"), "train")
    raw = {str(r["id"]): r for r in read_jsonl(cfg.path("raw_dir") / "scifact/claims_train.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    counts, missing = Counter(), []
    for row in read_jsonl(args.folder / "training_pairs.jsonl"):
        if row["label"] == "NEI":
            continue
        doc = corpus[row["doc_id"]]
        ids = row.get("context_sentence_ids")
        if ids is None:
            ids = list(range(len(doc.sentences))) if row["view"] == "abstract" else gold[row["claim_id"]].evidence_by_doc[row["doc_id"]]
        pieces = [doc.sentences[i] for i in ids]
        prefix = format_premise(doc.title, "", row.get("include_title", False))
        premise = format_premise(doc.title, " ".join(pieces), row.get("include_title", False))
        assert premise == row["premise"], "Saved premise differs from its original source"
        encoded = tokenizer(premise, row["claim"], truncation="only_first", max_length=512, return_offsets_mapping=True)
        visible_end = max(end for (start, end), segment in zip(encoded["offset_mapping"], encoded.sequence_ids(), strict=True) if segment == 0)
        ends = {sid: len(prefix) + len(" ".join(pieces[:i + 1]).rstrip()) for i, sid in enumerate(ids)}
        alternatives = [r["sentences"] for r in raw[row["claim_id"]]["evidence"][row["doc_id"]]
                        if r["label"] == row["label"] and r["sentences"]]
        complete = any(all(sid in ends and ends[sid] <= visible_end for sid in rationale) for rationale in alternatives)
        counts[f"{row['view']}_positive"] += 1
        counts[f"{row['view']}_truncated"] += visible_end < len(premise.rstrip())
        if not complete:
            counts[f"{row['view']}_without_complete_rationale"] += 1
            missing.append({"claim_id": row["claim_id"], "doc_id": row["doc_id"], "view": row["view"],
                            "visible_characters": visible_end, "premise_characters": len(premise)})
    report = {"purpose": "TRAIN input/rationale coverage audit; no evaluation examples", "counts": dict(counts),
              "training_pairs_sha256": sha256_file(args.folder / "training_pairs.jsonl"),
              "positive_views_without_complete_annotated_rationale": missing}
    write_json(args.folder / "context_coverage_audit.json", report)
    print(dict(counts))


if __name__ == "__main__":
    main()
