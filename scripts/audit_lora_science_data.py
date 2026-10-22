"""Independent source/annotation/exposure audit without loading neural weights."""
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from claim_checker.config import load_config, sha256_text
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
    plan_path = ROOT / "artifacts/improvement/lora_science_protocol_v2.json"
    plan = read_json(plan_path)
    folder = ROOT / "artifacts/improvement/lora_science_selection_data_v2"
    integrity = read_json(folder / "integrity.json")
    assert integrity["identity"]["plan_sha256"] == sha256_file(plan_path)
    for path, digest in plan["source_sha256"].items():
        assert sha256_file(ROOT / path) == digest, path
    cfg = load_config()
    gold = load_gold(cfg.path("processed_dir"), "train")
    forbidden = {normalize_claim_text(g.claim) for g in load_gold(cfg.path("processed_dir"), "validation").values()}
    corpus = load_corpus(cfg.path("processed_dir"))
    tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/models/nli_large_base/tokenizer.json"))
    tokenizer.no_truncation()
    papers, texts, counts = {}, {}, {}
    fit_mining = read_jsonl(folder / "fit_retrieval_hits.jsonl")
    assert {r["claim_id"] for r in fit_mining} == set(plan["fit_ids"])
    assert len(fit_mining) == len(plan["fit_ids"])
    assert sha256_file(folder / "fit_retrieval_hits.jsonl") == integrity["retrieval_hits_sha256"]
    negative_by_claim = {r["claim_id"]: r["negative_doc_id"] for r in fit_mining}
    for row in fit_mining:
        if row["negative_doc_id"] is not None:
            assert row["negative_doc_id"] in {h["doc_id"] for h in row["hits"]}
    for role in ("fit", "stop"):
        path = folder / f"{role}_rows.jsonl"
        assert sha256_file(path) == integrity[f"{role}_rows_sha256"]
        rows = read_jsonl(path)
        assert {r["claim_id"] for r in rows} == set(plan[f"{role}_ids"])
        papers[role] = {r["doc_id"] for r in rows}
        texts[role] = {normalize_claim_text(gold[r["claim_id"]].claim) for r in rows}
        assert not texts[role] & forbidden
        truncated_nei = 0
        for row in rows:
            g, doc = gold[row["claim_id"]], corpus[row["doc_id"]]
            assert g.split == "train" and row["claim"] == g.claim
            ids = row.get("context_sentence_ids", list(range(len(doc.sentences))))
            assert ids == sorted(set(ids)) and all(0 <= sid < len(doc.sentences) for sid in ids)
            assert row["premise"] == format_premise(doc.title, " ".join(doc.sentences[sid] for sid in ids), True)
            if row["view"] == "hard_negative":
                assert role == "fit" and doc.doc_id == negative_by_claim[g.claim_id]
                assert doc.doc_id not in g.reference_doc_ids and row["label"] == "NEI"
            elif row["view"] == "weak_nonrationale":
                assert role == "fit" and doc.doc_id in g.reference_doc_ids and row["label"] == "NEI"
                assert not set(ids) & set(g.evidence_by_doc.get(doc.doc_id, []))
                assert row["class_weight"] == .1
            else:
                assert doc.doc_id in g.reference_doc_ids and row["label"] == g.label_by_doc.get(doc.doc_id, "NEI")
            length = len(tokenizer.encode(row["premise"], row["claim"]).ids)
            if row["label"] != "NEI":
                assert length <= 512 and set(row["rationale_sentence_ids"]) <= set(ids)
                assert set(row["rationale_sentence_ids"]) <= set(g.evidence_by_doc[doc.doc_id])
            else:
                truncated_nei += length > 512
        counts[role] = {"claims": len({r["claim_id"] for r in rows}), "pairs": len(rows),
                        "labels": dict(Counter(r["label"] for r in rows)), "views": dict(Counter(r["view"] for r in rows)),
                        "weak_NEI_premises_over512_before_truncation": truncated_nei}
    audit_papers = {did for cid in plan["audit_ids"] for did in gold[cid].reference_doc_ids}
    audit_texts = {normalize_claim_text(gold[cid].claim) for cid in plan["audit_ids"]}
    assert not papers["fit"] & papers["stop"] and not (papers["fit"] | papers["stop"]) & audit_papers
    assert not texts["fit"] & texts["stop"] and not (texts["fit"] | texts["stop"]) & audit_texts
    manifest = read_json(ROOT / "artifacts/models/lora_science_select_v2/training_manifest.json")
    used = set(plan["fit_ids"]) | set(plan["stop_ids"])
    assert set(manifest["fitted_claim_ids"]) == used
    assert set(manifest["fitted_claim_text_sha256"]) == {sha256_text(normalize_claim_text(gold[c].claim)) for c in used}
    report = {"counts": counts, "source_and_label_mismatches": 0, "positive_training_truncations": 0,
              "fit_stop_audit_source_paper_overlap": 0, "fit_stop_audit_text_overlap": 0,
              "validation_fitting_text_overlap": 0, "complete_fit_and_stop_exposure_guard": True,
              "frozen_code_hashes_match": True, "plan_sha256": sha256_file(plan_path)}
    old_path = ROOT / "artifacts/improvement/lora_science_selection_data_v1/fit_retrieval_hits.jsonl"
    if old_path.exists():
        old, current = read_jsonl(old_path), {r["claim_id"]: r for r in fit_mining}
        shared = [r for r in old if r["claim_id"] in current]
        report["available_CPU_GPU_mining_prefix"] = {
            "queries": len(shared), "not_random_or_a_complete_parity_benchmark": True,
            "same_negative_source": sum(r["negative_doc_id"] == current[r["claim_id"]]["negative_doc_id"] for r in shared),
            "same_top20_source_order": sum([h["doc_id"] for h in r["hits"]] == [h["doc_id"] for h in current[r["claim_id"]]["hits"]] for r in shared)}
    write_json(folder / "independent_audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
