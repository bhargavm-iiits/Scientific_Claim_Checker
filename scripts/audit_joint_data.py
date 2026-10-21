"""Independently check saved joint supervision without loading a neural model."""
from collections import Counter
from pathlib import Path

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
    plan_path = ROOT / "artifacts/improvement/joint_nli_protocol_v1.json"
    plan = read_json(plan_path)
    cfg = load_config()
    corpus, gold = load_corpus(cfg.path("processed_dir")), load_gold(cfg.path("processed_dir"), "train")
    forbidden = {normalize_claim_text(g.claim) for g in load_gold(cfg.path("processed_dir"), "validation").values()}
    assert all(sha256_file(ROOT / name) == digest for name, digest in plan["source_sha256"].items())
    manifest = read_json(ROOT / "artifacts/models/joint_nli_select_v1/training_manifest.json")
    expected_used = set(plan["fit_ids"]) | set(plan["stop_ids"])
    assert set(manifest["fitted_claim_ids"]) == expected_used
    assert set(manifest["fitted_claim_text_sha256"]) == {sha256_text(normalize_claim_text(gold[c].claim)) for c in expected_used}
    folder = ROOT / "artifacts/improvement/joint_selection_data_v1"
    papers, texts, counts, hashes = {}, {}, {}, {}
    for role in ("fit", "stop"):
        path = folder / f"{role}_tokens.jsonl"
        rows = read_jsonl(path)
        hashes[role] = sha256_file(path)
        assert {r["claim_id"] for r in rows} == set(plan[f"{role}_ids"])
        papers[role] = {r["doc_id"] for r in rows}
        texts[role] = {normalize_claim_text(gold[r["claim_id"]].claim) for r in rows}
        assert not texts[role] & forbidden
        for row in rows:
            g, doc = gold[row["claim_id"]], corpus[row["doc_id"]]
            source_ids = row["token_source_ids"]
            inputs = row["inputs"]
            assert len(source_ids) == len(inputs["input_ids"]) <= 512
            assert len(inputs["rationale_targets"]) == len(inputs["source_mask"]) == len(source_ids)
            assert row["premise"].startswith(format_premise(doc.title, "", True))
            assert all(sid == -1 or 0 <= sid < len(doc.sentences) for sid in source_ids)
            positives = set(g.evidence_by_doc.get(doc.doc_id, []))
            for sid, mask, target in zip(source_ids, inputs["source_mask"], inputs["rationale_targets"], strict=True):
                assert mask == int(sid >= 0)
                assert target == (-100 if sid < 0 else int(sid in positives))
            if row["view"] == "hard_negative":
                assert doc.doc_id not in g.reference_doc_ids and row["label"] == "NEI" and row["class_weight"] == .25
            else:
                assert doc.doc_id in g.reference_doc_ids
                assert row["label"] == g.label_by_doc.get(doc.doc_id, "NEI")
            if row["label"] != "NEI":
                assert 1 in inputs["rationale_targets"], "A decisive pair must retain positive source supervision"
        counts[role] = {"rows": len(rows), "claims": len({r["claim_id"] for r in rows}),
                        "pair_label_distribution": dict(Counter(r["label"] for r in rows)),
                        "positive_source_tokens": sum(r["inputs"]["rationale_targets"].count(1) for r in rows)}
    audit_papers = {d for cid in plan["audit_ids"] for d in gold[cid].reference_doc_ids}
    audit_texts = {normalize_claim_text(gold[cid].claim) for cid in plan["audit_ids"]}
    assert not papers["fit"] & papers["stop"] and not (papers["fit"] | papers["stop"]) & audit_papers
    assert not texts["fit"] & texts["stop"] and not (texts["fit"] | texts["stop"]) & audit_texts
    result = {"counts": counts, "source_target_mismatches": 0, "decisive_rows_without_positive_source_tokens": 0,
              "fit_stop_audit_source_paper_overlap": 0, "fit_stop_audit_claim_text_overlap": 0,
              "validation_text_overlap": 0, "fit_and_stop_exposure_guard_complete": True,
              "frozen_source_hashes_match": True, "plan_sha256": sha256_file(plan_path), "token_file_sha256": hashes}
    write_json(folder / "independent_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()
