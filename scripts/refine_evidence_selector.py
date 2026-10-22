"""Evaluate exact cached prefix selections on development, never validation."""
from pathlib import Path

import numpy as np

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.models.evidence_classifier import select_sentence_ids

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "artifacts/improvement/trained_evidence_deberta_dev120_v2"
    previous = read_json(folder / "reader_calibration_refinement.json")
    rows = {r["claim_id"]: r for r in read_jsonl(folder / "predictions.jsonl")}
    ids = load_split(load_config(), "dev_tune")["claim_ids"]
    assert set(rows) == set(ids)
    neutral = {(r["claim_id"], c["doc_id"], c["sentence_ids"][0]): 1 - c["scores"]["NEI"]
               for r in read_jsonl(ROOT / "artifacts/improvement/nli_short_passages_dev120/predictions.jsonl")
               for c in r["contexts"] if c["width"] == 1}
    keys = ("alpha", "max_sentences", "k", "retrieval_gate")
    structural, seen = [], set()
    for candidate in previous["candidates"]:
        key = tuple(candidate[k] for k in keys)
        if key not in seen:
            structural.append(dict(zip(keys, key, strict=True)))
            seen.add(key)
        if len(structural) == 10:
            break
    cutoffs = read_json(ROOT / "artifacts/improvement/evidence_reader_refinement_plan.json")["reader_cutoffs"]
    selectors = [round(0.3 + i * 0.025, 3) for i in range(28)] + [0.99]
    names = ["SUPPORT", "CONTRADICT", "NEI", "ERROR"]
    gold = np.array([names.index(rows[cid]["gold"]) for cid in ids])
    errors = np.array([bool(rows[cid]["error"]) for cid in ids])
    best = None
    leaders, tested = [], 0
    for cfg in structural:
        lookup = {}
        for cid in ids:
            documents = []
            for doc in rows[cid]["documents"]:
                if doc["rank"] > cfg["k"] or (cfg["retrieval_gate"] is not None and doc["retrieval_score"] < cfg["retrieval_gate"]):
                    continue
                fused = [p ** cfg["alpha"] * neutral[(cid, doc["doc_id"], i)] ** (1 - cfg["alpha"])
                         for i, p in enumerate(doc["selector_probabilities"])]
                contexts = {tuple(c["ids"]): c["window_scores"] for c in doc["contexts"] if c["alpha"] == cfg["alpha"]}
                documents.append((fused, contexts))
            lookup[cid] = documents
        for threshold in selectors:
            maxima = np.zeros((len(ids), 2))
            for number, cid in enumerate(ids):
                for probabilities, contexts in lookup[cid]:
                    selected = tuple(select_sentence_ids(probabilities, threshold=threshold, maximum=cfg["max_sentences"]))
                    scores = contexts[selected] if selected else []  # Exact tuple or fail; never impute model output.
                    for score in scores:
                        label = max(score, key=score.get)
                        if label in ("SUPPORT", "CONTRADICT"):
                            index = names.index(label)
                            maxima[number, index] = max(maxima[number, index], score[label])
            has_s = maxima[None, None, :, 0] >= np.array(cutoffs)[:, None, None]
            has_c = maxima[None, None, :, 1] >= np.array(cutoffs)[None, :, None]
            predictions = np.where(has_s & ~has_c, 0, np.where(has_c & ~has_s, 1, 2))
            predictions[:, :, errors] = 3
            accuracy = (predictions == gold).mean(axis=2)
            macro = np.zeros_like(accuracy)
            for label in range(3):
                positive = predictions == label
                tp = (positive & (gold == label)).sum(axis=2)
                macro += 2 * tp / (positive.sum(axis=2) + (gold == label).sum()) / 3
            for i, support in enumerate(cutoffs):
                for j, contradict in enumerate(cutoffs):
                    candidate = {**cfg, "selector_threshold": threshold, "support_threshold": support,
                                 "contradict_threshold": contradict, "accuracy": float(accuracy[i, j]), "macro_f1": float(macro[i, j])}
                    tested += 1
                    if best is None or (candidate["macro_f1"], candidate["accuracy"]) > (best["macro_f1"], best["accuracy"]):
                        expected = verdict_metrics([rows[cid]["gold"] for cid in ids], [names[p] for p in predictions[i, j]])
                        assert np.isclose(expected["accuracy"], candidate["accuracy"], atol=1e-12)
                        assert np.isclose(expected["macro_f1"], candidate["macro_f1"], atol=1e-12)
                        best = {**candidate, "accuracy": expected["accuracy"], "macro_f1": expected["macro_f1"]}
                        leaders.append(best)
    report = {"chosen": best, "successive_best": leaders, "tested_candidates": tested, "population": "dev_tune120",
              "validation_used": False, "source_predictions_sha256": sha256_file(folder / "predictions.jsonl"),
              "plan_sha256": sha256_file(ROOT / "artifacts/improvement/evidence_selector_refinement_plan.json"),
              "integration_criterion_passed": best["accuracy"] > 0.85 and best["macro_f1"] > 0.85 + 1e-12}
    write_json(folder / "selector_calibration_refinement.json", report)
    print(best, flush=True)


if __name__ == "__main__":
    main()
