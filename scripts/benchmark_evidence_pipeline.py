"""Development-only rationale selection and NLI judgement on genuine retrieval."""
import argparse
import itertools
import time
from pathlib import Path

from claim_checker.agents.judge_nli import choose_label
from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import (
    append_jsonl,
    load_corpus,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.models.evidence_classifier import EvidenceClassifier, select_sentence_ids
from claim_checker.models.verifier import NLIVerifier
from claim_checker.pipeline.aggregate import aggregate_verdicts

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="trained_evidence_dev120_v1")
    parser.add_argument("--selector-run", default="evidence_scifact_train_v1")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a folder basename")
    selector_root = ROOT / "artifacts/models" / args.selector_run
    assert read_json(selector_root / "status.json")["complete"]
    selector_epoch = read_json(selector_root / "selection.json")["chosen"]["epoch"]
    selector_dir = selector_root / f"epoch{selector_epoch}"
    frozen = read_json(ROOT / "artifacts/improvement/frozen_selection_nli_title_family_v1.json")
    reader_dir = ROOT / f"artifacts/models/{frozen['training_name']}/epoch{frozen['chosen']['epoch']}"
    assert not read_json(selector_root / "training_manifest.json")["settings"]["final_fit"]
    assert not read_json(reader_dir.parent / "training_manifest.json")["settings"]["final_fit"]
    cfg = load_config()
    corpus, gold = load_corpus(cfg.path("processed_dir")), load_gold(cfg.path("processed_dir"), "train")
    population = load_split(cfg, "dev_tune")["claim_ids"]
    retrieval_path = ROOT / "artifacts/runs/dev_v1/retrieval.jsonl"
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(retrieval_path)}
    short_dir = ROOT / "artifacts/improvement/nli_short_passages_dev120"
    assert read_json(short_dir / "status.json")["complete"]
    assert read_json(short_dir / "manifest.json")["model_sha256"] == sha256_file(reader_dir / "model.safetensors")
    neutrality = {(r["claim_id"], c["doc_id"], c["sentence_ids"][0]): 1 - c["scores"]["NEI"]
                  for r in read_jsonl(short_dir / "predictions.jsonl") for c in r["contexts"] if c["width"] == 1}
    folder = ROOT / "artifacts/improvement" / args.name
    folder.mkdir(exist_ok=True)
    identity = {"population": population, "selector_epoch": selector_epoch,
                "selector_sha256": sha256_file(selector_dir / "model.safetensors"),
                "reader_sha256": sha256_file(reader_dir / "model.safetensors"),
                "selector_training_sha256": sha256_file(selector_root / "training_manifest.json"),
                "script_sha256": sha256_file(Path(__file__)),
                "selector_code_sha256": sha256_file(ROOT / "src/claim_checker/models/evidence_classifier.py"),
                "retrieval_sha256": sha256_file(retrieval_path),
                "single_sentence_scores_sha256": sha256_file(short_dir / "predictions.jsonl"),
                "plan_sha256": sha256_file(ROOT / "artifacts/improvement/evidence_fusion_development_plan.json")}
    manifest = folder / "manifest.json"
    if manifest.exists():
        assert read_json(manifest) == identity, "Use a new experiment name when its identity changes"
    else:
        write_json(manifest, identity)
    path = folder / "predictions.jsonl"
    done = {r["claim_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    selector = EvidenceClassifier(selector_dir, device="cuda")
    reader = NLIVerifier(reader_dir, device="cuda", include_title=True, window_overlap=2)
    for number, cid in enumerate(population, 1):
        if cid in done:
            continue
        started = time.perf_counter()
        claim = gold[cid].claim
        documents, error = [], None
        try:
            for hit in retrieval[cid][:5]:
                doc = corpus[hit["doc_id"]]
                probabilities = selector.score(claim, doc)
                contexts, unique_inputs = [], {}
                for alpha, threshold, maximum in itertools.product((0, 0.5, 1), (0.3, 0.5, 0.7, 0.9), (1, 2, 3)):
                    fused = [p ** alpha * neutrality[(cid, doc.doc_id, i)] ** (1 - alpha)
                             for i, p in enumerate(probabilities)]
                    ids = select_sentence_ids(fused, threshold=threshold, maximum=maximum)
                    contexts.append({"alpha": alpha, "threshold": threshold, "maximum": maximum, "ids": ids})
                    key = tuple(ids)
                    if key not in unique_inputs:
                        sentences = [doc.sentences[i] for i in ids]
                        windows = reader.windows(claim, sentences, title=doc.title)
                        unique_inputs[key] = [reader.format_premise(doc.title, " ".join(sentences[i] for i in window)) for window in windows]
                flattened, sections = [], {}
                for key, premises in unique_inputs.items():
                    sections[key] = (len(flattened), len(flattened) + len(premises))
                    flattened.extend(premises)
                scores = reader.score(flattened, [claim] * len(flattened))
                for context in contexts:
                    start, end = sections[tuple(context["ids"])]
                    context["window_scores"] = scores[start:end]
                documents.append({"doc_id": doc.doc_id, "rank": hit["rank"], "retrieval_score": hit["score"],
                                  "selector_probabilities": probabilities, "contexts": contexts})
        except ValueError as exc:
            error = str(exc)
        row = {"claim_id": cid, "gold": gold[cid].label, "documents": documents, "error": error}
        append_jsonl(path, row)
        done[cid] = row
        print(f"Scored {number}/{len(population)} retrieved development claims; {time.perf_counter()-started:.1f}s; error={error}", flush=True)
    assert set(done) == set(population)
    # Convert the small cached context choices once; the grid never calls a model.
    indexed = {}
    for cid, row in done.items():
        indexed[cid] = [(doc, {(c["alpha"], c["threshold"], c["maximum"]): c for c in doc["contexts"]}) for doc in row["documents"]]
    candidates = []
    for alpha, threshold, maximum, k, gate, support, contradict in itertools.product((0, 0.5, 1), (0.3, 0.5, 0.7, 0.9),
            (1, 2, 3), (1, 3, 5), (None, -2, 0), (0.5, 0.7, 0.9, 0.98), (0.5, 0.7, 0.9, 0.98)):
        predictions = []
        for cid in population:
            if done[cid]["error"]:
                predictions.append("ERROR")
                continue
            labels = []
            for doc, lookup in indexed[cid]:
                if doc["rank"] > k or (gate is not None and doc["retrieval_score"] < gate):
                    continue
                context = lookup[(alpha, threshold, maximum)]
                labels.extend(choose_label(s, {"SUPPORT": support, "CONTRADICT": contradict}) for s in context["window_scores"])
            predictions.append(aggregate_verdicts(labels)[0])
        metrics = verdict_metrics([done[cid]["gold"] for cid in population], predictions)
        candidates.append({"alpha": alpha, "selector_threshold": threshold, "max_sentences": maximum, "k": k,
                           "retrieval_gate": gate, "support_threshold": support, "contradict_threshold": contradict,
                           "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    candidates.sort(key=lambda c: (-c["macro_f1"], -c["accuracy"], c["max_sentences"], c["k"]))
    chosen = candidates[0]
    write_json(folder / "selection.json", {"chosen": chosen, "candidates": candidates, "selector_epoch": selector_epoch,
               "prediction_sha256": sha256_file(path), "validation_used_for_selection": False,
               "operational_errors": sum(bool(r["error"]) for r in done.values()),
               "integration_criterion_passed": chosen["accuracy"] > 0.85 and chosen["macro_f1"] > 0.85})
    write_json(folder / "status.json", {"complete": True, "population": len(population)})
    print("Frozen retrieved-development selection:", chosen, flush=True)


if __name__ == "__main__":
    main()
