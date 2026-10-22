"""Score complete development-only source passages, then freeze a bounded grid."""
import argparse
import itertools
import time
from pathlib import Path

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
from claim_checker.models.passage_windows import best_passage, passage_windows
from claim_checker.models.verifier import NLIVerifier
from claim_checker.pipeline.aggregate import aggregate_verdicts

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="fresh experiment folder name")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a folder basename")
    plan = ROOT / "artifacts/improvement/short_passage_selection_plan.json"
    frozen = read_json(ROOT / "artifacts/improvement/frozen_selection_nli_title_family_v1.json")
    name, epoch = frozen["training_name"], frozen["chosen"]["epoch"]
    model = ROOT / f"artifacts/models/{name}/epoch{epoch}"
    training = read_json(model.parent / "training_manifest.json")
    assert not training["settings"]["final_fit"]
    cfg = load_config()
    corpus, gold = load_corpus(cfg.path("processed_dir")), load_gold(cfg.path("processed_dir"), "train")
    ids = load_split(cfg, "dev_tune")["claim_ids"]
    retrieval_path = ROOT / "artifacts/runs/dev_v1/retrieval.jsonl"
    retrieval = {row["claim_id"]: row["hits"] for row in read_jsonl(retrieval_path)}
    folder = ROOT / "artifacts/improvement" / args.name
    folder.mkdir(exist_ok=True)
    identity = {"population": ids, "training_name": name, "epoch": epoch, "model_sha256": sha256_file(model / "model.safetensors"),
                "plan_sha256": sha256_file(plan), "retrieval_sha256": sha256_file(retrieval_path),
                "script_sha256": sha256_file(Path(__file__)),
                "passage_code_sha256": sha256_file(ROOT / "src/claim_checker/models/passage_windows.py")}
    manifest = folder / "manifest.json"
    if manifest.exists():
        assert read_json(manifest) == identity, "Use a new experiment if its identity changed"
    else:
        write_json(manifest, identity)
    path = folder / "predictions.jsonl"
    done = {row["claim_id"]: row for row in read_jsonl(path)} if path.exists() else {}
    verifier = NLIVerifier(model, device="cuda", include_title=True, window_overlap=2)
    for number, cid in enumerate(ids, 1):
        if cid in done:
            continue
        claim = gold[cid].claim
        started = time.perf_counter()
        contexts, premises = [], []
        for hit in retrieval[cid][:5]:
            doc = corpus[hit["doc_id"]]
            for width in (1, 2):
                for window in passage_windows(verifier.tokenizer, claim, doc.title, doc.sentences, width=width):
                    premise = verifier.format_premise(doc.title, " ".join(doc.sentences[i] for i in window))
                    contexts.append({"doc_id": doc.doc_id, "rank": hit["rank"], "retrieval_score": hit["score"],
                                     "width": width, "sentence_ids": window})
                    premises.append(premise)
        scores = verifier.score(premises, [claim] * len(premises))
        for context, score in zip(contexts, scores, strict=True):
            context["scores"] = score
        row = {"claim_id": cid, "gold": gold[cid].label, "contexts": contexts}
        append_jsonl(path, row)
        done[cid] = row
        print(f"Scored {number}/{len(ids)} development claims; {len(contexts)} short contexts; {time.perf_counter()-started:.1f}s", flush=True)
    assert set(done) == set(ids)
    candidates = []
    for widths, k, gate, support, contradict, margin in itertools.product(((1,), (2,), (1, 2)), (1, 3, 5),
            (None, -2, 0), (0.5, 0.7, 0.9, 0.98), (0.5, 0.7, 0.9, 0.98), (0, 0.1, 0.2)):
        predictions = []
        for cid in ids:
            by_doc = {}
            for context in done[cid]["contexts"]:
                if context["rank"] <= k and context["width"] in widths and (gate is None or context["retrieval_score"] >= gate):
                    by_doc.setdefault(context["doc_id"], []).append(context["scores"])
            labels = [best_passage(scores, support_threshold=support, contradict_threshold=contradict, margin=margin)[0]
                      for scores in by_doc.values()]
            predictions.append(aggregate_verdicts(labels)[0])
        metrics = verdict_metrics([done[cid]["gold"] for cid in ids], predictions)
        candidates.append({"widths": widths, "k": k, "retrieval_gate": gate, "support_threshold": support,
                           "contradict_threshold": contradict, "margin": margin,
                           "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    candidates.sort(key=lambda c: (-c["macro_f1"], -c["accuracy"], len(c["widths"]), c["k"]))
    chosen = candidates[0]
    write_json(folder / "selection.json", {"chosen": chosen, "candidates": candidates,
               "prediction_sha256": sha256_file(path), "validation_used_for_selection": False,
               "integration_criterion_passed": chosen["accuracy"] > 0.85 and chosen["macro_f1"] > 0.85})
    write_json(folder / "status.json", {"complete": True, "population": len(ids)})
    print("Best complete development recipe:", chosen, flush=True)


if __name__ == "__main__":
    main()
