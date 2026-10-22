"""Save annotation-free NLI scores on the fixed development retrieval results."""
from __future__ import annotations

import argparse
import time
from importlib.metadata import version
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import load_split, stratified_sample
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
from claim_checker.evaluation.training_provenance import assert_unfitted_population
from claim_checker.models.verifier import NLIVerifier
from claim_checker.pipeline.aggregate import aggregate_verdicts

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/models/nli_base")
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--window-overlap", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    training_manifest = (ROOT / args.model).parent / "training_manifest.json"
    if training_manifest.exists() and read_json(training_manifest).get("settings", {}).get("final_fit"):
        raise ValueError("This checkpoint was refitted on internal development data. Use validation evaluation, not dev_tune scoring.")
    cfg = load_config()
    corpus = load_corpus(cfg.path("processed_dir"))
    gold = load_gold(cfg.path("processed_dir"), "train")
    population, _ = stratified_sample(
        [gold[c] for c in load_split(cfg, "dev_tune")["claim_ids"]], args.size, 42,
    )
    if training_manifest.exists():
        assert_unfitted_population(read_json(training_manifest), population)
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    selected = {(r["claim_id"], r["doc_id"]): r["selected"]
                for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/pair_predictions.jsonl")}
    verifier = NLIVerifier(ROOT / args.model, device=args.device, include_title=args.include_title, window_overlap=args.window_overlap)
    folder = ROOT / "artifacts/improvement" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "predictions.jsonl"
    model_path = ROOT / args.model
    identity = {"settings": vars(args), "claim_ids": [g.claim_id for g in population],
                "libraries": {name: version(name) for name in ("torch", "transformers")},
                "model_sha256": sha256_file(model_path / "model.safetensors"),
                "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
                "model_config_sha256": sha256_file(model_path / "config.json"),
                "training_manifest_sha256": sha256_file(training_manifest) if training_manifest.exists() else None,
                "retrieval_sha256": sha256_file(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl"),
                "selection_sha256": sha256_file(ROOT / "artifacts/runs/dev_v1/pair_predictions.jsonl"),
                "script_sha256": sha256_file(Path(__file__)),
                "verifier_code_sha256": sha256_file(ROOT / "src/claim_checker/models/verifier.py")}
    identity["context_code_sha256"] = sha256_file(ROOT / "src/claim_checker/data/verifier_context.py")
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != identity:
        raise ValueError("Verifier experiment identity changed; use a new --name")
    if path.exists() and not manifest_path.exists():
        raise ValueError("Legacy predictions have no immutable identity; preserve them and use a new --name")
    write_json(manifest_path, identity)
    write_json(folder / "status.json", {"complete": False, "population": len(population)})
    done = {r["claim_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    for i, gold in enumerate(population, 1):
        if gold.claim_id in done:
            continue
        start = time.perf_counter()
        contexts = []
        for hit in retrieval[gold.claim_id][:args.top_k]:
            doc = corpus[hit["doc_id"]]
            for ids in verifier.windows(gold.claim, doc.sentences, title=doc.title):
                contexts.append({"doc_id": doc.doc_id, "rank": hit["rank"], "view": "abstract",
                                 "sentence_ids": ids, "text": verifier.format_premise(doc.title, " ".join(doc.sentences[n] for n in ids))})
            ids = selected.get((gold.claim_id, doc.doc_id), [])
            if ids:
                contexts.append({"doc_id": doc.doc_id, "rank": hit["rank"], "view": "selected",
                                 "sentence_ids": ids, "text": verifier.format_premise(doc.title, " ".join(doc.sentences[n] for n in ids))})
        scores = verifier.score([r["text"] for r in contexts], [gold.claim] * len(contexts))
        for context, score in zip(contexts, scores, strict=True):
            context["scores"] = score
            context.pop("text")
        verdict, _ = aggregate_verdicts(max(r["scores"], key=r["scores"].get)
                                       for r in contexts if r["view"] == "abstract")
        row = {"claim_id": gold.claim_id, "claim": gold.claim, "gold": gold.label,
               "verdict": verdict, "contexts": contexts, "seconds": time.perf_counter() - start}
        append_jsonl(path, row)
        done[gold.claim_id] = row
        metrics = verdict_metrics([r["gold"] for r in done.values()], [r["verdict"] for r in done.values()])
        print(f"{i}/{len(population)}: {metrics['accuracy']:.3f}/{metrics['macro_f1']:.3f}; {row['seconds']:.1f}s", flush=True)
    write_json(folder / "metrics.json", verdict_metrics(
        [done[g.claim_id]["gold"] for g in population], [done[g.claim_id]["verdict"] for g in population],
    ))
    write_json(folder / "status.json", {"complete": True, "population": len(population)})


if __name__ == "__main__":
    main()
