"""Fixed-population evaluation of a local Qwen judge; no gold inputs to inference."""
from __future__ import annotations

import argparse
import time
from importlib.metadata import version
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import load_split, normalize_claim_text, stratified_sample
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
from claim_checker.models.lora_judge import LoraJudge
from claim_checker.pipeline.aggregate import aggregate_verdicts

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)
    p.add_argument("--name", required=True)
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--split", choices=("dev_tune", "final", "accuracy_holdout"), default="dev_tune")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    cfg = load_config()
    corpus = load_corpus(cfg.path("processed_dir"))
    source = "train" if args.split == "dev_tune" else "validation"
    gold = load_gold(cfg.path("processed_dir"), source)
    population, _ = stratified_sample([gold[c] for c in load_split(cfg, args.split)["claim_ids"]], args.size, 42)
    retrieval_path = ROOT / ("artifacts/runs/dev_v1/retrieval.jsonl" if source == "train" else
                             ("artifacts/runs/verifier_holdout_v2/retrieval.jsonl" if args.split == "accuracy_holdout"
                              else "artifacts/runs/verifier_recheck_v2/retrieval.jsonl"))
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(retrieval_path)}
    adapter = ROOT / args.adapter if args.adapter else None
    if adapter:
        manifest = read_json(adapter.parent / "training_manifest.json")
        if manifest["contract_sha256"] != sha256_file(ROOT / "src/claim_checker/models/lora_judge.py"):
            raise ValueError("The training/inference contract changed; evaluate a matching implementation")
        if args.split == "dev_tune" and manifest["settings"]["final_fit"]:
            raise ValueError("Final-refit models cannot be evaluated as held-out on internal development")
        training_path = adapter.parent / "training_rows.jsonl"
        if sha256_file(training_path) != manifest["training_sha256"]:
            raise ValueError("Training rows no longer match their manifest")
        evaluation_texts = {normalize_claim_text(g.claim) for g in population}
        if any(normalize_claim_text(r["claim"]) in evaluation_texts for r in read_jsonl(training_path)):
            raise ValueError("Training/evaluation text overlap")
    base = ROOT / "artifacts/models/qwen25_15b_base"
    base_sha = sha256_file(base / "model.safetensors")
    if adapter and manifest.get("base_weights_sha256") != base_sha:
        raise ValueError("Inference base differs from the training base")
    identity = {"settings": vars(args), "base_revision": (base / "source_revision.txt").read_text(),
                "base_sha256": base_sha,
                "base_files_sha256": {name: sha256_file(base / name) for name in (
                    "config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt")},
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors") if adapter else None,
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json") if adapter else None,
                "training_manifest_sha256": sha256_file(adapter.parent / "training_manifest.json") if adapter else None,
                "libraries": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
                "contract_sha256": sha256_file(ROOT / "src/claim_checker/models/lora_judge.py"),
                "script_sha256": sha256_file(Path(__file__)),
                "aggregation_sha256": sha256_file(ROOT / "src/claim_checker/pipeline/aggregate.py"),
                "retrieval_sha256": sha256_file(retrieval_path),
                "gold_sha256": sha256_file(cfg.path("processed_dir") / f"claim_gold_{source}.jsonl"),
                "claim_ids": [g.claim_id for g in population]}
    folder = ROOT / "artifacts/improvement" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists() and read_json(folder / "manifest.json") != identity:
        raise ValueError("Experiment identity changed; choose a new name")
    write_json(folder / "manifest.json", identity)
    write_json(folder / "status.json", {"complete": False, "population": len(population)})
    path = folder / "predictions.jsonl"
    done = {r["claim_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    judge = LoraJudge(base, adapter, device=args.device, max_length=args.max_length)
    for i, gold in enumerate(population, 1):
        if gold.claim_id in done:
            continue
        started, docs, error, conflict = time.perf_counter(), [], None, False
        try:
            for hit in retrieval[gold.claim_id][:3]:
                doc = corpus[hit["doc_id"]]
                windows = judge.judge(gold.claim, doc)
                label, window_conflict = aggregate_verdicts(w["answer"]["verdict"] for w in windows)
                conflict = conflict or window_conflict
                docs.append({"doc_id": doc.doc_id, "rank": hit["rank"], "verdict": label,
                             "window_conflict": window_conflict, "windows": windows})
            verdict, doc_conflict = aggregate_verdicts(d["verdict"] for d in docs)
            conflict = conflict or doc_conflict
            if conflict:
                verdict = "NEI"
        except (ValueError, RuntimeError) as exc:
            verdict, error = "ERROR", f"{type(exc).__name__}: {exc}"
        row = {"claim_id": gold.claim_id, "claim": gold.claim, "gold": gold.label, "verdict": verdict,
               "conflict": conflict, "documents": docs, "error": error, "seconds": time.perf_counter() - started}
        append_jsonl(path, row)
        done[gold.claim_id] = row
        metrics = verdict_metrics([r["gold"] for r in done.values()], [r["verdict"] for r in done.values()])
        print(f"{i}/{len(population)}: accuracy={metrics['accuracy']:.3f}, MF1={metrics['macro_f1']:.3f}, {row['seconds']:.1f}s, error={error}", flush=True)
    metrics = verdict_metrics([done[g.claim_id]["gold"] for g in population],
                              [done[g.claim_id]["verdict"] for g in population])
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "status.json", {"complete": True, "population": len(population),
                                        "errors": sum(done[g.claim_id]["error"] is not None for g in population)})
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
