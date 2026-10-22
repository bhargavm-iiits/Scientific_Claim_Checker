"""Evaluate a separately labeled general-science benchmark with frozen identities.

Developer-written smoke examples never establish the requested 95% performance.
Gold labels and reference metadata are used only after ClaimChecker.check(text).
"""
import argparse
import math
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from claim_checker.config import load_config
from claim_checker.data.store import (
    append_jsonl,
    load_corpus,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from claim_checker.evaluation.metrics import operational_metrics, quote_validity, verdict_metrics
from claim_checker.evaluation.runner import prediction_record, run_directory
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import ClaimGold, Verdict, utc_now

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    claim: str
    label: Verdict


def accuracy_interval(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", choices=("smoke", "holdout"), default="smoke")
    parser.add_argument("--benchmark-metadata", type=Path, help="Required for holdout: source, label_review and limitations")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/general_science.yaml", overrides={"verifier": {"device": args.device}})
    claims = [BenchmarkClaim.model_validate(r) for r in read_jsonl(args.claims)]
    assert claims and len(claims) == len({r.claim_id for r in claims}), "Claims must have unique IDs"
    assert set(r.label for r in claims) == {"SUPPORT", "CONTRADICT", "NEI"}, "Report all three classes"
    metadata = read_json(args.benchmark_metadata) if args.benchmark_metadata else {}
    if args.purpose == "holdout":
        assert all(metadata.get(k) for k in ("source", "label_review", "limitations")), "Holdout requires documented label provenance"
    corpus = load_corpus(cfg.path("processed_dir"))
    folder = run_directory(cfg, args.run_id)
    identity = {
        "purpose": args.purpose, "benchmark_sha256": sha256_file(args.claims), "population": [r.claim_id for r in claims],
        "benchmark_metadata": metadata, "configuration": cfg.model_dump(mode="json"),
        "corpus_sha256": sha256_file(cfg.path("processed_dir") / "corpus.jsonl"),
        "source_manifest_sha256": sha256_file(cfg.path("processed_dir") / "source_manifest.json"),
        "index_manifest_sha256": sha256_file(cfg.path("index_dir") / "manifest.json"),
        "weights_sha256": sha256_file(ROOT / cfg.verifier.model_dir / "model.safetensors"),
        "reader_artifacts_sha256": {p.name: sha256_file(p) for p in sorted((ROOT / cfg.verifier.model_dir).iterdir())
                                    if p.is_file() and p.name != "model.safetensors"},
        "code_sha256": {str(p.relative_to(ROOT / "src/claim_checker")): sha256_file(p)
                        for p in sorted((ROOT / "src/claim_checker").rglob("*.py"))},
        "library_versions": {name: version(name) for name in ("torch", "transformers", "sentence-transformers", "numpy", "langgraph")},
        "evaluation_script_sha256": sha256_file(Path(__file__)),
    }
    frozen = folder / "identity.json"
    if frozen.exists():
        assert read_json(frozen) == identity, "Cannot resume with changed benchmark, model, data or code"
    else:
        folder.mkdir(parents=True, exist_ok=True)
        write_json(frozen, identity)
    registry = cfg.path("runs_dir") / "benchmark_usage.jsonl"
    uses = read_jsonl(registry) if registry.exists() else []
    prior = [r for r in uses if r["benchmark_sha256"] == identity["benchmark_sha256"] and r["run_id"] != args.run_id]
    if not any(r["run_id"] == args.run_id for r in uses):
        append_jsonl(registry, {"run_id": args.run_id, "purpose": args.purpose,
                              "benchmark_sha256": identity["benchmark_sha256"], "identity_sha256": sha256_file(frozen), "created_at": utc_now()})
    path = folder / "predictions.jsonl"
    saved = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in saved}
    assert len(done) == len(saved) and done <= set(identity["population"])
    checker = ClaimChecker(cfg, corpus=corpus)
    for index, claim in enumerate(claims, 1):
        if claim.claim_id in done:
            continue
        result = checker.check(claim.claim, claim_id=claim.claim_id)
        gold = ClaimGold(claim_id=claim.claim_id, claim=claim.claim, split="validation", cited_doc_ids=[],
                         reference_doc_ids=[], evidence_by_doc={}, label_by_doc={}, label=claim.label, conflict=False)
        row = prediction_record("general_science", gold, result)
        row["source_provenance"] = {d.doc_id: d.source for d in result.documents}
        append_jsonl(path, row)
        print(f"{index}/{len(claims)} general-science claims; status={result.status}", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == len(claims)
    metrics = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
    summary = {
        "purpose": args.purpose, "verdict": metrics, "operational": operational_metrics(rows), "quotes": quote_validity(rows, corpus),
        "accuracy_wilson_95pct_interval": accuracy_interval(metrics["correct"], len(rows)),
        "previous_runs_on_this_benchmark": len(prior), "independent_label_provenance_established_by_this_script": False,
        "thresholds_met_on_this_supplied_benchmark": metrics["accuracy"] > 0.95 and metrics["macro_f1"] > 0.95,
        "eligible_to_establish_general_95percent_accuracy": False,
        "limitations": ("Development smoke examples; not a representative blind benchmark." if args.purpose == "smoke" else
                        "Label independence, contamination and representativeness require review; recorded provenance is not certification."),
        "identity_sha256": sha256_file(frozen),
    }
    write_json(folder / "metrics.json", summary)
    report = (f"# General science {args.purpose} evaluation\n\nClaims: {len(rows)}. Accuracy: {metrics['accuracy']:.4%}; "
              f"Macro-F1: {metrics['macro_f1']:.6f}. Operational errors: {summary['operational']['errors']}.\n\n"
              f"{summary['limitations']} This does not establish 95% general-science accuracy.\n")
    (folder / "report.md").write_text(report, encoding="utf-8")
    print(f"accuracy={metrics['accuracy']:.6f}; Macro-F1={metrics['macro_f1']:.6f}; purpose={args.purpose}", flush=True)


if __name__ == "__main__":
    main()
