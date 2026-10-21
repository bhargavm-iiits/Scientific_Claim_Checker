"""Freeze and evaluate a training-free reader; validation is conditional on TRAIN audit."""
import argparse
import gc
import time
from pathlib import Path

from claim_checker.agents.evidence import build_quotes
from claim_checker.agents.explainer import explain
from claim_checker.agents.reasoned_reader import read_papers
from claim_checker.config import load_config
from claim_checker.data.splits import load_split, normalize_claim_text
from claim_checker.data.store import (
    append_jsonl,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from claim_checker.evaluation.metrics import (
    evidence_metrics,
    operational_metrics,
    quote_validity,
    retrieval_metrics,
    verdict_metrics,
)
from claim_checker.evaluation.runner import prediction_record
from claim_checker.models.ollama_client import OllamaClient
from claim_checker.pipeline.aggregate import aggregate
from claim_checker.pipeline.supervisor import validate_claim_text
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import ClaimResult, DocumentReport, ErrorInfo

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/improvement/reasoned_reader_protocol_v1.json"
INSTRUCTION = ROOT / "prompts/reasoned_reader_v1.txt"


def configuration():
    return load_config(overrides={
        "model": {"name": "qwen3.5:4b", "think": False, "temperature": 0, "seed": 42,
                  "context_tokens": 16384, "max_output_tokens": 2048,
                  "request_timeout_seconds": 240, "keep_alive": "10m"},
        "retrieval": {"pipeline_k": 3},
    })


def prepare(cfg, client):
    old = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    grouped = read_json(old)
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    ids = grouped["folds"][0]["audit"]
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    assert len(ids) == len(set(ids)) == 163
    assert all(train[c].split == "train" and normalize_claim_text(train[c].claim) not in forbidden for c in ids)
    plan = {
        "objective": "Original300SciFactclaims: accuracy>0.95 AND three-class Macro-F1>0.95",
        "architecture": "Training-free native reasoning; joint context, independently structured paper judgements; deterministic aggregation",
        "model": client.model_info(), "server_version": client.server_version(),
        "model_config": cfg.model.model_dump(mode="json"),
        "retrieval_config": cfg.retrieval.model_dump(mode="json"),
        "k": 3, "retrieval_gate": -2, "audit_ids": ids,
        "audit_population": "Reused163originalTRAINclaims; not blind, not validation300",
        "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
        "validation_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
        "source_group_protocol_sha256": sha256_file(old),
        "prompt_sha256": sha256_file(INSTRUCTION),
        "inference_code_sha256": sha256_file(ROOT / "src/claim_checker/agents/reasoned_reader.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "fine_tuning": False, "demonstrations": False, "threshold_searches": 0,
        "native_thinking": True, "manual_chat_template_or_budget_forcing": False,
        "validation_gate": {"minimum_audit_accuracy": 0.80, "minimum_audit_macro_f1": 0.80, "maximum_errors": 0},
        "validation_policy": "Only if audit gate passes: one unchanged300confirmation; do not tune on its results",
        "limitations": "No project fine-tuning is not proof of no generic pretraining contamination. Audit and validation are reused development evidence, not new blind populations.",
        "pdf_sha256": sha256_file(Path(r"C:\Users\Bhargav\Downloads\P08_Scientific_Claim_Checker.pdf")),
    }
    if PLAN.exists():
        assert read_json(PLAN) == plan, "Frozen recipe changed; preserve this run and use a new version"
    else:
        write_json(PLAN, plan)
    return plan


class ReasonedChecker:
    """Experiment-only adapter over the same real retrieval, sources and result schema."""
    def __init__(self, cfg, client):
        self.cfg, self.client = cfg, client
        self.resources = ClaimChecker(cfg)
        self.instruction = INSTRUCTION.read_text(encoding="utf-8")

    @property
    def corpus(self):
        return self.resources.corpus

    def check(self, claim, claim_id=None):
        start = time.perf_counter()
        before = dict(self.client.stats)
        hits, metadata = [], {"strategy": "pipeline", "reader": "reasoned_reader_v1"}
        try:
            text = validate_claim_text(claim, self.cfg.runtime.max_claim_chars, self.cfg.runtime.max_claim_tokens,
                                       self.resources.count_claim_tokens)
            retrieved = self.resources._retrieve(text, "hybrid", True, 10)
            hits = retrieved.hits
            metadata["retrieval_hits"] = [h.model_dump(mode="json") for h in hits]
            selected = [h for h in hits[:3] if h.score >= -2]
            docs = [self.corpus[h.doc_id] for h in selected]
            judgements, generation = read_papers(self.client, text, docs, self.instruction)
            verdict, conflict = aggregate(judgements)
            reports = [DocumentReport(doc_id=h.doc_id, title=d.title, rank=h.rank,
                                      retrieval_score=h.score, retrieval_method="hybrid",
                                      judgement=j, sentences=d.sentences, source=d.source)
                       for h, d, j in zip(selected, docs, judgements, strict=True)]
            quotes = [q for d, j in zip(docs, judgements, strict=True)
                      for q in build_quotes(d, j.cited_sentence_ids, j.verdict)]
            metadata.update(generation=generation, llm_calls=self.client.stats["calls"] - before["calls"],
                            llm_cache_hits=self.client.stats["cache_hits"] - before["cache_hits"])
            return ClaimResult(status="ok", claim=text, claim_id=claim_id, verdict=verdict, conflict=conflict,
                               documents=reports, evidence=quotes,
                               explanation=explain(verdict, conflict, judgements, {d.doc_id: d.title for d in docs}),
                               timings={"total": time.perf_counter() - start}, metadata=metadata)
        except Exception as exc:
            return ClaimResult(status="error", claim=claim, claim_id=claim_id,
                               error=ErrorInfo(error_type=type(exc).__name__, message=str(exc), stage="reasoned_reader"),
                               timings={"total": time.perf_counter() - start}, metadata=metadata)


def evaluate(checker, claims, name, plan):
    from claim_checker.retrieval.index import check_index

    folder = ROOT / "artifacts/improvement" / name
    folder.mkdir(exist_ok=True)
    identity = {"protocol_sha256": sha256_file(PLAN), "index": check_index(checker.cfg)["fingerprint"],
                "claim_ids": [g.claim_id for g in claims], "model_digest": checker.client.model_digest(),
                "core_code_sha256": {str(p.relative_to(ROOT / "src/claim_checker")): sha256_file(p)
                                     for p in sorted((ROOT / "src/claim_checker").rglob("*.py"))}}
    manifest = folder / "identity.json"
    if manifest.exists():
        assert read_json(manifest) == identity, "Evaluation identity changed"
    else:
        write_json(manifest, identity)
    path = folder / "predictions.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in existing}
    assert len(done) == len(existing) and done <= set(identity["claim_ids"])
    for index, gold in enumerate(claims, 1):
        if gold.claim_id in done:
            continue
        result = checker.check(gold.claim, gold.claim_id)
        row = prediction_record("reasoned_reader_v1", gold, result)
        row["generation"] = result.metadata.get("generation")
        append_jsonl(path, row)
        if index % 5 == 0 or result.status != "ok":
            print(f"{name}: {index}/{len(claims)} complete; latest status={result.status}; {result.timings['total']:.1f}s", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == len(claims) and {r["claim_id"] for r in rows} == set(identity["claim_ids"])
    verdict = verdict_metrics([r["gold_label"] for r in rows],
                              [r["verdict"] if r["status"] == "ok" else None for r in rows])
    report = {"verdict": verdict, "operational": operational_metrics(rows), "quotes": quote_validity(rows, checker.corpus),
              "retrieval": retrieval_metrics(rows), "citations": evidence_metrics(rows, "cited"),
              "population": plan["audit_population"] if name.endswith("audit_v1") else "Original reusedvalidation300, notblind",
              "generation": {"native_thinking_requested": True,
                             "thinking_fallback_claims": sum(bool((r.get("generation") or {}).get("thinking_fallback")) for r in rows)}}
    write_json(folder / "metrics.json", report)
    print(f"{name}: accuracy={verdict['accuracy']:.4%}, MacroF1={verdict['macro_f1']:.6f}; errors={report['operational']['errors']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "audit", "confirm", "all"), required=True)
    args = parser.parse_args()
    cfg = configuration()
    client = OllamaClient(cfg.model, ROOT / "artifacts/cache/reasoned_reader_v1")
    plan = prepare(cfg, client)
    if args.stage == "prepare":
        print("Frozen one training-free reader recipe; no audit or validation predictions yet.")
        return
    checker = ReasonedChecker(cfg, client)
    if args.stage in ("audit", "all"):
        train = load_gold(cfg.path("processed_dir"), "train")
        report = evaluate(checker, [train[c] for c in plan["audit_ids"]], "reasoned_reader_audit_v1", plan)
        gate = plan["validation_gate"]
        passed = (report["verdict"]["accuracy"] >= gate["minimum_audit_accuracy"]
                  and report["verdict"]["macro_f1"] >= gate["minimum_audit_macro_f1"]
                  and report["operational"]["errors"] <= gate["maximum_errors"])
        write_json(ROOT / "artifacts/improvement/reasoned_reader_decision_v1.json",
                   {"protocol_sha256": sha256_file(PLAN), "audit_gate_passed": passed, "validation300_used": False,
                    "audit": report, "target_met": False})
        if not passed:
            print("Audit gate failed; preserve rejected reader, no validation300 run.", flush=True)
            return
    if args.stage in ("confirm", "all"):
        decision = read_json(ROOT / "artifacts/improvement/reasoned_reader_decision_v1.json")
        assert decision["protocol_sha256"] == sha256_file(PLAN) and decision["audit_gate_passed"]
        gold = load_gold(cfg.path("processed_dir"), "validation")
        ids = load_split(cfg, "accuracy_holdout")["claim_ids"] + load_split(cfg, "final")["claim_ids"]
        assert len(ids) == len(set(ids)) == len(gold) == 300 and set(ids) == set(gold)
        report = evaluate(checker, [gold[c] for c in ids], "reasoned_reader_validation300_v1", plan)
        write_json(ROOT / "artifacts/improvement/final_reasoned_reader_results_v1.json",
                   {**report, "target_met": report["verdict"]["accuracy"] > .95 and report["verdict"]["macro_f1"] > .95,
                    "baseline_accuracy": 242 / 300, "baseline_macro_f1": .7959743387519879})
    del checker
    gc.collect()


if __name__ == "__main__":
    main()
