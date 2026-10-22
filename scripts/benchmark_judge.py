"""Development-only judge experiments with saved, genuine retrieval rankings.

No reference document IDs or labels enter a prompt. Training annotations may be used
only as in-context demonstrations, with all evaluation/tuning texts excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from claim_checker.agents.examples import ExampleLibrary, prepend_examples
from claim_checker.agents.judge import judge_document
from claim_checker.agents.paper_examples import PaperExampleLibrary, prepend_paper_examples
from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import load_config
from claim_checker.data.splits import load_split, stratified_sample
from claim_checker.data.store import append_jsonl, load_corpus, load_gold, read_json, write_json
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.models.ollama_client import OllamaClient
from claim_checker.pipeline.aggregate import aggregate_verdicts
from claim_checker.pipeline.consistency import (
    consistency_feedback,
    consistency_flags,
    downgrade_inconsistent,
)
from claim_checker.pipeline.supervisor import CITATION_FEEDBACK, downgrade, uncited_decisive
from claim_checker.schemas import VerdictAnswer

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    p.add_argument("--mode", choices=["v2_selected", "v2_full", "v3_full", "v4_full"], default="v2_selected")
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--name", required=True)
    demonstrations = p.add_mutually_exclusive_group()
    demonstrations.add_argument("--examples", action="store_true")
    demonstrations.add_argument("--paper-examples", action="store_true", help="Explicit TRAIN-annotation-assisted same-paper demonstrations")
    p.add_argument("--templates-dir", default="v2", help="Template folder for v2 modes")
    p.add_argument("--supervise", action="store_true", help="Apply the existing bounded v2 citation/consistency supervisor")
    args = p.parse_args()
    if args.paper_examples and args.mode.startswith("v2"):
        p.error("Paper demonstrations are currently supported only by the v3_full/v4_full component experiments")
    cfg = load_config(overrides={"model": {"name": args.model}, "judge_v2": {"templates_dir": args.templates_dir}})
    corpus = load_corpus(cfg.path("processed_dir"))
    train = load_gold(cfg.path("processed_dir"), "train")
    dev = load_split(cfg, "dev_tune")
    pop, _ = stratified_sample([train[c] for c in dev["claim_ids"]], args.size, 42)
    retrieval = {r["claim_id"]: r["hits"] for r in _rows(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    selected = {(r["claim_id"], r["doc_id"]): r["selected"] for r in _rows(ROOT / "artifacts/runs/dev_v1/pair_predictions.jsonl")}
    llm = OllamaClient(cfg.model, cfg.path("cache_dir"))
    prompts = PromptLibrary(cfg.path("prompts_dir"), v2_dir=args.templates_dir)
    example_path = ROOT / "artifacts/examples/train.jsonl"
    examples = (PaperExampleLibrary.load(example_path) if args.paper_examples else
                (ExampleLibrary.load(example_path) if args.examples else None))
    folder = ROOT / "artifacts/improvement" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    template = f"{args.templates_dir}/judge" if args.mode.startswith("v2") else ("v4/judge" if args.mode.startswith("v4") else "v3/judge")
    prompt_path = ROOT / f"prompts/{template}.jinja2"
    identity = {"model": args.model, "digest": llm.model_digest(), "mode": args.mode,
                "claim_ids": [g.claim_id for g in pop],
                "prompt_sha": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                "config": cfg.fingerprint("model", "judge", "judge_v2"),
                "retrieval_sha": hashlib.sha256((ROOT / "artifacts/runs/dev_v1/retrieval.jsonl").read_bytes()).hexdigest(),
                "selection_sha": hashlib.sha256((ROOT / "artifacts/runs/dev_v1/pair_predictions.jsonl").read_bytes()).hexdigest()}
    if examples is not None:
        identity["examples_sha"] = hashlib.sha256(example_path.read_bytes()).hexdigest()
    if args.paper_examples:
        identity["paper_examples"] = True
        identity["paper_selector_sha"] = hashlib.sha256((ROOT / "src/claim_checker/agents/paper_examples.py").read_bytes()).hexdigest()
        identity["paper_benchmark_sha"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        audit = read_json(example_path.parent / "paper_memory_audit.json")
        if audit["source_sha256"] != identity["examples_sha"] or audit["excluded_text_overlap"] or audit["source_or_label_mismatches"]:
            raise ValueError("The TRAIN demonstration artifact does not match its clean source audit")
    if args.supervise:
        identity["supervise"] = True
    if (folder / "manifest.json").exists() and read_json(folder / "manifest.json") != identity:
        raise ValueError("Experiment identity changed; use a new --name")
    write_json(folder / "manifest.json", identity)
    completed = {r["claim_id"]: r for r in _rows(folder / "predictions.jsonl")}
    for i, g in enumerate(pop, 1):
        if g.claim_id in completed:
            continue
        start = time.perf_counter()
        docs = []
        for hit in retrieval[g.claim_id][:3]:
            d = corpus[hit["doc_id"]]
            if args.mode.startswith("v2"):
                mode = "evidence" if args.mode.endswith("selected") else "full_abstract"
                j, _ = judge_document(llm, prompts, g.claim, d, mode,
                                      selected.get((g.claim_id, d.doc_id)), cfg.judge,
                                      prompt_set="v2", v2cfg=cfg.judge_v2)
                if args.supervise:
                    for attempt in range(cfg.judge.max_citation_repairs + 1):
                        missing = uncited_decisive([j])
                        flags = consistency_flags(j) if cfg.judge_v2.consistency_check and not missing else []
                        if not missing and not flags:
                            break
                        if attempt == cfg.judge.max_citation_repairs:
                            j = downgrade(j) if missing else downgrade_inconsistent(j, flags)
                            break
                        feedback = CITATION_FEEDBACK.format(verdict=j.verdict) if missing else consistency_feedback(j, flags)
                        j, _ = judge_document(llm, prompts, g.claim, d, mode,
                                              selected.get((g.claim_id, d.doc_id)), cfg.judge, feedback,
                                              prompt_set="v2", v2cfg=cfg.judge_v2)
                result = {"verdict": j.verdict, "reason": j.reason, "cited": j.cited_sentence_ids,
                          "confidence": j.confidence}
            else:
                messages = prompts.render(template, claim=g.claim, doc_id=d.doc_id,
                                          title=d.title, sentences=[{"id":n,"text":s} for n,s in enumerate(d.sentences)])
                paper_examples = examples.retrieve_for_paper(g.claim, d.doc_id) if args.paper_examples else []
                if args.paper_examples:
                    messages = prepend_paper_examples(messages, paper_examples)
                elif examples is not None:
                    messages = prepend_examples(messages, examples.retrieve(g.claim, doc_id=d.doc_id))
                response = llm.structured(messages, VerdictAnswer, purpose="judge_v3")
                answer = response.parsed
                ids = sorted({n for n in answer.cited_sentence_ids if 0 <= n < len(d.sentences)})
                verdict = answer.verdict if ids or answer.verdict == "NEI" else "NEI"
                result = {"verdict":verdict,"reason":answer.reason,"cited":ids,"confidence":answer.confidence}
                if args.paper_examples:
                    result["training_example_keys"] = [e["key"] for e in paper_examples]
            docs.append({"doc_id":d.doc_id,"rank":hit["rank"],"retrieval_score":hit["score"],**result})
        verdict, conflict = aggregate_verdicts(d["verdict"] for d in docs)
        row = {"claim_id":g.claim_id,"claim":g.claim,"gold":g.label,"verdict":verdict,
               "conflict":conflict,"documents":docs,"seconds":time.perf_counter()-start}
        append_jsonl(folder / "predictions.jsonl", row)
        completed[g.claim_id] = row
        m = verdict_metrics([r["gold"] for r in completed.values()], [r["verdict"] for r in completed.values()])
        print(f"{i}/{len(pop)} {g.claim_id}: {g.label} -> {verdict}; cumulative {m['accuracy']:.3f}/{m['macro_f1']:.3f} ({row['seconds']:.1f}s)", flush=True)
    rows = [completed[g.claim_id] for g in pop]
    m = verdict_metrics([r["gold"] for r in rows],[r["verdict"] for r in rows])
    write_json(folder / "metrics.json",m)
    print(json.dumps(m,indent=2))


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
