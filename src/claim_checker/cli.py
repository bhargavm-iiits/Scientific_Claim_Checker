"""Command-line interface: doctor, prepare-data, make-dev-split, build-index, check, retrieval-eval, evaluate, report, latency."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from claim_checker import __version__  # noqa: E402
from claim_checker.config import AppConfig, load_config, load_experiments  # noqa: E402
from claim_checker.errors import ClaimCheckerError  # noqa: E402

REQUIRED_IMPORTS = [
    ("datasets", "datasets"), ("huggingface_hub", "huggingface_hub"), ("numpy", "numpy"), ("pandas", "pandas"),
    ("sklearn", "scikit-learn"), ("torch", "torch"), ("sentence_transformers", "sentence-transformers"),
    ("rank_bm25", "rank-bm25"), ("ollama", "ollama"), ("httpx", "httpx"), ("pydantic", "pydantic"),
    ("yaml", "pyyaml"), ("jinja2", "jinja2"), ("langgraph", "langgraph"), ("streamlit", "streamlit"),
]


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(cfg: AppConfig, args) -> int:
    from claim_checker.data.sources import pkg_version

    checks: list[dict] = []

    def add(name, ok, detail="", hint="", critical=True):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "hint": hint, "critical": critical})

    add("python", sys.version_info >= (3, 11), f"{platform.python_version()} ({sys.executable})")
    for mod, dist in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
            add(f"import {mod}", True, pkg_version(dist) or "?")
        except Exception as exc:
            add(f"import {mod}", False, f"{type(exc).__name__}: {exc}", f"pip install {dist}")
    try:
        import torch

        add("torch cuda (informational)", True, f"cuda available={torch.cuda.is_available()}; embeddings/reranker on "
            f"{cfg.retrieval.embedding_device}/{cfg.retrieval.reranker_device}", critical=False)
    except Exception:
        pass
    for key in ("raw_dir", "processed_dir", "splits_dir", "index_dir", "runs_dir", "cache_dir"):
        p = cfg.path(key)
        try:
            p.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=p, delete=True):
                pass
            add(f"writable {cfg.paths.model_dump()[key]}", True, str(p))
        except Exception as exc:
            add(f"writable {key}", False, str(exc))

    from claim_checker.runtime import check_readiness

    rep = check_readiness(cfg)
    add("prepared data", rep.data_ready, "", "claim-checker prepare-data", critical=False)
    add("frozen final split", rep.splits_ready, "", "claim-checker prepare-data", critical=False)
    add("index fresh", rep.index_ready, rep.index_fingerprint or next(
        (p["message"] for p in rep.problems if p["component"] == "index"), ""), "claim-checker build-index", critical=False)
    local_only = cfg.product.judge_backend in {"nli", "lora"} and (
        cfg.product.evidence_mode == "full_abstract" or cfg.evidence_selector.enabled)
    if not local_only:
        add("ollama server", rep.server_version is not None, f"version {rep.server_version} at {cfg.model.host}",
            "Install Ollama and run `ollama serve`")
    add(f"model {rep.model_name or cfg.model.name}", rep.model_ready,
        "local checkpoint files" if local_only else f"digest {rep.model_digest}",
        "See Docs/ACCURACY_IMPROVEMENT_2026-10-22.md" if local_only else f"ollama pull {cfg.model.name}")

    if local_only and rep.model_ready and not args.skip_model_test:
        from claim_checker.runtime import ClaimChecker

        try:
            checker = ClaimChecker(cfg)
            if cfg.product.judge_backend == "nli":
                _ = checker.verifier
            else:
                _ = checker.lora_judge
            if cfg.evidence_selector.enabled:
                _ = checker.evidence_selector
            add("local model loading", True, "Configured classifier/adapter weights and contracts loaded.")
        except ClaimCheckerError as exc:
            add("local model loading", False, exc.message, exc.hint or "")
    elif rep.model_ready and not args.skip_model_test:
        from claim_checker.models.ollama_client import OllamaClient
        from claim_checker.schemas import DoctorProbe

        client = OllamaClient(cfg.model, cache_dir=None)
        messages = [
            {"role": "system", "content": "You check scientific claims against numbered sentences. Answer in JSON."},
            {"role": "user", "content": "Claim: Aspirin reduces fever.\n[0] Aspirin significantly lowered body "
             "temperature in febrile patients.\n[1] The study was conducted in 2001.\nReturn the verdict, the "
             "supporting sentence ids and a short reason."},
        ]
        t0 = time.perf_counter()
        try:
            resp = client.structured(messages, DoctorProbe, purpose="doctor")
            add("structured output", True, f"{resp.parsed.model_dump()} in {time.perf_counter() - t0:.1f}s "
                f"(attempts={resp.attempts})")
        except ClaimCheckerError as exc:
            add("structured output", False, exc.message, exc.hint or "")

    if args.json:
        print(json.dumps({"version": __version__, "checks": checks}, indent=2))
    else:
        for c in checks:
            mark = "OK  " if c["ok"] else ("FAIL" if c["critical"] else "WARN")
            line = f"[{mark}] {c['check']}: {c['detail']}"
            if not c["ok"] and c["hint"]:
                line += f"  -> {c['hint']}"
            print(line)
    return 0 if all(c["ok"] for c in checks if c["critical"]) else 1


# ---------------------------------------------------------------------------
# data / index
# ---------------------------------------------------------------------------


def cmd_prepare_data(cfg: AppConfig, args) -> int:
    if cfg.knowledge.kind == "general_science":
        from claim_checker.errors import ConfigError

        raise ConfigError("Use python scripts/prepare_general_science.py --download for the general-science collection; SciFact preparation is separate.")
    from claim_checker.data.prepare import prepare_data
    from claim_checker.data.sources import acquire_sources
    from claim_checker.data.splits import freeze_splits
    from claim_checker.data.store import load_gold

    bundle = acquire_sources(cfg, force_download=args.force_download, log=_log)
    quality = prepare_data(cfg, bundle, log=_log)
    processed = cfg.path("processed_dir")
    res = freeze_splits(cfg, load_gold(processed, "train"), load_gold(processed, "validation"), force=args.force_splits)
    for name, p in res["payloads"].items():
        _log(f"split {name}: {p['size']} claims {p['class_counts']} [{res['status'][name]}]")
    final = res["payloads"]["final"]
    _log(f"final: excluded {len(final['excluded_annotation_conflicts'])} annotation conflicts, "
         f"{final['eligible_claims']} eligible; development overlap removed: "
         f"{len(res['payloads']['development']['removed_overlap_with_final'])}")
    quality["splits_summary"] = {k: {"size": v["size"], "class_counts": v["class_counts"]} for k, v in res["payloads"].items()}
    quality["overlap"] = {
        "train_claims_removed_for_final_overlap": res["payloads"]["development"]["removed_overlap_with_final"],
        "train_claims_overlapping_other_validation": res["payloads"]["development"]["overlap_with_other_validation_claims_kept"],
    }
    from claim_checker.data.store import write_json

    write_json(processed / "data_quality.json", quality)
    _log(f"Route: {bundle.manifest['route']}. Manifest: {cfg.path('raw_dir') / 'source_manifest.json'}")
    return 0


def cmd_make_dev_split(cfg: AppConfig, args) -> int:
    from claim_checker.data.splits import freeze_dev_tune_split

    res = freeze_dev_tune_split(cfg, size=args.size, seed=args.seed, force=args.force)
    p = res["payload"]
    _log(f"split dev_tune: {p['size']} claims {p['class_counts']} (pool {p['pool_size']}, seed {p['seed']}) "
         f"[{res['status']}] -> {cfg.path('splits_dir') / 'dev_tune.json'}")
    return 0


def cmd_build_index(cfg: AppConfig, args) -> int:
    from claim_checker.data.store import load_corpus
    from claim_checker.retrieval.index import build_index

    corpus = load_corpus(cfg.path("processed_dir"))
    t0 = time.perf_counter()
    manifest = build_index(cfg, corpus, log=_log)
    _log(f"Index built in {time.perf_counter() - t0:.1f}s: {manifest['windows']} windows, "
         f"fingerprint {manifest['fingerprint']}, {manifest['documents_with_multiple_windows']} multi-window docs")
    return 0


# ---------------------------------------------------------------------------
# check / evaluate
# ---------------------------------------------------------------------------


def cmd_check(cfg: AppConfig, args) -> int:
    from claim_checker.runtime import ClaimChecker

    checker = ClaimChecker(cfg)
    oracle_ids = args.oracle_doc_ids
    if args.variant:
        spec = checker.experiments.variant(args.variant)
        if oracle_ids and not spec.oracle:
            print(f"ERROR (invalid_input): --oracle-doc-ids only applies to an oracle variant, not {spec.name!r}.",
                  file=sys.stderr)
            return 2
        res = checker.check_variant(args.claim, spec, oracle_doc_ids=oracle_ids)
    else:
        res = checker.check(args.claim, k=args.k, rerank=not args.no_rerank, evidence_mode=args.evidence_mode,
                            oracle_doc_ids=oracle_ids)
    if args.json:
        print(res.to_json())
    else:
        if res.status == "error":
            print(f"ERROR ({res.error.error_type}): {res.error.message}")
            if res.error.hint:
                print(f"Hint: {res.error.hint}")
        else:
            print(f"Verdict: {res.verdict}{'  (conflicting evidence)' if res.conflict else ''}")
            print(res.explanation)
            sources = {d.doc_id: d.source for d in res.documents}
            for q in res.evidence:
                print(f'  - [{q.doc_id} s{q.sentence_id + 1}] ({q.doc_verdict}) "{q.quote}"  -- {q.title}')
                source = sources.get(q.doc_id, {})
                if source.get("url"):
                    print(f"    Source: {source['url']}")
                    print(f"    {source.get('attribution', '')}")
            print(f"Timings (s): {res.timings}")
    return 0 if res.status == "ok" else 2


def cmd_retrieval_eval(cfg: AppConfig, args) -> int:
    from claim_checker.evaluation.runner import format_retrieval_table, run_retrieval_eval
    from claim_checker.runtime import ClaimChecker

    checker = ClaimChecker(cfg)
    out = run_retrieval_eval(cfg, checker, split=args.split, limit=args.limit, log=_log)
    print(f"Six-way retrieval comparison on {out['n_claims']} {args.split} claims:")
    print(format_retrieval_table(out["results"]))
    print(f"Saved to {cfg.path('retrieval_dir')}")
    return 0


def cmd_evaluate(cfg: AppConfig, args) -> int:
    if cfg.knowledge.kind == "general_science":
        from claim_checker.errors import ConfigError

        raise ConfigError("SciFact evaluation suites cannot measure the general-science collection. Use scripts/evaluate_general_science.py with a separate labeled benchmark.")
    from claim_checker.evaluation.analysis import analyse_run
    from claim_checker.evaluation.runner import run_evaluation
    from claim_checker.runtime import ClaimChecker

    experiments = load_experiments(root=cfg.root)
    checker = ClaimChecker(cfg, experiments=experiments)
    run_dir = run_evaluation(cfg, experiments, checker, suite=args.suite, variants=args.variants,
                             run_id=args.run_id, limit=args.limit, log=_log)
    out = analyse_run(run_dir, checker.corpus)
    _print_rows(out["rows"])
    print(f"Run saved to {run_dir}")
    return 0


def cmd_report(cfg: AppConfig, args) -> int:
    from claim_checker.data.store import load_corpus
    from claim_checker.evaluation.analysis import analyse_run
    from claim_checker.evaluation.runner import run_directory

    run_dir = run_directory(cfg, args.run_id, stage="report")
    if not (run_dir / "predictions.jsonl").exists():
        print(f"No predictions found in {run_dir}")
        return 1
    try:
        corpus = load_corpus(cfg.path("processed_dir"))
    except ClaimCheckerError:
        corpus = None
    out = analyse_run(run_dir, corpus)
    _print_rows(out["rows"])
    print(f"Report written to {run_dir / 'report.md'}")
    return 0


def cmd_latency(cfg: AppConfig, args) -> int:
    from claim_checker.evaluation.latency import (
        command_line,
        run_latency_benchmark,
        write_latency_outputs,
    )
    from claim_checker.runtime import ClaimChecker

    experiments = load_experiments(root=cfg.root)
    if args.suite not in experiments.suites:
        print(f"ERROR (config_error): unknown suite {args.suite!r}; known: {sorted(experiments.suites)}",
              file=sys.stderr)
        return 2
    split = experiments.suites[args.suite].split
    if args.judge_think is not None:
        args.judge_think = args.judge_think == "on"
    if args.no_cache:
        cfg = cfg.model_copy(deep=True)
        cfg.model.cache_enabled = False  # every model request goes to Ollama
    checker = ClaimChecker(cfg, experiments=experiments)
    out = run_latency_benchmark(cfg, checker, split=split, limit=args.limit, k=args.k, no_cache=args.no_cache,
                                repeat=args.repeat, command=command_line(getattr(args, "_argv", None)), log=_log,
                                prompt_set=args.prompt_set, judge_think=args.judge_think)
    name = args.name or (f"cold_full_k{args.k}" if args.no_cache else f"cached_full_k{args.k}")
    json_path, md_path = write_latency_outputs(cfg, out, name)
    tot = out["stages_seconds"].get("total", {})
    print(f"Warm-up {out['warmup_seconds']['total']:.2f} s; per claim mean {tot.get('mean') or 0:.2f} s, "
          f"median {tot.get('median') or 0:.2f} s, p95 {tot.get('p95') or 0:.2f} s over {out['n_claims']} claims")
    print(f"Saved {json_path} and {md_path}")
    return 0 if out["errors"] == 0 else 2


def cmd_compare(cfg: AppConfig, args) -> int:
    from claim_checker.evaluation.compare import (
        compare_runs,
        render_comparison_md,
        write_comparison,
    )
    from claim_checker.evaluation.runner import validate_run_id

    def parse(spec: str) -> tuple[str, str]:
        run, sep, variant = spec.partition(":")
        if not sep or not variant:
            raise ValueError(f"expected RUN_ID:VARIANT, got {spec!r}")
        return validate_run_id(run, stage="compare"), variant

    try:
        a, b = parse(args.a), parse(args.b)
        out = compare_runs(cfg.path("runs_dir"), a, b, n_boot=args.n_boot, seed=args.seed)
    except ValueError as exc:
        print(f"ERROR (invalid_input): {exc}", file=sys.stderr)
        return 2
    name = args.name or f"{a[0]}-{a[1]}_vs_{b[0]}-{b[1]}"
    validate_run_id(name, stage="compare")
    jp, mp = write_comparison(cfg.root, out, name)
    print(render_comparison_md(out))
    print(f"Saved {jp} and {mp}")
    return 0


def _print_rows(rows: list[dict]) -> None:
    def f(x):
        return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    cols = ["variant", "n_claims", "coverage", "accuracy", "macro_f1", "recall@3", "evidence_precision_selected",
            "evidence_recall_selected", "failure_rate", "latency_mean_s"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f(r[c]) for c in cols))


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="claim-checker", description="Local Scientific Claim Checker (SciFact)")
    p.add_argument("--config", help="path to a YAML config (default configs/default.yaml)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check environment, paths, data, index and the local model")
    d.add_argument("--json", action="store_true")
    d.add_argument("--skip-model-test", action="store_true", help="skip the structured-output request")

    pd = sub.add_parser("prepare-data", help="download, validate and normalise data; freeze splits")
    pd.add_argument("--force-download", action="store_true")
    pd.add_argument("--force-splits", action="store_true", help="overwrite a frozen final split (new experiment)")

    ds = sub.add_parser("make-dev-split", help="freeze the stratified dev_tune subset of the development split "
                        "(tuning population for v2; never the final split)")
    ds.add_argument("--size", type=int, default=120)
    ds.add_argument("--seed", type=int, default=None, help="default: splits.seed (42)")
    ds.add_argument("--force", action="store_true", help="overwrite an existing dev_tune.json with different content")

    sub.add_parser("build-index", help="build BM25 + dense window index with fingerprints")

    c = sub.add_parser("check", help="check one claim")
    c.add_argument("--claim", required=True)
    c.add_argument("--k", type=int, default=None)
    c.add_argument("--no-rerank", action="store_true")
    c.add_argument("--evidence-mode", choices=["evidence", "full_abstract"], default=None,
                   help="context supplied to the judge (default: the selected configuration)")
    c.add_argument("--variant", help="run a named variant from configs/experiments.yaml instead")
    c.add_argument("--oracle-doc-ids", nargs="+", default=None, metavar="DOC_ID",
                   help="oracle mode: judge these corpus abstracts instead of retrieving (required for "
                        "--variant oracle; normally the evaluation runner supplies the gold reference ids)")
    c.add_argument("--json", action="store_true")

    r = sub.add_parser("retrieval-eval", help="six-way retrieval comparison (BM25/dense/hybrid x rerank)")
    r.add_argument("--split", default="final", choices=["final", "smoke", "development", "dev_tune"])
    r.add_argument("--limit", type=int, default=None)

    e = sub.add_parser("evaluate", help="run evaluation variants on a frozen population (resumable)")
    e.add_argument("--suite", required=True)
    e.add_argument("--variants", nargs="+", default=None)
    e.add_argument("--run-id", default=None)
    e.add_argument("--limit", type=int, default=None)

    cp = sub.add_parser("compare", help="paired comparison of two evaluated variants (McNemar, bootstrap CIs)")
    cp.add_argument("--a", required=True, metavar="RUN_ID:VARIANT", help="baseline, e.g. final_v1:full")
    cp.add_argument("--b", required=True, metavar="RUN_ID:VARIANT", help="new system, e.g. final_v2:full_v2")
    cp.add_argument("--name", default=None, help="output stem under artifacts/comparisons/")
    cp.add_argument("--n-boot", type=int, default=10000)
    cp.add_argument("--seed", type=int, default=42)

    rp = sub.add_parser("report", help="recompute metrics and report for a run")
    rp.add_argument("--run-id", required=True)

    lt = sub.add_parser("latency", help="per-stage latency benchmark of the full pipeline (writes artifacts/latency/)")
    lt.add_argument("--suite", default="smoke", help="suite whose frozen split supplies the claims (default smoke)")
    lt.add_argument("--limit", type=int, default=10)
    lt.add_argument("--k", type=int, default=3)
    lt.add_argument("--no-cache", action="store_true", help="disable the LLM disk cache (cold measurement)")
    lt.add_argument("--repeat", action="store_true",
                    help="afterwards run the same claims again in the same process (cached repeat latency)")
    lt.add_argument("--name", default=None, help="output file stem (default cold_full_k<k> / cached_full_k<k>)")
    lt.add_argument("--prompt-set", choices=["v1", "v2"], default=None,
                    help="judge prompt set (default: the product default in configs/default.yaml)")
    lt.add_argument("--judge-think", choices=["on", "off"], default=None,
                    help="qwen3 thinking for the judge step (default: the product default)")
    return p


COMMANDS = {
    "doctor": cmd_doctor, "prepare-data": cmd_prepare_data, "make-dev-split": cmd_make_dev_split,
    "build-index": cmd_build_index, "check": cmd_check,
    "retrieval-eval": cmd_retrieval_eval, "evaluate": cmd_evaluate, "report": cmd_report, "latency": cmd_latency,
    "compare": cmd_compare,
}


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    args._argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.command](cfg, args)
    except ClaimCheckerError as exc:
        print(f"ERROR ({exc.error_type}): {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"Hint: {exc.hint}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
