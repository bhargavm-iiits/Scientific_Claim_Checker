"""Variant execution with per-claim checkpointing and exact-manifest resume.

Run layout ``artifacts/runs/<run_id>/``: manifest.json, predictions.jsonl (written last for each
claim, so it defines completion), pair_predictions.jsonl, retrieval.jsonl, events.jsonl.
If a run is interrupted after a claim's diagnostic rows were appended but before its prediction,
resuming first removes those orphaned diagnostic rows (see :func:`reconcile_diagnostics`) so the
re-run claim is not duplicated; a complete run's files are left byte-for-byte untouched.
A run resumes only if its identity (data, splits, population, index, model digest, prompt hashes,
configuration, variant definitions) is unchanged; otherwise it is refused.
"""

from __future__ import annotations

import platform
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from claim_checker import __version__
from claim_checker.config import (
    AppConfig,
    ExperimentsConfig,
    canonical_json,
    sha256_text,
    variant_dump,
)
from claim_checker.data.sources import pkg_version
from claim_checker.data.splits import SPLIT_SOURCE, load_split, split_fingerprint
from claim_checker.data.store import (
    CORPUS_FILE,
    GOLD_FILE,
    append_jsonl,
    iter_jsonl,
    load_gold,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from claim_checker.errors import (
    ConfigError,
    InputValidationError,
    ModelUnavailableError,
    RunMismatchError,
)
from claim_checker.evaluation.metrics import latency_stats, recall_at_k
from claim_checker.evaluation.training_provenance import assert_unfitted_population
from claim_checker.schemas import ClaimGold, ClaimResult, utc_now

PREDICTIONS = "predictions.jsonl"
PAIRS = "pair_predictions.jsonl"
RETRIEVAL = "retrieval.jsonl"
EVENTS = "events.jsonl"
MANIFEST = "manifest.json"


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "claim_checker": __version__,
        **{p: pkg_version(p) for p in ("torch", "sentence-transformers", "datasets", "langgraph", "ollama",
                                        "pydantic", "rank-bm25", "numpy", "scikit-learn")},
    }


def population(cfg: AppConfig, suite_split: str, limit: int | None) -> tuple[list[ClaimGold], dict]:
    split = load_split(cfg, suite_split)
    gold = load_gold(cfg.path("processed_dir"), SPLIT_SOURCE[suite_split])
    ids = split["claim_ids"][:limit] if limit else split["claim_ids"]
    missing = [i for i in ids if i not in gold]
    if missing:
        raise ConfigError(f"Split {suite_split} references unknown claims {missing[:5]}", stage="evaluate")
    return [gold[i] for i in ids], split


def run_identity(cfg: AppConfig, checker, suite: str, split_name: str, pop: list[ClaimGold],
                 needs_index: bool, prompt_sets: set[str] | None = None,
                 judge_backends: set[str] | None = None) -> dict:
    """Identity record that must match for a run to resume. For a run made only of v1 variants it is
    exactly the v1 record; a run with v2 variants adds the v2 prompt hashes and the ``judge_v2`` /
    ``product``-independent v2 configuration fingerprint (``config_fingerprint_v2``)."""
    processed = cfg.path("processed_dir")
    index_fp = None
    if needs_index:
        from claim_checker.retrieval.index import check_index

        index_fp = check_index(cfg)["fingerprint"]
    if judge_backends == {"lora"}:
        model_identity = {"name": "Qwen2.5-1.5B-Instruct+LoRA", "digest": sha256_file(Path(cfg.root) / cfg.lora.adapter_dir / "adapter_model.safetensors")}
    elif judge_backends == {"nli"}:
        model_identity = {"name": "Scientific NLI verifier", "digest": sha256_file(Path(cfg.root) / cfg.verifier.model_dir / "model.safetensors")}
    else:
        model_identity = {"name": cfg.model.name, "digest": checker.llm.model_digest()}
    ident = {
        "suite": suite,
        "split": split_name,
        "population_ids_sha256": sha256_text(canonical_json([g.claim_id for g in pop])),
        "population_size": len(pop),
        "data": {
            "corpus_sha256": sha256_file(processed / CORPUS_FILE),
            "gold_sha256": sha256_file(processed / GOLD_FILE.format(split=SPLIT_SOURCE[split_name])),
        },
        "split_fingerprint": split_fingerprint(cfg, split_name),
        "index_fingerprint": index_fp,
        "model": model_identity,
        "prompt_hashes": checker.prompts.hashes(),
        "config_fingerprint": cfg.fingerprint("model", "retrieval", "evidence", "judge", "runtime"),
    }
    reranker_path = Path(cfg.retrieval.reranker_model)
    if not reranker_path.is_absolute():
        reranker_path = Path(cfg.root) / reranker_path
    if reranker_path.is_dir():
        ident["local_reranker_files_sha256"] = {
            name: sha256_file(reranker_path / name)
            for name in ("model.safetensors", "config.json", "tokenizer.json")
        }
        training_path = reranker_path.parent / "training_manifest.json"
        if training_path.exists():
            training = read_json(training_path)
            assert_unfitted_population(training, pop)
            ident["local_reranker_training_sha256"] = sha256_file(training_path)
    if prompt_sets and prompt_sets - {"v1"}:
        ident["prompt_hashes_v2"] = checker.prompts.hashes_v2()
        ident["config_fingerprint_v2"] = cfg.fingerprint("judge_v2")
    if prompt_sets and "v3" in prompt_sets:
        ident["prompt_hashes_v3"] = checker.prompts.hashes_v3()
        ident["config_fingerprint_v3"] = cfg.fingerprint("judge_v3")
        examples_path = Path(cfg.judge_v3.examples_path)
        if not examples_path.is_absolute():
            examples_path = Path(cfg.root) / examples_path
        ident["training_examples_sha256"] = sha256_file(examples_path) if cfg.judge_v3.use_examples else None
    if judge_backends and "nli" in judge_backends:
        path = Path(cfg.verifier.model_dir)
        if not path.is_absolute():
            path = Path(cfg.root) / path
        ident["verifier_config"] = cfg.verifier.model_dump()
        ident["verifier_weights_sha256"] = sha256_file(path / "model.safetensors")
        ident["verifier_tokenizer_sha256"] = sha256_file(path / "tokenizer.json")
        ident["verifier_model_config_sha256"] = sha256_file(path / "config.json")
        training_manifest = path.parent / "training_manifest.json"
        if training_manifest.exists():
            ident["verifier_training_manifest_sha256"] = sha256_file(training_manifest)
            training = read_json(training_manifest)
            assert_unfitted_population(training, pop)
            if training.get("settings", {}).get("final_fit") and SPLIT_SOURCE[split_name] == "train":
                raise ConfigError("The final-fit verifier includes internal development claims in training. Evaluate it on validation, not a training-derived split.")
        ident["verifier_code_sha256"] = {
            file: sha256_file(Path(__file__).resolve().parents[1] / file)
            for file in ("agents/judge_nli.py", "models/verifier.py", "data/verifier_context.py", "pipeline/aggregate.py", "pipeline/graph.py", "agents/explainer.py")
        }
    if cfg.evidence_selector.enabled:
        path = Path(cfg.root) / cfg.evidence_selector.model_dir
        ident["evidence_selector_config"] = cfg.evidence_selector.model_dump()
        ident["evidence_selector_files_sha256"] = {name: sha256_file(path / name)
                                                   for name in ("model.safetensors", "config.json", "tokenizer.json")}
        training_path = path.parent / "training_manifest.json"
        ident["evidence_selector_training_sha256"] = sha256_file(training_path)
        if read_json(training_path)["settings"]["final_fit"] and SPLIT_SOURCE[split_name] == "train":
            raise ConfigError("The refitted rationale selector includes internal development; evaluate it on validation.")
        ident["evidence_selector_code_sha256"] = {file: sha256_file(Path(__file__).resolve().parents[1] / file)
                                                 for file in ("agents/trained_evidence.py", "models/evidence_classifier.py", "runtime.py")}
    if judge_backends and "lora" in judge_backends:
        base = Path(cfg.root) / cfg.lora.base_dir
        adapter = Path(cfg.root) / cfg.lora.adapter_dir
        ident["lora_config"] = cfg.lora.model_dump()
        ident["lora_files_sha256"] = {
            "base/" + name: sha256_file(base / name) for name in (
                "model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "merges.txt")}
        ident["lora_files_sha256"].update({"adapter/" + name: sha256_file(adapter / name)
                                           for name in ("adapter_model.safetensors", "adapter_config.json")})
        training_path = adapter.parent / "training_manifest.json"
        training = read_json(training_path)
        ident["lora_training_manifest_sha256"] = sha256_file(training_path)
        if training["settings"]["final_fit"] and SPLIT_SOURCE[split_name] == "train":
            raise ConfigError("The final-refit LoRA judge includes internal development claims; evaluate it on validation.")
        ident["lora_code_sha256"] = {file: sha256_file(Path(__file__).resolve().parents[1] / file)
                                    for file in ("agents/judge_lora.py", "models/lora_judge.py", "pipeline/graph.py", "pipeline/aggregate.py", "agents/explainer.py")}
    return ident


def gold_fields(g: ClaimGold) -> dict:
    return {
        "claim_id": g.claim_id,
        "claim": g.claim,
        "gold_label": g.label,
        "reference_doc_ids": g.reference_doc_ids,
        "evidence_by_doc": g.evidence_by_doc,
        "evidence_doc_ids": g.evidence_doc_ids,
        "label_by_doc": g.label_by_doc,
    }


def record_downgraded(res: ClaimResult) -> bool:
    """True if a decisive answer was downgraded to NEI for lack of a valid citation, either in a
    per-document judgement (pipeline) or in the single-prompt baseline's claim-level answer."""
    if any(d.judgement.downgraded for d in res.documents if d.judgement):
        return True
    return bool((res.metadata.get("single_prompt") or {}).get("downgraded"))


def prediction_record(variant: str, g: ClaimGold, res: ClaimResult) -> dict:
    hits = res.metadata.get("retrieval_hits")
    if (res.metadata.get("options") or {}).get("oracle_doc_ids") is not None:
        hits = None  # oracle documents are supplied, not retrieved: no retrieval metrics
    retrieved = [h["doc_id"] for h in hits] if hits is not None else None
    examined = [d.doc_id for d in res.documents] if res.status == "ok" else []
    selected = None
    if res.documents and any(d.selected_sentence_ids is not None for d in res.documents):
        selected = {d.doc_id: d.selected_sentence_ids or [] for d in res.documents}
    cited: dict[str, list[int]] | None = None
    if res.status == "ok" and res.documents and res.documents[0].judgement is not None:
        cited = {d.doc_id: (d.judgement.cited_sentence_ids if d.judgement else []) for d in res.documents}
    elif res.status == "ok" and res.metadata.get("strategy") == "single_prompt":
        cited = {d.doc_id: [] for d in res.documents}
        for q in res.evidence:
            cited.setdefault(q.doc_id, []).append(q.sentence_id)
    elif res.status == "ok" and res.metadata.get("strategy") == "pipeline":
        cited = {}  # No accepted documents still means zero predicted citations.
    if (selected is None and res.status == "ok" and not res.documents
            and res.metadata.get("strategy") == "pipeline"
            and (res.metadata.get("options") or {}).get("evidence_mode") == "evidence"):
        selected = {}
    return {
        "variant": variant,
        **gold_fields(g),
        "status": res.status,
        "verdict": res.verdict,
        "correct": res.status == "ok" and res.verdict == g.label,
        "conflict": res.conflict,
        "error": res.error.model_dump() if res.error else None,
        "retrieved_doc_ids": retrieved,
        "examined_doc_ids": examined if res.metadata.get("strategy") != "judge_only" else None,
        "selected": selected,
        "cited": cited,
        "doc_verdicts": {d.doc_id: d.judgement.verdict for d in res.documents if d.judgement},
        "quotes": [q.model_dump() for q in res.evidence],
        "timings": res.timings,
        "explanation": res.explanation,
        "llm_calls": res.metadata.get("llm_calls"),
        "llm_cache_hits": res.metadata.get("llm_cache_hits"),
        # True: retrieval served from the in-process cache (timings["retrieve"] is the real near-zero
        # lookup; no retrieve.* sub-timings); False: computed now; None: no retrieval / unknown.
        "retrieval_cache_hit": res.metadata.get("retrieval_cache_hit"),
        "retrieve_original_compute_s": res.metadata.get("retrieve_original_compute_s"),
        # Pipeline variants downgrade per-document judgements; the single-prompt baseline downgrades
        # its one claim-level answer (metadata["single_prompt"]["downgraded"]). Both count here.
        "downgraded": record_downgraded(res),
        # Single-prompt baseline only: the model's own answer before any grounding downgrade.
        "model_verdict": (res.metadata.get("single_prompt") or {}).get("model_verdict"),
        "finished_at": utc_now(),
        **({"prompt_set": res.metadata["prompt_set"], "judge_thinking": res.metadata.get("judge_thinking")}
           if res.metadata.get("prompt_set") in ("v2", "v3") else {}),
        **({"judge_backend": "nli", "verifier": res.metadata.get("verifier")}
           if res.metadata.get("judge_backend") == "nli" else {}),
        **({"evidence_backend": "trained_classifier", "evidence_selector": res.metadata.get("evidence_selector")}
           if res.metadata.get("evidence_backend") == "trained_classifier" else {}),
        **({"judge_backend": "lora", "lora": res.metadata.get("lora")}
           if res.metadata.get("judge_backend") == "lora" else {}),
    }


def pair_records(variant: str, g: ClaimGold, res: ClaimResult) -> list[dict]:
    out = []
    for d in res.documents:
        is_ref = d.doc_id in g.reference_doc_ids
        out.append({
            "variant": variant, "claim_id": g.claim_id, "doc_id": d.doc_id, "rank": d.rank,
            "is_reference_doc": is_ref,
            "gold_pair_label": g.label_by_doc.get(d.doc_id, "NEI") if is_ref else "unannotated",
            "gold_evidence": g.evidence_by_doc.get(d.doc_id, []),
            "selected": d.selected_sentence_ids,
            "predicted": d.judgement.verdict if d.judgement else None,
            "cited": d.judgement.cited_sentence_ids if d.judgement else None,
            "reason": d.judgement.reason if d.judgement else None,
            "confidence": d.judgement.confidence if d.judgement else None,
            "skipped_model": d.judgement.skipped_model if d.judgement else None,
            "downgraded": d.judgement.downgraded if d.judgement else None,
        })
        j = d.judgement
        if j is not None and j.prompt_set == "v2":  # v2 diagnostics (v1 rows keep their exact columns)
            out[-1].update({"prompt_set": "v2", "checklist": j.checklist, "thinking": j.thinking,
                            "consistency_flags": j.consistency_flags, "repaired": j.repaired})
    return out


def completed(run_dir: Path) -> set[tuple[str, str]]:
    path = run_dir / PREDICTIONS
    if not path.exists():
        return set()
    return {(r["variant"], r["claim_id"]) for r in iter_jsonl(path)}


DIAGNOSTIC_FILES = (RETRIEVAL, PAIRS, EVENTS)


def _clean_diagnostic_rows(rows: list[dict], done: set[tuple[str, str]]) -> list[dict]:
    """Rows of completed claims only, keeping one write ("attempt") per claim.

    Each attempt appends its rows contiguously: one row per claim in retrieval/events, one row per
    examined document in pair predictions. A new attempt starts when the (variant, claim_id)
    changes or when a row repeats within the current attempt (same ``doc_id`` for pair rows; any
    second row for one-row-per-claim files). Only the last attempt of a completed claim - the one
    written together with its prediction - is kept.
    """
    attempts: list[tuple[tuple, list[dict]]] = []
    seen: set = set()
    for r in rows:
        key = (r.get("variant"), r.get("claim_id"))
        sub = r.get("doc_id")
        if attempts and attempts[-1][0] == key and sub not in seen:
            attempts[-1][1].append(r)
        else:
            attempts.append((key, [r]))
            seen = set()
        seen.add(sub)
    last = {key: i for i, (key, _) in enumerate(attempts)}
    out: list[dict] = []
    for i, (key, block) in enumerate(attempts):
        if key in done and last[key] == i:
            out.extend(block)
    return out


def reconcile_diagnostics(run_dir: Path, done: set[tuple[str, str]], log=print) -> dict[str, int]:
    """Drop diagnostic rows (retrieval / pair / event) whose (variant, claim_id) has no prediction,
    plus stale duplicate blocks. A file is rewritten (atomically) only if something is removed, so
    resuming a complete run leaves every file unchanged. Returns removed-row counts per file."""
    removed: dict[str, int] = {}
    for name in DIAGNOSTIC_FILES:
        path = run_dir / name
        if not path.exists():
            continue
        rows = list(iter_jsonl(path))
        kept = _clean_diagnostic_rows(rows, done)
        if len(kept) != len(rows):
            write_jsonl(path, kept)
            removed[name] = len(rows) - len(kept)
            log(f"Removed {len(rows) - len(kept)} orphaned/duplicate row(s) from {name} "
                "(claims interrupted before their prediction was saved)")
    return removed


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_WINDOWS_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))}


def validate_run_id(run_id: str, *, stage: str = "evaluate") -> str:
    """A run id names one folder directly under ``paths.runs_dir``: letters, digits, ``.``, ``_``
    and ``-`` only, not made of dots alone (``.``/``..``), and not a Windows device name. Rejects
    path separators, drive letters and absolute paths, so a run id can never escape the runs folder."""
    rid = run_id if isinstance(run_id, str) else ""
    if (not RUN_ID_PATTERN.fullmatch(rid) or set(rid) == {"."} or ".." in rid
            or rid.split(".")[0].upper() in _WINDOWS_DEVICE_NAMES):
        raise InputValidationError(
            f"Invalid run id {run_id!r}.", stage=stage,
            hint="Use only letters, digits, '.', '_' and '-' (e.g. final_v2); no path separators or '..'.")
    return rid


def run_directory(cfg: AppConfig, run_id: str, *, stage: str = "evaluate") -> Path:
    """``runs_dir / run_id`` after validating ``run_id`` (and checking the result stays inside)."""
    root = cfg.path("runs_dir")
    run_dir = root / validate_run_id(run_id, stage=stage)
    if run_dir.resolve().parent != root.resolve():
        raise InputValidationError(f"Run id {run_id!r} resolves outside {root}.", stage=stage)
    return run_dir


def default_run_id(suite: str) -> str:
    return f"{suite}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def run_evaluation(
    cfg: AppConfig,
    experiments: ExperimentsConfig,
    checker,
    *,
    suite: str,
    variants: list[str] | None = None,
    run_id: str | None = None,
    limit: int | None = None,
    log=print,
) -> Path:
    if cfg.knowledge.kind == "general_science":
        raise ConfigError("SciFact suites cannot evaluate a different evidence collection; use a separately labeled general-science benchmark.", stage="evaluate")
    if suite not in experiments.suites:
        raise ConfigError(f"Unknown suite {suite!r}; known: {sorted(experiments.suites)}", stage="evaluate")
    suite_spec = experiments.suites[suite]
    variant_names = variants or suite_spec.variants
    specs = {v: experiments.variant(v) for v in variant_names}
    pop, _ = population(cfg, suite_spec.split, limit)
    needs_index = any(not s.oracle and s.strategy != "judge_only" for s in specs.values())

    run_id = run_id or default_run_id(suite)
    run_dir = run_directory(cfg, run_id)
    identity = run_identity(cfg, checker, suite, suite_spec.split, pop, needs_index,
                            {s.prompt_set for s in specs.values()},
                            {s.judge_backend for s in specs.values()})
    spec_dump = {k: variant_dump(v) for k, v in specs.items()}
    manifest_path = run_dir / MANIFEST
    if manifest_path.exists():
        old = read_json(manifest_path)
        old_identity = dict(old.get("identity", {}))
        new_identity = dict(identity)
        if not needs_index or old_identity.get("index_fingerprint") is None:
            old_identity.pop("index_fingerprint", None)
            new_identity.pop("index_fingerprint", None)
        if old_identity != new_identity:
            changed = sorted(k for k in set(old_identity) | set(new_identity) if old_identity.get(k) != new_identity.get(k))
            raise RunMismatchError(
                f"Run {run_id} exists with a different identity ({', '.join(changed)} changed); refusing to resume. "
                "Use a new --run-id.", stage="evaluate")
        for name, spec in spec_dump.items():
            if name in old.get("variants", {}) and old["variants"][name] != spec:
                raise RunMismatchError(f"Variant {name} definition changed since run {run_id} started.", stage="evaluate")
        manifest = old
        manifest["variants"] = {**old.get("variants", {}), **spec_dump}
        manifest.setdefault("resumed_at", []).append(utc_now())
        if needs_index and manifest["identity"].get("index_fingerprint") is None:
            manifest["identity"]["index_fingerprint"] = identity["index_fingerprint"]
        log(f"Resuming run {run_id}")
    else:
        manifest = {
            "run_id": run_id, "created_at": utc_now(), "identity": identity, "variants": spec_dump,
            "limit": limit, "config": cfg.model_dump(mode="json"), "environment": environment_info(),
            "claim_ids": [g.claim_id for g in pop],
            "gold_label_counts": {lab: sum(1 for g in pop if g.label == lab) for lab in ("SUPPORT", "CONTRADICT", "NEI")},
        }
    write_json(manifest_path, manifest)

    done = completed(run_dir)
    reconcile_diagnostics(run_dir, done, log=log)
    for vname in variant_names:
        spec = specs[vname]
        todo = [g for g in pop if (vname, g.claim_id) not in done]
        log(f"[{vname}] {len(pop) - len(todo)}/{len(pop)} already done; running {len(todo)}")
        for i, g in enumerate(todo, start=1):
            t0 = time.perf_counter()
            res = checker.check_variant(g.claim, spec, claim_id=g.claim_id,
                                        oracle_doc_ids=list(g.reference_doc_ids) if spec.oracle else None)
            if res.status == "error" and res.error and res.error.error_type == "model_unavailable":
                raise ModelUnavailableError(
                    f"Model became unavailable during {vname}/{g.claim_id}: {res.error.message}. "
                    "Completed claims are saved; rerun the same command to resume.",
                    stage="evaluate", hint=res.error.hint)
            hits = None if spec.oracle else res.metadata.get("retrieval_hits")
            if hits is not None:
                ranked = [h["doc_id"] for h in hits]
                append_jsonl(run_dir / RETRIEVAL, {
                    "variant": vname, "claim_id": g.claim_id, "hits": hits,
                    "reference_doc_ids": g.reference_doc_ids, "evidence_doc_ids": g.evidence_doc_ids,
                    "recall@3": recall_at_k(ranked, g.reference_doc_ids, 3),
                    "recall@10": recall_at_k(ranked, g.reference_doc_ids, 10)})
            for row in pair_records(vname, g, res):
                append_jsonl(run_dir / PAIRS, row)
            append_jsonl(run_dir / EVENTS, {"variant": vname, "claim_id": g.claim_id,
                                            "events": [e.model_dump() for e in res.events]})
            append_jsonl(run_dir / PREDICTIONS, prediction_record(vname, g, res))
            log(f"[{vname}] {i}/{len(todo)} claim {g.claim_id}: gold={g.label} pred={res.verdict} "
                f"status={res.status} ({time.perf_counter() - t0:.1f}s)")
    return run_dir


# ---------------------------------------------------------------------------
# Retrieval-only evaluation (six-way)
# ---------------------------------------------------------------------------


def run_retrieval_eval(cfg: AppConfig, checker, *, split: str = "final", limit: int | None = None, log=print) -> dict:
    """BM25 / dense / hybrid x rerank on/off on a frozen population. Writes artifacts/retrieval/."""
    pop, _ = population(cfg, split, limit)
    retriever = checker.retriever
    r = cfg.retrieval
    configs = [(m, rr) for m in ("bm25", "dense", "hybrid") for rr in (False, True)]
    per_claim = []
    lat: dict[str, list[float]] = {f"{m}{'+rerank' if rr else ''}": [] for m, rr in configs}
    retriever.rerank("warm up", [next(iter(checker.corpus))])  # load cross-encoder before timing
    retriever.dense.search("warm up", 1)
    for n, g in enumerate(pop, start=1):
        cand = retriever.candidates(g.claim)
        row = {"claim_id": g.claim_id, "reference_doc_ids": g.reference_doc_ids, "evidence_doc_ids": g.evidence_doc_ids,
               "rankings": {}}
        for m, rr in configs:
            name = f"{m}{'+rerank' if rr else ''}"
            base_t = cand.timings["bm25"] if m == "bm25" else cand.timings["dense"] if m == "dense" else \
                cand.timings["bm25"] + cand.timings["dense"] + cand.timings["fuse"]
            ranked = [d for d, _ in cand.candidates[m]]
            t = base_t
            if rr:
                rer, t_rr = retriever.rerank(g.claim, ranked[: r.rerank_top_n])
                ranked = [d for d, _, _ in rer]
                t += t_rr
            ranked = ranked[: r.keep_top]
            row["rankings"][name] = ranked
            lat[name].append(t)
        per_claim.append(row)
        if n % 10 == 0:
            log(f"  {n}/{len(pop)} claims")
    results = {}
    for name in lat:
        recs = [{"retrieved_doc_ids": row["rankings"][name], "reference_doc_ids": row["reference_doc_ids"],
                 "evidence_doc_ids": row["evidence_doc_ids"]} for row in per_claim]
        from claim_checker.evaluation.metrics import retrieval_metrics

        m = retrieval_metrics(recs, ks=(1, 3, 5, 10))
        m["latency"] = latency_stats(lat[name])
        results[name] = m
    out_dir = cfg.path("retrieval_dir")
    out = {"created_at": utc_now(), "split": split, "n_claims": len(pop), "limit": limit,
           "index_fingerprint": retriever.fingerprint, "config": cfg.retrieval.model_dump(), "results": results,
           "notes": "Latency = per-claim seconds on CPU for search (+ re-ranking of top "
                    f"{r.rerank_top_n}); recall uses cited reference documents present in the corpus."}
    write_json(out_dir / f"retrieval_eval_{split}.json", out)
    write_jsonl(out_dir / f"retrieval_per_claim_{split}.jsonl", per_claim)
    _write_retrieval_table(out_dir / f"retrieval_eval_{split}.csv", results)
    return out


def _write_retrieval_table(path: Path, results: dict[str, Any]) -> None:
    import csv

    cols = ["method", "recall@3", "recall@10", "evidence_doc_recall@3", "evidence_doc_recall@10",
            "recall@3_n_claims", "evidence_doc_recall@3_n_claims", "latency_mean_s", "latency_median_s", "latency_p95_s"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, m in results.items():
            w.writerow([name, m["recall@3"], m["recall@10"], m["evidence_doc_recall@3"], m["evidence_doc_recall@10"],
                        m["recall@3_n_claims"], m["evidence_doc_recall@3_n_claims"], m["latency"]["mean"],
                        m["latency"]["median"], m["latency"]["p95"]])


def format_retrieval_table(results: dict[str, Any]) -> str:
    def f(x):
        return "-" if x is None else f"{x:.3f}"

    lines = ["| method | Recall@3 | Recall@10 | EvidDocRecall@3 | EvidDocRecall@10 | mean s | p95 s |",
             "|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {f(m['recall@3'])} | {f(m['recall@10'])} | {f(m['evidence_doc_recall@3'])} | "
                     f"{f(m['evidence_doc_recall@10'])} | {f(m['latency']['mean'])} | {f(m['latency']['p95'])} |")
    return "\n".join(lines)
