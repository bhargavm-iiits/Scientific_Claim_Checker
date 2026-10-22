"""Reproducible cold-latency benchmark for the product configuration.

    python -m claim_checker latency --suite smoke --limit 10 --no-cache

Runs the full pipeline (hybrid + re-rank, evidence agent, k = 3 by default) on the first ``limit``
claims of a frozen split, in the current (fresh) process, and writes
``artifacts/latency/<name>.json`` plus a short ``.md`` summary. With ``--no-cache`` the LLM disk
cache is disabled, so every model request really goes to Ollama; the in-process retrieval cache is
new (one ``ClaimChecker`` per benchmark) and every measured claim is distinct, so no retrieval is
served from cache either. Both facts are verified per claim and recorded.

Warm-up is measured separately and excluded from the per-claim statistics:

* ``resources``: :meth:`ClaimChecker.warmup` (corpus, BM25, dense index, BGE encoder, cross-encoder,
  Ollama model check, one retrieval);
* ``first_claim``: one full pipeline run on a warm-up claim that is not in the measured set (loads
  the Ollama model into GPU memory if it was not resident, first CUDA kernels, prompt templates).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from claim_checker.config import AppConfig
from claim_checker.data.store import write_json
from claim_checker.errors import ConfigError
from claim_checker.evaluation.metrics import latency_stats
from claim_checker.evaluation.runner import environment_info, population
from claim_checker.schemas import utc_now

# Stages reported in the summary table, in pipeline order (others are kept in the JSON).
SUMMARY_STAGES = ("validate_input", "retrieve", "retrieve.bm25", "retrieve.dense", "retrieve.fuse",
                  "retrieve.rerank", "select_evidence", "judge", "aggregate", "supervise", "explain", "total")
FALLBACK_WARMUP_CLAIM = "Vitamin D supplementation increases bone mineral density in older adults."


def latency_dir(cfg: AppConfig) -> Path:
    return Path(cfg.root) / "artifacts" / "latency"


def _gpu_snapshot() -> list[dict] | None:
    """GPU name / utilisation / memory from ``nvidia-smi`` (None if unavailable)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            rows.append({"name": parts[0], "utilization_pct": parts[1], "memory_used_mib": parts[2],
                         "memory_total_mib": parts[3]})
    return rows


def _ollama_loaded_models(host: str) -> list[dict] | None:
    """Models resident in Ollama memory (``/api/ps``) - shows whether warm-up had to load the model."""
    try:
        import httpx

        models = httpx.get(f"{host}/api/ps", timeout=5).json().get("models", [])
        return [{"name": m.get("name"), "size_vram": m.get("size_vram"), "expires_at": m.get("expires_at")}
                for m in models]
    except Exception:
        return None


def _measure(checker, pop, k: int, log, label: str, judge_opts: dict | None = None) -> list[dict]:
    claims = []
    for i, g in enumerate(pop, start=1):
        res = checker.check(g.claim, k=k, claim_id=g.claim_id, **(judge_opts or {}))
        rec = {
            "claim_id": g.claim_id, "gold_label": g.label, "status": res.status, "verdict": res.verdict,
            "error_type": res.error.error_type if res.error else None,
            "timings": res.timings,
            "llm_calls": res.metadata.get("llm_calls"), "llm_cache_hits": res.metadata.get("llm_cache_hits"),
            "retrieval_cache_hit": res.metadata.get("retrieval_cache_hit"),
            "documents_examined": len(res.documents),
            "sentences_selected": sum(len(d.selected_sentence_ids or []) for d in res.documents),
            "judge_model_calls_skipped": sum(1 for d in res.documents if d.judgement and d.judgement.skipped_model),
            "prompt_set": res.metadata.get("prompt_set"),
            "judge_thinking": res.metadata.get("judge_thinking"),
        }
        claims.append(rec)
        log(f"  [{label}] {i}/{len(pop)} claim {g.claim_id}: {res.status} {res.verdict} total "
            f"{res.timings.get('total', 0):.2f} s (llm calls {rec['llm_calls']}, llm cache hits {rec['llm_cache_hits']}, "
            f"retrieval cache hit {rec['retrieval_cache_hit']})")
    return claims


def _summary(claims: list[dict]) -> dict:
    ok = [c for c in claims if c["status"] == "ok"]
    names = sorted({s for c in ok for s in c["timings"]},
                   key=lambda s: (SUMMARY_STAGES.index(s) if s in SUMMARY_STAGES else len(SUMMARY_STAGES), s))
    return {
        "errors": len(claims) - len(ok),
        "stages_seconds": {s: latency_stats(c["timings"].get(s) for c in ok) for s in names},
        "cache_check": {"llm_cache_hits_total": sum(c["llm_cache_hits"] or 0 for c in claims),
                        "retrieval_cache_hits": sum(1 for c in claims if c["retrieval_cache_hit"]),
                        "llm_calls_total": sum(c["llm_calls"] or 0 for c in claims)},
    }


def run_latency_benchmark(cfg: AppConfig, checker, *, split: str = "smoke", limit: int = 10, k: int = 3,
                          no_cache: bool = True, repeat: bool = False, command: str | None = None,
                          log=print, prompt_set: str | None = None, judge_think: bool | None = None) -> dict:
    """Measure warm-up and per-claim, per-stage latency. Returns the JSON-ready result."""
    if limit < 1:
        raise ConfigError("--limit must be at least 1", stage="latency")
    if no_cache and getattr(checker.llm, "cache_dir", None) is not None:
        raise ConfigError("LLM disk cache is enabled but a cold (--no-cache) benchmark was requested.",
                          stage="latency", hint="Construct the checker with model.cache_enabled = false.")
    # Judge configuration: explicit, else the product default (what ClaimChecker.check() uses).
    prompt_set = prompt_set or cfg.product.prompt_set
    judge_think = cfg.product.judge_think if judge_think is None else bool(judge_think)
    judge_opts = {"prompt_set": prompt_set, "judge_think": judge_think}
    pool, _ = population(cfg, split, None)
    pop = pool[:limit]
    warm_gold = pool[limit] if len(pool) > limit else None
    warm_claim = warm_gold.claim if warm_gold else FALLBACK_WARMUP_CLAIM

    gpu_before = _gpu_snapshot()
    ollama_before = _ollama_loaded_models(cfg.model.host)
    t0 = time.perf_counter()
    checker.warmup()
    warm_resources = time.perf_counter() - t0
    log(f"warm-up (resources): {warm_resources:.2f} s")
    t0 = time.perf_counter()
    first = checker.check(warm_claim, k=k, **judge_opts)
    warm_first = time.perf_counter() - t0
    if first.status != "ok":
        raise ConfigError(f"Warm-up claim failed: {first.error.message if first.error else first.status}",
                          stage="latency", hint=first.error.hint if first.error else None)
    log(f"warm-up (first claim): {warm_first:.2f} s")

    claims = _measure(checker, pop, k, log, "measured", judge_opts)
    summary = _summary(claims)
    repeat_out = None
    if repeat:  # same claims again, same process: retrieval (and, if enabled, LLM) answers are cached
        rep = _measure(checker, pop, k, log, "repeat", judge_opts)
        repeat_out = {"description": "second pass over the same claims in the same process", **_summary(rep),
                      "claims": rep}
    gpu_after = _gpu_snapshot()
    try:
        digest = checker.llm.model_digest()
    except Exception:
        digest = None
    return {
        "benchmark": "cold_full_pipeline" if no_cache else "full_pipeline_with_llm_cache",
        "created_at": utc_now(),
        "command": command,
        "process": "fresh Python process; one ClaimChecker (new in-process retrieval cache)",
        "split": split, "claim_ids": [g.claim_id for g in pop], "n_claims": len(pop),
        "configuration": {"k": k, "retrieval_method": "hybrid", "rerank": True, "evidence_mode": "evidence",
                          "llm_disk_cache": not no_cache, "model": cfg.model.name, "model_digest": digest,
                          "think": cfg.model.think, "temperature": cfg.model.temperature,
                          "prompt_set": prompt_set, "judge_think": judge_think,
                          "judge_think_budget_tokens": cfg.judge_v2.think_budget_tokens if judge_think else None,
                          "num_ctx": cfg.model.context_tokens, "embedding_device": cfg.retrieval.embedding_device,
                          "index_fingerprint": getattr(checker.retriever, "fingerprint", None)},
        "warmup_seconds": {
            "resources": round(warm_resources, 4),
            "first_claim": round(warm_first, 4),
            "total": round(warm_resources + warm_first, 4),
            "first_claim_id": warm_gold.claim_id if warm_gold else None,
            "first_claim_timings": first.timings,
            "ollama_models_resident_before": ollama_before,
        },
        **summary,
        "claims": claims,
        "repeat": repeat_out,
        "gpu_before": gpu_before, "gpu_after": gpu_after,
        "environment": environment_info(),
        "notes": ("Per-claim statistics exclude warm-up. Stage timings are wall-clock seconds of each "
                  "LangGraph node; retrieve.* are sub-steps of retrieve. The judge is skipped without a "
                  "model call for an abstract with no selected evidence, so judge time varies with the "
                  "evidence agent's output. Measured on one laptop with a local Ollama server; other "
                  "GPU work at the same time slows the model stages."),
    }


def _f(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def render_latency_md(out: dict, json_name: str) -> str:
    w = out["warmup_seconds"]
    cfg = out["configuration"]
    L = [f"# Latency benchmark - `{out['benchmark']}`", "",
         f"- Created: {out['created_at']}; command: `{out.get('command') or '-'}`",
         f"- {out['n_claims']} `{out['split']}` claims ({', '.join(out['claim_ids'])}); full pipeline, hybrid + "
         f"re-rank, evidence agent, k = {cfg['k']}; LLM disk cache {'on' if cfg['llm_disk_cache'] else '**off**'}; "
         f"model `{cfg['model']}` (think={cfg['think']}, temperature {cfg['temperature']}); judge prompt set "
         f"`{cfg.get('prompt_set', 'v1')}`, judge thinking {cfg.get('judge_think', False)}"
         + (f" (budget {cfg['judge_think_budget_tokens']} tokens)" if cfg.get("judge_think_budget_tokens") else "")
         + "; embeddings on "
         f"`{cfg['embedding_device']}`.",
         f"- Cache check: {out['cache_check']['llm_cache_hits_total']} LLM cache hits, "
         f"{out['cache_check']['retrieval_cache_hits']} retrieval cache hits, "
         f"{out['cache_check']['llm_calls_total']} model calls; {out['errors']} errors.",
         f"- Warm-up (excluded below): resources {_f(w['resources'])} s + first claim {_f(w['first_claim'])} s "
         f"= **{_f(w['total'])} s** per process.", "",
         "| stage | n | mean s | median s | p95 s |", "|---|---|---|---|---|"]
    for s, st in out["stages_seconds"].items():
        if s in SUMMARY_STAGES:
            name = f"**{s}**" if s == "total" else s
            L.append(f"| {name} | {st['n']} | {_f(st['mean'])} | {_f(st['median'])} | {_f(st['p95'])} |")
    rep = out.get("repeat")
    if rep:
        t = rep["stages_seconds"].get("total", {})
        cc = rep["cache_check"]
        L += ["", f"Repeat pass (same claims, same process): total mean {_f(t.get('mean'))} s, median "
              f"{_f(t.get('median'))} s, p95 {_f(t.get('p95'))} s; {cc['llm_cache_hits_total']} LLM cache hits, "
              f"{cc['llm_calls_total']} model calls, {cc['retrieval_cache_hits']} retrieval cache hits."]
    gpus = out.get("gpu_before") or []
    if gpus:
        L += ["", "GPU before the run: " + "; ".join(
            f"{g['name']} {g['utilization_pct']} % busy, {g['memory_used_mib']} / {g['memory_total_mib']} MiB" for g in gpus)]
    L += ["", out["notes"], "", f"Raw data: `{json_name}`.", ""]
    return "\n".join(L)


def write_latency_outputs(cfg: AppConfig, out: dict, name: str) -> tuple[Path, Path]:
    d = latency_dir(cfg)
    json_path, md_path = d / f"{name}.json", d / f"{name}.md"
    write_json(json_path, out)
    md_path.write_text(render_latency_md(out, f"artifacts/latency/{json_path.name}"), encoding="utf-8")
    return json_path, md_path


def command_line(argv: list[str] | None) -> str:
    args = argv if argv is not None else sys.argv[1:]
    return "python -m claim_checker " + " ".join(args)
