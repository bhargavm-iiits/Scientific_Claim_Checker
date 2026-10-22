"""Turn run predictions into metrics.json, results.csv, confusion_matrices.json,
failure_cases.csv and report.md."""

from __future__ import annotations

import csv
from pathlib import Path

from claim_checker.data.store import iter_jsonl, read_json, write_json
from claim_checker.evaluation.metrics import (
    cache_metrics,
    evidence_metrics,
    evidence_prediction,
    latency_stats,
    operational_metrics,
    quote_validity,
    retrieval_metrics,
    stage_latency,
    verdict_metrics,
)
from claim_checker.evaluation.runner import EVENTS, MANIFEST, PREDICTIONS

FAILURE_ORDER = ["CONTRADICT", "SUPPORT", "NEI"]


def load_predictions(run_dir: Path) -> dict[str, list[dict]]:
    """Latest record per (variant, claim_id), grouped by variant in first-seen order."""
    latest: dict[tuple[str, str], dict] = {}
    for r in iter_jsonl(run_dir / PREDICTIONS):
        latest[(r["variant"], r["claim_id"])] = r
    out: dict[str, list[dict]] = {}
    for (v, _), r in latest.items():
        out.setdefault(v, []).append(r)
    return out


DOWNGRADE_FLAG = "downgraded_no_citation"


def downgrades_from_events(run_dir: Path) -> set[tuple[str, str]]:
    """(variant, claim_id) pairs whose saved stage events contain a ``downgraded_no_citation`` flag.

    Records written before the single-prompt downgrade was propagated into ``predictions.jsonl``
    (e.g. the frozen ``final_v1`` run) carry ``downgraded=False`` for single-prompt claims even when
    the answer was downgraded; the event log is the authoritative trace. Only the last event row per
    claim (the attempt saved with its prediction) is used. Missing file -> empty set."""
    path = run_dir / EVENTS
    if not path.exists():
        return set()
    last: dict[tuple[str, str], list] = {}
    for row in iter_jsonl(path):
        last[(row.get("variant"), row.get("claim_id"))] = row.get("events") or []
    return {key for key, events in last.items()
            if any((e.get("data") or {}).get("flag") == DOWNGRADE_FLAG for e in events)}


def is_downgraded(r: dict, from_events: set[tuple[str, str]] | None = None) -> bool:
    return bool(r.get("downgraded")) or (r.get("variant"), r.get("claim_id")) in (from_events or set())


def variant_metrics(records: list[dict], corpus: dict | None, strategy: str | None,
                    downgraded_events: set[tuple[str, str]] | None = None) -> dict:
    gold = [r["gold_label"] for r in records]
    pred = [r["verdict"] if r["status"] == "ok" else None for r in records]
    m = {
        "verdict": verdict_metrics(gold, pred),
        "operational": operational_metrics(records),
        "latency_total": latency_stats(r["timings"].get("total") for r in records),
        "latency_stages": stage_latency(records),
    }
    if any(r.get("retrieved_doc_ids") is not None for r in records):
        m["retrieval"] = retrieval_metrics(records)
    if any(r.get("selected") is not None for r in records):
        m["evidence_selected"] = evidence_metrics(records, "selected")
    if any(evidence_prediction(r, "cited") is not None for r in records):
        m["evidence_cited"] = evidence_metrics(records, "cited")
    if corpus is not None:
        m["quotes"] = quote_validity(records, corpus)
    downgraded = [r["claim_id"] for r in records if is_downgraded(r, downgraded_events)]
    m["downgraded_claims"] = len(downgraded)
    m["downgraded_claim_ids"] = downgraded
    m["cache"] = cache_metrics(records)
    return m


def failure_stage(r: dict) -> str:
    """Earliest-failing-stage heuristic."""
    if r["status"] != "ok":
        return "operational"
    gold = r["gold_label"]
    ev_docs = set(r.get("evidence_doc_ids") or [])
    examined = r.get("examined_doc_ids")
    if examined is None:
        return "judgement error (no retrieval: judge-only baseline)"
    if gold in ("SUPPORT", "CONTRADICT"):
        if not ev_docs & set(examined):
            return "retrieval miss"
        sel = r.get("selected")
        if sel is not None:
            hit = any(set(sel.get(d, [])) & set(r["evidence_by_doc"].get(d, [])) for d in ev_docs & set(examined))
            if not hit:
                return "evidence miss"
        return "judgement error"
    return "judgement error (false decisive verdict)" if not r.get("conflict") else "judgement error (conflict)"


def select_failures(by_variant: dict[str, list[dict]], primary: str = "full", minimum: int = 10) -> list[dict]:
    failures = []
    order = [primary] + [v for v in by_variant if v != primary] if primary in by_variant else list(by_variant)
    for v in order:
        wrong = [r for r in by_variant[v] if not r["correct"]]
        buckets = {lab: [r for r in wrong if r["gold_label"] == lab] for lab in FAILURE_ORDER}
        # Round-robin across classes with CONTRADICT first (and given a double share).
        picked: list[dict] = []
        while any(buckets.values()):
            for lab in ["CONTRADICT", "CONTRADICT", "SUPPORT", "NEI"]:
                if buckets[lab]:
                    picked.append(buckets[lab].pop(0))
        for r in picked:
            failures.append({**r, "variant": v, "failure_stage": failure_stage(r)})
        if v == primary and len(failures) >= minimum:
            break
        if len(failures) >= max(minimum, 30):
            break
    return failures


def _fmt(x, pct=False) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def results_rows(metrics: dict, variants_meta: dict) -> list[dict]:
    rows = []
    for v, m in metrics.items():
        vm, op = m["verdict"], m["operational"]
        pc = vm.get("per_class", {})
        ret = m.get("retrieval", {})
        evs = m.get("evidence_selected") or {}
        evc = m.get("evidence_cited") or {}
        rows.append({
            "variant": v,
            "description": variants_meta.get(v, {}).get("description", ""),
            "n_claims": vm["n_claims"],
            "coverage": vm["coverage"],
            "accuracy": vm["accuracy"],
            "macro_f1": vm.get("macro_f1"),
            "f1_support": pc.get("SUPPORT", {}).get("f1"),
            "f1_contradict": pc.get("CONTRADICT", {}).get("f1"),
            "f1_nei": pc.get("NEI", {}).get("f1"),
            "recall@3": ret.get("recall@3"),
            "recall@10": ret.get("recall@10"),
            "evidence_doc_recall_examined": ret.get("evidence_doc_recall_examined"),
            "evidence_precision_selected": evs.get("precision"),
            "evidence_recall_selected": evs.get("recall"),
            "evidence_precision_cited": evc.get("precision"),
            "evidence_recall_cited": evc.get("recall"),
            "quote_validity": (m.get("quotes") or {}).get("validity"),
            "conflict_rate": op["conflict_rate_of_requested"],
            "failure_rate": op["operational_failure_rate"],
            "latency_mean_s": m["latency_total"]["mean"],
            "latency_median_s": m["latency_total"]["median"],
            "latency_p95_s": m["latency_total"]["p95"],
        })
    return rows


def write_csv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    cols = cols or (list(rows[0]) if rows else [])
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


FAILURE_COLS = ["variant", "claim_id", "claim", "gold_label", "verdict", "status", "failure_stage", "conflict",
                "reference_doc_ids", "evidence_doc_ids", "examined_doc_ids", "gold_evidence", "selected", "cited",
                "doc_verdicts", "error", "explanation"]


def analyse_run(run_dir: str | Path, corpus: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / MANIFEST)
    by_variant = load_predictions(run_dir)
    variants_meta = manifest.get("variants", {})
    from_events = downgrades_from_events(run_dir)
    metrics = {v: variant_metrics(recs, corpus, variants_meta.get(v, {}).get("strategy"), from_events)
               for v, recs in by_variant.items()}
    write_json(run_dir / "metrics.json", {"run_id": manifest.get("run_id"), "identity": manifest.get("identity"),
                                          "metrics": metrics})
    write_json(run_dir / "confusion_matrices.json", {v: m["verdict"].get("confusion") for v, m in metrics.items()})
    rows = results_rows(metrics, variants_meta)
    write_csv(run_dir / "results.csv", rows)
    failures = select_failures(by_variant)
    fail_rows = []
    for r in failures:
        fail_rows.append({**r, "gold_evidence": r.get("evidence_by_doc"),
                          "error": (r.get("error") or {}).get("message") if r.get("error") else ""})
    write_csv(run_dir / "failure_cases.csv", fail_rows, FAILURE_COLS)
    report = render_report(manifest, rows, metrics, failures)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return {"metrics": metrics, "rows": rows, "failures": failures, "report": report}


def render_report(manifest: dict, rows: list[dict], metrics: dict, failures: list[dict]) -> str:
    ident = manifest.get("identity", {})
    L = [f"# Evaluation report - run `{manifest.get('run_id')}`", "",
         f"- Suite: **{ident.get('suite')}** (split `{ident.get('split')}`), claims requested per variant: "
         f"**{ident.get('population_size')}**" + (f" (limit {manifest.get('limit')})" if manifest.get("limit") else ""),
         f"- Gold label counts: {manifest.get('gold_label_counts')}",
         f"- Model: `{ident.get('model', {}).get('name')}` digest `{ident.get('model', {}).get('digest')}`",
         f"- Index fingerprint: `{ident.get('index_fingerprint')}`; prompt hashes: `{ident.get('prompt_hashes')}`",
         f"- Created: {manifest.get('created_at')}", "",
         "Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong "
         "(coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. "
         "Evidence precision excludes sentences selected in documents without reference annotations "
         "(unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational "
         "error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.",
         "", "## Results", "",
         "| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join([r["variant"], str(r["n_claims"])] + [_fmt(r[c]) for c in (
            "coverage", "accuracy", "macro_f1", "f1_support", "f1_contradict", "f1_nei", "recall@3", "recall@10",
            "evidence_precision_selected", "evidence_recall_selected", "quote_validity", "conflict_rate",
            "failure_rate", "latency_mean_s", "latency_p95_s")]) + " |")
    L += ["", "## Confusion matrices (rows = gold, columns = predicted)", ""]
    for v, m in metrics.items():
        cm = (m["verdict"].get("confusion") or {}).get("matrix")
        if not cm:
            continue
        L += [f"**{v}**", "", "| gold \\ pred | SUPPORT | CONTRADICT | NEI | ERROR |", "|---|---|---|---|---|"]
        for g, row in cm.items():
            L.append(f"| {g} | {row['SUPPORT']} | {row['CONTRADICT']} | {row['NEI']} | {row['ERROR']} |")
        L.append("")
    L += ["## Evidence metrics (denominators)", ""]
    for v, m in metrics.items():
        for key in ("evidence_selected", "evidence_cited"):
            e = m.get(key)
            if e:
                L.append(f"- {v} / {key}: precision {_fmt(e['precision'])} = {e['true_positive_sentences']}/"
                         f"{e['predicted_sentences_in_annotated_docs']} sentences; recall {_fmt(e['recall'])} = "
                         f"{e['true_positive_sentences']}/{e['reference_sentences']} sentences "
                         f"({e.get('claims_in_recall_denominator', e['claims'])} claims, including "
                         f"{e.get('errored_claims_included', 0)} with an operational error = "
                         f"{e.get('errored_reference_sentences_included', 0)} reference sentences in "
                         f"{e.get('errored_annotated_pairs_included', 0)} annotated claim-document pairs); "
                         f"{e['predicted_sentences_in_unannotated_docs']} predicted sentences in unannotated documents "
                         "(excluded from precision).")
    L += ["", "## Operational summary", ""]
    for v, m in metrics.items():
        op = m["operational"]
        ids = m.get("downgraded_claim_ids") or []
        L.append(f"- {v}: {op['ok']}/{op['requested']} ok, {op['errors']} errors {op['errors_by_type']}, "
                 f"{op['conflicts']} conflicts, {m.get('downgraded_claims', 0)} claims with a downgraded judgement"
                 + (f" (decisive answer without a valid citation -> NEI; claims {', '.join(map(str, ids))})"
                    if ids else "") + ".")
    L += render_cache_section(metrics)
    L += ["", f"## Failure cases ({len(failures)} selected; CONTRADICT emphasised)", "",
          "Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.",
          "", "| variant | claim | gold | pred | stage |", "|---|---|---|---|---|"]
    for f in failures:
        claim = f["claim"].replace("|", "/")
        L.append(f"| {f['variant']} | {f['claim_id']}: {claim[:90]} | {f['gold_label']} | {f['verdict']} | {f['failure_stage']} |")
    L += ["", "Human-reviewed analysis of these failures belongs in `failure_analysis.md`.", ""]
    return "\n".join(L)


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f} %"


def render_cache_section(metrics: dict) -> list[str]:
    """Per-variant LLM-cache and retrieval-cache usage, plus a one-line latency caveat when any
    retrieval was (or, for records saved without the flag, demonstrably was) served from cache.
    Tolerates metrics computed without a ``cache`` entry (older runs)."""
    L = ["", "## Cache usage (latency context)", "",
         "| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | "
         "misses | unknown | retrieval hit rate |",
         "|---|---|---|---|---|---|---|---|---|"]
    known_hits: dict[str, int] = {}
    inferred: dict[str, int] = {}
    unknown_total = 0
    for v, m in metrics.items():
        c = m.get("cache") or {}
        llm, ret = c.get("llm") or {}, c.get("retrieval") or {}
        n_ret = ret.get("claims_with_retrieval", 0)
        if n_ret:
            ret_cells = [str(n_ret), str(ret.get("cache_hits", 0)), str(ret.get("cache_misses", 0)),
                         str(ret.get("unknown", 0)), _pct(ret.get("cache_hit_rate"))]
        else:
            ret_cells = ["0", "-", "-", "-", "n/a (no retrieval)"]
        L.append("| " + " | ".join([v, str(llm.get("model_calls", "-")), str(llm.get("cache_hits", "-")),
                                    _pct(llm.get("cache_hit_rate"))] + ret_cells) + " |")
        if ret.get("cache_hits"):
            known_hits[v] = ret["cache_hits"]
        if ret.get("unknown_with_copied_subtimings"):
            inferred[v] = ret["unknown_with_copied_subtimings"]
        unknown_total += ret.get("unknown", 0) or 0
    L.append("")
    if known_hits or inferred:
        parts = []
        if known_hits:
            parts.append(f"{sum(known_hits.values())} claim(s) were served from the in-process retrieval cache "
                         f"({', '.join(f'{v} {n}' for v, n in known_hits.items())})")
        if inferred:
            parts.append(f"{sum(inferred.values())} claim(s) without a cache flag have `retrieve.*` sub-timings "
                         f"larger than their own `retrieve` stage, i.e. copied from an earlier cached computation "
                         f"({', '.join(f'{v} {n}' for v, n in inferred.items())})")
        L.append("- **Caveat (retrieval cache):** " + "; ".join(parts) + ". For these claims `retrieve` is the "
                 "near-zero lookup actually measured and `total` latency excludes the search (about one "
                 "uncached retrieval per claim); compare latency only between variants with the same cache state.")
    if unknown_total:
        L.append(f"- Retrieval-cache status is unknown for {unknown_total} claim(s) recorded before the "
                 "`retrieval_cache_hit` flag existed.")
    L.append("- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a "
             "variant faster than a cold run.")
    return L
