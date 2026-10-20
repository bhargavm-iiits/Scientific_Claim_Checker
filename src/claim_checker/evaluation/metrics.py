"""Metric definitions with explicit denominators.

Verdict metrics use **all requested claims**: operational errors count as wrong (predicted label
``ERROR``) and verdict coverage is reported separately. Evidence precision only counts predicted
sentences in documents that carry reference evidence for the claim, because unannotated documents
are not proven irrelevant (a SciFact limitation); predictions in unannotated documents are counted
separately.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
ERROR_LABEL = "ERROR"


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def latency_stats(values: Iterable[float | None]) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "unit": "seconds"}
    arr = np.asarray(vals)
    return {"n": len(vals), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)), "unit": "seconds"}


def verdict_metrics(gold: Sequence[str], pred: Sequence[str | None]) -> dict:
    """``pred`` is None for an operational error (counted as wrong)."""
    from sklearn.metrics import precision_recall_fscore_support

    if len(gold) != len(pred):
        raise ValueError("gold and pred must have equal length")
    n = len(gold)
    y_pred = [p if p is not None else ERROR_LABEL for p in pred]
    covered = sum(1 for p in pred if p is not None)
    correct = sum(1 for g, p in zip(gold, y_pred, strict=True) if g == p)
    correct_cov = sum(1 for g, p in zip(gold, pred, strict=True) if p is not None and g == p)
    out: dict = {
        "n_claims": n,
        "unit": "claims",
        "covered": covered,
        "coverage": _safe_div(covered, n),
        "correct": correct,
        "accuracy": _safe_div(correct, n),
        "accuracy_on_covered": _safe_div(correct_cov, covered),
    }
    if n == 0:
        out.update(macro_f1=None, per_class={}, confusion={})
        return out
    p, r, f, s = precision_recall_fscore_support(gold, y_pred, labels=LABELS, average=None, zero_division=0)
    out["per_class"] = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(LABELS)
    }
    present = [lab for lab in LABELS if lab in gold]
    out["macro_f1"] = float(np.mean([out["per_class"][lab]["f1"] for lab in present])) if present else None
    out["macro_f1_labels"] = present
    out["confusion"] = confusion_matrix(gold, y_pred)
    return out


def confusion_matrix(gold: Sequence[str], pred: Sequence[str]) -> dict:
    cols = LABELS + [ERROR_LABEL]
    m = {g: dict.fromkeys(cols, 0) for g in LABELS}
    for g, p in zip(gold, pred, strict=True):
        m[g][p if p in cols else ERROR_LABEL] += 1
    return {"rows": "gold", "columns": "predicted", "matrix": m}


def recall_at_k(ranked: Sequence[str], reference: Iterable[str], k: int) -> float | None:
    ref = set(reference)
    if not ref:
        return None
    return len(ref & set(ranked[:k])) / len(ref)


def retrieval_metrics(records: Sequence[dict], ks: Sequence[int] = (3, 10)) -> dict:
    """Records need ``retrieved_doc_ids``, ``reference_doc_ids``, ``evidence_doc_ids``.

    Recall@k = mean over claims with >= 1 reference document of |ref ∩ top-k| / |ref|.
    Evidence-bearing recall uses only documents with annotated evidence (claims that have any).
    """
    out: dict = {"unit": "claims (mean of per-claim document recall)"}
    usable = [r for r in records if r.get("retrieved_doc_ids") is not None]
    for k in ks:
        ref_vals = [recall_at_k(r["retrieved_doc_ids"], r["reference_doc_ids"], k) for r in usable]
        ref_vals = [v for v in ref_vals if v is not None]
        ev_vals = [recall_at_k(r["retrieved_doc_ids"], r["evidence_doc_ids"], k) for r in usable]
        ev_vals = [v for v in ev_vals if v is not None]
        out[f"recall@{k}"] = float(np.mean(ref_vals)) if ref_vals else None
        out[f"recall@{k}_n_claims"] = len(ref_vals)
        out[f"evidence_doc_recall@{k}"] = float(np.mean(ev_vals)) if ev_vals else None
        out[f"evidence_doc_recall@{k}_n_claims"] = len(ev_vals)
    ex = [recall_at_k(r["examined_doc_ids"], r["evidence_doc_ids"], len(r["examined_doc_ids"]) or 1)
          for r in usable if r.get("examined_doc_ids") is not None]
    ex = [v for v in ex if v is not None]
    out["evidence_doc_recall_examined"] = float(np.mean(ex)) if ex else None
    out["evidence_doc_recall_examined_n_claims"] = len(ex)
    return out


def evidence_prediction(record: dict, field: str):
    """An empty retrieved pipeline still predicts zero citations, not 'stage absent'.

    Older saved records used null when a relevance gate rejected every document.
    Normalize those records on read without changing the historical predictions.
    Evidence selection remains absent when the full-abstract mode skips that stage.
    """
    prediction = record.get(field)
    if (field == "cited" and prediction is None and record.get("status") == "ok"
            and record.get("examined_doc_ids") == [] and record.get("retrieved_doc_ids") is not None):
        return {}
    return prediction


def evidence_metrics(records: Sequence[dict], field: str = "selected") -> dict:
    """Micro sentence-level precision/recall against reference evidence.

    ``record[field]``: ``{doc_id: [sentence ids]}`` predicted; ``record["evidence_by_doc"]`` gold.

    Recall denominator: all reference evidence sentences of the evaluated claims (end-to-end, so
    retrieval misses reduce it). Operational errors stay in the evaluation population: a claim
    whose check failed (``status != "ok"``, no prediction) contributes its reference sentences to
    the recall denominator with zero predicted sentences. A successful record with no prediction
    for ``field`` (the stage does not apply, e.g. judge-only) is not evaluated.

    Precision denominator: predicted sentences inside documents that have reference evidence for
    that claim (errored claims predict nothing, so precision is unaffected).
    """
    tp = pred_in_annotated = pred_unannotated = gold_total = 0
    pairs = 0
    claims = 0
    errored_claims = errored_pairs = errored_gold = 0
    for r in records:
        pred = evidence_prediction(r, field)
        gold = {d: set(v) for d, v in (r.get("evidence_by_doc") or {}).items() if v}
        if pred is None:
            if r.get("status", "ok") != "ok":  # operational error: counted, nothing predicted
                errored_claims += 1
                errored_pairs += len(gold)
                n_gold = sum(len(v) for v in gold.values())
                errored_gold += n_gold
                gold_total += n_gold
            continue
        claims += 1
        gold_total += sum(len(v) for v in gold.values())
        for d, ids in pred.items():
            ids = set(ids)
            if d in gold:
                pairs += 1
                pred_in_annotated += len(ids)
                tp += len(ids & gold[d])
            else:
                pred_unannotated += len(ids)
    return {
        "unit": "evidence sentences (micro)",
        "claims": claims,
        "claims_in_recall_denominator": claims + errored_claims,
        "errored_claims_included": errored_claims,
        "errored_annotated_pairs_included": errored_pairs,
        "errored_reference_sentences_included": errored_gold,
        "annotated_pairs_with_predictions": pairs,
        "true_positive_sentences": tp,
        "predicted_sentences_in_annotated_docs": pred_in_annotated,
        "predicted_sentences_in_unannotated_docs": pred_unannotated,
        "reference_sentences": gold_total,
        "precision": _safe_div(tp, pred_in_annotated),
        "precision_denominator": pred_in_annotated,
        "recall": _safe_div(tp, gold_total),
        "recall_denominator": gold_total,
        "note": "Unannotated documents are not proven irrelevant; their selections are excluded from precision. "
                "Claims with an operational error count in the recall denominator with zero predicted sentences.",
    }


def quote_validity(records: Sequence[dict], corpus: dict) -> dict:
    total = valid = 0
    for r in records:
        for q in r.get("quotes") or []:
            total += 1
            doc = corpus.get(q["doc_id"])
            if doc is not None and 0 <= q["sentence_id"] < len(doc.sentences) and doc.sentences[q["sentence_id"]] == q["quote"]:
                valid += 1
    return {"unit": "quotations", "quotes": total, "valid": valid, "validity": _safe_div(valid, total)}


def operational_metrics(records: Sequence[dict]) -> dict:
    n = len(records)
    errors = [r for r in records if r["status"] != "ok"]
    ok = n - len(errors)
    conflicts = sum(1 for r in records if r["status"] == "ok" and r.get("conflict"))
    by_type: dict[str, int] = {}
    for r in errors:
        t = (r.get("error") or {}).get("error_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    return {
        "unit": "claims",
        "requested": n,
        "ok": ok,
        "errors": len(errors),
        "operational_failure_rate": _safe_div(len(errors), n),
        "errors_by_type": by_type,
        "conflicts": conflicts,
        "conflict_rate_of_ok": _safe_div(conflicts, ok),
        "conflict_rate_of_requested": _safe_div(conflicts, n),
    }


# Per-record timings are rounded to 4 decimals, so a fresh computation's nested sub-steps can sum to a
# hair more than the enclosing stage; anything beyond this tolerance cannot be a fresh measurement.
_NESTED_TIMING_TOLERANCE_S = 0.001


def subtimings_exceed_stage(timings: dict | None, stage: str = "retrieve") -> bool:
    """True if the ``<stage>.*`` sub-step timings sum to more than the stage's own wall time.

    The sub-steps run *inside* the stage, so for a fresh computation their sum cannot exceed it; a
    larger sum means the sub-timings were copied from an earlier (cached) computation. Used to
    recognise retrieval-cache hits in records saved before the explicit flag existed.
    """
    timings = timings or {}
    if stage not in timings:
        return False
    sub = sum(float(v) for k, v in timings.items() if k.startswith(f"{stage}."))
    return sub > float(timings[stage]) + _NESTED_TIMING_TOLERANCE_S


def cache_metrics(records: Sequence[dict]) -> dict:
    """LLM-cache and in-process retrieval-cache usage of one variant.

    LLM: model requests sent (``llm_calls``, schema repairs included) vs answered from the on-disk
    cache (``llm_cache_hits``); hit rate = hits / (calls + hits).

    Retrieval: one lookup per claim whose retrieval ran (``retrieved_doc_ids`` present; oracle,
    judge-only and claims that failed before retrieval are not applicable). Each record's
    ``retrieval_cache_hit`` is True (served from cache: ``timings["retrieve"]`` is the real near-zero
    lookup and ``total`` excludes the search), False (computed), or missing = **unknown** (records
    saved before the flag existed). Hit rate = hits / (hits + misses), unknowns excluded. Unknown
    records whose ``retrieve.*`` sub-timings exceed their ``retrieve`` stage time are counted in
    ``unknown_with_copied_subtimings``: those sub-timings are copies and the search was cached.
    """
    llm_calls = sum(int(r.get("llm_calls") or 0) for r in records)
    llm_hits = sum(int(r.get("llm_cache_hits") or 0) for r in records)
    applicable = [r for r in records if r.get("retrieved_doc_ids") is not None]
    hits = misses = unknown = copied = 0
    for r in applicable:
        flag = r.get("retrieval_cache_hit")
        if flag is True:
            hits += 1
        elif flag is False:
            misses += 1
        else:
            unknown += 1
            copied += subtimings_exceed_stage(r.get("timings"))
    return {
        "llm": {
            "unit": "model requests",
            "model_calls": llm_calls,
            "cache_hits": llm_hits,
            "cache_hit_rate": _safe_div(llm_hits, llm_calls + llm_hits),
        },
        "retrieval": {
            "unit": "claims whose retrieval ran (one lookup each)",
            "claims_with_retrieval": len(applicable),
            "cache_hits": hits,
            "cache_misses": misses,
            "unknown": unknown,
            "cache_hit_rate": _safe_div(hits, hits + misses),
            "unknown_with_copied_subtimings": copied,
            "note": "A cache hit's `retrieve` time is the real near-zero lookup, so `total` latency excludes "
                    "the search; its original sub-step seconds are kept in `retrieve_original_compute_s`, "
                    "not in timings. Unknown = saved before the flag existed.",
        },
    }


def stage_latency(records: Sequence[dict]) -> dict:
    stages: dict[str, list[float]] = {}
    for r in records:
        for k, v in (r.get("timings") or {}).items():
            stages.setdefault(k, []).append(v)
    return {k: latency_stats(v) for k, v in sorted(stages.items())}
