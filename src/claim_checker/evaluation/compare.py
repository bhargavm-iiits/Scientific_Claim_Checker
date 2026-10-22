"""Paired comparison of two evaluated variants on the same claims (e.g. final_v1/full vs
final_v2/full_v2): accuracy, macro-F1 and per-class F1 with bootstrap CIs, the paired accuracy
difference with a bootstrap CI, McNemar's exact test, error-type counts and judge-consistency counts.

    python -m claim_checker compare --a final_v1:full --b final_v2:full_v2 --name final_v1_vs_v2

Writes ``artifacts/comparisons/<name>.json`` and ``.md``. Only saved run files are read.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np

from claim_checker.data.store import iter_jsonl, write_json
from claim_checker.evaluation.analysis import load_predictions
from claim_checker.evaluation.metrics import LABELS, verdict_metrics
from claim_checker.evaluation.runner import EVENTS, PAIRS
from claim_checker.pipeline.consistency import _NEGATES_SUPPORT
from claim_checker.schemas import utc_now


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of the b vs c discordant pairs at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def _pred(r: dict) -> str | None:
    return r["verdict"] if r["status"] == "ok" else None


def _macro_f1(gold, pred) -> float:
    return verdict_metrics(gold, pred)["macro_f1"]


def bootstrap(gold: list[str], pa: list, pb: list, n_boot: int = 10000, seed: int = 42) -> dict:
    """Percentile 95% CIs, resampling claims (paired: the same resample for both systems)."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    g = np.array(gold, dtype=object)
    a = np.array([x if x is not None else "ERROR" for x in pa], dtype=object)
    b = np.array([x if x is not None else "ERROR" for x in pb], dtype=object)
    acc_a, acc_b, f1_a, f1_b = [], [], [], []
    from sklearn.metrics import f1_score

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gi, ai, bi = g[idx], a[idx], b[idx]
        acc_a.append(float(np.mean(gi == ai)))
        acc_b.append(float(np.mean(gi == bi)))
        present = [lab for lab in LABELS if lab in set(gi)]
        f1_a.append(float(f1_score(gi, ai, labels=present, average="macro", zero_division=0)))
        f1_b.append(float(f1_score(gi, bi, labels=present, average="macro", zero_division=0)))
    acc_a, acc_b, f1_a, f1_b = map(np.array, (acc_a, acc_b, f1_a, f1_b))

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    return {"n_boot": n_boot, "seed": seed, "method": "percentile, paired resampling of claims",
            "accuracy_a": ci(acc_a), "accuracy_b": ci(acc_b), "accuracy_diff_b_minus_a": ci(acc_b - acc_a),
            "macro_f1_a": ci(f1_a), "macro_f1_b": ci(f1_b), "macro_f1_diff_b_minus_a": ci(f1_b - f1_a),
            "p_accuracy_diff_le_0": float(np.mean(acc_b - acc_a <= 0)),
            "p_macro_f1_diff_le_0": float(np.mean(f1_b - f1_a <= 0))}


def error_counts(gold: list[str], pred: list) -> dict:
    fs = sum(1 for g, p in zip(gold, pred, strict=True) if p == "SUPPORT" and g != "SUPPORT")
    return {
        "false_support": fs,
        "nei_to_support": sum(1 for g, p in zip(gold, pred, strict=True) if g == "NEI" and p == "SUPPORT"),
        "contradict_to_support": sum(1 for g, p in zip(gold, pred, strict=True) if g == "CONTRADICT" and p == "SUPPORT"),
        "false_contradict": sum(1 for g, p in zip(gold, pred, strict=True) if p == "CONTRADICT" and g != "CONTRADICT"),
        "support_to_nei": sum(1 for g, p in zip(gold, pred, strict=True) if g == "SUPPORT" and p == "NEI"),
        "errors_total": sum(1 for g, p in zip(gold, pred, strict=True) if g != p),
        "operational_errors": sum(1 for p in pred if p is None),
    }


def judge_consistency(run_dir: Path, variant: str, claim_ids: set[str]) -> dict:
    """Pair-level counts over the claims compared: final SUPPORT document verdicts whose own reason
    argues against support (same keyword rule as the v2 supervisor, applicable to v1 too), and, for v2,
    supervisor consistency re-asks / downgrades and thinking statistics."""
    out = {"support_doc_verdicts": 0, "support_reason_negates_support": 0, "support_reason_negates_claim_ids": []}
    last_pairs: dict[tuple[str, str], dict] = {}
    if (run_dir / PAIRS).exists():
        for row in iter_jsonl(run_dir / PAIRS):
            if row.get("variant") == variant and row.get("claim_id") in claim_ids:
                last_pairs[(row["claim_id"], row["doc_id"])] = row
    ids = set()
    think_chars, forced, model_judgements = [], 0, 0
    for (cid, _), row in last_pairs.items():
        if row.get("predicted") == "SUPPORT":
            out["support_doc_verdicts"] += 1
            if _NEGATES_SUPPORT.search(row.get("reason") or ""):
                out["support_reason_negates_support"] += 1
                ids.add(cid)
        th = row.get("thinking")
        if th and not row.get("skipped_model"):
            model_judgements += 1
            think_chars.append(int(th.get("thinking_chars") or 0))
            forced += int(th.get("budget_forced") or 0)
    out["support_reason_negates_claim_ids"] = sorted(ids, key=lambda x: (len(x), x))
    flags: dict[str, int] = {}
    if (run_dir / EVENTS).exists():
        last: dict[str, list] = {}
        for row in iter_jsonl(run_dir / EVENTS):
            if row.get("variant") == variant and row.get("claim_id") in claim_ids:
                last[row["claim_id"]] = row.get("events") or []
        for events in last.values():
            for e in events:
                f = (e.get("data") or {}).get("flag")
                if f in ("consistency_repair", "downgraded_inconsistent", "citation_repair", "downgraded_no_citation",
                         "nei_checklist_decisive", "think_fallback"):
                    flags[f] = flags.get(f, 0) + 1
    out["supervisor_flag_claims"] = flags
    if think_chars:
        arr = np.array(think_chars)
        out["thinking"] = {"model_judgements": model_judgements, "budget_forced": forced,
                           "thinking_chars_mean": float(arr.mean()), "thinking_chars_median": float(np.median(arr)),
                           "thinking_chars_max": int(arr.max())}
    return out


def compare_runs(runs_dir: Path, a: tuple[str, str], b: tuple[str, str], *, n_boot: int = 10000,
                 seed: int = 42) -> dict:
    (run_a, var_a), (run_b, var_b) = a, b
    pa = {r["claim_id"]: r for r in load_predictions(runs_dir / run_a).get(var_a, [])}
    pb = {r["claim_id"]: r for r in load_predictions(runs_dir / run_b).get(var_b, [])}
    if not pa or not pb:
        raise ValueError(f"no predictions for {run_a}:{var_a} or {run_b}:{var_b}")
    common = [c for c in pa if c in pb]
    gold = [pa[c]["gold_label"] for c in common]
    if any(pb[c]["gold_label"] != pa[c]["gold_label"] for c in common):
        raise ValueError("gold labels differ between the two runs for the same claim ids")
    ya, yb = [_pred(pa[c]) for c in common], [_pred(pb[c]) for c in common]
    ma, mb = verdict_metrics(gold, ya), verdict_metrics(gold, yb)
    ca = [g == p for g, p in zip(gold, ya, strict=True)]
    cb = [g == p for g, p in zip(gold, yb, strict=True)]
    only_a = sum(1 for x, y in zip(ca, cb, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(ca, cb, strict=True) if y and not x)

    def summary(m):
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "correct": m["correct"],
                "per_class": m["per_class"], "confusion": m["confusion"]["matrix"]}

    lat_a = [pa[c]["timings"].get("total") for c in common]
    lat_b = [pb[c]["timings"].get("total") for c in common]
    return {
        "created_at": utc_now(),
        "a": {"run": run_a, "variant": var_a, **summary(ma), "errors": error_counts(gold, ya),
              "judge_consistency": judge_consistency(runs_dir / run_a, var_a, set(common)),
              "latency_mean_s_as_recorded": float(np.mean([x for x in lat_a if x is not None]))},
        "b": {"run": run_b, "variant": var_b, **summary(mb), "errors": error_counts(gold, yb),
              "judge_consistency": judge_consistency(runs_dir / run_b, var_b, set(common)),
              "latency_mean_s_as_recorded": float(np.mean([x for x in lat_b if x is not None]))},
        "n_claims": len(common),
        "gold_label_counts": {lab: gold.count(lab) for lab in LABELS},
        "paired": {"both_correct": sum(1 for x, y in zip(ca, cb, strict=True) if x and y),
                   "only_a_correct": only_a, "only_b_correct": only_b,
                   "both_wrong": sum(1 for x, y in zip(ca, cb, strict=True) if not x and not y),
                   "accuracy_diff_b_minus_a": (sum(cb) - sum(ca)) / len(common),
                   "macro_f1_diff_b_minus_a": mb["macro_f1"] - ma["macro_f1"],
                   "mcnemar_exact_p": mcnemar_exact(only_a, only_b),
                   "claims_only_a_correct": [c for c, x, y in zip(common, ca, cb, strict=True) if x and not y],
                   "claims_only_b_correct": [c for c, x, y in zip(common, ca, cb, strict=True) if y and not x]},
        "bootstrap": bootstrap(gold, ya, yb, n_boot=n_boot, seed=seed),
        "notes": "Latency 'as recorded' is the per-claim total saved in each run (cache state may differ between "
                 "runs; see the runs' reports). Accuracy counts operational errors as wrong.",
    }


def _f(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def render_comparison_md(out: dict) -> str:
    a, b, p, bs = out["a"], out["b"], out["paired"], out["bootstrap"]
    la, lb = f"{a['run']}:{a['variant']}", f"{b['run']}:{b['variant']}"
    L = [f"# Paired comparison: `{la}` (A) vs `{lb}` (B)", "",
         f"- {out['n_claims']} claims in both runs; gold {out['gold_label_counts']}. Created {out['created_at']}.",
         f"- Bootstrap: {bs['n_boot']} paired resamples of claims (seed {bs['seed']}), percentile 95% CIs.", "",
         "| metric | A | B | B - A |", "|---|---|---|---|",
         f"| accuracy | {_f(a['accuracy'])} [{_f(bs['accuracy_a'][0])}, {_f(bs['accuracy_a'][1])}] | "
         f"{_f(b['accuracy'])} [{_f(bs['accuracy_b'][0])}, {_f(bs['accuracy_b'][1])}] | "
         f"{p['accuracy_diff_b_minus_a']:+.3f} [{bs['accuracy_diff_b_minus_a'][0]:+.3f}, "
         f"{bs['accuracy_diff_b_minus_a'][1]:+.3f}] |",
         f"| macro-F1 | {_f(a['macro_f1'])} [{_f(bs['macro_f1_a'][0])}, {_f(bs['macro_f1_a'][1])}] | "
         f"{_f(b['macro_f1'])} [{_f(bs['macro_f1_b'][0])}, {_f(bs['macro_f1_b'][1])}] | "
         f"{p['macro_f1_diff_b_minus_a']:+.3f} [{bs['macro_f1_diff_b_minus_a'][0]:+.3f}, "
         f"{bs['macro_f1_diff_b_minus_a'][1]:+.3f}] |"]
    for lab in LABELS:
        L.append(f"| F1 {lab} | {_f(a['per_class'][lab]['f1'])} | {_f(b['per_class'][lab]['f1'])} | "
                 f"{b['per_class'][lab]['f1'] - a['per_class'][lab]['f1']:+.3f} |")
    for k in ("false_support", "nei_to_support", "contradict_to_support", "false_contradict", "support_to_nei",
              "errors_total", "operational_errors"):
        L.append(f"| {k.replace('_', ' ')} (claims) | {a['errors'][k]} | {b['errors'][k]} | "
                 f"{b['errors'][k] - a['errors'][k]:+d} |")
    ja, jb = a["judge_consistency"], b["judge_consistency"]
    L.append(f"| SUPPORT doc verdicts whose reason argues against support (keyword rule) | "
             f"{ja['support_reason_negates_support']}/{ja['support_doc_verdicts']} | "
             f"{jb['support_reason_negates_support']}/{jb['support_doc_verdicts']} | |")
    L += ["", f"McNemar exact test (paired correctness): only A correct = {p['only_a_correct']}, only B correct = "
          f"{p['only_b_correct']}, both correct = {p['both_correct']}, both wrong = {p['both_wrong']}; two-sided "
          f"p = {p['mcnemar_exact_p']:.4f}. Bootstrap share of resamples with accuracy(B) - accuracy(A) <= 0: "
          f"{bs['p_accuracy_diff_le_0']:.4f}.", ""]
    for side, s in (("A", a), ("B", b)):
        cm = s["confusion"]
        L += [f"Confusion {side} (`{s['run']}:{s['variant']}`; rows gold, columns predicted)", "",
              "| gold \\ pred | SUPPORT | CONTRADICT | NEI | ERROR |", "|---|---|---|---|---|"]
        L += [f"| {g} | {r['SUPPORT']} | {r['CONTRADICT']} | {r['NEI']} | {r['ERROR']} |" for g, r in cm.items()]
        L.append("")
    if jb.get("supervisor_flag_claims") or jb.get("thinking"):
        L.append(f"B supervisor flags (claims with the event): {jb.get('supervisor_flag_claims')}; thinking: "
                 f"{jb.get('thinking')}.")
        L.append("")
    L += [out["notes"], ""]
    return "\n".join(L)


def write_comparison(root: Path, out: dict, name: str) -> tuple[Path, Path]:
    d = Path(root) / "artifacts" / "comparisons"
    jp, mp = d / f"{name}.json", d / f"{name}.md"
    write_json(jp, out)
    mp.write_text(render_comparison_md(out), encoding="utf-8")
    return jp, mp
