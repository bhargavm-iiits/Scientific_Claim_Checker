"""Rendering helpers for the Streamlit claim checker.

The module has two layers:

* **Pure helpers** (no Streamlit calls): input validation through the core validator, verdict
  banner wording, 1-based sentence numbering, JSON export, readiness summaries, dataset example
  selection and loading of saved evaluation artifacts. These are unit-tested offline.
* **Render functions** (``render_*``): draw those values with Streamlit. Result cards, the
  pipeline timeline and notices are HTML from :mod:`ui.components` (every text escaped) drawn
  with ``st.html``; styling lives in ``ui/assets/theme.css``.

Nothing here retrieves, selects evidence or judges: every result comes from the shared
``claim_checker`` pipeline. Quotations are the stored source sentences the core attached to the
result; this module only formats them. Sentence positions are zero-based in every record and are
converted to one-based numbers only when rendered.
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from claim_checker.agents.explainer import explain
from claim_checker.errors import ClaimCheckerError, InputValidationError
from claim_checker.pipeline.state import STAGES
from claim_checker.pipeline.supervisor import validate_claim_text
from claim_checker.schemas import ClaimResult, DocumentReport, ErrorInfo, EvidenceQuote
from ui import components

EVALUATE_COMMAND = "python -m claim_checker evaluate --suite final"
REPORT_COMMAND = "python -m claim_checker report --run-id {run_id}"
RETRIEVAL_COMMAND = "python -m claim_checker retrieval-eval --split final"
CONFIDENCE_LABEL = "model-reported, uncalibrated"
RELEVANCE_NOTE = "a ranking relevance score, not a probability that the claim is true"

STAGE_LABELS = {
    "validate_input": "Validate input and resources",
    "retrieve": "Search abstracts (BM25 + dense, RRF, re-rank)",
    "select_evidence": "Evidence agent selects sentences",
    "judge": "Judge each abstract",
    "aggregate": "Combine document decisions",
    "supervise": "Supervisor checks citations",
    "explain": "Write the explanation",
}

VERDICT_MEANINGS = {
    "SUPPORT": "At least one examined abstract contains sentences that support the claim, and none "
               "contradicts it.",
    "CONTRADICT": "At least one examined abstract contains sentences that contradict the claim, and none "
                  "supports it.",
    "NEI": "Not enough information: none of the examined abstracts gave decisive evidence either way "
           "(or they disagreed - see the conflict notice). NEI does not mean the claim is false.",
}

# Error types that mean "fix your setup" rather than "try again".
SETUP_ERROR_TYPES = {"data_not_ready", "index_not_ready", "stale_index", "model_unavailable", "config_error",
                     "data_integrity"}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|$<>~])")


def escape_md(text: Any) -> str:
    """Backslash-escape Markdown punctuation so dataset/model text renders literally.

    Scientific abstracts contain ``$``, ``*``, ``_`` and ``[ ]`` which Streamlit would otherwise
    interpret as LaTeX, emphasis or links.
    """
    return _MD_SPECIAL.sub(r"\\\1", "" if text is None else str(text))


def code_span(text: Any) -> str:
    """Text for inside a Markdown code span: rendered literally, so only backticks are removed
    (backslash-escaping would show the backslashes)."""
    return " ".join(("" if text is None else str(text)).replace("`", "").split())


def sentence_number(sentence_id: int) -> int:
    """Zero-based internal position -> one-based number shown to people (render time only)."""
    return int(sentence_id) + 1


def format_sentence_numbers(ids: Iterable[int] | None) -> str:
    ids = list(ids or [])
    return ", ".join(str(sentence_number(i)) for i in ids)


def format_confidence(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} ({CONFIDENCE_LABEL})"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 1000:.0f} ms" if value < 1 else f"{value:.1f} s"


# ---------------------------------------------------------------------------
# Input validation (core validator; runs before any expensive work)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputCheck:
    ok: bool
    text: str = ""
    message: str = ""
    hint: str | None = None


def validate_claim_input(
    raw: str | None, max_chars: int, max_tokens: int, count_tokens: Callable[[str], int] | None = None
) -> InputCheck:
    """Validate the claim with the core's ``validate_claim_text``.

    The UI calls this without a token counter (that needs the embedding tokenizer); the pipeline's
    ``validate_input`` node then enforces the tokenizer-aware budget before retrieval or any model
    call, and :func:`input_check_from_result` turns that rejection back into an input message.
    """
    try:
        text = validate_claim_text(raw, max_chars, max_tokens, count_tokens)
    except InputValidationError as exc:
        return InputCheck(ok=False, message=exc.message, hint=exc.hint)
    return InputCheck(ok=True, text=text)


def is_input_rejection(result: ClaimResult) -> bool:
    return result.status == "error" and result.error is not None and result.error.error_type == "invalid_input"


def input_check_from_result(result: ClaimResult) -> InputCheck:
    err = result.error
    return InputCheck(ok=False, message=err.message if err else "Invalid input.", hint=err.hint if err else None)


def submission_allowed(running: bool, ready: bool) -> bool:
    """The Check button is enabled only when nothing is running and resources are ready."""
    return (not running) and ready


def error_result(claim: str, exc: BaseException, stage: str = "ui") -> ClaimResult:
    """Wrap an exception raised outside the pipeline (e.g. while building the checker) as a
    validated ``status="error"`` record so it is shown and downloaded like any other result."""
    if isinstance(exc, ClaimCheckerError):
        info = ErrorInfo(**{**exc.to_dict(), "stage": exc.stage or stage})
    else:
        info = ErrorInfo(error_type="unexpected_error", message=f"{type(exc).__name__}: {exc}", stage=stage)
    return ClaimResult(status="error", claim=claim or "", verdict=None,
                       explanation=explain(None, False, [], {}, error=info), error=info)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Banner:
    kind: str  # success | error | info | warning
    title: str
    body: str
    icon: str


def verdict_banner(result: ClaimResult) -> Banner:
    """Wording for the headline. Operational errors never look like an NEI verdict."""
    if result.status == "error" or result.verdict is None:
        err = result.error
        msg = err.message if err else "The pipeline ended without a result."
        hint = f" Suggested fix: {err.hint}" if err and err.hint else ""
        stage = f" (stage: {err.stage})" if err and err.stage else ""
        return Banner(
            kind="error",
            title="Check failed - no verdict was given (operational error)",
            body=(f"{msg}{stage}.{hint} This is not a NOT ENOUGH INFORMATION result: nothing was concluded "
                  "about the claim."),
            icon=":material/build_circle:",
        )
    if result.verdict == "NEI" and result.conflict:
        return Banner(kind="warning", title="NEI - the abstracts gave conflicting evidence",
                      body="Some examined abstracts support the claim and others contradict it. Under this "
                           "prototype's policy a conflict is reported as NEI with a conflict flag; read the "
                           "quotations from both sides below.",
                      icon=":material/compare_arrows:")
    if result.verdict == "SUPPORT":
        return Banner(kind="success", title="SUPPORT - the retrieved evidence supports the claim",
                      body=VERDICT_MEANINGS["SUPPORT"], icon=":material/check_circle:")
    if result.verdict == "CONTRADICT":
        return Banner(kind="error", title="CONTRADICT - the retrieved evidence contradicts the claim",
                      body=VERDICT_MEANINGS["CONTRADICT"], icon=":material/cancel:")
    return Banner(kind="info", title="NEI - not enough information in the examined abstracts",
                  body=VERDICT_MEANINGS["NEI"], icon=":material/help:")


def result_json(result: ClaimResult) -> str:
    """The exact validated record shown on screen, serialised by the core schema."""
    return result.to_json(indent=2)


def result_filename(result: ClaimResult, stamp: str) -> str:
    label = (result.verdict or "ERROR") if result.status == "ok" else "ERROR"
    safe = re.sub(r"[^0-9A-Za-z_-]", "", stamp.replace(":", "").replace(" ", "_"))
    return f"claim_check_{safe}_{label}.json"


def quotes_by_document(result: ClaimResult) -> dict[str, list[EvidenceQuote]]:
    """Cited quotations grouped by document in the documents' rank order, sentences ascending."""
    order = [d.doc_id for d in result.documents]
    grouped: dict[str, list[EvidenceQuote]] = {}
    for q in result.evidence:
        grouped.setdefault(q.doc_id, []).append(q)
    for qs in grouped.values():
        qs.sort(key=lambda q: q.sentence_id)
    ranked = {d: grouped[d] for d in order if d in grouped}
    ranked.update({d: v for d, v in grouped.items() if d not in ranked})
    return ranked


def judgement_notes(doc: DocumentReport) -> str:
    j = doc.judgement
    if j is None:
        return "not judged"
    notes = []
    if j.skipped_model:
        notes.append("no evidence sentence selected; NEI without a model call")
    if j.downgraded:
        notes.append("downgraded to NEI: decisive verdict had no valid citation")
    if j.repaired:
        notes.append("model output repaired once")
    if j.dropped_invalid_ids:
        notes.append(f"{len(j.dropped_invalid_ids)} invalid sentence id(s) rejected")
    if j.cached:
        notes.append("cached model response")
    return "; ".join(notes)


def document_rows(result: ClaimResult) -> list[dict]:
    rows = []
    for d in result.documents:
        j = d.judgement
        rows.append({
            "rank": d.rank,
            "document ID": d.doc_id,
            "title": d.title,
            "verdict": j.verdict if j else None,
            "cited sentences (1-based)": format_sentence_numbers(j.cited_sentence_ids) if j else "",
            "selected by evidence agent": ("-" if d.selected_sentence_ids is None
                                           else format_sentence_numbers(d.selected_sentence_ids) or "none"),
            f"confidence ({CONFIDENCE_LABEL})": j.confidence if j else None,
            "reason": j.reason if j else "",
            "relevance score": d.retrieval_score,
            "notes": judgement_notes(d),
        })
    return rows


def retrieval_rows(result: ClaimResult) -> list[dict]:
    hits = result.metadata.get("retrieval_hits") or []
    examined = {d.doc_id for d in result.documents}
    return [{"rank": h.get("rank"), "document ID": h.get("doc_id"), "title": h.get("title", ""),
             "method": h.get("method", ""), "relevance score": h.get("score"),
             "examined": h.get("doc_id") in examined} for h in hits]


def stage_rows(result: ClaimResult) -> list[dict]:
    """One row per graph stage from the result's events and timings."""
    rows = []
    for stage in STAGES:
        evs = [e for e in result.events if e.stage == stage]
        if any(e.status == "error" for e in evs):
            status = "error"
        elif any(e.message == "completed" for e in evs):
            status = "warning" if any(e.status == "warning" for e in evs) else "ok"
        else:
            status = "not run"
        notes = "; ".join(e.message for e in evs if e.status in ("warning", "error", "info") and e.message)
        rows.append({"stage": stage, "description": STAGE_LABELS.get(stage, stage), "status": status,
                     "seconds": result.timings.get(stage), "notes": notes})
    for key, val in sorted(result.timings.items()):
        if key.startswith("retrieve."):
            rows.append({"stage": key, "description": f"  retrieval sub-step: {key.split('.', 1)[1]}",
                         "status": "ok", "seconds": val, "notes": ""})
    return rows


class StageProgress:
    """Live stage status for a running check. ``update`` is passed to
    ``ClaimChecker.check(progress=...)`` and is called after each graph node finishes.

    Per-stage seconds come from the node's own ``completed`` event (``data["seconds"]``) when
    present, otherwise from the wall-clock gap since the previous update. A stage that runs more
    than once (a supervisor citation repair re-runs ``judge``) accumulates its time."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.status: dict[str, str] = {}
        self.seconds: dict[str, float] = {}
        self._clock = clock
        self._t0 = clock()
        self._last = self._t0

    @property
    def elapsed(self) -> float:
        return self._last - self._t0

    def update(self, stage: str, events: Sequence[Any]) -> None:
        now = self._clock()
        measured = [getattr(e, "data", {}).get("seconds") for e in events
                    if getattr(e, "message", "") == "completed" and isinstance(getattr(e, "data", None), dict)]
        measured = [float(s) for s in measured if isinstance(s, int | float)]
        dt = sum(measured) if measured else max(0.0, now - self._last)
        self.seconds[stage] = self.seconds.get(stage, 0.0) + dt
        self._last = now
        if any(getattr(e, "status", "") == "error" for e in events):
            self.status[stage] = "error"
        elif any(getattr(e, "status", "") == "warning" for e in events):
            self.status[stage] = "warning"
        else:
            self.status[stage] = "ok"

    def html(self, *, warming: bool = False) -> str:
        """Live pipeline board (pending / running / done / failed per stage)."""
        return components.live_timeline_html(STAGES, self.status, self.seconds, elapsed=self.elapsed,
                                             warming=warming)

    def markdown(self) -> str:
        marks = {"ok": "done", "warning": "done (with warnings)", "error": "FAILED"}
        failed = "error" in self.status.values() or "handle_error" in self.status
        lines, running_marked = [], False
        for stage in STAGES:
            state = self.status.get(stage)
            if state is not None:
                mark = marks[state]
            elif failed:
                mark = "not run"
            elif not running_marked:
                mark, running_marked = "running...", True
            else:
                mark = "waiting"
            lines.append(f"- {STAGE_LABELS.get(stage, stage)}: **{mark}**")
        return "\n".join(lines)


def stage_summary(result: ClaimResult) -> str:
    marks = {"ok": "done", "warning": "done (with warnings)", "error": "FAILED", "not run": "not run"}
    parts = [f"{r['stage']}: {marks[r['status']]}" for r in stage_rows(result) if "." not in r["stage"]]
    total = result.timings.get("total")
    return " | ".join(parts) + (f" | total {format_seconds(total)}" if total is not None else "")


def status_label(result: ClaimResult) -> str:
    if is_input_rejection(result):
        return "Input rejected before any model call"
    if result.status != "ok":
        return "Check failed (operational error, no verdict)"
    suffix = " with conflicting evidence" if result.conflict else ""
    return f"Finished: {result.verdict}{suffix} in {format_seconds(result.timings.get('total'))}"


def history_rows(history: Sequence[dict]) -> list[dict]:
    rows = []
    for h in history:
        r: ClaimResult = h["result"]
        rows.append({"#": h["id"], "time": h["time"], "claim": r.claim,
                     "outcome": (r.verdict + (" (conflict)" if r.conflict else "")) if r.status == "ok"
                     else "ERROR (no verdict)",
                     "k": h.get("settings", {}).get("k"),
                     "seconds": r.timings.get("total")})
    return rows


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def readiness_items(report: Any) -> list[dict]:
    """Checklist rows from the core ``ReadinessReport`` (duck-typed so tests can fake it)."""
    g = lambda name, default=None: getattr(report, name, default)  # noqa: E731
    model = g("model_name", "") or "model"
    digest = g("model_digest")
    items = [
        {"component": "Prepared SciFact data", "ok": bool(g("data_ready", False)), "required": True,
         "detail": "data/processed/corpus.jsonl"},
        {"component": "Search index", "ok": bool(g("index_ready", False)), "required": True,
         "detail": f"fingerprint {g('index_fingerprint')}" if g("index_fingerprint") else "artifacts/index"},
        {"component": f"Model {model}", "ok": bool(g("model_ready", False)), "required": True,
         "detail": f"digest {digest[:12]}" if digest else "local weights found" if g("model_backend", "ollama") != "ollama" and g("model_ready") else "not available"},
        {"component": "Frozen evaluation splits", "ok": bool(g("splits_ready", False)), "required": False,
         "detail": "needed only for evaluation runs"},
    ]
    if g("model_backend", "ollama") == "ollama":
        items.insert(2, {"component": "Ollama server", "ok": g("server_version") is not None, "required": True,
                         "detail": f"version {g('server_version')}" if g("server_version") else "not reachable"})
    return items


def readiness_problems(report: Any) -> list[dict]:
    return list(getattr(report, "problems", []) or [])


def is_ready(report: Any) -> bool:
    return bool(getattr(report, "ready", False))


# ---------------------------------------------------------------------------
# Dataset example claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleClaim:
    claim_id: str
    claim: str
    reference_label: str
    split: str

    def option_label(self) -> str:
        return f"[dataset label {self.reference_label}] {self.claim}"


# Curated example chips: real SciFact validation claims, one per reference label. Their answers
# are usually in the local LLM cache, so they make a quick demo.
FEATURED_EXAMPLES = (
    ExampleClaim(claim_id="148", claim="Autophagy declines in aged organisms.", reference_label="SUPPORT",
                 split="validation"),
    ExampleClaim(claim_id="793", claim="Mitochondria are uninvolved in apoptosis.", reference_label="CONTRADICT",
                 split="validation"),
    ExampleClaim(claim_id="659", claim="Ivermectin is used to treat lymphatic filariasis.", reference_label="NEI",
                 split="validation"),
)


def _id_key(claim_id: str) -> tuple[int, str]:
    return (int(claim_id), "") if str(claim_id).isdigit() else (10**12, str(claim_id))


def pick_examples(golds: Iterable[Any], per_label: int = 2, seed: int = 42, max_chars: int = 180) -> list[ExampleClaim]:
    """Deterministic sample of real annotated claims, ``per_label`` for each reference label."""
    by_label: dict[str, list[Any]] = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    for g in golds:
        if g.label in by_label and not getattr(g, "conflict", False) and len(g.claim) <= max_chars:
            by_label[g.label].append(g)
    rng = random.Random(seed)
    out: list[ExampleClaim] = []
    for label in ("SUPPORT", "CONTRADICT", "NEI"):
        pool = sorted(by_label[label], key=lambda g: _id_key(g.claim_id))
        for g in sorted(rng.sample(pool, min(per_label, len(pool))), key=lambda g: _id_key(g.claim_id)):
            out.append(ExampleClaim(claim_id=g.claim_id, claim=g.claim, reference_label=label, split=g.split))
    return out


def load_example_claims(processed_dir: str | Path, split: str = "train", per_label: int = 2) -> list[ExampleClaim]:
    """Example claims from ``claim_gold_<split>.jsonl``; empty if the data is not prepared yet."""
    from claim_checker.data.store import load_gold

    try:
        golds = load_gold(processed_dir, split).values()
    except (ClaimCheckerError, OSError, ValueError):
        return []
    return pick_examples(golds, per_label=per_label)


# ---------------------------------------------------------------------------
# Saved evaluation artifacts
# ---------------------------------------------------------------------------

RUN_FILES = ("manifest.json", "metrics.json", "results.csv", "confusion_matrices.json", "failure_cases.csv",
             "report.md", "failure_analysis.md", "predictions.jsonl")


@dataclass
class SavedRun:
    run_id: str
    path: Path
    manifest: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    confusion: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    report: str = ""
    failure_analysis: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def analysed(self) -> bool:
        return bool(self.results or self.metrics)


def list_runs(runs_dir: str | Path) -> list[Path]:
    """Run folders containing at least one known artifact, newest first."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and any((p / f).exists() for f in RUN_FILES)]
    return sorted(runs, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)


def _num(value: str) -> Any:
    if value is None:
        return None
    v = value.strip()
    if v == "" or v.lower() in ("none", "nan"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return value


def read_csv_rows(path: Path, numeric: bool = True) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if numeric:
        rows = [{k: _num(v) for k, v in r.items()} for r in rows]
    return rows


def load_run(run_dir: str | Path) -> SavedRun:
    path = Path(run_dir)
    run = SavedRun(run_id=path.name, path=path)
    for name, attr in (("manifest.json", "manifest"), ("metrics.json", "metrics"),
                       ("confusion_matrices.json", "confusion")):
        f = path / name
        if f.exists():
            try:
                setattr(run, attr, json.loads(f.read_text(encoding="utf-8")) or {})
            except (OSError, ValueError) as exc:
                run.problems.append(f"{name}: {exc}")
    for name, attr, numeric in (("results.csv", "results", True), ("failure_cases.csv", "failures", False)):
        f = path / name
        if f.exists():
            try:
                setattr(run, attr, read_csv_rows(f, numeric=numeric))
            except (OSError, ValueError, csv.Error) as exc:
                run.problems.append(f"{name}: {exc}")
    for name, attr in (("report.md", "report"), ("failure_analysis.md", "failure_analysis")):
        f = path / name
        if f.exists():
            try:
                setattr(run, attr, f.read_text(encoding="utf-8"))
            except OSError as exc:
                run.problems.append(f"{name}: {exc}")
    if run.manifest.get("run_id"):
        run.run_id = str(run.manifest["run_id"])
    return run


def run_identity_rows(run: SavedRun) -> list[dict]:
    m = run.manifest
    ident = m.get("identity", {}) or {}
    model = ident.get("model", {}) or {}
    rows = [
        ("Run ID", run.run_id),
        ("Created", m.get("created_at")),
        ("Suite", ident.get("suite")),
        ("Claim split", ident.get("split")),
        ("Claims requested per variant", ident.get("population_size")),
        ("Limit", m.get("limit")),
        ("Reference label counts", m.get("gold_label_counts")),
        ("Model", model.get("name")),
        ("Model digest", model.get("digest")),
        ("Index fingerprint", ident.get("index_fingerprint")),
    ]
    return [{"field": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else str(v)}
            for k, v in rows if v not in (None, "")]


def per_class_rows(metrics_json: dict) -> list[dict]:
    rows = []
    for variant, m in (metrics_json.get("metrics") or {}).items():
        vm = m.get("verdict", {}) or {}
        for label, pc in (vm.get("per_class") or {}).items():
            rows.append({"variant": variant, "class": label, "precision": pc.get("precision"),
                         "recall": pc.get("recall"), "F1": pc.get("f1"), "gold claims (support)": pc.get("support")})
    return rows


def confusion_rows(entry: dict | None) -> list[dict]:
    matrix = (entry or {}).get("matrix") or {}
    rows = []
    for gold, preds in matrix.items():
        rows.append({"gold \\ predicted": gold, **{p: n for p, n in preds.items()}})
    return rows


def load_retrieval_results(retrieval_dir: str | Path) -> list[dict]:
    root = Path(retrieval_dir)
    if not root.is_dir():
        return []
    out = []
    for f in sorted(root.glob("retrieval_eval_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["_file"] = f.name
        out.append(data)
    return out


def retrieval_table_rows(data: dict) -> list[dict]:
    rows = []
    for method, m in (data.get("results") or {}).items():
        lat = m.get("latency") or {}
        row = {"method": method}
        for k in (1, 3, 5, 10):
            if f"recall@{k}" in m:
                row[f"Recall@{k}"] = m.get(f"recall@{k}")
        for k in (3, 10):
            if f"evidence_doc_recall@{k}" in m:
                row[f"evidence-doc Recall@{k}"] = m.get(f"evidence_doc_recall@{k}")
        row["claims"] = m.get("recall@3_n_claims")
        row["mean latency (s)"] = lat.get("mean")
        row["p95 latency (s)"] = lat.get("p95")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


def html_block(markup: str, *, width: str | int = "stretch") -> None:
    """Draw component HTML (sanitised by Streamlit, not Markdown-processed). Empty markup is skipped."""
    if markup:
        st.html(markup, width=width)


def abstract_sentences(corpus: Mapping[str, Any] | None, result: ClaimResult) -> dict[str, list[str]]:
    """Stored sentences of the examined documents, for the expandable abstracts. Read from the
    checker's already-loaded corpus; empty when unavailable (the view then shows cited quotes only)."""
    if not corpus or result.status != "ok":
        return {}
    out: dict[str, list[str]] = {}
    for d in result.documents:
        doc = corpus.get(d.doc_id)
        sentences = getattr(doc, "sentences", None)
        if sentences:
            out[d.doc_id] = list(sentences)
    return out


def system_html(report: Any) -> str:
    rows = []
    for item in readiness_items(report):
        state = "ok" if item["ok"] else ("bad" if item["required"] else "warn")
        word = "ready" if item["ok"] else "missing"
        rows.append(f'<li class="cl-s-{state}"><span class="cl-dot" aria-hidden="true"></span><div>'
                    f"<b>{components.esc(item['component'])}</b> <span class=\"cl-sr\">{word}</span>"
                    f"<small>{components.esc(item['detail'])}</small></div></li>")
    return f'<ul class="cl-sys">{"".join(rows)}</ul>'


def render_setup_problems(report: Any) -> None:
    problems = readiness_problems(report)
    if problems or not is_ready(report):
        html_block(components.setup_problems_html(problems))


def render_input_problem(check: InputCheck) -> None:
    html_block(components.notice_html("warn", "The claim was not checked.", check.message, check.hint,
                                      icon="!"))


def render_download(result: ClaimResult, stamp: str, key: str) -> None:
    st.download_button(
        "Download JSON",
        data=result_json(result),
        file_name=result_filename(result, stamp),
        mime="application/json",
        key=key,
        icon=":material/download:",
        on_click="ignore",
        width="stretch",
    )


def render_raw_json(result: ClaimResult) -> None:
    with st.expander("Raw JSON record", icon=":material/data_object:"):
        st.caption("The exact validated record (zero-based `sentence_id`). Use the copy icon to copy it.")
        st.code(result_json(result), language="json")


def _cache_note(result: ClaimResult) -> str | None:
    meta = result.metadata or {}
    if meta.get("retrieval_cache_hit"):
        orig = meta.get("retrieve_original_compute_s")
        when = f" (originally {components.fmt_seconds(orig)})" if isinstance(orig, int | float) else ""
        return f"Retrieval served from the in-process cache{when}; sub-step timings were not re-measured."
    return None


def render_pipeline(result: ClaimResult) -> None:
    html_block(components.section_title("Pipeline"))
    html_block(components.timeline_html(stage_rows(result), result.timings.get("total"),
                                        cache_note=_cache_note(result)))


def render_result(result: ClaimResult, *, fresh: bool, stamp: str, entry_id: int,
                  sentences: Mapping[str, Sequence[str]] | None = None) -> None:
    """Full result view. ``fresh`` distinguishes a just-generated result from a stored one."""
    noun = "source passages" if result.metadata.get("knowledge", {}).get("kind") == "general_science" else "abstracts"
    html_block(components.claim_echo_html(result.claim, fresh=fresh, stamp=stamp, entry_id=entry_id))
    html_block(components.verdict_hero_html(result))
    html_block(components.conflict_html(result))
    ok = result.status == "ok"
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        if ok:
            html_block(components.section_title("Evidence", count=len(result.evidence),
                                                note=f"quoted from the {noun}"))
            docs = {d.doc_id: d for d in result.documents}
            html_block(components.evidence_html(quotes_by_document(result), docs, ok=ok,
                                               document_noun="source passage" if noun == "source passages" else "abstract"))
        html_block(components.section_title("Explanation"))
        html_block(components.explanation_html(result))
    with right:
        render_pipeline(result)
        details = components.run_details_html(result)
        if details:
            html_block(components.section_title("Run details"))
            html_block(details)
        render_download(result, stamp, key=f"download_{entry_id}")
        render_raw_json(result)
    if ok:
        html_block(components.section_title("Sources", count=len(result.documents),
                                            note=f"examined {noun} in rank order"))
        html_block(components.sources_html(result, sentences))
        cands = components.candidates_html(result)
        if cands:
            n = len((result.metadata or {}).get("retrieval_hits") or [])
            with st.expander(f"All retrieved candidates ({n})", icon=":material/format_list_numbered:"):
                st.caption(f"Ranked by the hybrid search and re-ranker. Scores are {RELEVANCE_NOTE}.")
                html_block(cands)


# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

HOW_IT_WORKS = """
**What the checker does.** You type one scientific claim. The checker searches a fixed local collection
of SciFact biomedical abstracts, reads the most relevant ones and reports whether they support or
contradict the claim, quoting the exact sentences it relied on. Everything runs on this computer with a
local model (Ollama); nothing is sent to a cloud service.

**The steps**

1. **Validate** - the claim must be non-empty, at most 2,000 characters and within the tokenizer budget.
   Data, index and model availability are checked before any expensive work.
2. **Search** - two searches run over all abstracts: BM25 (matching words) and dense retrieval
   (matching meaning with BGE embeddings). Their rankings are merged with reciprocal rank fusion.
3. **Re-rank** - a cross-encoder rescores the top candidates more carefully. Its scores measure
   *relevance to the claim*, not the probability that the claim is true.
4. **Select evidence** - for each of the top *k* abstracts (3 by default), the evidence agent picks the
   sentence numbers that bear on the claim. It may pick none.
5. **Judge** - the judge decides SUPPORT, CONTRADICT or NEI for each abstract separately and cites
   sentence numbers. Cited numbers are checked against the real abstract; quotations are then copied
   from the stored sentences, so the model cannot invent quotation text.
6. **Combine and supervise** - document decisions are combined by a fixed rule (below). A decisive
   judgement that cites no valid sentence is sent back once, then downgraded to NEI.
7. **Explain** - a plain-English explanation is written from the validated decisions (no extra model call).
"""

AGGREGATION_RULES = [
    {"document decisions": "at least one SUPPORT, no CONTRADICT", "final verdict": "SUPPORT"},
    {"document decisions": "at least one CONTRADICT, no SUPPORT", "final verdict": "CONTRADICT"},
    {"document decisions": "all NEI", "final verdict": "NEI"},
    {"document decisions": "both SUPPORT and CONTRADICT", "final verdict": "NEI with conflict flag"},
    {"document decisions": "a stage failed (model, index or data problem)", "final verdict": "error - no verdict"},
]

NEI_VS_ERROR = """
**NEI is not an error.** NEI is a real answer: the abstracts that were examined do not settle the claim.
It does not mean the claim is false, only that this collection of abstracts is insufficient.

**An operational error is not NEI.** If Ollama is not running, the model is missing, the index is stale
or a request fails, the result has `status = error` and **no verdict at all**, together with a suggested
fix. Errors are shown in a separate "Check failed" message and never disguised as scientific uncertainty.
"""

LIMITS = """
**Things to keep in mind**

- Evidence is limited to the abstracts in the local SciFact collection (not full papers, not the web).
- Confidence numbers are model-reported and uncalibrated. Treat them as the model's self-assessment only.
- Document IDs are dataset identifiers. The app does not invent paper links or DOIs.
- Sentence numbers start at 1 on screen; the downloaded JSON stores zero-based positions (`sentence_id`).
- The model is small (qwen3:4b). Read the quoted sentences before relying on a verdict.
"""


def verdict_meanings_html() -> str:
    tiles = "".join(
        f'<div class="cl-vtile cl-v-{s.key}"><div class="cl-vtile-head">'
        f'<span class="cl-vtile-icon" aria-hidden="true">{s.icon}</span>'
        f'<span class="cl-vtile-code">{components.esc(s.code)}</span></div>'
        f'<div class="cl-vtile-label">{components.esc(s.label)}</div>'
        f"<p>{components.esc(VERDICT_MEANINGS[code])}</p></div>"
        for code, s in components.VERDICT_STYLES.items()
    )
    return f'<div class="cl-vtiles">{tiles}</div>'


def render_how_it_works(knowledge_kind: str = "scifact") -> None:
    with st.container(key="cl-prose"):
        if knowledge_kind == "general_science":
            st.markdown("""### General science evidence

This collection contains cached OpenStax textbook passages across physics,
chemistry, biology and astronomy, plus an attributed USGS water-science excerpt. It is a fixed
collection, not a live web search.

1. Search the local source passages and rank them for your claim.
2. Read the most relevant passages with a local NLI model.
3. Report SUPPORT, CONTRADICT or NEI and copy cited sentences from the stored text.
4. Show the original source link and publisher attribution beside each source.

NEI means the examined sources do not settle the claim. It does not mean the
claim is false. Conflicting source decisions are also reported as NEI.

This mode has its own corpus, index and evaluation results. SciFact accuracy
does not describe general-science accuracy. No 95% general-science result has
been established. Formulas, conditions and claims outside these textbooks may
require evidence that is not available here.
""")
            return
        st.markdown(HOW_IT_WORKS)
        html_block(components.section_title("What the verdicts mean"))
        html_block(verdict_meanings_html())
        html_block(components.section_title("How document decisions are combined"))
        st.dataframe(AGGREGATION_RULES, hide_index=True, width="stretch")
        st.markdown(NEI_VS_ERROR)
        st.markdown(LIMITS)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


def no_evaluation_message() -> str:
    return (f"No saved evaluation yet - run `{EVALUATE_COMMAND}` from the project folder "
            "(the `claim-checker evaluate --suite final` script is equivalent). If a run then shows no "
            f"metrics, run `{REPORT_COMMAND.format(run_id='<run_id>')}`.")


def render_saved_run(run: SavedRun) -> None:
    st.markdown(f"#### Run `{code_span(run.run_id)}`")
    for p in run.problems:
        st.warning(f"Could not read {escape_md(p)}")
    ident = run_identity_rows(run)
    if ident:
        with st.expander("Run identity (population, model, index)", expanded=False):
            st.dataframe(ident, hide_index=True, width="stretch")
    if not run.analysed:
        st.info(f"This run has predictions but no metrics yet. Run `{REPORT_COMMAND.format(run_id=run.run_id)}` "
                "to compute them.")
        return

    st.markdown("##### Variant comparison")
    if run.results:
        st.dataframe(run.results, hide_index=True, width="stretch")
        st.caption("Accuracy and macro-F1 are over all requested claims; operational errors count as wrong and "
                   "coverage shows the share that received a verdict. Recall@k counts reference abstracts in the "
                   "top-k. Latency is in seconds per claim.")
    else:
        st.caption("results.csv not found.")

    pc = per_class_rows(run.metrics)
    if pc:
        st.markdown("##### Per-class verdict performance")
        variants = sorted({r["variant"] for r in pc})
        pick = st.selectbox("Variant", ["all variants", *variants], key=f"pc_variant_{run.run_id}")
        st.dataframe([r for r in pc if pick == "all variants" or r["variant"] == pick], hide_index=True,
                     width="stretch")
        st.caption("Unit: claims. Support = number of gold claims with that label.")

    cm_variants = [v for v, e in (run.confusion or {}).items() if e and e.get("matrix")]
    if cm_variants:
        st.markdown("##### Confusion matrices")
        v = st.selectbox("Variant", cm_variants, key=f"cm_variant_{run.run_id}")
        st.dataframe(confusion_rows(run.confusion[v]), hide_index=True, width="stretch")
        st.caption("Rows are gold (reference) labels, columns are predictions; ERROR = operational failure "
                   "with no verdict. Unit: claims.")

    if run.failures:
        st.markdown(f"##### Failure examples ({len(run.failures)})")
        cols = ["variant", "claim_id", "claim", "gold_label", "verdict", "status", "failure_stage", "conflict"]
        st.dataframe([{c: f.get(c) for c in cols} for f in run.failures], hide_index=True, width="stretch")
        st.caption("Failure stage is a heuristic for the earliest failing stage (operational / retrieval miss / "
                   "evidence miss / judgement error).")
        if run.failure_analysis:
            with st.expander("Human-reviewed failure analysis (failure_analysis.md)"):
                st.markdown(run.failure_analysis)
    if run.report:
        with st.expander("Full saved report (report.md)"):
            st.markdown(run.report)


def render_retrieval_results(results: list[dict]) -> None:
    st.markdown("#### Retrieval comparison")
    if not results:
        st.caption(f"No saved retrieval comparison yet - run `{RETRIEVAL_COMMAND}`.")
        return
    for data in results:
        st.markdown(f"**{escape_md(data.get('_file'))}** - split `{code_span(data.get('split'))}`, "
                    f"{escape_md(data.get('n_claims'))} claims, created {escape_md(data.get('created_at'))}")
        st.dataframe(retrieval_table_rows(data), hide_index=True, width="stretch")
        st.caption("Recall@k = mean per-claim share of reference abstracts found in the top-k. "
                   f"Scores and rankings measure relevance, not truth. {escape_md(data.get('notes', ''))}")


def render_evaluation_tab(runs_dir: str | Path, retrieval_dir: str | Path) -> None:
    st.info("**Saved results.** Everything on this tab was computed earlier by the evaluation commands and is "
            "read from disk. Nothing here runs the model, and these numbers are not the result of your claim.",
            icon=":material/inventory_2:")
    runs = list_runs(runs_dir)
    if not runs:
        st.info(no_evaluation_message(), icon=":material/info:")
    else:
        names = [p.name for p in runs]
        choice = st.selectbox("Saved evaluation run (newest first)", names, key="eval_run_choice")
        render_saved_run(load_run(Path(runs_dir) / choice))
    st.divider()
    render_retrieval_results(load_retrieval_results(retrieval_dir))
