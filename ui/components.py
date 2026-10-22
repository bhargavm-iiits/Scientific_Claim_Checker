"""HTML building blocks for the ClaimLens UI (dark research console).

Every function here is *pure*: it takes plain values or core records and returns an HTML string
that the app passes to ``st.html`` (sanitised by Streamlit with DOMPurify, not run through
Markdown, so ``$``/``*``/``_`` in abstracts render literally). Every piece of user, corpus or model
text goes through :func:`esc` before it is placed in markup. Styling lives in
``ui/assets/theme.css``; class names are prefixed ``cl-``.

Nothing here computes a verdict: everything shown is read from the validated ``ClaimResult``.
Sentence positions are zero-based in the record and shown one-based (``S1`` is the first
sentence).
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from claim_checker.schemas import ClaimResult, DocumentReport, EvidenceQuote

CONFIDENCE_NOTE = "model-reported, uncalibrated"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    """HTML-escape any value (``None`` -> empty string). Quotes are escaped too, so the result
    is safe inside attribute values."""
    return html.escape("" if value is None else str(value), quote=True)


_CODE_SPAN = re.compile(r"`([^`\n]{1,200})`")


def rich(value: Any) -> str:
    """Escape, then turn `backtick` spans into ``<code>`` and newlines into ``<br>``.

    Used for hints and explanations that quote commands. The transformation runs on the
    already-escaped text, so it cannot introduce markup from the input."""
    out = _CODE_SPAN.sub(r"<code>\1</code>", esc(value))
    return out.replace("\n", "<br>")


def sentence_label(sentence_id: int) -> str:
    """Zero-based position -> one-based ``S<n>`` label (render time only)."""
    return f"S{int(sentence_id) + 1}"


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "–"
    if value < 1:
        return f"{value * 1000:.0f} ms"
    if value < 100:
        return f"{value:.1f} s"
    return f"{value:.0f} s"


def fmt_score(value: float | None) -> str:
    if value is None:
        return "–"
    v = float(value)
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


# ---------------------------------------------------------------------------
# Verdict vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerdictStyle:
    key: str  # css modifier: support | contradict | nei | error | none
    code: str  # SUPPORT / CONTRADICT / NEI / ERROR
    icon: str
    label: str  # human-readable


VERDICT_STYLES: dict[str, VerdictStyle] = {
    "SUPPORT": VerdictStyle("support", "SUPPORT", "✔", "Supported"),
    "CONTRADICT": VerdictStyle("contradict", "CONTRADICT", "✖", "Contradicted"),
    "NEI": VerdictStyle("nei", "NEI", "?", "Not enough information"),
}
ERROR_STYLE = VerdictStyle("error", "ERROR", "!", "Check failed")
NONE_STYLE = VerdictStyle("none", "NOT JUDGED", "–", "Not judged")

VERDICT_BLURBS = {
    "SUPPORT": "An examined abstract contains sentences that support the claim, and none contradicts it.",
    "CONTRADICT": "An examined abstract contains sentences that contradict the claim, and none supports it.",
    "NEI": "No examined abstract settles the claim either way. NEI does not mean the claim is false.",
}


def verdict_style(verdict: str | None) -> VerdictStyle:
    return VERDICT_STYLES.get(verdict or "", NONE_STYLE)


def result_style(result: ClaimResult) -> VerdictStyle:
    if result.status != "ok" or result.verdict is None:
        return ERROR_STYLE
    return verdict_style(result.verdict)


def verdict_pill(verdict: str | None, *, small: bool = False) -> str:
    """Text + icon pill (colour is never the only signal)."""
    s = verdict_style(verdict)
    size = " cl-pill-sm" if small else ""
    return (f'<span class="cl-vpill cl-v-{s.key}{size}"><span aria-hidden="true">{s.icon}</span>'
            f"{esc(s.code)}</span>")


def doc_chip(doc_id: str, sentence_id: int | None = None) -> str:
    tail = f" · {sentence_label(sentence_id)}" if sentence_id is not None else ""
    return f'<span class="cl-chip-mono" title="SciFact corpus identifier">doc {esc(doc_id)}{tail}</span>'


# ---------------------------------------------------------------------------
# Derived values (read-only views of the result)
# ---------------------------------------------------------------------------


def reported_confidence(result: ClaimResult) -> tuple[float, str] | None:
    """Confidence to show next to the verdict, or ``None``.

    Uses a claim-level ``metadata["confidence"]`` if the backend provides one; otherwise the
    highest model-reported confidence among the document judgements that agree with the final
    verdict. Never invented: ``None`` when no such number exists (e.g. NEI decided without a
    model call, a conflict, or an error)."""
    if result.status != "ok" or result.verdict is None:
        return None
    meta = result.metadata or {}
    value = meta.get("confidence")
    if isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0:
        return float(value), "claim-level"
    if result.conflict:
        return None
    vals = [j.confidence for j in result.judgements
            if j.verdict == result.verdict and j.confidence is not None and not j.skipped_model]
    if not vals:
        return None
    return max(vals), "highest agreeing document"


def verdict_counts(result: ClaimResult) -> dict[str, int]:
    counts = {"SUPPORT": 0, "CONTRADICT": 0, "NEI": 0}
    for j in result.judgements:
        counts[j.verdict] = counts.get(j.verdict, 0) + 1
    return counts


def verdict_summary(result: ClaimResult) -> str:
    """One plain sentence derived from the per-document decisions."""
    if result.status != "ok" or result.verdict is None:
        return "The pipeline stopped before reaching a verdict. Nothing was concluded about the claim."
    n = len(result.documents)
    noun = "source passage" if result.metadata.get("knowledge", {}).get("kind") == "general_science" else "abstract"
    c = verdict_counts(result)
    if result.conflict:
        return (f"The examined {noun}s disagree: {c['SUPPORT']} support the claim and "
                f"{c['CONTRADICT']} contradict it, so the result is NEI with a conflict flag.")
    if n == 0:
        return VERDICT_BLURBS.get(result.verdict, "").replace("abstract", noun)
    if result.verdict == "SUPPORT":
        return f"{c['SUPPORT']} of {plural(n, f'examined {noun}')} support the claim; none contradicts it."
    if result.verdict == "CONTRADICT":
        return f"{c['CONTRADICT']} of {plural(n, f'examined {noun}')} contradict the claim; none supports it."
    return f"None of the {plural(n, f'examined {noun}')} gave decisive evidence either way."


def hit_lookup(result: ClaimResult) -> dict[str, dict]:
    hits = (result.metadata or {}).get("retrieval_hits") or []
    return {str(h.get("doc_id")): h for h in hits if isinstance(h, dict) and h.get("doc_id") is not None}


# ---------------------------------------------------------------------------
# Header, status and empty state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusPill:
    label: str
    value: str
    state: str  # ok | warn | bad
    detail: str = ""


def status_pills(report: Any) -> list[StatusPill]:
    """Header pills from the core ``ReadinessReport`` (duck-typed)."""
    g = lambda name, default=None: getattr(report, name, default)  # noqa: E731
    version = g("server_version")
    model = g("model_name", "") or "model"
    digest = g("model_digest")
    fp = g("index_fingerprint")
    pills = [
        StatusPill("Ollama", f"v{version}" if version else "offline", "ok" if version else "bad",
                   "Local Ollama server" + (f" {version}" if version else " is not reachable")),
        StatusPill("Model", model, "ok" if g("model_ready", False) else ("warn" if version else "bad"),
                   f"digest {digest[:12]}" if digest else "model not available"),
        StatusPill("Index", "ready" if g("index_ready", False) else "missing",
                   "ok" if g("index_ready", False) else "bad", f"fingerprint {fp}" if fp else "search index"),
        StatusPill("Data", "ready" if g("data_ready", False) else "missing",
                   "ok" if g("data_ready", False) else "bad", "prepared source corpus"),
    ]
    backend = g("model_backend", "ollama")
    if backend != "ollama":
        ready = g("model_ready", False)
        pills[0] = StatusPill("Reader", backend.upper(), "ok" if ready else "bad", "Local classifier reader")
        pills[1] = StatusPill("Model", model, "ok" if ready else "bad", "Local checkpoint")
    return pills


def header_html(pills: Sequence[StatusPill], subtitle: str) -> str:
    items = "".join(
        f'<li class="cl-status cl-s-{esc(p.state)}" title="{esc(p.detail)}">'
        f'<span class="cl-dot" aria-hidden="true"></span><span class="cl-status-label">{esc(p.label)}</span>'
        f'<span class="cl-status-value">{esc(p.value)}</span>'
        f'{"" if p.state == "ok" else "<span class=cl-sr> (not ready)</span>"}</li>'
        for p in pills
    )
    return (
        '<header class="cl-header">'
        '<div class="cl-brand"><span class="cl-logo" aria-hidden="true">◆</span>'
        '<div><div class="cl-brand-name">ClaimLens</div>'
        f'<div class="cl-brand-sub">{esc(subtitle)}</div></div></div>'
        f'<ul class="cl-statusbar" aria-label="System status">{items}</ul>'
        "</header>"
    )


def section_title(title: str, *, count: int | None = None, note: str | None = None) -> str:
    badge = f'<span class="cl-count">{count}</span>' if count is not None else ""
    sub = f'<span class="cl-section-note">{esc(note)}</span>' if note else ""
    return f'<div class="cl-section"><h3>{esc(title)}{badge}</h3>{sub}</div>'


def empty_state_html(*, corpus_size: int | None, model_name: str, embedding_model: str,
                     reranker_model: str, k: int, knowledge_kind: str = "scifact", model_backend: str = "ollama") -> str:
    general = knowledge_kind == "general_science"
    noun = "science source passages" if general else "SciFact abstracts"
    corpus = f"{corpus_size:,} {noun}" if corpus_size else f"the local {noun}"
    tiles = "".join(
        f'<div class="cl-vtile cl-v-{s.key}"><div class="cl-vtile-head">'
        f'<span class="cl-vtile-icon" aria-hidden="true">{s.icon}</span>'
        f'<span class="cl-vtile-code">{esc(s.code)}</span></div>'
        f'<div class="cl-vtile-label">{esc(s.label)}</div><p>{esc(VERDICT_BLURBS[code].replace("abstract", "source passage") if general else VERDICT_BLURBS[code])}</p></div>'
        for code, s in VERDICT_STYLES.items()
    )
    steps = [
        ("Retrieve", "BM25 + dense (BGE) fused with RRF, then cross-encoder re-rank"),
        ("Select evidence", f"An evidence agent picks the relevant sentences in the top {k} abstracts"),
        ("Judge", "Each abstract is judged separately and must cite sentence numbers"),
        ("Aggregate", "A fixed rule combines the decisions; quotes are copied from the corpus"),
    ]
    if general:
        steps[1] = ("Read sources", f"Read the top {k} retrieved textbook passages")
        steps[2] = ("Judge", "Each source passage is judged and decisive answers cite source sentences")
    flow = "".join(
        f'<li><span class="cl-step-n">{i}</span><div><b>{esc(t)}</b><span>{esc(d)}</span></div></li>'
        for i, (t, d) in enumerate(steps, start=1)
    )
    model_fact = f"{model_name} runs locally" if model_backend != "ollama" else f"{model_name} via local Ollama"
    facts = [corpus, model_fact, embedding_model.split("/")[-1],
             reranker_model.split("/")[-1], "runs offline"]
    fact_html = "".join(f'<span class="cl-fact">{esc(f)}</span>' for f in facts)
    overview = ("ClaimLens searches cached OpenStax physics, chemistry, biology and astronomy passages, plus an attributed USGS water-science excerpt. "
                "Answers cite the original source text and publisher links. This collection is broader than SciFact, "
                "but it does not cover every scientific claim. Queries are processed locally."
                if general else
                "ClaimLens searches a local collection of biomedical abstracts, reads the most relevant ones and "
                "tells you whether they support or contradict your claim, quoting the exact sentences it relied "
                "on. Nothing leaves this computer.")
    return (
        '<section class="cl-card cl-empty">'
        '<div class="cl-empty-intro"><div class="cl-kicker">Overview</div>'
        f"<h2>Check a scientific claim against {esc(corpus)}.</h2>"
        f"<p>{esc(overview)}</p>"
        f'<div class="cl-facts">{fact_html}</div></div>'
        f'<ol class="cl-flow">{flow}</ol>'
        f'<div class="cl-vtiles">{tiles}</div>'
        "</section>"
    )


def notice_html(kind: str, title: str, body: str = "", hint: str | None = None, *, icon: str = "!") -> str:
    """Inline notice (``kind``: warn | bad | ok | info). ``body``/``hint`` are escaped here."""
    hint_html = f'<div class="cl-notice-hint">{rich(hint)}</div>' if hint else ""
    role = "alert" if kind in ("warn", "bad") else "status"
    return (f'<div class="cl-notice cl-n-{esc(kind)}" role="{role}"><span class="cl-notice-icon" '
            f'aria-hidden="true">{esc(icon)}</span><div><div class="cl-notice-title">{esc(title)}</div>'
            f'<div class="cl-notice-body">{rich(body)}</div>{hint_html}</div></div>')


def setup_problems_html(problems: Iterable[Mapping[str, Any]]) -> str:
    rows = "".join(
        f'<li><span class="cl-chip-mono">{esc(str(p.get("component", "setup")))}</span>'
        f'<div><div>{esc(p.get("message", ""))}</div>'
        f'<div class="cl-notice-hint">{rich(p.get("hint", ""))}</div></div></li>'
        for p in problems
    )
    return (
        '<div class="cl-notice cl-n-bad" role="alert"><span class="cl-notice-icon" aria-hidden="true">!</span>'
        '<div><div class="cl-notice-title">The checker is not ready yet</div>'
        '<div class="cl-notice-body">Fix the items below, then press <b>Re-check</b> in the sidebar.</div>'
        f'<ul class="cl-problems">{rows}</ul></div></div>'
    )


# ---------------------------------------------------------------------------
# Result: claim echo, verdict hero, conflict, error
# ---------------------------------------------------------------------------


def claim_echo_html(claim: str, *, fresh: bool, stamp: str, entry_id: int) -> str:
    if fresh:
        tag = f'<span class="cl-tag cl-tag-live">Fresh result · live pipeline · {esc(stamp)}</span>'
    else:
        tag = (f'<span class="cl-tag">Earlier check #{entry_id} · {esc(stamp)} · shown from memory, '
               "not re-run</span>")
    return (f'<div class="cl-claim-echo"><div class="cl-claim-meta"><span class="cl-kicker">Claim</span>{tag}'
            f'</div><blockquote>{esc(claim)}</blockquote></div>')


def _stat(value: str, label: str) -> str:
    return f'<div class="cl-stat"><b>{esc(value)}</b><span>{esc(label)}</span></div>'


def verdict_hero_html(result: ClaimResult) -> str:
    """Big glowing verdict card. Errors get :func:`error_card_html` instead."""
    if result.status != "ok" or result.verdict is None:
        return error_card_html(result)
    s = result_style(result)
    label = "Conflicting evidence" if result.conflict else s.label
    meta = result.metadata or {}
    conf = reported_confidence(result)
    if conf is not None:
        value, basis = conf
        pct = max(0.0, min(1.0, value)) * 100
        meter = (
            '<div class="cl-meter"><div class="cl-meter-head"><span>Confidence</span>'
            f'<span class="cl-meter-note">{esc(CONFIDENCE_NOTE)} · {esc(basis)}</span></div>'
            f'<div class="cl-meter-row"><div class="cl-meter-track" role="meter" aria-valuemin="0" '
            f'aria-valuemax="1" aria-valuenow="{value:.2f}" aria-label="Model-reported confidence">'
            f'<div class="cl-meter-fill" style="width:{pct:.0f}%"></div></div>'
            f'<b class="cl-meter-val">{value:.2f}</b></div></div>'
        )
    else:
        meter = ""
    n_docs = len(result.documents)
    stats = [
        _stat(str(n_docs), "sources examined"),
        _stat(str(len(result.evidence)), "evidence sentences"),
        _stat(fmt_seconds(result.timings.get("total")), "total time"),
    ]
    calls, hits = meta.get("llm_calls"), meta.get("llm_cache_hits")
    if isinstance(calls, int) and calls > 0:
        stats.append(_stat(f"{hits or 0}/{calls}", "LLM calls cached"))
    return (
        f'<section class="cl-hero cl-v-{s.key}" role="status" aria-label="Verdict: {esc(label)}">'
        f'<div class="cl-badge"><span class="cl-badge-icon" aria-hidden="true">{s.icon}</span>'
        f'<span class="cl-badge-code">{esc(s.code)}</span></div>'
        f'<div class="cl-hero-body"><div class="cl-kicker">Verdict</div>'
        f'<h2 class="cl-hero-title">{esc(label)}</h2>'
        f'<p class="cl-hero-summary">{esc(verdict_summary(result))}</p>{meter}'
        f'<div class="cl-stats">{"".join(stats)}</div></div></section>'
    )


def conflict_html(result: ClaimResult) -> str:
    if result.status != "ok" or not result.conflict:
        return ""
    supp = [d.doc_id for d in result.documents if d.judgement and d.judgement.verdict == "SUPPORT"]
    contra = [d.doc_id for d in result.documents if d.judgement and d.judgement.verdict == "CONTRADICT"]
    s_html = " ".join(doc_chip(d) for d in supp) or "none"
    c_html = " ".join(doc_chip(d) for d in contra) or "none"
    noun = "source passages" if result.metadata.get("knowledge", {}).get("kind") == "general_science" else "abstracts"
    return (
        '<div class="cl-notice cl-n-warn" role="alert"><span class="cl-notice-icon" aria-hidden="true">⇄</span>'
        '<div><div class="cl-notice-title">Conflicting evidence</div>'
        f'<div class="cl-notice-body">Some examined {noun} support the claim and others contradict it. By the '
        "documented conflict policy the final verdict is NEI with a conflict flag; read the quotations from "
        "both sides.</div>"
        f'<div class="cl-conflict-row"><span>Supporting</span>{s_html}</div>'
        f'<div class="cl-conflict-row"><span>Contradicting</span>{c_html}</div>'
        f'<span class="cl-sr">Supporting: {esc(", ".join(f"document {d}" for d in supp) or "none")}. '
        f'Contradicting: {esc(", ".join(f"document {d}" for d in contra) or "none")}.</span>'
        "</div></div>"
    )


def error_card_html(result: ClaimResult) -> str:
    err = result.error
    rows = []
    if err is not None:
        rows.append(("Error type", f'<code>{esc(err.error_type)}</code>'))
        if err.stage:
            rows.append(("Stage", f'<code>{esc(err.stage)}</code>'))
        rows.append(("Message", esc(err.message)))
        if err.hint:
            rows.append(("Suggested fix", rich(err.hint)))
    else:
        rows.append(("Message", "The pipeline ended without a result."))
    grid = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows)
    return (
        '<section class="cl-hero cl-v-error" role="alert" aria-label="Check failed">'
        f'<div class="cl-badge"><span class="cl-badge-icon" aria-hidden="true">{ERROR_STYLE.icon}</span>'
        f'<span class="cl-badge-code">{ERROR_STYLE.code}</span></div>'
        '<div class="cl-hero-body"><div class="cl-kicker">Operational error</div>'
        '<h2 class="cl-hero-title">Check failed - no verdict was given</h2>'
        '<p class="cl-hero-summary">This is not a NOT ENOUGH INFORMATION result: the pipeline stopped because '
        "of a setup or runtime problem, and nothing was concluded about the claim.</p>"
        f'<dl class="cl-errgrid">{grid}</dl></div></section>'
    )


# ---------------------------------------------------------------------------
# Evidence, explanation, sources
# ---------------------------------------------------------------------------


def evidence_html(grouped: Mapping[str, Sequence[EvidenceQuote]], docs: Mapping[str, DocumentReport],
                  *, ok: bool, document_noun: str = "abstract") -> str:
    if not grouped:
        msg = (f"No sentence was cited: no examined {document_noun} gave decisive evidence. Quotations appear only "
               "when a judgement cites sentences; the Sources below show what was read."
               if ok else "No evidence: the check did not complete.")
        return f'<div class="cl-card cl-muted-card">{esc(msg)}</div>'
    cards = []
    for doc_id, quotes in grouped.items():
        d = docs.get(doc_id)
        rank = f'<span class="cl-rank">#{d.rank}</span>' if d else ""
        for q in quotes:
            s = verdict_style(q.doc_verdict)
            cards.append(
                f'<article class="cl-quote cl-v-{s.key}">'
                f'<p class="cl-quote-text"><span class="cl-quote-mark" aria-hidden="true">“</span>'
                f"{esc(q.quote)}</p>"
                f'<div class="cl-quote-foot">{rank}<span class="cl-quote-title">{esc(q.title)}</span>'
                f'<span class="cl-quote-tags">{doc_chip(q.doc_id, q.sentence_id)}'
                f"{verdict_pill(q.doc_verdict, small=True)}</span></div></article>"
            )
    note = (f'<p class="cl-fineprint">Quotations are copied from the stored {esc(document_noun)} sentences by application '
            "code; the model only chooses sentence numbers. Sentence numbers start at 1 (S1).</p>")
    return f'<div class="cl-quotes">{"".join(cards)}</div>{note}'


def explanation_html(result: ClaimResult) -> str:
    text = (result.explanation or "").strip()
    body = rich(text) if text else "<em>No explanation was produced.</em>"
    return f'<div class="cl-card cl-explain"><p>{body}</p></div>'


def _judgement_notes(doc: DocumentReport) -> list[str]:
    j = doc.judgement
    if j is None:
        return ["not judged"]
    notes = []
    if j.skipped_model:
        notes.append("no evidence sentence selected - NEI without a model call")
    if j.downgraded:
        notes.append("downgraded to NEI: decisive verdict had no valid citation")
    if j.repaired:
        notes.append("model output repaired once")
    if j.dropped_invalid_ids:
        notes.append(f"{len(j.dropped_invalid_ids)} invalid sentence id(s) rejected")
    if j.cached:
        notes.append("cached model response")
    return notes


def _abstract_html(doc: DocumentReport, sentences: Sequence[str] | None,
                   quotes: Sequence[EvidenceQuote]) -> str:
    cited = set(doc.judgement.cited_sentence_ids) if doc.judgement else set()
    selected = set(doc.selected_sentence_ids or [])
    vkey = verdict_style(doc.judgement.verdict if doc.judgement else None).key
    legend = ('<div class="cl-legend"><span class="cl-lg-cited">cited by the judge</span>'
              '<span class="cl-lg-sel">selected by the evidence agent</span></div>')
    if sentences:
        items = []
        for i, sent in enumerate(sentences):
            cls = []
            tags = []
            if i in cited:
                cls.append(f"cl-cited cl-v-{vkey}")
                tags.append('<span class="cl-stag">cited</span>')
            if i in selected:
                cls.append("cl-selected")
                if i not in cited:
                    tags.append('<span class="cl-stag cl-stag-sel">selected</span>')
            items.append(f'<li class="{" ".join(cls)}"><span class="cl-sn">{sentence_label(i)}</span>'
                         f'<span class="cl-st">{esc(sent)}{"".join(tags)}</span></li>')
        body = f'<ol class="cl-abstract">{"".join(items)}</ol>'
        name = "Source passage" if doc.source.get("collection") == "general_science" else "Abstract"
        summary = f"{name} · {plural(len(sentences), 'sentence')}"
    else:
        items = [f'<li class="cl-cited cl-v-{vkey}"><span class="cl-sn">{sentence_label(q.sentence_id)}</span>'
                 f'<span class="cl-st">{esc(q.quote)}</span></li>' for q in quotes]
        sel = ", ".join(sentence_label(i) for i in sorted(selected))
        extra = f'<p class="cl-fineprint">Selected by the evidence agent: {esc(sel)}</p>' if sel else ""
        body = (f'<ol class="cl-abstract">{"".join(items)}</ol>{extra}' if items else
                '<p class="cl-fineprint">The full abstract text is not available in this view.</p>' + extra)
        summary = "Cited sentences"
    return (f'<details class="cl-details"><summary>{esc(summary)}</summary>{legend if sentences else ""}'
            f"{body}</details>")


def source_citation_html(source: Mapping) -> str:
    """Link to stored source provenance; reject executable URLs and escape attributes."""
    url = source.get("url")
    if not isinstance(url, str):
        return ""
    try:
        parts = urlsplit(url)
        valid = parts.scheme in {"http", "https"} and parts.hostname and not parts.username and not parts.password
    except ValueError:
        valid = False
    if not valid:
        return ""
    attribution = source.get("attribution", source.get("publisher", ""))
    return (f'<p class="cl-fineprint"><a href="{esc(url)}" target="_blank" rel="noopener noreferrer">View original source</a>'
            f'<br>{esc(attribution)}</p>')


def sources_html(result: ClaimResult, sentences: Mapping[str, Sequence[str]] | None = None) -> str:
    """One card per examined document, in rank order, with an expandable abstract."""
    if not result.documents:
        noun = "source passages" if result.metadata.get("knowledge", {}).get("kind") == "general_science" else "abstracts"
        return f'<div class="cl-card cl-muted-card">No {noun} were examined.</div>'
    hits = hit_lookup(result)
    by_doc: dict[str, list[EvidenceQuote]] = {}
    for q in result.evidence:
        by_doc.setdefault(q.doc_id, []).append(q)
    cards = []
    for d in result.documents:
        j = d.judgement
        method = d.retrieval_method or (hits.get(d.doc_id) or {}).get("method", "")
        score_label = "re-rank score" if "rerank" in method else (f"{method} score" if method else "score")
        scores = [(score_label, fmt_score(d.retrieval_score))]
        for name, val in ((hits.get(d.doc_id) or {}).get("stage_scores") or {}).items():
            if name == "rerank_windows":
                scores.append(("re-rank windows", str(int(val))))
            elif isinstance(val, int | float):
                scores.append((f"{name} score", fmt_score(val)))
        if j is not None and j.confidence is not None and not j.skipped_model:
            scores.append(("confidence", f"{j.confidence:.2f}"))
        score_html = "".join(f'<div><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in scores)
        reason = (f'<p class="cl-reason"><span>Judge</span>{esc(j.reason)}</p>'
                  if j is not None and j.reason and not j.skipped_model else "")
        notes = "".join(f'<span class="cl-note">{esc(n)}</span>' for n in _judgement_notes(d))
        vkey = verdict_style(j.verdict if j else None).key
        cards.append(
            f'<article class="cl-source cl-v-{vkey}"><div class="cl-source-head">'
            f'<span class="cl-rank-lg">#{d.rank}</span><div class="cl-source-title">'
            f"<h4>{esc(d.title)}</h4><div class=\"cl-source-tags\">{doc_chip(d.doc_id)}"
            f'{verdict_pill(j.verdict if j else None, small=True)}{notes}</div></div></div>'
            f'<div class="cl-scores">{score_html}</div>{source_citation_html(d.source)}{reason}'
            f"{_abstract_html(d, (sentences or {}).get(d.doc_id) or d.sentences, by_doc.get(d.doc_id, []))}</article>"
        )
    note = ('<p class="cl-fineprint">Relevance scores rank abstracts by relevance to the claim; they are not a '
            f"probability that the claim is true. Confidence is {CONFIDENCE_NOTE}. Document IDs are SciFact "
            "corpus identifiers (no paper links are invented).</p>")
    if result.metadata.get("knowledge", {}).get("kind") == "general_science":
        note = ('<p class="cl-fineprint">Sources are cached textbook and USGS article passages, with original source links and publisher attribution. '
                'Relevance scores and model confidence are not probabilities that a claim is true.</p>')
    return f'<div class="cl-sources">{"".join(cards)}</div>{note}'


def candidates_html(result: ClaimResult) -> str:
    """Retrieved candidates beyond the examined top-k (compact list)."""
    examined = {d.doc_id for d in result.documents}
    hits = [h for h in (result.metadata or {}).get("retrieval_hits") or [] if isinstance(h, dict)]
    if not hits:
        return ""
    rows = "".join(
        f'<li class="{"cl-examined" if h.get("doc_id") in examined else ""}">'
        f'<span class="cl-rank">#{esc(h.get("rank"))}</span><span class="cl-cand-title">{esc(h.get("title", ""))}'
        f'</span>{doc_chip(str(h.get("doc_id")))}<span class="cl-cand-score">{esc(fmt_score(h.get("score")))}'
        f'</span><span class="cl-cand-flag">{"examined" if h.get("doc_id") in examined else ""}</span></li>'
        for h in hits
    )
    return f'<ol class="cl-cands">{rows}</ol>'


# ---------------------------------------------------------------------------
# Pipeline timeline (finished result and live run)
# ---------------------------------------------------------------------------

STAGE_SHORT = {
    "validate_input": ("Validate", "input and resources"),
    "retrieve": ("Retrieve", "BM25 + dense → RRF → re-rank"),
    "select_evidence": ("Select evidence", "evidence agent picks sentences"),
    "judge": ("Judge", "one verdict per abstract"),
    "aggregate": ("Aggregate", "combine document decisions"),
    "supervise": ("Supervise", "citation check"),
    "explain": ("Explain", "plain-English summary"),
}
SUBSTEP_LABELS = {"bm25": "BM25", "dense": "dense (BGE)", "fuse": "RRF fuse", "rerank": "cross-encoder"}
STATE_WORDS = {"ok": "done", "warning": "done with warnings", "error": "failed", "not run": "not run",
               "active": "running", "pending": "waiting"}


def _stage_state_class(status: str) -> str:
    return {"ok": "done", "warning": "warn", "error": "error", "not run": "skip", "active": "active",
            "pending": "pending"}.get(status, "pending")


def timeline_html(rows: Sequence[Mapping[str, Any]], total: float | None, *, cache_note: str | None = None) -> str:
    """Finished pipeline: ``rows`` are ``views.stage_rows`` (main stages + ``retrieve.*`` sub-steps)."""
    main = [r for r in rows if "." not in str(r["stage"])]
    subs = {str(r["stage"]).split(".", 1)[1]: r for r in rows if str(r["stage"]).startswith("retrieve.")}
    secs = [float(r["seconds"]) for r in main if isinstance(r.get("seconds"), int | float)]
    scale = max(secs) if secs else 0.0
    items = []
    for r in main:
        stage = str(r["stage"])
        title, sub = STAGE_SHORT.get(stage, (stage, ""))
        sec = r.get("seconds")
        width = (float(sec) / scale * 100) if scale and isinstance(sec, int | float) else 0
        state = _stage_state_class(str(r["status"]))
        note = (f'<div class="cl-tl-note" title="{esc(r["notes"])}">{esc(r["notes"])}</div>'
                if r.get("notes") else "")
        sub_html = ""
        if stage == "retrieve" and subs:
            sub_items = "".join(
                f'<li><span>{esc(SUBSTEP_LABELS.get(k, k))}</span><b>{esc(fmt_seconds(v.get("seconds")))}</b></li>'
                for k, v in subs.items()
            )
            sub_html = f'<ul class="cl-tl-subs">{sub_items}</ul>'
        items.append(
            f'<li class="cl-tl-{state}"><span class="cl-tl-dot" aria-hidden="true"></span>'
            f'<div class="cl-tl-main"><div class="cl-tl-row"><span class="cl-tl-name">{esc(title)}</span>'
            f'<span class="cl-tl-time">{esc(fmt_seconds(sec) if sec is not None else STATE_WORDS.get(str(r["status"]), ""))}'
            f'</span></div><div class="cl-tl-sub">{esc(sub)}<span class="cl-sr"> - '
            f'{esc(STATE_WORDS.get(str(r["status"]), str(r["status"])))}</span></div>'
            f'<div class="cl-tl-bar"><i style="width:{width:.0f}%"></i></div>{sub_html}{note}</div></li>'
        )
    cache = f'<p class="cl-fineprint">{esc(cache_note)}</p>' if cache_note else ""
    return (f'<div class="cl-card cl-timeline"><ol>{"".join(items)}</ol>'
            f'<div class="cl-tl-total"><span>Total</span><b>{esc(fmt_seconds(total))}</b></div>{cache}</div>')


def live_timeline_html(stages: Sequence[str], status: Mapping[str, str], seconds: Mapping[str, float], *,
                       elapsed: float, warming: bool) -> str:
    """Live pipeline board while a check runs (pending / active / done / failed per stage)."""
    failed = "error" in status.values() or "handle_error" in status
    items, active_marked = [], False
    for stage in stages:
        st_ = status.get(stage)
        if st_ is not None:
            state = st_
        elif failed:
            state = "not run"
        elif not active_marked:
            state, active_marked = "active", True
        else:
            state = "pending"
        title, sub = STAGE_SHORT.get(stage, (stage, ""))
        if stage == "validate_input" and state == "active" and warming:
            sub = "loading corpus, indexes and models (first run only)"
        sec = seconds.get(stage)
        right = fmt_seconds(sec) if sec is not None and state in ("ok", "warning", "error") else STATE_WORDS[state]
        items.append(
            f'<li class="cl-tl-{_stage_state_class(state)}"><span class="cl-tl-dot" aria-hidden="true"></span>'
            f'<div class="cl-tl-main"><div class="cl-tl-row"><span class="cl-tl-name">{esc(title)}</span>'
            f'<span class="cl-tl-time">{esc(right)}</span></div><div class="cl-tl-sub">{esc(sub)}'
            f'<span class="cl-sr"> - {esc(STATE_WORDS[state])}</span></div></div></li>'
        )
    done = sum(1 for s in stages if status.get(s) in ("ok", "warning"))
    pct = done / len(stages) * 100 if stages else 0
    warm = (
        '<div class="cl-warm"><b>Warming up.</b> The first check in a new server session loads the corpus, '
        "the BM25 and dense indexes, the BGE embedding model and the cross-encoder (about 20&#8209;30&nbsp;s). Later "
        "checks reuse them and are much faster.</div>" if warming else ""
    )
    return (
        '<section class="cl-card cl-live" aria-live="polite">'
        '<div class="cl-live-head"><span class="cl-spinner" aria-hidden="true"></span>'
        '<div><div class="cl-live-title">Checking the claim with the local pipeline…</div>'
        f'<div class="cl-live-sub">{done} of {len(stages)} stages done'
        f"{f' · {esc(fmt_seconds(elapsed))} elapsed at the last stage' if elapsed > 0 else ' · starting'}"
        "</div></div></div>"
        f'<div class="cl-progress"><i style="width:{pct:.0f}%"></i></div>{warm}'
        f'<div class="cl-timeline cl-timeline-live"><ol>{"".join(items)}</ol></div></section>'
    )


def run_details_html(result: ClaimResult) -> str:
    """Key facts about the run (tolerates missing or unknown metadata fields)."""
    m = result.metadata or {}
    opts = m.get("options") if isinstance(m.get("options"), dict) else {}
    settings = m.get("model_settings") if isinstance(m.get("model_settings"), dict) else {}
    rows: list[tuple[str, str]] = []

    def add(label: str, value: Any, mono: bool = False) -> None:
        if value in (None, "", [], {}):
            return
        v = esc(value)
        rows.append((label, f"<code>{v}</code>" if mono else v))

    add("Model", m.get("model"))
    digest = m.get("model_digest")
    add("Model digest", str(digest)[:19] if digest else None, mono=True)
    if opts:
        parts = [f"k={opts.get('k')}" if opts.get("k") is not None else "",
                 str(opts.get("retrieval_method") or ""),
                 ("re-rank on" if opts.get("rerank") else "re-rank off") if "rerank" in opts else "",
                 str(opts.get("evidence_mode") or "")]
        add("Settings", " · ".join(p for p in parts if p))
    add("Variant", opts.get("variant") or m.get("variant"))
    add("Prompt set", m.get("prompt_set"))
    think = m.get("think", settings.get("think"))
    add("Thinking", None if think is None else ("on" if think else "off"))
    add("Index", m.get("index_fingerprint"), mono=True)
    if isinstance(m.get("llm_calls"), int):
        add("LLM calls", f"{m.get('llm_calls')} ({m.get('llm_cache_hits', 0)} cached, "
                         f"{m.get('llm_repairs', 0)} repaired)")
    add("Created", m.get("created_at"))
    add("Package", m.get("package_version"))
    grid = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows)
    return f'<div class="cl-card cl-details-card"><dl class="cl-kv">{grid}</dl></div>' if rows else ""


# ---------------------------------------------------------------------------
# Sidebar history
# ---------------------------------------------------------------------------


def history_card_html(entry: Mapping[str, Any], *, current: bool) -> str:
    r: ClaimResult = entry["result"]
    s = result_style(r)
    label = s.code + (" · conflict" if r.status == "ok" and r.conflict else "")
    cur = " cl-hist-current" if current else ""
    return (
        f'<div class="cl-hist cl-v-{s.key}{cur}"><div class="cl-hist-top"><span class="cl-hist-v">'
        f'<span class="cl-dot" aria-hidden="true"></span>{esc(label)}</span>'
        f'<span class="cl-hist-time">#{esc(entry.get("id"))}</span></div>'
        f'<div class="cl-hist-claim">{esc(r.claim)}</div>'
        f'<div class="cl-hist-meta">{esc(entry.get("time"))} · {esc(fmt_seconds(r.timings.get("total")))}</div>'
        "</div>"
    )


def counter_html(chars: int, max_chars: int, max_tokens: int) -> str:
    over = " cl-over" if chars > max_chars else ""
    return (f'<div class="cl-counter{over}" id="cl-counter" data-max="{max_chars}">'
            f'<span class="cl-counter-n">{chars:,}</span> / {max_chars:,} characters · up to {max_tokens} '
            "tokens · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to check</div>")
