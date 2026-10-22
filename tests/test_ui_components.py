"""Offline tests for the ClaimLens HTML components (``ui/components.py``) and theme assets.

Components are pure functions returning HTML strings, so they are tested without Streamlit:
verdict rendering for every outcome, escaping of corpus/model/user text, the "never invent a
confidence" rule, abstract highlighting and the pipeline timelines.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from claim_checker.schemas import (
    ClaimResult,
    DocumentJudgement,
    DocumentReport,
    ErrorInfo,
    EvidenceQuote,
)
from tests.conftest import scripted
from ui import components as C
from ui import theme, views

ROOT = Path(__file__).resolve().parents[1]
EVIL = '<script>alert("x")</script><img src=x onerror=alert(1)>'


def judged(doc_id: str, verdict: str, cited=(), confidence=0.8, rank=1, skipped=False, selected=None):
    j = DocumentJudgement(doc_id=doc_id, verdict=verdict, cited_sentence_ids=list(cited), reason=f"reason {doc_id}",
                          confidence=None if skipped else confidence, skipped_model=skipped)
    return DocumentReport(doc_id=doc_id, title=f"Title {doc_id}", rank=rank, retrieval_score=0.5,
                          retrieval_method="hybrid+rerank", selected_sentence_ids=selected, judgement=j)


def result(verdict="SUPPORT", conflict=False, docs=None, evidence=None, **meta) -> ClaimResult:
    docs = docs if docs is not None else [judged("101", verdict, cited=[2])]
    evidence = evidence if evidence is not None else [
        EvidenceQuote(doc_id=d.doc_id, title=d.title, sentence_id=i, quote=f"quote {d.doc_id}-{i}",
                      doc_verdict=d.judgement.verdict)
        for d in docs if d.judgement for i in d.judgement.cited_sentence_ids]
    return ClaimResult(status="ok", claim="A claim.", verdict=verdict, conflict=conflict, documents=docs,
                       evidence=evidence, explanation="Because `x` < y.\nSecond line.",
                       timings={"validate_input": 0.01, "retrieve": 1.2, "retrieve.bm25": 0.02, "judge": 3.0,
                                "total": 4.5}, metadata=meta)


def error_result() -> ClaimResult:
    err = ErrorInfo(error_type="model_unavailable", message="Cannot reach Ollama <host>", stage="validate_input",
                    hint="Run `ollama serve`")
    return ClaimResult(status="error", claim="c", verdict=None, error=err, timings={"total": 0.1})


# ---------------------------------------------------------------------------
# Text helpers and escaping
# ---------------------------------------------------------------------------


def test_esc_and_rich_escape_everything_before_formatting():
    assert C.esc(None) == ""
    assert C.esc('<a href="x">&</a>') == "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;"
    out = C.rich("Run `ollama <serve>`\nthen wait")
    assert out == "Run <code>ollama &lt;serve&gt;</code><br>then wait"
    assert "<script" not in C.rich(EVIL)


def test_code_span_keeps_identifiers_literal():
    assert views.code_span("dev_v1") == "dev_v1"  # no "dev\_v1" inside a code span
    assert views.code_span("a`b\nc") == "ab c" and views.code_span(None) == ""


def test_sentence_labels_are_one_based():
    assert C.sentence_label(0) == "S1" and C.doc_chip("13734012", 3).endswith("doc 13734012 · S4</span>")


@pytest.mark.parametrize("value,expected", [(None, "–"), (0.0123, "12 ms"), (7.94, "7.9 s"), (123.4, "123 s")])
def test_fmt_seconds(value, expected):
    assert C.fmt_seconds(value) == expected


def test_corpus_and_model_text_is_escaped_in_every_card():
    doc = judged("101", "SUPPORT", cited=[0])
    doc = doc.model_copy(update={"title": EVIL, "judgement": doc.judgement.model_copy(update={"reason": EVIL})})
    q = EvidenceQuote(doc_id="101", title=EVIL, sentence_id=0, quote=EVIL, doc_verdict="SUPPORT")
    res = ClaimResult(status="ok", claim=EVIL, verdict="SUPPORT", documents=[doc], evidence=[q], explanation=EVIL,
                      metadata={"model": EVIL, "retrieval_hits": [{"doc_id": "101", "rank": 1, "title": EVIL,
                                                                    "score": 1.0}]})
    blobs = [
        C.claim_echo_html(res.claim, fresh=True, stamp=EVIL, entry_id=1),
        C.verdict_hero_html(res),
        C.evidence_html(views.quotes_by_document(res), {"101": doc}, ok=True),
        C.explanation_html(res),
        C.sources_html(res, {"101": [EVIL, "second"]}),
        C.candidates_html(res),
        C.run_details_html(res),
        C.history_card_html({"id": 1, "time": EVIL, "result": res}, current=True),
        C.notice_html("warn", EVIL, EVIL, EVIL),
        C.setup_problems_html([{"component": EVIL, "message": EVIL, "hint": EVIL}]),
        C.error_card_html(ClaimResult(status="error", claim=EVIL, error=ErrorInfo(
            error_type=EVIL, message=EVIL, stage=EVIL, hint=EVIL))),
    ]
    for blob in blobs:
        assert "<script" not in blob and "<img" not in blob, blob
    assert "&lt;script&gt;" in blobs[0] and "&lt;img src=x onerror=alert(1)&gt;" in blobs[2]


# ---------------------------------------------------------------------------
# Verdict hero for every outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict,label,icon", [
    ("SUPPORT", "Supported", "✔"), ("CONTRADICT", "Contradicted", "✖"), ("NEI", "Not enough information", "?")])
def test_verdict_hero_uses_text_and_icon_for_each_verdict(verdict, label, icon):
    html = C.verdict_hero_html(result(verdict))
    key = C.VERDICT_STYLES[verdict].key
    assert f'class="cl-hero cl-v-{key}"' in html
    assert f">{label}</h2>" in html and f">{verdict}</span>" in html and icon in html
    assert 'aria-label="Verdict: ' in html
    assert "1</b><span>sources examined" in html and "4.5 s</b><span>total time" in html


def test_error_result_renders_as_error_card_never_as_a_verdict():
    html = C.verdict_hero_html(error_result())
    assert 'class="cl-hero cl-v-error"' in html and 'role="alert"' in html
    assert "Check failed - no verdict was given" in html and "not a NOT ENOUGH INFORMATION result" in html
    assert "<code>model_unavailable</code>" in html and "<code>validate_input</code>" in html
    assert "Cannot reach Ollama &lt;host&gt;" in html and "<code>ollama serve</code>" in html
    for word in ("Supported", "Contradicted", "Not enough information", "role=\"meter\""):
        assert word not in html
    assert C.result_style(error_result()) is C.ERROR_STYLE


def test_conflict_banner_lists_both_sides():
    docs = [judged("101", "SUPPORT", cited=[1]), judged("202", "CONTRADICT", cited=[0], rank=2)]
    res = result("NEI", conflict=True, docs=docs)
    banner = C.conflict_html(res)
    assert "Conflicting evidence" in banner and "doc 101" in banner and "doc 202" in banner
    hero = C.verdict_hero_html(res)
    assert ">Conflicting evidence</h2>" in hero and "1 support the claim and 1 contradict it" in hero
    assert C.conflict_html(result("SUPPORT")) == ""


# ---------------------------------------------------------------------------
# Confidence is shown only when the backend provides one
# ---------------------------------------------------------------------------


def test_confidence_comes_from_agreeing_model_judgements():
    docs = [judged("1", "SUPPORT", cited=[0], confidence=0.6), judged("2", "SUPPORT", cited=[0], confidence=0.9),
            judged("3", "NEI", skipped=True)]
    assert C.reported_confidence(result("SUPPORT", docs=docs)) == (0.9, "highest agreeing document")
    html = C.verdict_hero_html(result("SUPPORT", docs=docs))
    assert "model-reported, uncalibrated" in html and "0.90" in html and 'aria-valuenow="0.90"' in html


def test_claim_level_confidence_in_metadata_takes_precedence():
    assert C.reported_confidence(result("SUPPORT", confidence=0.42)) == (0.42, "claim-level")
    assert C.reported_confidence(result("SUPPORT", confidence="high"))[1] == "highest agreeing document"


def test_no_confidence_is_invented():
    nei = result("NEI", docs=[judged("1", "NEI", skipped=True), judged("2", "NEI", skipped=True)], evidence=[])
    assert C.reported_confidence(nei) is None
    assert 'role="meter"' not in C.verdict_hero_html(nei)
    assert C.reported_confidence(error_result()) is None
    conflict = result("NEI", conflict=True, docs=[judged("1", "SUPPORT", cited=[0]),
                                                  judged("2", "CONTRADICT", cited=[0])])
    assert C.reported_confidence(conflict) is None


# ---------------------------------------------------------------------------
# Evidence, sources, run details
# ---------------------------------------------------------------------------


def test_evidence_cards_colour_and_label_each_quote():
    docs = [judged("101", "SUPPORT", cited=[0, 2]), judged("202", "CONTRADICT", cited=[1], rank=2)]
    res = result("NEI", conflict=True, docs=docs)
    html = C.evidence_html(views.quotes_by_document(res), {d.doc_id: d for d in docs}, ok=True)
    assert html.count('class="cl-quote ') == 3
    assert "cl-quote cl-v-support" in html and "cl-quote cl-v-contradict" in html
    assert "doc 101 · S1" in html and "doc 101 · S3" in html and "doc 202 · S2" in html
    empty = C.evidence_html({}, {}, ok=True)
    assert "No sentence was cited" in empty
    assert "did not complete" in C.evidence_html({}, {}, ok=False)


def test_sources_highlight_cited_and_selected_sentences():
    doc = judged("101", "SUPPORT", cited=[2], selected=[1, 2])
    res = result("SUPPORT", docs=[doc], retrieval_hits=[
        {"doc_id": "101", "rank": 1, "title": "Title 101", "score": 6.8, "method": "hybrid+rerank",
         "stage_scores": {"hybrid": 0.033, "rerank_windows": 1}}])
    html = C.sources_html(res, {"101": ["first", "second", "third", "fourth"]})
    assert "Abstract · 4 sentences" in html
    assert '<li class="cl-cited cl-v-support cl-selected"><span class="cl-sn">S3</span>' in html
    assert '<li class="cl-selected"><span class="cl-sn">S2</span>' in html
    assert '<li class=""><span class="cl-sn">S1</span>' in html
    assert "re-rank score" in html and "hybrid score" in html and "re-rank windows" in html
    assert "confidence</span><b>0.80" in html and "reason 101" in html


def test_sources_degrade_to_cited_quotes_without_corpus():
    res = result("SUPPORT", docs=[judged("101", "SUPPORT", cited=[2])])
    html = C.sources_html(res, None)
    assert "Cited sentences" in html and "quote 101-2" in html
    assert "No abstracts were examined" in C.sources_html(result("NEI", docs=[], evidence=[]))


def test_abstract_sentences_helper_reads_only_examined_documents(corpus):
    res = result("SUPPORT", docs=[judged("101", "SUPPORT", cited=[2])])
    assert views.abstract_sentences(corpus, res) == {"101": corpus["101"].sentences}
    assert views.abstract_sentences(None, res) == {}
    assert views.abstract_sentences(corpus, error_result()) == {}


def test_run_details_tolerate_unknown_and_missing_metadata():
    assert C.run_details_html(result("SUPPORT")) == ""
    res = result("SUPPORT", model="qwen3:4b", options={"k": 3, "retrieval_method": "hybrid", "rerank": True,
                                                      "evidence_mode": "evidence", "variant": "full"},
                 prompt_set="v2", think=True, llm_calls=4, llm_cache_hits=1, brand_new_field={"x": 1})
    html = C.run_details_html(res)
    assert "k=3 · hybrid · re-rank on · evidence" in html and "full" in html
    assert "Prompt set</dt><dd>v2" in html and "Thinking</dt><dd>on" in html
    assert "4 (1 cached, 0 repaired)" in html and "brand_new_field" not in html


# ---------------------------------------------------------------------------
# Timelines, header, history, empty state
# ---------------------------------------------------------------------------


def test_finished_timeline_shows_durations_bars_and_substeps(make_checker):
    res = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])})).check("Vitamin D increases BMD.")
    html = C.timeline_html(views.stage_rows(res), res.timings.get("total"))
    for title in ("Validate", "Retrieve", "Select evidence", "Judge", "Aggregate", "Supervise", "Explain"):
        assert f'<span class="cl-tl-name">{title}</span>' in html
    assert html.count("cl-tl-done") == 7 and "BM25" in html and "Total" in html
    assert 'style="width:100%"' in html  # the slowest stage fills its bar


def test_failed_timeline_marks_error_and_skipped_stages():
    rows = [{"stage": "validate_input", "status": "error", "seconds": 0.1, "notes": "boom"},
            {"stage": "retrieve", "status": "not run", "seconds": None, "notes": ""}]
    html = C.timeline_html(rows, 0.1)
    assert "cl-tl-error" in html and "cl-tl-skip" in html and "not run" in html and "boom" in html


def test_live_timeline_states_and_warmup_message():
    stages = ("validate_input", "retrieve", "judge")
    html = C.live_timeline_html(stages, {"validate_input": "ok"}, {"validate_input": 1.5}, elapsed=1.5,
                                warming=True)
    assert html.count("cl-tl-done") == 1 and html.count("cl-tl-active") == 1 and html.count("cl-tl-pending") == 1
    assert "1 of 3 stages done" in html and "Warming up" in html and "1.5 s" in html
    cold = C.live_timeline_html(stages, {}, {}, elapsed=0.0, warming=True)
    assert "loading corpus, indexes and models" in cold and "starting" in cold
    assert "Warming up" not in C.live_timeline_html(stages, {}, {}, elapsed=0, warming=False)


def test_status_pills_reflect_readiness():
    from claim_checker.runtime import ReadinessReport

    ok = ReadinessReport(data_ready=True, index_ready=True, model_ready=True, server_version="0.4",
                         model_name="qwen3:4b", model_digest="sha256:abc", index_fingerprint="fp")
    assert [p.state for p in C.status_pills(ok)] == ["ok", "ok", "ok", "ok"]
    down = ReadinessReport(data_ready=True, index_ready=False, model_name="qwen3:4b")
    assert [p.state for p in C.status_pills(down)] == ["bad", "bad", "bad", "ok"]
    no_model = ReadinessReport(data_ready=True, index_ready=True, server_version="0.4", model_name="m")
    assert C.status_pills(no_model)[1].state == "warn"
    header = C.header_html(C.status_pills(down), "sub")
    assert "ClaimLens" in header and "(not ready)" in header and "cl-s-bad" in header


def test_history_card_shows_verdict_text_and_marks_current():
    entry = {"id": 3, "time": "12:00:00", "result": result("CONTRADICT")}
    html = C.history_card_html(entry, current=True)
    assert "cl-v-contradict" in html and "cl-hist-current" in html and ">CONTRADICT<" in html
    assert "#3" in html and "4.5 s" in html
    err = C.history_card_html({"id": 4, "time": "t", "result": error_result()}, current=False)
    assert "cl-v-error" in err and "ERROR" in err


def test_empty_state_lists_verdicts_and_configured_facts():
    html = C.empty_state_html(corpus_size=5183, model_name="qwen3:4b", embedding_model="BAAI/bge-small-en-v1.5",
                              reranker_model="cross-encoder/ms-marco-MiniLM-L6-v2", k=3)
    assert "5,183 SciFact abstracts" in html and "bge-small-en-v1.5" in html and "top 3 abstracts" in html
    for code in ("SUPPORT", "CONTRADICT", "NEI"):
        assert f">{code}</span>" in html


def test_counter_flags_over_long_input():
    assert "cl-over" not in C.counter_html(10, 2000, 256)
    over = C.counter_html(2001, 2000, 256)
    assert "cl-over" in over and "2,001" in over and 'data-max="2000"' in over


# ---------------------------------------------------------------------------
# Theme assets
# ---------------------------------------------------------------------------


def test_stylesheet_defines_verdict_palette_and_focus_outline():
    css = theme.stylesheet()
    for token in ("#0B1020", "#10B981", "#F43F5E", "#F59E0B", "#7C83FF", ":focus-visible",
                  "prefers-reduced-motion", ".cl-hero", ".cl-badge"):
        assert token in css
    assert "@import" not in css  # the optional web-font import lives in its own <style> block
    assert css.count("{") == css.count("}")


def test_streamlit_config_uses_matching_dark_theme():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    t = cfg["theme"]
    assert t["base"] == "dark" and t["backgroundColor"] == "#0B1020" and t["primaryColor"] == "#7C83FF"
    assert cfg["client"]["toolbarMode"] == "minimal"
