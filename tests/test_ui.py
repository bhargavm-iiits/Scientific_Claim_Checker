"""Offline UI tests: no model, no downloads, no running Ollama.

Pure helpers in ``ui/views.py`` are tested directly (HTML components in
``tests/test_ui_components.py``). The Streamlit app is driven with
``streamlit.testing.v1.AppTest``; the shared ``ClaimChecker`` is the real pipeline wired to the
synthetic corpus, scripted fake Ollama transport and fake retriever from ``conftest.py``. Result
cards are ``st.html`` blocks, read back through :func:`html_text`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from claim_checker import runtime
from claim_checker.errors import ModelUnavailableError
from claim_checker.runtime import ReadinessReport
from claim_checker.schemas import ClaimGold, ClaimResult, EvidenceQuote
from tests.conftest import FakeTransport, scripted
from ui import views

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "ui" / "app.py"
COUNTER_SCRIPT_TAG = "<script>\n(function () {"  # the app's own static counter script
SUPPORT_SCRIPT = ({"101": [2]}, {"101": ("SUPPORT", [2])})
CONFLICT_SCRIPT = ({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("CONTRADICT", [1])})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def ready_report(**overrides) -> ReadinessReport:
    rep = ReadinessReport(data_ready=True, splits_ready=True, index_ready=True, model_ready=True,
                          server_version="0.0-fake", model_name="qwen3:4b", model_digest="sha256:fakedigest",
                          index_fingerprint="fakefp")
    for k, v in overrides.items():
        setattr(rep, k, v)
    return rep


def test_local_classifier_readiness_has_no_ollama_requirement():
    report = ready_report(model_name="Scientific NLI verifier", model_backend="nli", server_version=None)
    items = views.readiness_items(report)
    assert all(item["component"] != "Ollama server" for item in items)
    assert all(item["ok"] for item in items if item["required"])
    assert next(item for item in items if item["component"].startswith("Model "))["detail"] == "digest sha256:faked"


def gold(claim_id: str, claim: str, label: str | None, conflict: bool = False) -> ClaimGold:
    return ClaimGold(claim_id=claim_id, claim=claim, split="train", cited_doc_ids=["101"],
                     reference_doc_ids=["101"], evidence_by_doc={}, label_by_doc={}, label=label,
                     conflict=conflict)


GOLDS = [
    gold("1", "Vitamin D increases bone density.", "SUPPORT"),
    gold("3", "IL-6 is elevated in sepsis.", "SUPPORT"),
    gold("5", "Vitamin D has no effect on bone density.", "CONTRADICT"),
    gold("7", "Stars form inside cells.", "CONTRADICT"),
    gold("9", "Coffee improves memory in mice.", "NEI"),
    gold("11", "Aspirin cures astronomy.", "NEI"),
    gold("13", "Conflicting claim.", None, conflict=True),
]


@pytest.fixture
def app_root(tmp_path, monkeypatch):
    """A temporary project root for the app's own configuration (examples, saved runs)."""
    root = tmp_path / "root"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    with open(processed / "claim_gold_train.jsonl", "w", encoding="utf-8") as fh:
        for g in GOLDS:
            fh.write(g.model_dump_json() + "\n")
    monkeypatch.setenv("CLAIM_CHECKER_ROOT", str(root))
    st.cache_resource.clear()
    st.cache_data.clear()
    yield root
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def app_env(app_root, monkeypatch, make_checker):
    """Patch readiness and the checker factory; capture download_button payloads."""
    state: dict = {"readiness": ready_report(), "checker": None, "downloads": [], "constructed": 0}

    monkeypatch.setattr(runtime, "check_readiness", lambda cfg=None, **kw: state["readiness"])

    def factory(*args, **kwargs):
        state["constructed"] += 1
        if state["checker"] is None:
            state["checker"] = make_checker(scripted(*SUPPORT_SCRIPT))
        return state["checker"]

    monkeypatch.setattr(runtime, "ClaimChecker", factory)
    original = st.download_button

    def recording_download_button(label, data, *args, **kwargs):
        state["downloads"].append({"label": label, "data": data, "file_name": kwargs.get("file_name")})
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", recording_download_button)
    state["root"] = app_root
    return state


def start_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def submit(at: AppTest, claim: str | None = None) -> AppTest:
    if claim is not None:
        at.text_area(key="claim_text").input(claim)
    button(at, "Check claim").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def html_text(at: AppTest) -> str:
    """Bodies of every ``st.html`` block (component cards, notices, timeline)."""
    return "\n".join(e.proto.body for e in at.get("html"))


def all_text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "error", "warning", "info", "success"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts) + "\n" + html_text(at)


def unescape(text: str) -> str:
    return text.replace("\\", "")


def test_ui_loads_explicit_local_classifier_profile(app_env, monkeypatch):
    profile = app_env["root"] / "profile.yaml"
    profile.write_text("product:\n  judge_backend: nli\n  evidence_mode: full_abstract\n", encoding="utf-8")
    monkeypatch.setenv("CLAIM_CHECKER_CONFIG", str(profile))
    app_env["readiness"] = ready_report(model_name="Scientific NLI verifier", model_backend="nli")
    factory = runtime.ClaimChecker
    configurations = []

    def recording_factory(cfg, *args, **kwargs):
        configurations.append(cfg)
        return factory(cfg, *args, **kwargs)

    monkeypatch.setattr(runtime, "ClaimChecker", recording_factory)
    at = start_app()
    assert not next(toggle for toggle in at.toggle if toggle.label == "Evidence agent").value
    assert "Scientific NLI verifier" in all_text(at)
    button(at, "Load models").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert configurations and configurations[0].product.judge_backend == "nli"
    assert configurations[0].product.evidence_mode == "full_abstract"


def test_ui_can_switch_to_separate_general_science_collection(app_env, monkeypatch):
    app_env["readiness"] = ready_report(model_name="General science NLI verifier", model_backend="nli")
    factory = runtime.ClaimChecker
    configurations = []

    def recording_factory(cfg, *args, **kwargs):
        configurations.append(cfg)
        return factory(cfg, *args, **kwargs)

    monkeypatch.setattr(runtime, "ClaimChecker", recording_factory)
    at = start_app()
    at.selectbox(key="cc_collection").select("General science").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "OpenStax + USGS" in all_text(at)
    assert any(b.label == "Water is a Universal Solvent" for b in at.button)
    evidence = next(toggle for toggle in at.toggle if toggle.label == "Evidence agent")
    assert evidence.disabled and not evidence.value
    button(at, "Load models").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert configurations[-1].knowledge.kind == "general_science"
    assert configurations[-1].path("processed_dir") != app_env["root"] / "data/processed"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t "])
def test_blank_input_is_rejected(raw):
    check = views.validate_claim_input(raw, 2000, 256)
    assert not check.ok and "empty" in check.message


def test_oversized_input_is_rejected_by_characters():
    check = views.validate_claim_input("a" * 2001, 2000, 256)
    assert not check.ok and "2001" in check.message and "2000" in check.message
    assert views.validate_claim_input("a" * 2000, 2000, 256).ok


def test_token_budget_uses_core_validator():
    check = views.validate_claim_input("word " * 300, 2000, 256, count_tokens=lambda s: len(s.split()))
    assert not check.ok and "tokens" in check.message


def test_valid_input_is_normalised():
    check = views.validate_claim_input("  Vitamin D\n increases   bone density. ", 2000, 256)
    assert check.ok and check.text == "Vitamin D increases bone density."


def test_sentence_numbers_are_one_based_only_when_rendered():
    assert views.sentence_number(0) == 1
    assert views.format_sentence_numbers([0, 4]) == "1, 5"
    assert views.format_sentence_numbers(None) == ""


def test_escape_md_neutralises_markdown_and_latex():
    out = views.escape_md("cost $5 and $10 *bold* [link](x) 3.5%")
    assert "\\$5" in out and "\\*bold\\*" in out and "\\[link\\]" in out
    assert views.escape_md(None) == ""


def test_confidence_and_relevance_labels():
    assert "model-reported, uncalibrated" in views.format_confidence(0.8)
    assert views.format_confidence(None) == "-"
    assert "not a probability" in views.RELEVANCE_NOTE


def test_submission_allowed_only_when_idle_and_ready():
    assert views.submission_allowed(running=False, ready=True)
    assert not views.submission_allowed(running=True, ready=True)
    assert not views.submission_allowed(running=False, ready=False)


def test_result_json_equals_validated_result(make_checker):
    result = make_checker(scripted(*SUPPORT_SCRIPT)).check("Vitamin D increases bone mineral density.")
    assert result.status == "ok" and result.verdict == "SUPPORT"
    payload = views.result_json(result)
    assert ClaimResult.model_validate_json(payload) == result
    assert json.loads(payload) == json.loads(result.model_dump_json())
    assert views.result_filename(result, "12:30:05").endswith("_SUPPORT.json")


def test_document_rows_and_quotes_are_grounded(make_checker, corpus):
    result = make_checker(scripted(*SUPPORT_SCRIPT)).check("Vitamin D increases bone mineral density.")
    grouped = views.quotes_by_document(result)
    assert list(grouped) == ["101"]
    q = grouped["101"][0]
    assert q.sentence_id == 2 and q.quote == corpus["101"].sentences[2]
    rows = views.document_rows(result)
    assert [r["document ID"] for r in rows] == ["101", "202", "303"]
    assert rows[0]["cited sentences (1-based)"] == "3"
    assert rows[0]["verdict"] == "SUPPORT"
    assert "without a model call" in rows[1]["notes"]


def test_stage_rows_cover_every_graph_stage(make_checker):
    result = make_checker(scripted(*SUPPORT_SCRIPT)).check("Vitamin D increases bone mineral density.")
    rows = views.stage_rows(result)
    main = [r for r in rows if "." not in r["stage"]]
    assert [r["stage"] for r in main] == list(views.STAGES)
    assert all(r["status"] in ("ok", "warning") for r in main)
    assert "total" in views.stage_summary(result)


def test_verdict_banners_keep_errors_distinct_from_nei():
    err = views.error_result("claim", ModelUnavailableError("Ollama down", stage="model", hint="Run `ollama serve`"))
    b = views.verdict_banner(err)
    assert b.kind == "error" and "no verdict" in b.title and "not a NOT ENOUGH INFORMATION" in b.body
    assert "ollama serve" in b.body
    assert err.verdict is None and err.error.error_type == "model_unavailable"

    nei = ClaimResult(status="ok", claim="c", verdict="NEI")
    assert views.verdict_banner(nei).kind == "info" and views.verdict_banner(nei).title.startswith("NEI")
    conflict = ClaimResult(status="ok", claim="c", verdict="NEI", conflict=True)
    assert "conflicting" in views.verdict_banner(conflict).title
    assert views.verdict_banner(ClaimResult(status="ok", claim="c", verdict="SUPPORT")).kind == "success"
    assert views.verdict_banner(ClaimResult(status="ok", claim="c", verdict="CONTRADICT")).title.startswith(
        "CONTRADICT")


def test_error_result_wraps_unexpected_exceptions():
    res = views.error_result("claim", RuntimeError("boom"), stage="pipeline")
    assert res.status == "error" and res.verdict is None
    assert res.error.error_type == "unexpected_error" and "boom" in res.error.message
    assert "not a scientific judgement" in res.explanation


def test_input_rejection_from_pipeline_is_recognised(make_checker):
    result = make_checker(scripted(*SUPPORT_SCRIPT)).check("word " * 300)
    assert views.is_input_rejection(result)
    assert "tokens" in views.input_check_from_result(result).message


def test_readiness_items_and_problems():
    rep = ready_report(model_ready=False, server_version=None, model_digest=None)
    rep.problems.append({"component": "model", "message": "Ollama is not reachable.", "hint": "Run `ollama serve`."})
    items = {i["component"]: i for i in views.readiness_items(rep)}
    assert items["Prepared SciFact data"]["ok"] and items["Search index"]["ok"]
    assert not items["Model qwen3:4b"]["ok"] and not items["Ollama server"]["ok"]
    assert not items["Frozen evaluation splits"]["required"]
    assert views.readiness_problems(rep)[0]["hint"] == "Run `ollama serve`."
    assert not views.is_ready(rep) and views.is_ready(ready_report())


def test_pick_examples_is_deterministic_and_balanced():
    a = views.pick_examples(GOLDS, per_label=1)
    b = views.pick_examples(list(reversed(GOLDS)), per_label=1)
    assert a == b
    assert [e.reference_label for e in a] == ["SUPPORT", "CONTRADICT", "NEI"]
    two = views.pick_examples(GOLDS, per_label=2)
    assert len(two) == 6 and all(e.claim_id != "13" for e in two)  # conflicting annotation excluded
    assert "dataset label SUPPORT" in two[0].option_label()
    assert views.pick_examples(GOLDS, per_label=2, max_chars=25) and all(
        len(e.claim) <= 25 for e in views.pick_examples(GOLDS, per_label=2, max_chars=25))


def test_example_claims_load_from_real_gold_file(app_root, tmp_path):
    examples = views.load_example_claims(app_root / "data" / "processed", per_label=2)
    assert {e.claim for e in examples} <= {g.claim for g in GOLDS}
    assert views.load_example_claims(tmp_path / "missing") == []


def write_saved_run(runs_dir: Path, run_id: str = "final_test") -> Path:
    run = runs_dir / run_id
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "created_at": "2026-10-20T10:00:00+00:00", "gold_label_counts": {"SUPPORT": 2},
        "identity": {"suite": "final", "split": "final", "population_size": 3,
                     "model": {"name": "qwen3:4b", "digest": "abc"}, "index_fingerprint": "fp"}}), encoding="utf-8")
    per_class = {lab: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1}
                 for lab in ("SUPPORT", "CONTRADICT", "NEI")}
    (run / "metrics.json").write_text(json.dumps({"run_id": run_id, "metrics": {
        "full": {"verdict": {"per_class": per_class}}}}), encoding="utf-8")
    matrix = {g: {"SUPPORT": 1, "CONTRADICT": 0, "NEI": 0, "ERROR": 0} for g in ("SUPPORT", "CONTRADICT", "NEI")}
    (run / "confusion_matrices.json").write_text(json.dumps({"full": {"rows": "gold", "columns": "predicted",
                                                                       "matrix": matrix}}), encoding="utf-8")
    (run / "results.csv").write_text("variant,n_claims,accuracy,macro_f1,recall@3\nfull,3,0.667,0.5,\n",
                                     encoding="utf-8")
    (run / "failure_cases.csv").write_text(
        "variant,claim_id,claim,gold_label,verdict,status,failure_stage,conflict\n"
        "full,42,A distinctive failing claim,CONTRADICT,SUPPORT,ok,judgement error,False\n", encoding="utf-8")
    (run / "report.md").write_text("# Evaluation report - run `final_test`\n", encoding="utf-8")
    return run


def test_saved_runs_absent_and_present(tmp_path):
    assert views.list_runs(tmp_path / "nope") == []
    runs_dir = tmp_path / "runs"
    (runs_dir / "empty_dir").mkdir(parents=True)
    write_saved_run(runs_dir)
    runs = views.list_runs(runs_dir)
    assert [p.name for p in runs] == ["final_test"]
    run = views.load_run(runs[0])
    assert run.analysed and not run.problems
    assert run.results[0]["accuracy"] == pytest.approx(0.667) and run.results[0]["recall@3"] is None
    assert run.failures[0]["claim_id"] == "42"  # failure rows keep identifiers as text
    assert len(views.per_class_rows(run.metrics)) == 3
    cm = views.confusion_rows(run.confusion["full"])
    assert cm[0]["gold \\ predicted"] == "SUPPORT" and cm[0]["ERROR"] == 0
    ident = {r["field"]: r["value"] for r in views.run_identity_rows(run)}
    assert ident["Suite"] == "final" and ident["Model"] == "qwen3:4b"


def test_saved_run_with_corrupt_file_reports_problem(tmp_path):
    run = write_saved_run(tmp_path / "runs")
    (run / "metrics.json").write_text("{not json", encoding="utf-8")
    loaded = views.load_run(run)
    assert any("metrics.json" in p for p in loaded.problems)


def test_retrieval_results_loading(tmp_path):
    assert views.load_retrieval_results(tmp_path / "none") == []
    d = tmp_path / "retrieval"
    d.mkdir()
    (d / "retrieval_eval_final.json").write_text(json.dumps({"split": "final", "n_claims": 2, "results": {
        "bm25": {"recall@1": 0.1, "recall@3": 0.5, "recall@5": 0.6, "recall@10": 0.7, "recall@3_n_claims": 2,
                 "evidence_doc_recall@3": 0.4, "evidence_doc_recall@10": 0.6,
                 "latency": {"mean": 0.01, "p95": 0.02}}}}), encoding="utf-8")
    data = views.load_retrieval_results(d)
    rows = views.retrieval_table_rows(data[0])
    assert rows[0]["method"] == "bm25" and rows[0]["Recall@3"] == 0.5 and rows[0]["claims"] == 2


def test_no_evaluation_message_names_the_command():
    msg = views.no_evaluation_message()
    assert "No saved evaluation yet" in msg and "python -m claim_checker evaluate --suite final" in msg


def test_app_unanalysed_run_points_to_report_command(app_env):
    run = app_env["root"] / "artifacts" / "runs" / "smoke_partial"
    run.mkdir(parents=True)
    (run / "predictions.jsonl").write_text("", encoding="utf-8")
    at = start_app()
    assert any("no metrics yet" in i.value and "report --run-id smoke_partial" in i.value for i in at.info)


# ---------------------------------------------------------------------------
# App behaviour (AppTest)
# ---------------------------------------------------------------------------


def test_app_ready_state_renders_tabs_examples_and_no_model_load(app_env):
    at = start_app()
    assert [t.label for t in at.tabs] == ["Check a claim", "How it works", "Evaluation results"]
    assert not button(at, "Check claim").disabled
    assert app_env["constructed"] == 0  # rendering the page does not build the pipeline
    labels = at.selectbox(key="cc_example_choice").options
    assert len(labels) == 6 and all("dataset label" in s for s in labels)
    text = all_text(at)
    assert "No saved evaluation yet" in text  # evaluation tab without runs
    assert "NEI is not an error" in unescape(text)  # how-it-works tab
    html = html_text(at)
    assert "ClaimLens" in html and "cl-statusbar" in html  # header with status pills
    assert "5,183 SciFact abstracts" in html  # empty state uses the configured corpus size
    for verdict in ("Supported", "Contradicted", "Not enough information"):
        assert verdict in html
    chips = [b for b in at.button if (b.key or "").startswith("cl-chip-")]
    assert [b.label for b in chips] == [e.claim for e in views.FEATURED_EXAMPLES]


def test_app_not_ready_shows_setup_guidance_and_disables_submit(app_env):
    rep = ready_report(model_ready=False, server_version=None, model_digest=None)
    rep.problems.append({"component": "model", "message": "Ollama is not reachable at http://127.0.0.1:11434.",
                         "hint": "Install Ollama, run `ollama serve`, then `ollama pull qwen3:4b`."})
    app_env["readiness"] = rep
    at = start_app()
    assert button(at, "Check claim").disabled
    html = html_text(at)
    assert "The checker is not ready yet" in html
    assert "<code>ollama pull qwen3:4b</code>" in html and "Ollama is not reachable" in html
    assert "cl-s-bad" in html  # red status pill in the header


def test_app_rejects_blank_input_before_any_work(app_env):
    at = start_app()
    submit(at, "   ")
    html = html_text(at)
    assert "The claim was not checked" in html and "empty" in html
    assert app_env["constructed"] == 0 and not at.session_state["cc_history"]


def test_app_rejects_oversized_input_before_any_work(app_env):
    at = start_app()
    submit(at, "x" * 2001)
    html = html_text(at)
    assert "The claim was not checked" in html and "2001" in html and "2000" in html
    assert app_env["constructed"] == 0 and not at.session_state["cc_history"]
    # The over-long text is kept for editing, never silently truncated and checked.
    assert at.text_area(key="claim_text").value == "x" * 2001


def test_app_token_budget_rejection_is_an_input_message(app_env):
    at = start_app()
    submit(at, "word " * 300)  # 1500 chars but 300 tokens with the fake counter (limit 256)
    html = html_text(at)
    assert "The claim was not checked" in html and "tokens" in html
    assert not at.session_state["cc_history"]
    assert app_env["checker"].transport.calls == []  # rejected before any model call


def test_app_full_check_shows_verdict_evidence_and_matching_download(app_env):
    at = start_app()
    seen = {}
    checker = runtime.ClaimChecker()
    original = checker.check

    def spy(*args, **kwargs):
        seen["running"] = st.session_state["cc_running"]
        seen["kwargs"] = kwargs
        return original(*args, **kwargs)

    checker.check = spy
    submit(at, "Vitamin D supplementation increases bone mineral density.")
    assert seen["running"] is True  # the submit button is locked while the pipeline runs
    assert callable(seen["kwargs"].pop("progress"))  # live stage status callback
    assert seen["kwargs"] == {"k": 3, "retrieval_method": "hybrid", "rerank": True,
                              "evidence_mode": "evidence"}
    assert at.session_state["cc_running"] is False and at.session_state["cc_pending"] is None
    assert not button(at, "Check claim").disabled

    html = html_text(at)
    assert 'class="cl-hero cl-v-support"' in html and ">Supported<" in html and ">SUPPORT<" in html
    assert "Fresh result" in html
    assert "Vitamin D supplementation increased bone mineral density by 3.5%" in html
    assert "doc 101 · S3" in html and "Vitamin D and bone density" in html  # 1-based sentence label
    assert "model-reported, uncalibrated" in html and "0.80" in html  # scripted judge confidence
    # Sources card shows the full stored abstract with the cited sentence highlighted.
    assert "Abstract · 4 sentences" in html and "cl-cited" in html
    assert "Check failed" not in html

    history = at.session_state["cc_history"]
    assert len(history) == 1
    shown: ClaimResult = history[0]["result"]
    assert shown.verdict == "SUPPORT"
    payload = app_env["downloads"][-1]["data"]
    assert ClaimResult.model_validate_json(payload) == shown
    quote = json.loads(payload)["evidence"][0]
    assert quote["sentence_id"] == 2  # zero-based in the JSON, shown as "Sentence 3"


def test_app_advanced_settings_are_passed_to_pipeline(app_env):
    at = start_app()
    checker = runtime.ClaimChecker()
    calls = []
    original = checker.check
    checker.check = lambda *a, **k: calls.append(k) or original(*a, **k)
    at.radio(key="opt_k").set_value(1)
    at.toggle(key="opt_rerank").set_value(False)
    at.toggle(key="opt_evidence_agent").set_value(False)
    at.selectbox(key="opt_retrieval_method").set_value("bm25")
    submit(at, "Vitamin D supplementation increases bone mineral density.")
    assert [{k: v for k, v in c.items() if k != "progress"} for c in calls] == [
        {"k": 1, "retrieval_method": "bm25", "rerank": False, "evidence_mode": "full_abstract"}]
    assert at.session_state["cc_history"][0]["settings"]["retrieval_method"] == "bm25"


def test_app_shows_live_stage_progress_from_core_callback(app_env, monkeypatch):
    rendered: list[str] = []

    class RecordingProgress(views.StageProgress):
        def html(self, *, warming: bool = False) -> str:
            out = super().html(warming=warming)
            rendered.append(out)
            return out

    monkeypatch.setattr(views, "StageProgress", RecordingProgress)
    at = start_app()
    submit(at, "Vitamin D supplementation increases bone mineral density.")
    # Initial board, then one update per finished graph node, streamed while the check runs.
    assert len(rendered) >= 1 + 7
    assert rendered[0].count("cl-tl-active") == 1 and "0 of 7 stages done" in rendered[0]
    assert "Warming up" in rendered[0]  # first check in this server process
    assert "1 of 7 stages done" in rendered[1] and "cl-tl-done" in rendered[1]
    assert "cl-tl-active" in rendered[1]
    assert "7 of 7 stages done" in rendered[-1] and "cl-tl-active" not in rendered[-1]
    assert at.session_state["cc_history"][0]["result"].verdict == "SUPPORT"


def test_stage_progress_marks_running_waiting_and_failed():
    from claim_checker.schemas import StageEvent

    tracker = views.StageProgress()
    tracker.update("validate_input", [StageEvent(stage="validate_input", message="completed")])
    md = tracker.markdown()
    assert "Validate input and resources: **done**" in md
    assert "Search abstracts (BM25 + dense, RRF, re-rank): **running...**" in md
    assert "Write the explanation: **waiting**" in md
    tracker.update("retrieve", [StageEvent(stage="retrieve", status="error", message="boom")])
    md = tracker.markdown()
    assert "**FAILED**" in md and "Write the explanation: **not run**" in md
    html = tracker.html()
    assert "cl-tl-error" in html and "cl-tl-skip" in html and "cl-tl-active" not in html


def test_stage_progress_times_stages_from_events_or_clock():
    from claim_checker.schemas import StageEvent

    ticks = iter([0.0, 2.0, 5.0])
    tracker = views.StageProgress(clock=lambda: next(ticks))
    tracker.update("validate_input", [StageEvent(stage="validate_input", message="completed",
                                                 data={"seconds": 0.25})])
    tracker.update("retrieve", [StageEvent(stage="retrieve", status="info", message="note")])
    assert tracker.seconds == {"validate_input": 0.25, "retrieve": 3.0}
    assert tracker.elapsed == 5.0
    assert "250 ms" in tracker.html() and "3.0 s" in tracker.html()


def test_app_operational_error_is_distinct_from_nei(app_env, make_checker, unavailable):
    app_env["checker"] = make_checker(transport=FakeTransport(fail=unavailable))
    at = start_app()
    submit(at, "Vitamin D supplementation increases bone mineral density.")
    html = html_text(at)
    assert 'class="cl-hero cl-v-error"' in html and "Check failed - no verdict was given" in html
    assert "not a NOT ENOUGH INFORMATION result" in html and "<code>ollama serve</code>" in html
    assert "<code>model_unavailable</code>" in html
    assert "cl-hero cl-v-nei" not in html and "Not enough information</h2>" not in html
    result: ClaimResult = at.session_state["cc_history"][0]["result"]
    assert result.status == "error" and result.verdict is None
    assert json.loads(app_env["downloads"][-1]["data"])["verdict"] is None


def test_app_conflict_notice(app_env, make_checker):
    app_env["checker"] = make_checker(scripted(*CONFLICT_SCRIPT))
    at = start_app()
    submit(at, "Vitamin D supplementation changes bone mineral density.")
    html = html_text(at)
    assert "Conflicting evidence" in html and "cl-hero cl-v-nei" in html
    assert "document 101" in html and "document 202" in html
    assert "doc 101" in html and "doc 202" in html
    assert 'role="meter"' not in html  # no confidence is shown for a conflict


def test_app_example_button_fills_the_form(app_env):
    at = start_app()
    sel = at.selectbox(key="cc_example_choice")
    sel.set_value(2)
    at.run()
    button(at, "Use example").click()
    at.run()
    expected = views.pick_examples(GOLDS, per_label=2)[2].claim
    assert at.text_area(key="claim_text").value == expected
    assert app_env["constructed"] == 0  # choosing an example does not start a check


def test_app_example_chip_fills_the_form(app_env):
    at = start_app()
    next(b for b in at.button if b.key == "cl-chip-contradict").click()
    at.run()
    assert at.text_area(key="claim_text").value == "Mitochondria are uninvolved in apoptosis."
    assert app_env["constructed"] == 0 and not at.session_state["cc_history"]


def test_app_history_distinguishes_earlier_results(app_env):
    at = start_app()
    submit(at, "Vitamin D supplementation increases bone mineral density.")
    submit(at, "IL-6 levels are elevated in sepsis.")
    assert [h["id"] for h in at.session_state["cc_history"]] == [2, 1]
    assert html_text(at).count('class="cl-hist ') == 2  # two sidebar history cards
    next(b for b in at.button if b.key == "cl-hbtn-1").click()
    at.run()
    html = html_text(at)
    assert "Earlier check #1" in html and "not re-run" in html and "Fresh result" not in html
    assert "Vitamin D supplementation increases bone mineral density." in html


def test_app_escapes_claim_text_in_html(app_env):
    at = start_app()
    nasty = 'Vitamin D <script>alert(1)</script> <img src=x onerror="alert(2)"> & $x$ *bold*'
    submit(at, nasty)
    html = html_text(at)
    assert "<script>" not in html.replace(COUNTER_SCRIPT_TAG, "") and "<img" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html and "onerror=&quot;alert(2)&quot;" in html
    assert "&amp; $x$ *bold*" in html  # st.html is not Markdown: no LaTeX/emphasis processing


def test_app_new_check_returns_to_empty_state(app_env):
    at = start_app()
    submit(at, "Vitamin D supplementation increases bone mineral density.")
    button(at, "New check").click()
    at.run()
    assert at.session_state["cc_current"] is None and at.text_area(key="claim_text").value == ""
    assert "Check a scientific claim against" in html_text(at)
    assert len(at.session_state["cc_history"]) == 1  # history is kept


def test_app_recovers_from_stale_running_flag(app_env):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["cc_running"] = True  # left over from an interrupted run, no pending request
    at.run()
    assert not at.exception
    assert not button(at, "Check claim").disabled


def test_app_evaluation_tab_shows_saved_run(app_env):
    write_saved_run(app_env["root"] / "artifacts" / "runs")
    at = start_app()
    assert at.selectbox(key="eval_run_choice").value == "final_test"
    text = unescape(all_text(at))
    assert "Saved results" in text and "Variant comparison" in text and "Confusion matrices" in text
    assert "Failure examples (1)" in text
    frames = [df.value for df in at.dataframe]
    assert any("A distinctive failing claim" in str(f.to_dict()) for f in frames)


def test_evidence_quotes_render_from_record_not_model_text():
    """Rendering uses the stored quote attached by the core, with 1-based numbering."""
    q = EvidenceQuote(doc_id="7", title="T", sentence_id=0, quote="First sentence $x$.", doc_verdict="SUPPORT")
    res = ClaimResult(status="ok", claim="c", verdict="SUPPORT", evidence=[q])
    grouped = views.quotes_by_document(res)
    assert grouped == {"7": [q]}
    assert views.sentence_number(grouped["7"][0].sentence_id) == 1
