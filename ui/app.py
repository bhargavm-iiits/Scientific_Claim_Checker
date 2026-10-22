"""ClaimLens - Streamlit entry point for the Scientific Claim Checker.

Run from the project root:

    python -m streamlit run ui/app.py --server.address 127.0.0.1

The app only orchestrates the session; every check goes through the shared
``claim_checker.runtime.ClaimChecker`` (the same pipeline used by the CLI and the notebook).
Heavy resources are created once per server process with ``st.cache_resource`` and loaded lazily
on the first check, so rendering the page never loads a model.

Layout: ``ui/theme.py`` injects the stylesheet, ``ui/components.py`` builds the (escaped) HTML
cards, ``ui/views.py`` holds the pure helpers and render functions, and this module owns the
session flow (form -> pending request -> live run -> stored result -> history).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `streamlit run ui/app.py` puts only ui/ on sys.path
    sys.path.insert(0, str(ROOT))
try:
    import claim_checker  # noqa: F401  (installed editable in the normal environment)
except ImportError:  # running from a source checkout without installation
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from claim_checker import runtime  # noqa: E402
from claim_checker.config import AppConfig, load_config  # noqa: E402
from claim_checker.schemas import ClaimResult  # noqa: E402
from ui import components, theme, views  # noqa: E402

PROFILE_PATH = os.environ.get("CLAIM_CHECKER_CONFIG") or None

# Session-state keys
K_CLAIM = "claim_text"
K_K = "opt_k"
K_RERANK = "opt_rerank"
K_EVIDENCE = "opt_evidence_agent"
K_METHOD = "opt_retrieval_method"
K_RUNNING = "cc_running"
K_PENDING = "cc_pending"
K_HISTORY = "cc_history"
K_CURRENT = "cc_current"
K_FRESH = "cc_fresh"
K_INPUT_PROBLEM = "cc_input_problem"
K_READINESS = "cc_readiness"
K_EXAMPLE = "cc_example_choice"
K_NOTICE = "cc_notice"
K_COLLECTION = "cc_collection"
K_PROFILE = "cc_active_profile"

READINESS_TTL_SECONDS = 60.0
MAX_HISTORY = 25
RETRIEVAL_METHODS = {"hybrid": "Hybrid · BM25 + dense (RRF)", "bm25": "BM25 only", "dense": "Dense (BGE) only"}
FEATURED = {"support": views.FEATURED_EXAMPLES[0], "contradict": views.FEATURED_EXAMPLES[1],
            "nei": views.FEATURED_EXAMPLES[2]}


# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_checker(root: str, config_path: str | None = None) -> runtime.ClaimChecker:
    """One lazily-initialised checker per server process (models load on first use)."""
    return runtime.ClaimChecker(load_config(config_path, root=root))


@st.cache_resource(show_spinner=False)
def warm_flags(root: str, config_path: str | None = None) -> dict:
    """Process-wide flag: have the heavy models been loaded by a check or the warm-up button?"""
    return {"warm": False}


def active_profile() -> str | None:
    return st.session_state.get(K_PROFILE, PROFILE_PATH)


@st.cache_data(show_spinner=False)
def get_examples(processed_dir: str) -> list[views.ExampleClaim]:
    return views.load_example_claims(processed_dir, split="train", per_label=2)


def init_state(cfg: AppConfig) -> None:
    ss = st.session_state
    ss.setdefault(K_CLAIM, "")
    ss.setdefault(K_RUNNING, False)
    ss.setdefault(K_PENDING, None)
    ss.setdefault(K_HISTORY, [])
    ss.setdefault(K_CURRENT, None)
    ss.setdefault(K_FRESH, None)
    ss.setdefault(K_INPUT_PROBLEM, None)
    ss.setdefault(K_NOTICE, None)
    allowed = list(cfg.retrieval.allowed_k)
    ss.setdefault(K_K, cfg.retrieval.pipeline_k if cfg.retrieval.pipeline_k in allowed else allowed[0])
    ss.setdefault(K_RERANK, True)
    ss.setdefault(K_EVIDENCE, cfg.product.evidence_mode == "evidence")
    ss.setdefault(K_METHOD, "hybrid")
    # A run flag without a pending request can only be left over from an interrupted run.
    if ss[K_RUNNING] and ss[K_PENDING] is None:
        ss[K_RUNNING] = False


def get_readiness(cfg: AppConfig, force: bool = False):
    """Cheap readiness check (files + Ollama HTTP); never loads a model. Cached per session."""
    cached = st.session_state.get(K_READINESS)
    if not force and cached and time.time() - cached[0] < READINESS_TTL_SECONDS:
        return cached[1]
    try:
        report = runtime.check_readiness(cfg, check_model=True, timeout=2.0)
    except Exception as exc:  # readiness must never crash the page
        report = runtime.ReadinessReport(model_name=cfg.model.name)
        report.problems.append({"component": "setup", "message": f"Readiness check failed: {exc}",
                                "hint": "Run `claim-checker doctor` for details."})
    st.session_state[K_READINESS] = (time.time(), report)
    return report


def invalidate_readiness() -> None:
    st.session_state.pop(K_READINESS, None)


def result_sentences(cfg: AppConfig, result: ClaimResult) -> dict[str, list[str]]:
    """Abstract sentences for the Sources cards, from the checker's corpus (already loaded by the
    check that produced the result). Degrades to an empty mapping on any problem."""
    if result.status != "ok" or not result.documents:
        return {}
    try:
        return views.abstract_sentences(get_checker(cfg.root, active_profile()).corpus, result)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _fill_claim(text: str) -> None:
    st.session_state[K_CLAIM] = text
    st.session_state[K_INPUT_PROBLEM] = None


def _use_example(examples: list[views.ExampleClaim]) -> None:
    idx = st.session_state.get(K_EXAMPLE)
    if idx is not None and 0 <= idx < len(examples):
        _fill_claim(examples[idx].claim)


def _show_history_entry(entry_id: int) -> None:
    st.session_state[K_CURRENT] = entry_id


def _clear_claim() -> None:
    _fill_claim("")


def _new_check() -> None:
    _fill_claim("")
    st.session_state[K_CURRENT] = None


# ---------------------------------------------------------------------------
# Running a check
# ---------------------------------------------------------------------------


def run_check(cfg: AppConfig, pending: dict, progress=None) -> ClaimResult:
    """Call the shared pipeline. ``ClaimChecker.check`` reports operational problems as
    ``status="error"``; anything raised outside it is wrapped the same way. ``progress`` is the
    core's per-stage callback (live stage status)."""
    claim = pending["claim"]
    try:
        checker = get_checker(cfg.root, active_profile())
    except Exception as exc:
        return views.error_result(claim, exc, stage="load_checker")
    try:
        return checker.check(claim, k=pending["k"], retrieval_method=pending["retrieval_method"],
                             rerank=pending["rerank"], evidence_mode=pending["evidence_mode"], progress=progress)
    except Exception as exc:
        return views.error_result(claim, exc, stage="pipeline")


def record_result(result: ClaimResult, pending: dict) -> None:
    ss = st.session_state
    if views.is_input_rejection(result):  # token budget etc.: an input message, not a result
        ss[K_INPUT_PROBLEM] = views.input_check_from_result(result)
        return
    history: list[dict] = ss[K_HISTORY]
    entry_id = (history[0]["id"] + 1) if history else 1
    entry = {"id": entry_id, "time": datetime.now().strftime("%H:%M:%S"), "result": result,
             "settings": {k: pending[k] for k in ("k", "rerank", "evidence_mode", "retrieval_method")}}
    ss[K_HISTORY] = [entry, *history][:MAX_HISTORY]
    ss[K_CURRENT] = entry_id
    ss[K_FRESH] = entry_id
    if result.status == "error" and result.error and result.error.error_type in views.SETUP_ERROR_TYPES:
        invalidate_readiness()


def execute_pending(cfg: AppConfig) -> None:
    """Run the pending request with a live pipeline board, store the result, then rerun."""
    pending = st.session_state[K_PENDING]
    flags = warm_flags(cfg.root, active_profile())
    try:
        board = st.empty()
        tracker = views.StageProgress()
        warming = not flags["warm"]
        board.html(tracker.html(warming=warming))

        def on_stage(stage, events):
            tracker.update(stage, events)
            board.html(tracker.html(warming=warming))

        result = run_check(cfg, pending, progress=on_stage)
        record_result(result, pending)  # stored before any further Streamlit call
        if result.status == "ok":
            flags["warm"] = True
    finally:
        st.session_state[K_PENDING] = None
        st.session_state[K_RUNNING] = False
    st.rerun()  # re-render with the submit button enabled again


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def header(cfg: AppConfig, report) -> None:
    views.html_block(components.header_html(
        components.status_pills(report),
        f"Scientific Claim Checker · {cfg.knowledge.name} · local {report.model_name or cfg.model.name}",
    ))


def sidebar(cfg: AppConfig, report, running: bool) -> None:
    ss = st.session_state
    with st.sidebar:
        views.html_block('<div class="cl-side-title">Settings</div>')
        allowed = list(cfg.retrieval.allowed_k)
        general = cfg.knowledge.kind == "general_science"
        st.radio("Source passages to examine (k)" if general else "Abstracts to examine (k)", allowed, key=K_K, horizontal=True, disabled=running,
                 help="How many top-ranked sources the reader examines.")
        st.selectbox("Retrieval", list(RETRIEVAL_METHODS), format_func=RETRIEVAL_METHODS.get, key=K_METHOD,
                     disabled=running, help="How candidate sources are found before re-ranking.")
        st.toggle("Cross-encoder re-rank", key=K_RERANK, disabled=running,
                  help="Rescore the top candidates with the cross-encoder before picking the top k.")
        st.toggle("Evidence agent", key=K_EVIDENCE, disabled=running or general,
                  help="On: an agent selects sentences before judging. Off: the judge reads each whole source passage.")

        views.html_block('<div class="cl-side-title">System</div>')
        views.html_block(views.system_html(report))
        c1, c2 = st.columns(2)
        if c1.button("Re-check", disabled=running, help="Re-run the readiness check (files and Ollama).",
                     width="stretch"):
            get_readiness(cfg, force=True)
            st.rerun()
        if c2.button("Load models", disabled=running or not views.is_ready(report), width="stretch",
                     help="Optional: load the corpus, index, embedding model and re-ranker now so the first "
                          "check is faster (about 20-30 s)."):
            with st.spinner("Loading corpus, index, embedding model and re-ranker…"):
                try:
                    get_checker(cfg.root, active_profile()).warmup()
                    warm_flags(cfg.root, active_profile())["warm"] = True
                    ss[K_NOTICE] = ("ok", "Models loaded and ready.")
                except Exception as exc:
                    hint = getattr(exc, "hint", None)
                    ss[K_NOTICE] = ("bad", f"Warm-up failed: {exc}" + (f" {hint}" if hint else ""))
                    invalidate_readiness()
            st.rerun()
        notice = ss.get(K_NOTICE)
        if notice:
            kind, text = notice
            views.html_block(components.notice_html(kind, "Models ready" if kind == "ok" else "Warm-up failed",
                                                    text, icon="✓" if kind == "ok" else "!"))

        history: list[dict] = ss[K_HISTORY]
        views.html_block(f'<div class="cl-side-title">History · {len(history)}</div>')
        if not history:
            views.html_block('<div class="cl-side-note">Checks you run appear here. They are kept for this '
                             "browser session only - nothing is written to disk.</div>")
        for h in history:
            with st.container(key=f"cl-hist-{h['id']}"):
                views.html_block(components.history_card_html(h, current=h["id"] == ss[K_CURRENT]))
                st.button(f"Show check #{h['id']}", key=f"cl-hbtn-{h['id']}", on_click=_show_history_entry,
                          args=(h["id"],), disabled=running, width="stretch")
        if history:
            views.html_block('<div class="cl-side-note">Opening an earlier check re-displays its stored result; '
                             "it is not re-run.</div>")
        if report.model_backend == "ollama":
            detail = f"{components.esc(cfg.model.name)} via local Ollama at {components.esc(cfg.model.host)}"
        else:
            detail = f"{components.esc(report.model_name)} runs locally"
        views.html_block(f'<div class="cl-side-note">Model {detail}. Nothing is sent to a cloud service.</div>')


def claim_form(cfg: AppConfig, ready: bool, running: bool) -> str | None:
    """Hero input card. Returns the raw text when the form was submitted."""
    max_chars, max_tokens = cfg.runtime.max_claim_chars, cfg.runtime.max_claim_tokens
    with st.form("claim_form", clear_on_submit=False, border=True):
        # No widget max_chars on purpose: Streamlit would silently truncate pasted text and check a
        # different claim. Over-long input is rejected explicitly by the core validator instead.
        raw = st.text_area(
            "Scientific claim",
            key=K_CLAIM,
            height=118,
            placeholder="Enter a scientific claim, e.g. “Vitamin D supplementation increases bone mineral "
                        "density in older adults.”",
            help=f"One scientific statement, up to {max_chars:,} characters and {max_tokens} tokens.",
            disabled=running,
        )
        c1, c2, c3 = st.columns([5, 1.1, 1.6], vertical_alignment="center")
        with c1:
            st.html(components.counter_html(len(st.session_state.get(K_CLAIM) or ""), max_chars, max_tokens)
                    + theme.COUNTER_JS, unsafe_allow_javascript=True)
        c2.form_submit_button("Clear", on_click=_clear_claim, disabled=running, width="stretch")
        submitted = c3.form_submit_button("Check claim", type="primary",
                                          disabled=not views.submission_allowed(running, ready),
                                          icon=":material/fact_check:", width="stretch")
    return raw if submitted else None


def example_chips(cfg: AppConfig, running: bool) -> None:
    with st.container(key="cl-examples", horizontal=True, gap="small", vertical_alignment="center"):
        views.html_block('<div class="cl-ex-label">Try</div>', width="content")
        if cfg.knowledge.kind == "general_science":
            for index, claim in enumerate(("Water is a Universal Solvent", "Electrons carry positive electric charge.",
                                           "The mineral sample BX-204 contains 2.31% copper.")):
                st.button(claim, key=f"cl-science-example-{index}", on_click=_fill_claim, args=(claim,),
                          disabled=running, help="Demonstration claim; no reference label is shown to the model.")
            return
        for key, ex in FEATURED.items():
            with st.container(key=f"cl-ex-{key}", width="content"):
                st.button(ex.claim, key=f"cl-chip-{key}", on_click=_fill_claim, args=(ex.claim,),
                          disabled=running, help=f"SciFact {ex.split} claim · dataset label {ex.reference_label} "
                                                 "(reference annotation, not a result)")
        examples = get_examples(str(cfg.path("processed_dir")))
        if examples:
            with st.popover("More", icon=":material/library_books:", disabled=running,
                            help="More real claims from the SciFact training split"):
                st.caption("Real claims from the SciFact training split. The label is the dataset's reference "
                           "annotation, for comparison only - it is never shown to the model.")
                st.selectbox("Example", options=list(range(len(examples))),
                             format_func=lambda i: examples[i].option_label(), key=K_EXAMPLE,
                             label_visibility="collapsed")
                st.button("Use example", on_click=_use_example, args=(examples,), width="stretch")


def check_tab(cfg: AppConfig, report, running: bool) -> None:
    ss = st.session_state
    ready = views.is_ready(report)
    if not ready:
        views.render_setup_problems(report)

    raw = claim_form(cfg, ready, running)
    example_chips(cfg, running)

    if raw is not None and not running:
        check = views.validate_claim_input(raw, cfg.runtime.max_claim_chars, cfg.runtime.max_claim_tokens)
        if not check.ok:
            ss[K_INPUT_PROBLEM] = check
        else:
            ss[K_INPUT_PROBLEM] = None
            ss[K_PENDING] = {"claim": check.text, "k": int(ss[K_K]), "rerank": bool(ss[K_RERANK]),
                             "retrieval_method": str(ss[K_METHOD]),
                             "evidence_mode": "evidence" if ss[K_EVIDENCE] else "full_abstract"}
            ss[K_RUNNING] = True
            st.rerun()

    if ss[K_INPUT_PROBLEM] is not None:
        views.render_input_problem(ss[K_INPUT_PROBLEM])

    if running and ss[K_PENDING] is not None:
        execute_pending(cfg)

    history: list[dict] = ss[K_HISTORY]
    current = next((h for h in history if h["id"] == ss[K_CURRENT]), None)
    if current is not None:
        views.render_result(current["result"], fresh=current["id"] == ss[K_FRESH], stamp=current["time"],
                            entry_id=current["id"], sentences=result_sentences(cfg, current["result"]))
        st.button("New check", on_click=_new_check, icon=":material/add:", disabled=running)
    else:
        views.html_block(components.empty_state_html(
            corpus_size=report.corpus_size or cfg.data.expected_corpus_size, model_name=report.model_name or cfg.model.name,
            embedding_model=cfg.retrieval.embedding_model, reranker_model=cfg.retrieval.reranker_model,
            k=int(ss[K_K]), knowledge_kind=cfg.knowledge.kind, model_backend=cfg.product.judge_backend,
        ))


def main() -> None:
    st.set_page_config(page_title="ClaimLens · Scientific Claim Checker", page_icon=":material/diamond:",
                       layout="wide", initial_sidebar_state="expanded")
    theme.inject_theme()
    initial_cfg = load_config(PROFILE_PATH)
    labels = {"SciFact biomedical": "scifact", "General science": "general_science"}
    default = 1 if initial_cfg.knowledge.kind == "general_science" else 0
    choice = st.sidebar.selectbox("Evidence collection", list(labels), index=default, key=K_COLLECTION,
                                  disabled=bool(st.session_state.get(K_RUNNING)))
    kind = labels[choice]
    path = PROFILE_PATH if kind == initial_cfg.knowledge.kind else (str(ROOT / "configs/general_science.yaml") if kind == "general_science" else None)
    if K_PROFILE not in st.session_state or st.session_state[K_PROFILE] != path:
        for key in (K_READINESS, K_EVIDENCE, K_K, K_EXAMPLE):
            st.session_state.pop(key, None)
        st.session_state[K_CURRENT] = None
        st.session_state[K_PROFILE] = path
    cfg = initial_cfg if path == PROFILE_PATH else load_config(path)
    init_state(cfg)
    running = bool(st.session_state[K_RUNNING])
    report = get_readiness(cfg)

    header(cfg, report)
    sidebar(cfg, report, running)

    tab_check, tab_how, tab_eval = st.tabs(["Check a claim", "How it works", "Evaluation results"])
    # Static and saved-results tabs are drawn first so they stay usable while a check runs.
    with tab_how:
        if cfg.product.judge_backend != "ollama" and cfg.knowledge.kind == "scifact":
            st.info("This profile uses a local scientific NLI judge. The workflow below describes the original Ollama profile; source citations and the shared checking stages are retained.")
        views.render_how_it_works(cfg.knowledge.kind)
    with tab_eval:
        views.render_evaluation_tab(cfg.path("runs_dir"), cfg.path("retrieval_dir"))
    with tab_check:
        check_tab(cfg, report, running)


main()
