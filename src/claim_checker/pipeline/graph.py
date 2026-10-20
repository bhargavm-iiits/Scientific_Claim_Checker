"""LangGraph workflow:

    validate_input -> retrieve -> select_evidence -> judge -> aggregate -> supervise -> explain -> END
                                                       ^                      |
                                                       +---- bounded repair --+
    any node failing operationally -> handle_error -> END   (status=error, verdict unset)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from claim_checker.agents.evidence import select_evidence
from claim_checker.agents.examples import ExampleLibrary
from claim_checker.agents.explainer import explain
from claim_checker.agents.judge import judge_document
from claim_checker.agents.judge_lora import judge_lora
from claim_checker.agents.judge_nli import judge_nli
from claim_checker.agents.judge_v3 import judge_v3
from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import AppConfig
from claim_checker.errors import ClaimCheckerError, DataIntegrityError, InputValidationError
from claim_checker.models.ollama_client import OllamaClient
from claim_checker.pipeline.aggregate import aggregate
from claim_checker.pipeline.consistency import (
    consistency_feedback,
    consistency_flags,
    downgrade_inconsistent,
    nei_disagrees_with_checklist,
)
from claim_checker.pipeline.state import GraphState, PipelineOptions
from claim_checker.pipeline.supervisor import (
    CITATION_FEEDBACK,
    downgrade,
    quote_problems,
    uncited_decisive,
    validate_claim_text,
)
from claim_checker.schemas import (
    CorpusDocument,
    ErrorInfo,
    RetrievalHit,
    StageEvent,
    normalize_doc_id,
)


@dataclass
class PipelineResources:
    """Everything the graph needs. Callables let the runtime load heavy resources lazily."""

    cfg: AppConfig
    get_corpus: Callable[[], dict[str, CorpusDocument]]
    get_client: Callable[[], OllamaClient]
    prompts: PromptLibrary
    # (claim, method, rerank, keep) -> .hits, .timings (freshly measured only), optionally
    # .cache_hit / .original_timings (see RetrievalLookup)
    retrieve: Callable[[str, str, bool, int], Any]
    ensure_ready: Callable[[bool], None]  # (need_index) -> raises operational errors
    count_tokens: Callable[[str], int] | None = None
    get_examples: Callable[[], ExampleLibrary | None] | None = None
    get_verifier: Callable[[], Any] | None = None
    get_lora_judge: Callable[[], Any] | None = None
    ensure_local_ready: Callable[[bool], None] | None = None
    get_evidence_selector: Callable[[], Any] | None = None


@dataclass(frozen=True)
class RetrievalLookup:
    """What ``ClaimChecker._retrieve`` returns: the hits plus honest timing provenance.

    ``timings`` holds only sub-step timings (``bm25``, ``dense``, ``fuse``, ``rerank``) measured
    during *this* call, so it is empty when the result came from the in-process retrieval cache.
    In that case ``cache_hit`` is True and ``original_timings`` keeps the sub-timings of the earlier
    computation that filled the cache - recorded for reference, never re-presented as measured.
    """

    hits: list
    timings: dict[str, float]
    cache_hit: bool
    original_timings: dict[str, float] | None = None


RETRIEVAL_CACHE_NOTE = ("served from the in-process retrieval cache: the search was not re-run and its "
                        "sub-step timings were not re-measured (original computation in "
                        "retrieve_original_compute_s)")


def retrieval_cache_metadata(lookup: Any) -> dict:
    """Result-``metadata`` keys saying whether retrieval was served from the in-process cache:
    ``retrieval_cache_hit`` (bool) and, on a hit, ``retrieve_original_compute_s`` (the cached
    computation's sub-step seconds). Empty for a retrieval object without a ``cache_hit``
    attribute (e.g. an injected test callable): a missing flag means "unknown", never "miss"."""
    hit = getattr(lookup, "cache_hit", None)
    if hit is None:
        return {}
    meta: dict = {"retrieval_cache_hit": bool(hit)}
    if hit:
        orig = getattr(lookup, "original_timings", None) or {}
        meta["retrieve_original_compute_s"] = {k: round(float(v), 4) for k, v in orig.items()}
    return meta


ORACLE_HINT = ("Oracle mode is an evaluation upper bound: the evaluation runner supplies each claim's gold "
               "reference abstracts. To try it by hand pass the ids explicitly, e.g. "
               "`claim-checker check --variant oracle --oracle-doc-ids 13734012 --claim \"...\"`.")


def _oracle_id(value: Any) -> str:
    try:
        return normalize_doc_id(value)
    except ValueError:
        return str(value).strip()


def _events(stage: str, raw: list[dict]) -> list[StageEvent]:
    return [StageEvent(stage=stage, status=e.get("status", "info"), message=e.get("message", ""),
                       data=e.get("data", {})) for e in raw]


def _node(name: str, fn: Callable[[GraphState], dict]) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict:
        t0 = time.perf_counter()
        try:
            update = fn(state) or {}
        except ClaimCheckerError as exc:
            update = {"status": "error", "error": ErrorInfo(**{**exc.to_dict(), "stage": exc.stage or name})}
        except Exception as exc:  # unexpected failure -> still a visible operational error
            update = {"status": "error", "error": ErrorInfo(
                error_type="unexpected_error", message=f"{type(exc).__name__}: {exc}", stage=name)}
        dt = time.perf_counter() - t0
        events = list(update.pop("events", []))
        if update.get("status") == "error":
            err: ErrorInfo = update["error"]
            events.append(StageEvent(stage=name, status="error", message=err.message,
                                     data={"error_type": err.error_type}))
        else:
            events.append(StageEvent(stage=name, status="ok", message="completed", data={"seconds": round(dt, 4)}))
        update["events"] = events
        timings = dict(update.pop("timings", {}) or {})
        timings[name] = timings.get(name, 0.0) + dt
        update["timings"] = timings
        return update

    return run


def _route(next_node: str) -> Callable[[GraphState], str]:
    def route(state: GraphState) -> str:
        return "handle_error" if state.get("status") == "error" else next_node

    return route


def build_graph(res: PipelineResources):
    cfg = res.cfg

    def validate_input(state: GraphState) -> dict:
        opts: PipelineOptions = state["options"]
        claim = validate_claim_text(state.get("claim"), cfg.runtime.max_claim_chars,
                                    cfg.runtime.max_claim_tokens, None)
        if not 1 <= opts.k <= cfg.retrieval.keep_top:
            raise InputValidationError(f"k must be between 1 and {cfg.retrieval.keep_top}", stage="validate_input")
        if opts.oracle_doc_ids is not None and not [d for d in opts.oracle_doc_ids if str(d).strip()]:
            # An empty oracle list is a misconfiguration, never "no evidence": without this check it
            # would examine zero abstracts and return a scientific-looking NEI.
            raise InputValidationError("Oracle mode was requested without any reference document ids.",
                                       stage="validate_input", hint=ORACLE_HINT)
        # Token budget first: it needs only the tokenizer, so an over-budget claim is rejected before
        # the index, embedding model, re-ranker or Ollama are loaded or contacted.
        counted = False
        if res.count_tokens is not None:
            try:
                claim = validate_claim_text(claim, cfg.runtime.max_claim_chars, cfg.runtime.max_claim_tokens,
                                            res.count_tokens)
                counted = True
            except InputValidationError:
                raise
            except Exception:  # tokenizer not available yet (e.g. setup incomplete): readiness first
                counted = False
        local_evidence = opts.evidence_mode == "full_abstract" or cfg.evidence_selector.enabled
        ready = res.ensure_local_ready if opts.judge_backend in {"nli", "lora"} and local_evidence and res.ensure_local_ready else res.ensure_ready
        ready(opts.oracle_doc_ids is None)
        if res.count_tokens is not None and not counted:
            claim = validate_claim_text(claim, cfg.runtime.max_claim_chars, cfg.runtime.max_claim_tokens,
                                        res.count_tokens)
        return {"claim": claim, "status": "running"}

    def retrieve(state: GraphState) -> dict:
        opts: PipelineOptions = state["options"]
        corpus = res.get_corpus()
        if opts.oracle_doc_ids is not None:
            ids = list(dict.fromkeys(_oracle_id(d) for d in opts.oracle_doc_ids if str(d).strip()))
            missing = [d for d in ids if d not in corpus]
            if missing:
                # Silently dropping unknown ids would judge a subset (or nothing) and could turn a
                # data problem into an NEI verdict; fail visibly instead.
                raise DataIntegrityError(
                    f"Oracle document id(s) not found in the corpus: {missing}.", stage="retrieve",
                    hint="Pass SciFact corpus document ids (as in the gold reference_doc_ids); "
                         "check that `prepare-data` produced the corpus these ids come from.")
            hits = [RetrievalHit(doc_id=d, rank=i, method="oracle", score=0.0, title=corpus[d].title)
                    for i, d in enumerate(ids, start=1)]
            events = [{"status": "info", "message": f"oracle mode: {len(ids)} reference abstracts supplied"}]
            return {"hits": hits, "doc_ids": ids, "events": _events("retrieve", events)}
        keep = max(cfg.retrieval.keep_top, opts.k)
        out = res.retrieve(state["claim"], opts.retrieval_method, opts.rerank, keep)
        hits = list(out.hits)
        # Only sub-timings measured now; on a cache hit there are none (the node's own wall time,
        # recorded by _node as "retrieve", is then the real near-zero lookup cost).
        timings = {f"retrieve.{k}": v for k, v in (out.timings or {}).items()}
        examined = hits[:opts.k]
        gate = cfg.verifier.min_retrieval_score if opts.judge_backend == "nli" else None
        if gate is not None:
            examined = [hit for hit in examined if hit.score >= gate]
        update: dict = {"hits": hits, "doc_ids": [h.doc_id for h in examined], "timings": timings}
        cache_meta = retrieval_cache_metadata(out)
        if cache_meta:
            update["retrieval_cache"] = cache_meta
        if cache_meta.get("retrieval_cache_hit"):
            update["events"] = _events("retrieve", [{
                "status": "info", "message": RETRIEVAL_CACHE_NOTE,
                "data": {"flag": "retrieval_cache_hit", "original_compute_s": cache_meta["retrieve_original_compute_s"]},
            }])
        if gate is not None:
            update.setdefault("events", []).extend(_events("retrieve", [{"status": "info",
                "message": "Applied the development-calibrated retrieval relevance filter.",
                "data": {"min_retrieval_score": gate, "kept": len(examined), "requested_top_k": opts.k}}]))
        return update

    def select(state: GraphState) -> dict:
        opts: PipelineOptions = state["options"]
        if opts.evidence_mode != "evidence":
            return {"selections": {}, "events": _events("select_evidence", [
                {"status": "info", "message": "evidence agent off: judge reads full abstracts"}])}
        if cfg.evidence_selector.enabled:
            from claim_checker.agents.trained_evidence import select_trained_evidence

            if res.get_evidence_selector is None:
                raise ClaimCheckerError("The trained evidence selector is not configured", stage="select_evidence")
            selections, events = {}, []
            for d in state["doc_ids"]:
                verifier = res.get_verifier() if cfg.evidence_selector.alpha < 1 and res.get_verifier else None
                selection, event = select_trained_evidence(res.get_evidence_selector(), verifier, state["claim"],
                                                          res.get_corpus()[d], cfg.evidence_selector)
                selections[d] = selection
                events.extend(event)
            return {"selections": selections, "events": _events("select_evidence", events)}
        corpus, client = res.get_corpus(), res.get_client()
        selections, events = {}, []
        for d in state["doc_ids"]:
            sel, ev = select_evidence(client, res.prompts, state["claim"], corpus[d], cfg.evidence)
            selections[d] = sel
            events += ev
        return {"selections": selections, "events": _events("select_evidence", events)}

    def judge(state: GraphState) -> dict:
        opts: PipelineOptions = state["options"]
        corpus, client = res.get_corpus(), res.get_client()
        judgements = dict(state.get("judgements") or {})
        pending = state.get("pending_repair") or []
        targets = pending if pending else state["doc_ids"]
        repair_feedback = state.get("repair_feedback") or {}
        consistency_pending = state.get("consistency_pending") or {}
        events = []
        for d in targets:
            feedback = None
            if pending:
                feedback = repair_feedback.get(d) or CITATION_FEEDBACK.format(verdict=judgements[d].verdict)
            sel = (state.get("selections") or {}).get(d)
            if opts.judge_backend == "lora":
                if res.get_lora_judge is None:
                    raise ClaimCheckerError("LoRA judge is not configured", stage="judge")
                j = judge_lora(res.get_lora_judge(), state["claim"], corpus[d])
                ev = []
            elif opts.judge_backend == "nli":
                if res.get_verifier is None:
                    raise ClaimCheckerError("NLI verifier is not configured", stage="judge")
                ids = list(range(len(corpus[d].sentences))) if opts.evidence_mode == "full_abstract" else (sel.sentence_ids if sel else [])
                cutoffs = {"SUPPORT": cfg.verifier.support_threshold if cfg.verifier.support_threshold is not None else cfg.verifier.threshold,
                           "CONTRADICT": cfg.verifier.contradict_threshold if cfg.verifier.contradict_threshold is not None else cfg.verifier.threshold}
                j = judge_nli(res.get_verifier(), state["claim"], corpus[d], ids, threshold=cutoffs)
                ev = []
            elif opts.prompt_set == "v3":
                ids = list(range(len(corpus[d].sentences))) if opts.evidence_mode == "full_abstract" else (sel.sentence_ids if sel else [])
                examples = res.get_examples() if res.get_examples and cfg.judge_v3.use_examples else None
                j, ev = judge_v3(client, res.prompts, state["claim"], corpus[d], ids, cfg.judge_v3,
                                 examples, feedback)
            else:
                j, ev = judge_document(client, res.prompts, state["claim"], corpus[d], opts.evidence_mode,
                                       sel.sentence_ids if sel else None, cfg.judge, feedback=feedback,
                                       prompt_set=opts.prompt_set, think=opts.judge_think, v2cfg=cfg.judge_v2)
            if pending and d in consistency_pending:
                j = j.model_copy(update={"consistency_flags": [f"before_repair:{f}" for f in consistency_pending[d]]})
            judgements[d] = j
            events += ev
        return {"judgements": judgements, "pending_repair": [], "events": _events("judge", events)}

    def aggregate_node(state: GraphState) -> dict:
        judgements = [state["judgements"][d] for d in state["doc_ids"] if d in (state.get("judgements") or {})]
        verdict, conflict = aggregate(judgements)
        return {"verdict": verdict, "conflict": conflict}

    def supervise(state: GraphState) -> dict:
        opts: PipelineOptions = state["options"]
        if opts.prompt_set == "v2" and cfg.judge_v2.consistency_check:
            return supervise_v2(state)
        corpus = res.get_corpus()
        judgements = dict(state.get("judgements") or {})
        events = []
        problems = quote_problems(judgements.values(), corpus)
        if problems:
            raise ClaimCheckerError("Invalid citations after validation: " + "; ".join(problems), stage="supervise")
        missing = uncited_decisive(judgements.values())
        repairs = state.get("citation_repairs", 0)
        if missing and repairs < cfg.judge.max_citation_repairs:
            events.append({"status": "warning", "message": f"decisive verdict without citation in {missing}; re-asking judge",
                           "data": {"flag": "citation_repair", "doc_ids": missing}})
            return {"pending_repair": missing, "citation_repairs": repairs + 1,
                    "events": _events("supervise", events)}
        if missing:
            for d in missing:
                judgements[d] = downgrade(judgements[d])
            events.append({"status": "warning", "message": f"downgraded to NEI (no valid citation): {missing}",
                           "data": {"flag": "downgraded_no_citation", "doc_ids": missing}})
            verdict, conflict = aggregate(judgements[d] for d in state["doc_ids"] if d in judgements)
            return {"judgements": judgements, "verdict": verdict, "conflict": conflict, "pending_repair": [],
                    "events": _events("supervise", events)}
        return {"pending_repair": []}

    def supervise_v2(state: GraphState) -> dict:
        """v2 supervisor: the v1 citation policy plus the verdict-consistency check. Both problems share
        the single repair round (``judge.max_citation_repairs``); what is still wrong afterwards is
        downgraded to NEI (``downgraded_no_citation`` / ``downgraded_inconsistent``)."""
        corpus = res.get_corpus()
        judgements = dict(state.get("judgements") or {})
        events = []
        problems = quote_problems(judgements.values(), corpus)
        if problems:
            raise ClaimCheckerError("Invalid citations after validation: " + "; ".join(problems), stage="supervise")
        order = [d for d in state["doc_ids"] if d in judgements]
        missing = uncited_decisive(judgements[d] for d in order)
        inconsistent: dict[str, list[str]] = {}
        for d in order:
            flags = [] if d in missing else consistency_flags(judgements[d])
            if flags:
                inconsistent[d] = flags
        repairs = state.get("citation_repairs", 0)
        if (missing or inconsistent) and repairs < cfg.judge.max_citation_repairs:
            feedback = {d: CITATION_FEEDBACK.format(verdict=judgements[d].verdict) for d in missing}
            feedback.update({d: consistency_feedback(judgements[d], f) for d, f in inconsistent.items()})
            if missing:
                events.append({"status": "warning", "message": f"decisive verdict without citation in {missing}; "
                               "re-asking judge", "data": {"flag": "citation_repair", "doc_ids": missing}})
            if inconsistent:
                events.append({"status": "warning", "message": "verdict inconsistent with the judge's own "
                               f"checklist/reason in {sorted(inconsistent)}; re-asking judge",
                               "data": {"flag": "consistency_repair", "doc_flags": inconsistent}})
            return {"pending_repair": [d for d in order if d in feedback], "repair_feedback": feedback,
                    "consistency_pending": inconsistent, "citation_repairs": repairs + 1,
                    "events": _events("supervise", events)}
        nei_odd = [d for d in order if nei_disagrees_with_checklist(judgements[d])]
        if nei_odd:
            events.append({"status": "info", "message": f"NEI although the checklist reads as decisive in {nei_odd} "
                           "(reported only)", "data": {"flag": "nei_checklist_decisive", "doc_ids": nei_odd}})
        if not (missing or inconsistent):
            return {"pending_repair": [], "events": _events("supervise", events)}
        for d in missing:
            judgements[d] = downgrade(judgements[d])
        for d, flags in inconsistent.items():
            judgements[d] = downgrade_inconsistent(judgements[d], flags)
        if missing:
            events.append({"status": "warning", "message": f"downgraded to NEI (no valid citation): {missing}",
                           "data": {"flag": "downgraded_no_citation", "doc_ids": missing}})
        if inconsistent:
            events.append({"status": "warning", "message": "downgraded to NEI (verdict still inconsistent after "
                           f"the re-ask): {sorted(inconsistent)}",
                           "data": {"flag": "downgraded_inconsistent", "doc_flags": inconsistent}})
        verdict, conflict = aggregate(judgements[d] for d in order)
        return {"judgements": judgements, "verdict": verdict, "conflict": conflict, "pending_repair": [],
                "events": _events("supervise", events)}

    def explain_node(state: GraphState) -> dict:
        corpus = res.get_corpus()
        judgements = [state["judgements"][d] for d in state["doc_ids"] if d in (state.get("judgements") or {})]
        titles = {d: corpus[d].title for d in state["doc_ids"]}
        opts: PipelineOptions = state["options"]
        text = explain(state["verdict"], state.get("conflict", False), judgements, titles,
                       n_documents=len(state["doc_ids"]),
                       context="supplied" if opts.oracle_doc_ids is not None else "retrieved",
                       document_kind="passage" if res.cfg.knowledge.kind == "general_science" else "abstract")
        return {"explanation": text, "status": "ok"}

    def handle_error(state: GraphState) -> dict:
        return {"status": "error", "verdict": None, "conflict": False,
                "explanation": explain(None, False, [], {}, error=state.get("error"))}

    def after_supervise(state: GraphState) -> str:
        if state.get("status") == "error":
            return "handle_error"
        return "judge" if state.get("pending_repair") else "explain"

    g = StateGraph(GraphState)
    g.add_node("validate_input", _node("validate_input", validate_input))
    g.add_node("retrieve", _node("retrieve", retrieve))
    g.add_node("select_evidence", _node("select_evidence", select))
    g.add_node("judge", _node("judge", judge))
    g.add_node("aggregate", _node("aggregate", aggregate_node))
    g.add_node("supervise", _node("supervise", supervise))
    g.add_node("explain", _node("explain", explain_node))
    g.add_node("handle_error", handle_error)
    g.add_edge(START, "validate_input")
    for a, b in (("validate_input", "retrieve"), ("retrieve", "select_evidence"),
                 ("select_evidence", "judge"), ("judge", "aggregate"), ("aggregate", "supervise")):
        g.add_conditional_edges(a, _route(b), {b: b, "handle_error": "handle_error"})
    g.add_conditional_edges("supervise", after_supervise,
                            {"judge": "judge", "explain": "explain", "handle_error": "handle_error"})
    g.add_conditional_edges("explain", _route(END), {END: END, "handle_error": "handle_error"})
    g.add_edge("handle_error", END)
    return g.compile()
