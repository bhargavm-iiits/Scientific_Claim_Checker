import json

import pytest

from claim_checker.errors import InputValidationError
from claim_checker.pipeline.aggregate import aggregate_verdicts
from claim_checker.schemas import ClaimResult
from tests.conftest import FakeTransport, purpose_of, scripted


@pytest.mark.parametrize("verdicts,expected", [
    (["SUPPORT", "NEI"], ("SUPPORT", False)),
    (["CONTRADICT", "NEI", "NEI"], ("CONTRADICT", False)),
    (["NEI", "NEI"], ("NEI", False)),
    ([], ("NEI", False)),
    (["SUPPORT", "CONTRADICT"], ("NEI", True)),
])
def test_aggregation_table(verdicts, expected):
    assert aggregate_verdicts(verdicts) == expected
    with pytest.raises(ValueError):
        aggregate_verdicts(["MAYBE"])


def test_full_pipeline_returns_grounded_support(make_checker, corpus):
    checker = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])}))
    res = checker.check("Vitamin D supplementation increases bone mineral density", k=3)
    assert res.status == "ok" and res.verdict == "SUPPORT" and not res.conflict
    assert [(q.doc_id, q.sentence_id) for q in res.evidence] == [("101", 2)]
    assert res.evidence[0].quote == corpus["101"].sentences[2]
    assert [d.doc_id for d in res.documents] == ["101", "202", "303"]
    # documents without selected evidence were judged NEI without a model call
    assert res.documents[1].judgement.skipped_model and res.documents[1].judgement.verdict == "NEI"
    stages = [e.stage for e in res.events]
    for s in ("validate_input", "retrieve", "select_evidence", "judge", "aggregate", "supervise", "explain"):
        assert s in stages
    assert {"retrieve", "judge", "total"} <= set(res.timings)
    # JSON download contains the same validated result
    assert ClaimResult.model_validate_json(res.to_json()) == res


def test_opposing_documents_produce_conflict_flag(make_checker):
    checker = make_checker(scripted({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("CONTRADICT", [1])}))
    res = checker.check("Vitamin D supplementation increases bone mineral density", k=2)
    assert res.status == "ok" and res.verdict == "NEI" and res.conflict
    assert "conflicting evidence" in res.explanation


def test_decisive_without_citation_is_repaired_then_downgraded(make_checker):
    checker = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [])}))
    res = checker.check("Vitamin D increases bone density", k=1)
    judge_calls = [c for c in checker.transport.calls if purpose_of(c) == "judge"]
    assert len(judge_calls) == 2  # original + one supervisor re-ask (bounded)
    assert "supervisor" in judge_calls[1]["messages"][1]["content"]
    assert res.status == "ok" and res.verdict == "NEI"
    j = res.documents[0].judgement
    assert j.downgraded and j.verdict == "NEI"
    assert any(e.data.get("flag") == "downgraded_no_citation" for e in res.events)


def test_citation_repair_can_succeed(make_checker):
    answers = iter([("SUPPORT", []), ("SUPPORT", [2])])
    base = scripted({"101": [2]}, {})

    def respond(p):
        if purpose_of(p) == "judge":
            v, c = next(answers)
            return json.dumps({"verdict": v, "cited_sentence_ids": c, "reason": "r", "confidence": 0.7})
        return base(p)

    res = make_checker(respond).check("Vitamin D increases bone density", k=1)
    assert res.verdict == "SUPPORT" and res.documents[0].judgement.repaired and res.evidence


def test_ollama_unavailable_is_an_operational_error_not_nei(make_checker, unavailable):
    checker = make_checker(transport=FakeTransport(fail=unavailable))
    res = checker.check("Vitamin D increases bone density")
    assert res.status == "error" and res.verdict is None
    assert res.error.error_type == "model_unavailable" and res.error.hint
    assert "not a scientific judgement" in res.explanation


@pytest.mark.parametrize("claim", ["", "   ", "x" * 2001, " ".join(["word"] * 300)])
def test_blank_or_oversized_input_rejected_before_retrieval(make_checker, claim):
    checker = make_checker(scripted({}, {}))
    res = checker.check(claim)
    assert res.status == "error" and res.error.error_type == "invalid_input"
    assert checker._retriever.calls == 0 and checker.transport.calls == []


def test_invalid_model_output_twice_ends_in_terminal_error(make_checker):
    checker = make_checker(lambda p: "garbage")
    res = checker.check("Vitamin D increases bone density", k=1)
    assert res.status == "error" and res.error.error_type == "model_invalid_output"
    assert len(checker.transport.calls) == 2


def test_evidence_off_mode_gives_judge_the_full_abstract(make_checker, corpus):
    checker = make_checker(scripted({}, {"101": ("SUPPORT", [2])}))
    res = checker.check("Vitamin D increases bone density", k=1, evidence_mode="full_abstract")
    judge_calls = [c for c in checker.transport.calls if purpose_of(c) == "judge"]
    assert not [c for c in checker.transport.calls if purpose_of(c) == "evidence"]
    for i, s in enumerate(corpus["101"].sentences):
        assert f"[{i}] {s}" in judge_calls[0]["messages"][1]["content"]
    assert res.verdict == "SUPPORT"


def test_oracle_mode_skips_retrieval(make_checker):
    checker = make_checker(scripted({"202": [1]}, {"202": ("CONTRADICT", [1])}))
    res = checker.check_variant("Vitamin D increases bone density", "oracle", oracle_doc_ids=["202"])
    assert checker._retriever.calls == 0
    assert res.verdict == "CONTRADICT" and res.documents[0].retrieval_method == "oracle"


def test_baselines(make_checker, corpus):
    checker = make_checker(scripted({}, {}))
    r1 = checker.check_variant("Vitamin D increases bone density", "judge_only")
    assert r1.status == "ok" and r1.verdict == "SUPPORT" and r1.documents == [] and r1.evidence == []
    r2 = checker.check_variant("Vitamin D increases bone density", "retrieve_judge_single_prompt")
    assert r2.status == "ok" and r2.verdict == "SUPPORT"
    assert [(q.doc_id, q.sentence_id, q.quote) for q in r2.evidence] == [("101", 2, corpus["101"].sentences[2])]
    assert len(r2.documents) == 3


def test_single_prompt_invalid_citation_downgraded(make_checker):
    def respond(p):
        return json.dumps({"verdict": "CONTRADICT", "citations": [{"doc_id": "999", "sentence_id": 0}],
                           "reason": "r", "confidence": 0.5})

    res = make_checker(respond).check_variant("Some claim", "retrieve_judge_single_prompt")
    assert res.status == "ok" and res.verdict == "NEI" and res.evidence == []
    assert any(e.data.get("flag") == "downgraded_no_citation" for e in res.events)


def test_no_decisive_evidence_gives_valid_nei(make_checker):
    # The evidence agent selects nothing anywhere: a scientific NEI (status ok), not an error.
    checker = make_checker(scripted({}, {}))
    res = checker.check("Stars are made of cheese", k=3)
    assert res.status == "ok" and res.verdict == "NEI" and not res.conflict and res.error is None
    assert res.evidence == [] and all(d.judgement.skipped_model for d in res.documents)
    assert "does not mean the claim is false" in res.explanation


def test_graph_failure_is_returned_as_structured_error(make_checker, monkeypatch):
    checker = make_checker(scripted({}, {}))

    def boom(*a, **k):
        raise RuntimeError("Recursion limit of 25 reached")

    monkeypatch.setattr(checker._graph, "invoke", boom)
    res = checker.check("Vitamin D increases bone density")
    assert res.status == "error" and res.verdict is None and res.error.error_type == "graph_error"


def test_single_prompt_citation_ids_are_normalized(make_checker, corpus):
    def respond(p):
        return json.dumps({"verdict": "SUPPORT", "citations": [{"doc_id": " 0101 ", "sentence_id": 2}],
                           "reason": "r", "confidence": 0.5})

    res = make_checker(respond).check_variant("Vitamin D increases bone density", "retrieve_judge_single_prompt")
    assert res.verdict == "SUPPORT" and [(q.doc_id, q.quote) for q in res.evidence] == [("101", corpus["101"].sentences[2])]


def test_explainer_verb_agreement_and_one_based_sentences():
    from claim_checker.agents.explainer import explain
    from claim_checker.schemas import DocumentJudgement

    one = [DocumentJudgement(doc_id="1", verdict="SUPPORT", cited_sentence_ids=[0], reason="r")]
    two = one + [DocumentJudgement(doc_id="2", verdict="SUPPORT", cited_sentence_ids=[3], reason="r")]
    assert "1 of 3 examined abstracts supports it" in explain("SUPPORT", False, one, {}, n_documents=3)
    text = explain("SUPPORT", False, two, {}, n_documents=3)
    assert "2 of 3 examined abstracts support it" in text and "(sentence 4)" in text


def test_single_prompt_explanation_is_claim_level_and_names_cited_documents(make_checker, corpus):
    # Regression: the explainer was given DocumentJudgement(doc_id="ALL") and printed
    # "1 of 3 examined abstracts supports it. document ALL: ...".
    res = make_checker(scripted({}, {})).check_variant("Vitamin D increases bone density",
                                                       "retrieve_judge_single_prompt")
    text = res.explanation
    assert "ALL" not in text and "1 of 3" not in text
    assert "one claim-level verdict" in text and "SUPPORTED by the retrieved abstracts" in text
    assert f'"{corpus["101"].title}" (document 101) (sentence 3)' in text
    assert "Model reason: doc 101 shows an increase" in text and "(uncalibrated): 0.70" in text


def test_single_prompt_downgrade_explanation_says_why(make_checker):
    def respond(p):
        return json.dumps({"verdict": "CONTRADICT", "citations": [{"doc_id": "999", "sentence_id": 0}],
                           "reason": "r", "confidence": 0.5})

    res = make_checker(respond).check_variant("Some claim", "retrieve_judge_single_prompt")
    assert res.verdict == "NEI" and "ALL" not in res.explanation
    assert "answered CONTRADICT but cited no valid sentence" in res.explanation


def test_oracle_explanation_says_supplied_not_retrieved(make_checker):
    checker = make_checker(scripted({"202": [1]}, {"202": ("CONTRADICT", [1])}))
    res = checker.check_variant("Vitamin D increases bone density", "oracle", oracle_doc_ids=["202"])
    assert "CONTRADICTED by the supplied reference abstracts" in res.explanation
    assert "1 of 1 examined reference abstract contradicts it" in res.explanation
    assert "retrieved" not in res.explanation
    nei = make_checker(scripted({}, {})).check_variant("Stars are cheese", "oracle", oracle_doc_ids=["404"])
    assert nei.verdict == "NEI" and "supplied abstracts are insufficient" in nei.explanation
    assert "retrieved" not in nei.explanation
    full = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])})).check("Vitamin D increases bone density")
    assert "SUPPORTED by the retrieved abstracts" in full.explanation


def test_judge_schema_invalid_repair_keeps_first_answer_and_filters_ids(make_checker, corpus):
    # Attempt 1 is schema-valid but cites id 3 which was not supplied (context is [2]); the repair
    # is garbage. The first answer is kept, the invalid id dropped, and a fallback event recorded.
    answers = iter([json.dumps({"verdict": "SUPPORT", "cited_sentence_ids": [2, 3], "reason": "r",
                                "confidence": 0.8}), "garbage"])
    base = scripted({"101": [2]}, {})

    def respond(p):
        return next(answers) if purpose_of(p) == "judge" else base(p)

    res = make_checker(respond).check("Vitamin D increases bone density", k=1)
    assert res.status == "ok" and res.verdict == "SUPPORT"
    j = res.documents[0].judgement
    assert j.cited_sentence_ids == [2] and j.dropped_invalid_ids == [3]
    assert any(e.data.get("flag") == "repair_fallback" for e in res.events)
    assert [(q.doc_id, q.quote) for q in res.evidence] == [("101", corpus["101"].sentences[2])]


def test_progress_callback_streams_stages_and_returns_same_result(make_checker):
    script = scripted({"101": [2]}, {"101": ("SUPPORT", [2])})
    plain = make_checker(script).check("Vitamin D increases bone density", k=2)
    seen: list[tuple[str, list[str]]] = []
    live = make_checker(script).check("Vitamin D increases bone density", k=2,
                                      progress=lambda stage, evs: seen.append((stage, [e.status for e in evs])))
    assert [s for s, _ in seen] == ["validate_input", "retrieve", "select_evidence", "judge", "aggregate",
                                    "supervise", "explain"]
    assert all("ok" in statuses for _, statuses in seen)
    strip = {"timings", "metadata", "events"}
    assert live.model_dump(exclude=strip) == plain.model_dump(exclude=strip)
    assert [(e.stage, e.status, e.message) for e in live.events] == [(e.stage, e.status, e.message) for e in plain.events]


def test_progress_callback_errors_never_fail_the_check(make_checker, unavailable):
    def bad(stage, events):
        raise RuntimeError("display broke")

    res = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])})).check("Vitamin D increases bone", progress=bad)
    assert res.status == "ok" and res.verdict == "SUPPORT" and res.metadata["progress_callback_errors"] == 7
    stages = []
    err = make_checker(transport=FakeTransport(fail=unavailable)).check(
        "Vitamin D increases bone density", progress=lambda s, e: stages.append(s))
    assert err.status == "error" and stages == ["validate_input", "handle_error"]


@pytest.mark.parametrize("variant", [None, "retrieve_judge_single_prompt", "judge_only"])
def test_token_budget_is_checked_before_resource_readiness(make_checker, unavailable, variant):
    # Ollama is down: an over-budget claim must still be reported as an input problem (the
    # tokenizer-only check runs before ensure_ready), not as "model unavailable".
    checker = make_checker(transport=FakeTransport(fail=unavailable))
    claim = " ".join(["word"] * 300)
    res = checker.check(claim) if variant is None else checker.check_variant(claim, variant)
    assert res.status == "error" and res.error.error_type == "invalid_input" and "tokens" in res.error.message
    ok = checker.check("Vitamin D increases bone density") if variant is None else \
        checker.check_variant("Vitamin D increases bone density", variant)
    assert ok.error.error_type == "model_unavailable"


def test_tokenizer_unavailable_falls_back_to_readiness_first(make_checker, unavailable, cfg, experiments, corpus):
    from claim_checker.models.ollama_client import OllamaClient
    from claim_checker.runtime import ClaimChecker
    from tests.conftest import FakeRetriever

    def broken_counter(text):
        raise OSError("tokenizer files missing")

    checker = ClaimChecker(cfg, experiments=experiments, corpus=corpus, retriever=FakeRetriever(corpus),
                           llm=OllamaClient(cfg.model, transport=FakeTransport(fail=unavailable)),
                           token_counter=broken_counter)
    res = checker.check("Vitamin D increases bone density")
    assert res.error.error_type == "model_unavailable" and res.error.hint  # setup hint, not a tokenizer crash
    assert checker.count_claim_tokens is not None
    assert make_checker().count_claim_tokens("a b c") == 3
    with pytest.raises(InputValidationError):
        make_checker().validate_claim("word " * 300)


class _TimedRetriever:
    """FakeRetriever variant with distinctive sub-timings, so copies are recognisable."""

    fingerprint = "fakefp"

    def __init__(self, corpus):
        from tests.conftest import FakeRetriever

        self._inner = FakeRetriever(corpus)
        self.calls = 0

    def retrieve(self, claim, method="hybrid", rerank=True, keep=10):
        from claim_checker.retrieval.index import RetrievalOutput

        self.calls += 1
        out = self._inner.retrieve(claim, method=method, rerank=rerank, keep=keep)
        return RetrievalOutput(hits=out.hits, timings={"bm25": 0.25, "dense": 0.5, "fuse": 0.0001, "rerank": 1.25})


def test_retrieval_cache_hit_is_flagged_and_subtimings_not_copied(make_checker, corpus):
    retriever = _TimedRetriever(corpus)
    checker = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])}), retriever=retriever)
    claim = "Vitamin D supplementation increases bone mineral density"
    first = checker.check(claim, k=3)
    assert first.metadata["retrieval_cache_hit"] is False
    assert "retrieve_original_compute_s" not in first.metadata
    assert first.timings["retrieve.rerank"] == 1.25 and first.timings["retrieve.bm25"] == 0.25
    assert not any(e.data.get("flag") == "retrieval_cache_hit" for e in first.events)

    second = checker.check(claim, k=3)
    assert retriever.calls == 1  # served from the in-process cache
    assert second.status == "ok" and second.verdict == first.verdict
    assert second.metadata["retrieval_cache_hit"] is True
    # No copied sub-timings presented as measured; the real (near-zero) lookup time is kept.
    assert not [k for k in second.timings if k.startswith("retrieve.")]
    assert "retrieve" in second.timings and second.timings["retrieve"] < 0.25
    assert second.timings["total"] < 0.25  # honest wall time: the search did not run
    assert second.metadata["retrieve_original_compute_s"] == {"bm25": 0.25, "dense": 0.5, "fuse": 0.0001,
                                                              "rerank": 1.25}
    flagged = [e for e in second.events if e.data.get("flag") == "retrieval_cache_hit"]
    assert len(flagged) == 1 and flagged[0].stage == "retrieve" and flagged[0].status == "info"
    assert flagged[0].data["original_compute_s"]["rerank"] == 1.25
    # Repeated hits keep reporting the original computation (the cached entry is not mutated).
    third = checker.check(claim, k=3)
    assert third.metadata["retrieve_original_compute_s"]["rerank"] == 1.25
    assert ClaimResult.model_validate_json(second.to_json()) == second


def test_retrieval_cache_shared_across_variants_is_flagged(make_checker, corpus):
    retriever = _TimedRetriever(corpus)
    checker = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])}), retriever=retriever)
    claim = "Vitamin D increases bone density"
    sp = checker.check_variant(claim, "retrieve_judge_single_prompt")
    assert sp.metadata["retrieval_cache_hit"] is False
    full = checker.check_variant(claim, "full")  # same (claim, hybrid, rerank, keep) key
    assert retriever.calls == 1 and full.metadata["retrieval_cache_hit"] is True
    assert full.metadata["retrieve_original_compute_s"]["rerank"] == 1.25
    assert not [k for k in full.timings if k.startswith("retrieve.")]
    sp2 = checker.check_variant(claim, "retrieve_judge_single_prompt")
    assert sp2.metadata["retrieval_cache_hit"] is True and sp2.timings["retrieve"] < 0.25
    assert any(e.stage == "retrieve" and e.data.get("flag") == "retrieval_cache_hit" for e in sp2.events)
    off = checker.check_variant(claim, "rerank_off")  # different key: computed, not cached
    assert retriever.calls == 2 and off.metadata["retrieval_cache_hit"] is False
    # No retrieval at all: no flag (not "miss").
    oracle = checker.check_variant(claim, "oracle", oracle_doc_ids=["101"])
    judge_only = checker.check_variant(claim, "judge_only")
    assert "retrieval_cache_hit" not in oracle.metadata and "retrieval_cache_hit" not in judge_only.metadata


def test_retrieval_without_cache_flag_is_unknown_not_miss():
    from claim_checker.pipeline.graph import RetrievalLookup, retrieval_cache_metadata
    from claim_checker.retrieval.index import RetrievalOutput

    assert retrieval_cache_metadata(RetrievalOutput(hits=[], timings={"bm25": 0.1})) == {}
    assert retrieval_cache_metadata(RetrievalLookup(hits=[], timings={}, cache_hit=False)) == \
        {"retrieval_cache_hit": False}
    assert retrieval_cache_metadata(RetrievalLookup(hits=[], timings={}, cache_hit=True,
                                                    original_timings={"rerank": 1.23456})) == \
        {"retrieval_cache_hit": True, "retrieve_original_compute_s": {"rerank": 1.2346}}


# ---------------------------------------------------------------------------
# Review fixes: echoed "Document <id>" citations, oracle misconfiguration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["Document 101", "document 101", "DOCUMENT: 101", "Doc 101", "doc#101",
                                 "Doc: 0101", " Document  101 ", "101", 101])
def test_citation_doc_id_strips_echoed_document_label(raw):
    from claim_checker.agents.judge import citation_doc_id

    assert citation_doc_id(raw) == "101"


def test_citation_doc_id_keeps_unlabelled_or_bare_label_values():
    from claim_checker.agents.judge import citation_doc_id

    assert citation_doc_id("Document") == "Document"  # nothing after the label: left as is (invalid)
    assert citation_doc_id("abc-7") == "abc-7"


def test_single_prompt_accepts_citation_echoing_document_header(make_checker, corpus):
    # Regression (final_v1 claims 350, 437, 793, 852): the model echoed the prompt header
    # "Document {doc_id}:" in its citation; the valid citation was rejected and the decisive
    # verdict downgraded to NEI.
    def respond(p):
        return json.dumps({"verdict": "SUPPORT", "citations": [{"doc_id": "Document 101", "sentence_id": 2}],
                           "reason": "r", "confidence": 0.8})

    checker = make_checker(respond)
    res = checker.check_variant("Vitamin D increases bone density", "retrieve_judge_single_prompt")
    assert res.status == "ok" and res.verdict == "SUPPORT"
    assert [(q.doc_id, q.sentence_id) for q in res.evidence] == [("101", 2)]
    assert res.metadata["single_prompt"]["downgraded"] is False
    assert len(checker.transport.calls) == 1  # accepted first time: no repair round trip
    assert not any(e.data.get("flag") == "downgraded_no_citation" for e in res.events)


def test_single_prompt_downgrade_reaches_prediction_record(make_checker):
    from claim_checker.evaluation.runner import prediction_record
    from claim_checker.schemas import ClaimGold

    def respond(p):
        return json.dumps({"verdict": "CONTRADICT", "citations": [{"doc_id": "999", "sentence_id": 0}],
                           "reason": "r", "confidence": 0.5})

    res = make_checker(respond).check_variant("Some claim", "retrieve_judge_single_prompt")
    g = ClaimGold(claim_id="9", claim="Some claim", split="validation", cited_doc_ids=["101"],
                  reference_doc_ids=["101"], evidence_by_doc={}, label_by_doc={}, label="NEI")
    rec = prediction_record("retrieve_judge_single_prompt", g, res)
    assert rec["verdict"] == "NEI" and rec["downgraded"] is True and rec["model_verdict"] == "CONTRADICT"
    ok = make_checker(scripted({}, {})).check_variant("Some claim", "retrieve_judge_single_prompt")
    rec_ok = prediction_record("retrieve_judge_single_prompt", g, ok)
    assert rec_ok["downgraded"] is False and rec_ok["model_verdict"] == "SUPPORT"


@pytest.mark.parametrize("ids", [[], ["  "]])
def test_oracle_without_ids_is_an_input_error_not_nei(make_checker, ids):
    checker = make_checker(scripted({}, {}))
    for res in (checker.check_variant("Vitamin D increases bone density", "oracle", oracle_doc_ids=ids),
                checker.check("Vitamin D increases bone density", oracle_doc_ids=ids)):
        assert res.status == "error" and res.verdict is None
        assert res.error.error_type == "invalid_input" and res.error.stage == "validate_input"
        assert "--oracle-doc-ids" in res.error.hint
    # check_variant with no ids at all (None) is the same misconfiguration
    res = checker.check_variant("Vitamin D increases bone density", "oracle")
    assert res.status == "error" and res.error.error_type == "invalid_input"
    assert checker.transport.calls == [] and checker._retriever.calls == 0


def test_oracle_id_missing_from_corpus_is_a_data_error(make_checker):
    checker = make_checker(scripted({"202": [1]}, {"202": ("CONTRADICT", [1])}))
    res = checker.check_variant("Vitamin D increases bone density", "oracle", oracle_doc_ids=["202", "99999"])
    assert res.status == "error" and res.verdict is None
    assert res.error.error_type == "data_integrity" and res.error.stage == "retrieve"
    assert "99999" in res.error.message and res.error.hint
    assert checker.transport.calls == []  # nothing judged on a partial document set


def test_oracle_ids_are_normalized_and_deduplicated(make_checker):
    checker = make_checker(scripted({"202": [1]}, {"202": ("CONTRADICT", [1])}))
    res = checker.check_variant("Vitamin D increases bone density", "oracle", oracle_doc_ids=[" 0202 ", "202"])
    assert res.status == "ok" and [d.doc_id for d in res.documents] == ["202"]


def test_cli_check_oracle_requires_and_forwards_doc_ids(make_checker, monkeypatch, cfg, capsys):
    from claim_checker import cli, runtime

    checker = make_checker(scripted({"202": [1]}, {"202": ("CONTRADICT", [1])}))
    monkeypatch.setattr(runtime, "ClaimChecker", lambda cfg: checker)
    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    claim = ["--claim", "Vitamin D increases bone density"]
    assert cli.main(["check", *claim, "--variant", "oracle"]) == 2
    out = capsys.readouterr().out
    assert "ERROR (invalid_input)" in out and "--oracle-doc-ids" in out
    assert cli.main(["check", *claim, "--variant", "oracle", "--oracle-doc-ids", "202"]) == 0
    assert "Verdict: CONTRADICT" in capsys.readouterr().out
    assert cli.main(["check", *claim, "--variant", "full", "--oracle-doc-ids", "202"]) == 2
    assert "only applies to an oracle variant" in capsys.readouterr().err
