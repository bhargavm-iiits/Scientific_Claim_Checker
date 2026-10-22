"""Additive result fields requested by the UI: claim-level confidence, document sentences, one-based
reason display (v2), explanation punctuation and the warm-state helpers. Offline."""

from __future__ import annotations

import json

from claim_checker.agents.explainer import explain, one_based_reason
from claim_checker.runtime import claim_confidence
from claim_checker.schemas import DocumentJudgement, ErrorInfo
from tests.conftest import FakeTransport, purpose_of, scripted, v2_answer


def test_claim_confidence_rules():
    s = DocumentJudgement(doc_id="1", verdict="SUPPORT", cited_sentence_ids=[0], confidence=0.7)
    s2 = DocumentJudgement(doc_id="2", verdict="SUPPORT", cited_sentence_ids=[0], confidence=0.9)
    skipped = DocumentJudgement(doc_id="3", verdict="NEI", skipped_model=True)
    assert claim_confidence("SUPPORT", False, [s, s2, skipped]) == 0.9
    assert claim_confidence("SUPPORT", True, [s, s2]) is None  # conflict
    assert claim_confidence(None, False, [s]) is None  # error
    assert claim_confidence("NEI", False, [skipped]) is None  # NEI without a model call


def test_pipeline_result_has_confidence_and_document_sentences(make_checker, corpus):
    res = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])})).check("Vitamin D increases bone density", k=2)
    assert res.metadata["confidence"] == 0.8
    assert res.documents[0].sentences == corpus["101"].sentences
    assert json.loads(res.to_json())["documents"][1]["sentences"] == corpus["202"].sentences


def test_baselines_report_confidence(make_checker):
    checker = make_checker(scripted({}, {}))
    assert checker.check_judge_only("Vitamin D increases bone density").metadata["confidence"] == 0.6
    assert checker.check_single_prompt("Vitamin D increases bone density").metadata["confidence"] == 0.7


def test_one_based_reason_rewrites_only_supplied_ids():
    text = "Sentence [0] states X; [3] and sentences 5 show Y; [9] is unknown; see sentence 2."
    assert one_based_reason(text, [0, 2, 3, 5]) == (
        "Sentence 1 states X; [S4] and sentences 6 show Y; [9] is unknown; see sentence 3.")


def test_v2_reason_display_is_one_based_and_v1_reason_untouched(make_checker):
    def respond(p):
        if purpose_of(p) == "judge":
            return json.dumps(v2_answer("SUPPORT", [2], reason="Sentence [2] reports the increase."))
        return scripted({"101": [2]}, {})(p)

    res = make_checker(respond).check("Vitamin D increases bone density", k=1, prompt_set="v2")
    j = res.documents[0].judgement
    assert j.reason == "Sentence [2] reports the increase."  # model text kept, zero-based
    assert j.reason_display == "Sentence 3 reports the increase."
    assert "Sentence 3 reports the increase." in res.explanation
    v1 = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])})).check("Vitamin D increases bone density", k=1)
    assert v1.documents[0].judgement.reason_display is None


def test_error_explanation_has_single_final_period():
    err = ErrorInfo(error_type="model_unavailable", message="Ollama is not reachable (ConnectError).",
                    hint="Run `ollama serve`.")
    text = explain(None, False, [], {}, error=err)
    assert "(ConnectError). Suggested fix: Run `ollama serve`." in text and ".." not in text
    err2 = ErrorInfo(error_type="x", message="no period here", hint="do this")
    assert explain(None, False, [], {}, error=err2).endswith("no period here. Suggested fix: do this.")


def test_is_warm_reflects_loaded_resources(make_checker):
    checker = make_checker(transport=FakeTransport(scripted({}, {})))
    assert not checker.is_warm  # injected retriever, but no corpus access / model check yet
    checker.check("Vitamin D increases bone density", k=1)
    assert checker.is_warm
    loaded = checker.resources_loaded()
    assert loaded["corpus"] and loaded["retriever"] and loaded["model_checked"]
