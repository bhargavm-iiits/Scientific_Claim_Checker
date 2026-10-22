import pytest
from conftest import make_doc

from claim_checker.agents.judge_nli import choose_label, judge_nli
from claim_checker.pipeline.aggregate import aggregate
from claim_checker.schemas import DocumentJudgement


class FakeVerifier:
    def windows(self, claim, sentences):
        return [list(range(len(sentences)))]

    def score(self, premises, hypotheses):
        return [{"SUPPORT": 0.96, "CONTRADICT": 0.01, "NEI": 0.03} for _ in premises]


def test_verifier_threshold_does_not_turn_uncertain_answers_into_decisive_labels():
    scores = {"SUPPORT": 0.8, "CONTRADICT": 0.1, "NEI": 0.1}
    assert choose_label(scores, 0.9) == "NEI"
    assert choose_label(scores, 0.7) == "SUPPORT"


def test_verifier_quotes_keep_original_sentence_ids():
    doc = make_doc("101", "Title", ["Background.", "Irrelevant.", "Finding A.", "Finding B."])
    result = judge_nli(FakeVerifier(), "claim", doc, [2, 3], threshold=0.9)
    assert result.verdict == "SUPPORT"
    assert result.cited_sentence_ids == [2, 3]
    assert result.context_sentence_ids == [2, 3]
    assert "Background." not in result.reason


def test_verifier_rejects_sentence_ids_outside_abstract():
    doc = make_doc("101", "Title", ["Background."])
    with pytest.raises(ValueError, match="outside"):
        judge_nli(FakeVerifier(), "claim", doc, [8], threshold=0.9)


def test_verifier_empty_evidence_is_nei_without_inference():
    result = judge_nli(object(), "claim", make_doc("101", "Title", ["Background."]), [], threshold=0.9)
    assert result.verdict == "NEI" and result.skipped_model


def test_nli_pipeline_returns_grounded_quotes_without_an_ollama_judge_call(make_checker):
    checker = make_checker()
    checker._verifier = FakeVerifier()
    result = checker.check("A scientific claim.", k=1, evidence_mode="full_abstract", judge_backend="nli")
    assert result.status == "ok" and result.verdict == "SUPPORT"
    assert result.metadata["judge_backend"] == "nli"
    assert result.evidence
    for quote in result.evidence:
        assert quote.quote == checker.corpus[quote.doc_id].sentences[quote.sentence_id]
    assert not checker.transport.calls


def test_individual_class_cutoffs_are_applied():
    cutoffs = {"SUPPORT": 0.7, "CONTRADICT": 0.98}
    assert choose_label({"SUPPORT": 0.9, "CONTRADICT": 0.05, "NEI": 0.05}, cutoffs) == "SUPPORT"
    assert choose_label({"SUPPORT": 0.05, "CONTRADICT": 0.9, "NEI": 0.05}, cutoffs) == "NEI"


def test_cli_uses_full_abstract_backend_from_config_without_an_ollama_call(make_checker, monkeypatch, capsys):
    from claim_checker import cli, runtime

    checker = make_checker()
    checker.cfg.product.judge_backend = "nli"
    checker.cfg.product.evidence_mode = "full_abstract"
    checker._verifier = FakeVerifier()
    monkeypatch.setattr(runtime, "ClaimChecker", lambda cfg: checker)
    monkeypatch.setattr(cli, "load_config", lambda path=None: checker.cfg)
    assert cli.main(["check", "--claim", "A scientific claim."]) == 0
    assert "Verdict: SUPPORT" in capsys.readouterr().out
    assert not checker.transport.calls


def test_window_conflict_is_not_hidden_by_a_decisive_second_document():
    conflicted = DocumentJudgement(doc_id="1", verdict="NEI", consistency_flags=["window_conflict"])
    supporting = DocumentJudgement(doc_id="2", verdict="SUPPORT", cited_sentence_ids=[0])
    assert aggregate([conflicted, supporting]) == ("NEI", True)


def test_nli_only_evaluation_identity_needs_no_ollama(cfg, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from claim_checker.data.store import sha256_file
    from claim_checker.evaluation import runner

    processed = cfg.path("processed_dir")
    processed.mkdir(parents=True)
    (processed / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    (processed / "claim_gold_train.jsonl").write_text("{}\n", encoding="utf-8")
    weights = tmp_path / "verifier"
    weights.mkdir()
    for name in ("model.safetensors", "tokenizer.json", "config.json"):
        (weights / name).write_text("identity-only fixture", encoding="utf-8")
    cfg = cfg.model_copy(update={"verifier": cfg.verifier.model_copy(update={"model_dir": str(weights)})})
    monkeypatch.setattr(runner, "split_fingerprint", lambda *args: "fixed-test-split")
    # There is deliberately no llm property: touching it is a regression.
    checker = SimpleNamespace(prompts=SimpleNamespace(hashes=lambda: {}))
    identity = runner.run_identity(cfg, checker, "local", "dev_tune", [], False, {"v1"}, {"nli"})
    assert identity["model"] == {"name": "Scientific NLI verifier", "digest": sha256_file(weights / "model.safetensors")}


def test_title_context_is_supplied_without_changing_quote_indices():
    class TitleVerifier(FakeVerifier):
        include_title = True

        def format_premise(self, title, text):
            return f"Title: {title}\nAbstract: {text}"

        def windows(self, claim, sentences, title=""):
            assert title == "Study title"
            return super().windows(claim, sentences)

        def score(self, premises, hypotheses):
            assert all(p.startswith("Title: Study title\nAbstract: ") for p in premises)
            return super().score(premises, hypotheses)

    doc = make_doc("101", "Study title", ["Background.", "Finding A.", "Finding B."])
    result = judge_nli(TitleVerifier(), "Claim", doc, [1, 2], threshold=0.5)
    assert result.verdict == "SUPPORT"
    assert result.cited_sentence_ids == [1, 2]
    assert result.context_sentence_ids == [1, 2]


def test_relevance_filter_keeps_requested_population_and_reports_no_evidence(make_checker):
    checker = make_checker()
    checker.cfg.verifier.min_retrieval_score = 1e6
    checker._verifier = FakeVerifier()
    result = checker.check("A scientific claim.", k=3, evidence_mode="full_abstract", judge_backend="nli")
    assert result.status == "ok" and result.verdict == "NEI"
    assert result.documents == []
    assert result.metadata["retrieval_hits"]
    assert any(e.data.get("kept") == 0 for e in result.events)


def test_oracle_documents_bypass_retrieval_score_filter(make_checker):
    checker = make_checker()
    checker.cfg.verifier.min_retrieval_score = 1e6
    checker._verifier = FakeVerifier()
    result = checker.check("A scientific claim.", oracle_doc_ids=["101"], evidence_mode="full_abstract", judge_backend="nli")
    assert result.status == "ok" and result.verdict == "SUPPORT"
    assert [d.doc_id for d in result.documents] == ["101"]
