"""The trainable Qwen judge must preserve the app's evidence and error contracts."""
import pytest
from conftest import make_doc

from claim_checker.agents.judge_lora import judge_lora
from claim_checker.config import load_config
from claim_checker.runtime import check_readiness


class FakeLoraJudge:
    def judge(self, claim, doc):
        return [{"sentence_ids": list(range(len(doc.sentences))),
                 "answer": {"verdict": "SUPPORT", "cited_sentence_ids": [len(doc.sentences) - 1]}}]


def test_lora_pipeline_quotes_original_source_and_avoids_ollama(make_checker):
    checker = make_checker()
    checker._lora_judge = FakeLoraJudge()
    checker._check_resources = True
    checker.llm.ensure_ready = lambda: pytest.fail("Local judge must not contact Ollama")
    result = checker.check("A scientific claim.", k=1, evidence_mode="full_abstract", judge_backend="lora")
    assert result.status == "ok" and result.verdict == "SUPPORT"
    assert result.metadata["judge_backend"] == "lora"
    assert result.metadata["model"] == "Qwen2.5-1.5B-Instruct+LoRA"
    assert result.evidence
    for quote in result.evidence:
        assert quote.quote == checker.corpus[quote.doc_id].sentences[quote.sentence_id]
    assert not checker.transport.calls


def test_lora_selected_evidence_is_rejected_instead_of_silently_ignored(make_checker):
    checker = make_checker()
    result = checker.check("A scientific claim.", judge_backend="lora", evidence_mode="evidence")
    assert result.status == "error" and result.verdict is None
    assert "full_abstract" in result.error.message


def test_lora_bad_citation_is_an_operational_failure(make_checker):
    class Invalid(FakeLoraJudge):
        def judge(self, claim, doc):
            return [{"sentence_ids": [0], "answer": {"verdict": "SUPPORT", "cited_sentence_ids": [999]}}]
    checker = make_checker()
    checker._lora_judge = Invalid()
    result = checker.check("A scientific claim.", judge_backend="lora", evidence_mode="full_abstract")
    assert result.status == "error" and result.verdict is None


def test_lora_window_conflict_is_visible():
    class Conflict(FakeLoraJudge):
        def judge(self, claim, doc):
            return [{"sentence_ids": [i], "answer": {"verdict": label, "cited_sentence_ids": [i]}}
                    for i, label in enumerate(("SUPPORT", "CONTRADICT"))]
    judgement = judge_lora(Conflict(), "Claim", make_doc("1", "Title", ["Finding A.", "Finding B."]))
    assert judgement.verdict == "NEI" and "window_conflict" in judgement.consistency_flags
    assert judgement.cited_sentence_ids == [0, 1]


def test_local_readiness_does_not_require_ollama(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: pytest.fail("Readiness must not contact Ollama"))
    cfg = load_config(overrides={"product": {"judge_backend": "lora", "prompt_set": "v1", "evidence_mode": "full_abstract"},
                                 "lora": {"base_dir": str(tmp_path / "base"), "adapter_dir": str(tmp_path / "adapter")}})
    result = check_readiness(cfg)
    assert not result.model_ready
    assert any("Missing local files" in p["message"] for p in result.problems)


def test_local_warmup_loads_the_configured_judge_without_ollama(make_checker):
    checker = make_checker()
    checker.cfg = checker.cfg.model_copy(update={"product": checker.cfg.product.model_copy(update={
        "judge_backend": "lora", "prompt_set": "v1", "evidence_mode": "full_abstract"})})
    checker._lora_judge = FakeLoraJudge()
    checker.llm.ensure_ready = lambda: pytest.fail("Local warmup must not contact Ollama")
    checker.warmup()
    assert checker.is_warm and checker.resources_loaded()["model_checked"]
