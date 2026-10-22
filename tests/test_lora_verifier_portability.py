"""Portable inference and experiment configuration require no GPU or downloads."""
from types import SimpleNamespace

import pytest

from scripts.run_lora_scientific_verifier_v2 import experiment_config


def test_cpu_configuration_uses_cpu_for_all_neural_resources():
    cfg = experiment_config("a/local/model", device="cpu")
    assert cfg.retrieval.embedding_device == cfg.retrieval.reranker_device == cfg.verifier.device == "cpu"


def test_inference_honors_configured_cpu_device(monkeypatch, cfg, tmp_path):
    import scripts.lora_scientific_verifier_v2 as module

    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"synthetic-unit-fixture")
    devices = []

    def factory(path, **kwargs):
        devices.append(kwargs["device"])
        return SimpleNamespace(labels=["SUPPORT", "NEI", "CONTRADICT"])

    monkeypatch.setattr(module, "CachingVerifier", factory)
    monkeypatch.setattr(module, "ClaimChecker", lambda config: SimpleNamespace(corpus={}))
    module.FeedbackChecker(cfg, folder)
    assert devices == ["cpu"]


def test_all_qualifying_evidence_sentences_remain_eligible(monkeypatch):
    import scripts.lora_scientific_verifier_v2 as module
    from claim_checker.schemas import CorpusDocument, DocumentJudgement

    doc = CorpusDocument(doc_id="1", title="Trial", text="", sentences=[f"Finding {i}." for i in range(5)])
    original = DocumentJudgement(doc_id="1", verdict="SUPPORT", cited_sentence_ids=[0, 1])
    monkeypatch.setattr(module, "judge_nli", lambda *args, **kwargs: original)
    verifier = SimpleNamespace(format_premise=lambda title, sentence: sentence,
                               score=lambda premises, hypotheses: [{"SUPPORT": .9} for _ in premises])
    result = module.judge_document(verifier, "A claim", doc, [0, 1, 2, 3, 4], threshold=.5)
    assert result.cited_sentence_ids == [0, 1, 2, 3, 4]
    assert original.cited_sentence_ids == [0, 1]


def test_cached_scores_are_scoped_and_keep_input_order(monkeypatch):
    import scripts.lora_scientific_verifier_v2 as module

    calls = []

    def compute(self, premises, hypotheses, batch_size=4):
        calls.extend(zip(premises, hypotheses, strict=True))
        return [{"SUPPORT": float(len(premise))} for premise in premises]

    monkeypatch.setattr(module.NLIVerifier, "score", compute)
    verifier = module.CachingVerifier.__new__(module.CachingVerifier)
    verifier.reset_cache()
    assert verifier.score(["abc", "a", "abc"], ["H"] * 3) == [{"SUPPORT": 3.}, {"SUPPORT": 1.}, {"SUPPORT": 3.}]
    assert len(calls) == 2
    returned = verifier.score(["abc"], ["H"])
    returned[0]["SUPPORT"] = -99
    assert verifier.score(["abc"], ["H"])[0]["SUPPORT"] == 3.
    assert len(calls) == 2
    verifier.reset_cache()
    verifier.score(["abc"], ["H"])
    assert len(calls) == 3
    with pytest.raises(ValueError, match="one hypothesis"):
        verifier.score(["abc"], [])
