from types import SimpleNamespace

import pytest

from claim_checker.agents.trained_evidence import select_trained_evidence
from claim_checker.config import EvidenceSelectorConfig
from claim_checker.data.store import write_json
from claim_checker.errors import ConfigError
from claim_checker.evaluation import runner
from claim_checker.runtime import check_readiness
from tests.conftest import make_doc


class FakeVerifier:
    def windows(self, claim, sentences):
        return [list(range(len(sentences)))] if sentences else []

    def score(self, premises, hypotheses):
        return [{"SUPPORT": 0.99, "CONTRADICT": 0.005, "NEI": 0.005} for _ in premises]


class FakeSelector:
    def score(self, claim, doc):
        return [0.99 if i == 2 else 0.1 for i in range(len(doc.sentences))]


def test_trained_selector_pipeline_keeps_source_ids_and_needs_no_ollama(make_checker):
    checker = make_checker()
    checker.cfg.evidence_selector.enabled = True
    checker.cfg.product.judge_backend = "nli"
    checker.cfg.product.evidence_mode = "evidence"
    checker._evidence_selector, checker._verifier = FakeSelector(), FakeVerifier()
    result = checker.check("A scientific claim.", k=1)
    assert result.status == "ok" and result.verdict == "SUPPORT"
    assert result.documents[0].selected_sentence_ids == [2]
    assert result.evidence[0].sentence_id == 2
    assert result.evidence[0].quote == checker.corpus[result.evidence[0].doc_id].sentences[2]
    assert result.metadata["evidence_backend"] == "trained_classifier"
    assert not checker.transport.calls


def test_trained_selector_warmup_is_local(make_checker):
    checker = make_checker()
    checker.cfg.evidence_selector.enabled = True
    checker.cfg.product.judge_backend = "nli"
    checker._evidence_selector, checker._verifier = FakeSelector(), FakeVerifier()
    checker.warmup()
    assert checker.is_warm and not checker.transport.calls


def test_fusion_uses_non_neutral_probability_and_keeps_original_index():
    selector = SimpleNamespace(score=lambda claim, doc: [0.9, 0.8])
    verifier = SimpleNamespace(format_premise=lambda title, text: text,
                               score=lambda premises, hypotheses: [{"NEI": 0.99}, {"NEI": 0.01}])
    cfg = EvidenceSelectorConfig(enabled=True, alpha=0.5, threshold=0.5, maximum=1)
    selected, _ = select_trained_evidence(selector, verifier, "claim", make_doc("101", "Paper", ["first", "second"]), cfg)
    assert selected.sentence_ids == [1]


def test_invalid_selector_scores_fail_instead_of_dropping_source_sentences():
    selector = SimpleNamespace(score=lambda claim, doc: [float("nan")])
    with pytest.raises(ValueError, match="invalid"):
        select_trained_evidence(selector, None, "claim", make_doc("101", "Paper", ["one"]), EvidenceSelectorConfig())


def test_trained_selector_readiness_checks_files_without_contacting_ollama(cfg, tmp_path, monkeypatch):
    cfg.product.judge_backend = "nli"
    cfg.evidence_selector.enabled = True
    cfg.verifier.model_dir = str(tmp_path / "reader")
    cfg.evidence_selector.model_dir = str(tmp_path / "selector/epoch1")
    for directory in (cfg.verifier.model_dir, cfg.evidence_selector.model_dir):
        folder = tmp_path / directory
        folder.mkdir(parents=True)
        for name in ("model.safetensors", "config.json", "tokenizer.json"):
            (folder / name).write_text("readiness-only fixture", encoding="utf-8")
    write_json(tmp_path / "selector/training_manifest.json", {"settings": {"final_fit": False}})
    cfg.path("processed_dir").mkdir(parents=True)
    (cfg.path("processed_dir") / "corpus.jsonl").write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr("claim_checker.retrieval.index.check_index", lambda cfg: {"fingerprint": "test"})
    calls = []
    monkeypatch.setattr("httpx.get", lambda *a, **k: calls.append(a))
    report = check_readiness(cfg)
    assert report.model_ready and report.ready and not calls
    assert "trained rationale selector" in report.model_name


def test_selector_identity_tracks_weights_and_blocks_final_fit_on_development(cfg, tmp_path, monkeypatch):
    cfg.evidence_selector.enabled = True
    reader, selector = tmp_path / "reader", tmp_path / "selector/epoch1"
    cfg.verifier.model_dir, cfg.evidence_selector.model_dir = str(reader), str(selector)
    for folder in (reader, selector):
        folder.mkdir(parents=True)
        for name in ("model.safetensors", "config.json", "tokenizer.json"):
            (folder / name).write_text("identity-only fixture", encoding="utf-8")
    cfg.path("processed_dir").mkdir(parents=True)
    for name in ("corpus.jsonl", "claim_gold_train.jsonl"):
        (cfg.path("processed_dir") / name).write_text("{}\n", encoding="utf-8")
    manifest = selector.parent / "training_manifest.json"
    write_json(manifest, {"settings": {"final_fit": False}})
    monkeypatch.setattr(runner, "split_fingerprint", lambda *a: "fixed")
    checker = SimpleNamespace(prompts=SimpleNamespace(hashes=lambda: {}))
    first = runner.run_identity(cfg, checker, "local", "dev_tune", [], False, {"v1"}, {"nli"})
    (selector / "model.safetensors").write_text("different weights", encoding="utf-8")
    second = runner.run_identity(cfg, checker, "local", "dev_tune", [], False, {"v1"}, {"nli"})
    assert first["evidence_selector_files_sha256"] != second["evidence_selector_files_sha256"]
    write_json(manifest, {"settings": {"final_fit": True}})
    with pytest.raises(ConfigError, match="refitted rationale selector"):
        runner.run_identity(cfg, checker, "local", "dev_tune", [], False, {"v1"}, {"nli"})


def test_local_doctor_does_not_require_an_ollama_server(cfg, monkeypatch, capsys):
    from claim_checker import cli
    from claim_checker.runtime import ReadinessReport

    cfg.product.judge_backend = "nli"
    cfg.evidence_selector.enabled = True
    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    monkeypatch.setattr("claim_checker.runtime.check_readiness", lambda cfg: ReadinessReport(
        data_ready=True, index_ready=True, model_ready=True, splits_ready=True,
        model_name="Scientific NLI verifier + trained rationale selector"))
    assert cli.main(["doctor", "--skip-model-test", "--json"]) == 0
    assert '"check": "ollama server"' not in capsys.readouterr().out
