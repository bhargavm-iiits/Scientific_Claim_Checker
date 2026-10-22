from types import SimpleNamespace

import pytest

from claim_checker.data.store import write_json
from claim_checker.errors import ConfigError
from claim_checker.evaluation import runner


def test_local_reranker_weight_changes_are_fingerprinted_and_fitted_claims_rejected(cfg, tmp_path, monkeypatch):
    processed = cfg.path("processed_dir")
    processed.mkdir(parents=True)
    for name in ("corpus.jsonl", "claim_gold_train.jsonl"):
        (processed / name).write_text("{}\n", encoding="utf-8")
    reader, ranker = tmp_path / "reader", tmp_path / "ranker/epoch1"
    for directory in (reader, ranker):
        directory.mkdir(parents=True)
        for name in ("model.safetensors", "tokenizer.json", "config.json"):
            (directory / name).write_text("identity-only fixture", encoding="utf-8")
    cfg.verifier.model_dir = str(reader)
    cfg.retrieval.reranker_model = str(ranker)
    monkeypatch.setattr(runner, "split_fingerprint", lambda *args: "fixture")
    checker = SimpleNamespace(prompts=SimpleNamespace(hashes=lambda: {}))
    population = [SimpleNamespace(claim_id="17", claim="A scientific evaluation claim")]
    first = runner.run_identity(cfg, checker, "local", "dev_tune", population, False, {"v1"}, {"nli"})
    (ranker / "model.safetensors").write_text("changed model weights", encoding="utf-8")
    second = runner.run_identity(cfg, checker, "local", "dev_tune", population, False, {"v1"}, {"nli"})
    assert first["local_reranker_files_sha256"] != second["local_reranker_files_sha256"]
    write_json(ranker.parent / "training_manifest.json", {"fitted_claim_ids": ["17"]})
    with pytest.raises(ConfigError, match="not held-out"):
        runner.run_identity(cfg, checker, "local", "dev_tune", population, False, {"v1"}, {"nli"})
