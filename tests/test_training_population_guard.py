from types import SimpleNamespace

import pytest

from claim_checker.config import sha256_text
from claim_checker.data.splits import normalize_claim_text
from claim_checker.errors import ConfigError
from claim_checker.evaluation.training_provenance import assert_unfitted_population


def test_population_guard_rejects_fitted_id_even_for_selection_checkpoint():
    with pytest.raises(ConfigError, match="not held-out"):
        assert_unfitted_population({"settings": {"final_fit": False}, "fitted_claim_ids": ["17"]},
                                  [SimpleNamespace(claim_id="17", claim="A training claim")])


def test_population_guard_rejects_normalized_duplicate_with_new_id():
    training = {"fitted_claim_ids": ["17"],
                "fitted_claim_text_sha256": [sha256_text(normalize_claim_text("A scientific claim."))]}
    with pytest.raises(ConfigError, match="not held-out"):
        assert_unfitted_population(training, [SimpleNamespace(claim_id="new-id", claim="  A SCIENTIFIC  CLAIM.  ")])


def test_population_guard_accepts_disjoint_claims_and_legacy_manifest():
    fresh = [SimpleNamespace(claim_id="18", claim="An independently held-out claim")]
    assert_unfitted_population({"fitted_claim_ids": ["17"], "fitted_claim_text_sha256": []}, fresh)
    assert_unfitted_population({"settings": {"final_fit": True}}, fresh)


@pytest.mark.parametrize("duplicate_by", ["id", "text"])
def test_nli_evaluation_identity_enforces_actual_fitted_population(cfg, tmp_path, monkeypatch, duplicate_by):
    from claim_checker.data.store import write_json
    from claim_checker.evaluation import runner

    processed = cfg.path("processed_dir")
    processed.mkdir(parents=True)
    for name in ("corpus.jsonl", "claim_gold_train.jsonl"):
        (processed / name).write_text("{}\n", encoding="utf-8")
    model = tmp_path / "reader"
    model.mkdir()
    for name in ("model.safetensors", "tokenizer.json", "config.json"):
        (model / name).write_text("identity-only fixture", encoding="utf-8")
    cfg.verifier.model_dir = str(model)
    write_json(model.parent / "training_manifest.json", {
        "settings": {"final_fit": False}, "fitted_claim_ids": ["fit-id"],
        "fitted_claim_text_sha256": [sha256_text(normalize_claim_text("The fitting claim."))]})
    monkeypatch.setattr(runner, "split_fingerprint", lambda *args: "fixture")
    checker = SimpleNamespace(prompts=SimpleNamespace(hashes=lambda: {}))
    claim = SimpleNamespace(claim_id="fit-id" if duplicate_by == "id" else "different-id",
                            claim="Unrelated wording" if duplicate_by == "id" else " THE FITTING  CLAIM. ")
    with pytest.raises(ConfigError, match="not held-out"):
        runner.run_identity(cfg, checker, "local", "dev_tune", [claim], False, {"v1"}, {"nli"})
    fresh = SimpleNamespace(claim_id="fresh-id", claim="A genuinely separate claim")
    assert runner.run_identity(cfg, checker, "local", "dev_tune", [fresh], False, {"v1"}, {"nli"})["population_size"] == 1
