"""Training guards must reject leakage, wrong labels and fabricated source inputs."""
from types import SimpleNamespace

import pytest

from claim_checker.data.verifier_context import format_premise
from scripts import train_roberta_verifier as trainer


@pytest.fixture
def audit_inputs(monkeypatch, cfg, corpus):
    gold = {
        "fit": SimpleNamespace(claim_id="fit", claim="Vitamin D increases bone density.", split="train",
                               reference_doc_ids=["101"], label_by_doc={"101": "SUPPORT"}, evidence_by_doc={"101": [2]}),
        "stop": SimpleNamespace(claim_id="stop", claim="Vitamin D has no effect in older women.", split="train",
                                reference_doc_ids=["202"], label_by_doc={"202": "SUPPORT"}, evidence_by_doc={"202": [1]}),
    }
    validation = {"val": SimpleNamespace(claim="The validation claim must stay outside fitting.")}
    monkeypatch.setattr(trainer, "load_gold", lambda path, split: gold if split == "train" else validation)
    monkeypatch.setattr(trainer, "load_corpus", lambda path: corpus)

    def row(cid, did, sentence):
        return {"claim_id": cid, "claim": gold[cid].claim, "doc_id": did, "label": "SUPPORT", "view": "rationale",
                "context_sentence_ids": [sentence], "rationale_sentence_ids": [sentence],
                "premise": format_premise(corpus[did].title, corpus[did].sentences[sentence], True)}

    return cfg, [row("fit", "101", 2)], [row("stop", "202", 1)], {
        "fit_ids": ["fit"], "stop_ids": ["stop"], "eligible_ids": ["fit", "stop"]}, gold, validation


def test_roberta_audit_accepts_source_exact_disjoint_training(audit_inputs):
    cfg, fitting, stopping, protocol, _, _ = audit_inputs
    report = trainer.audit_rows(cfg, fitting, stopping, protocol, False)
    assert report["fit_claims"] == report["stop_claims"] == 1
    assert report["source_annotation_mismatches"] == report["validation_text_overlap"] == 0


@pytest.mark.parametrize("fault", ["validation_text", "shared_paper", "wrong_label", "fabricated_premise", "missing_fit_claim"])
def test_roberta_audit_rejects_contaminated_or_inaccurate_training(audit_inputs, fault):
    cfg, fitting, stopping, protocol, gold, validation = audit_inputs
    if fault == "validation_text":
        validation["val"].claim = gold["fit"].claim.upper()
    elif fault == "shared_paper":
        gold["stop"].reference_doc_ids.append("101")
    elif fault == "wrong_label":
        fitting[0]["label"] = "CONTRADICT"
    elif fault == "fabricated_premise":
        fitting[0]["premise"] = "An invented source sentence."
    else:
        fitting.clear()
    with pytest.raises(AssertionError):
        trainer.audit_rows(cfg, fitting, stopping, protocol, False)
