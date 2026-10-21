"""Source alignment, weak supervision and zero-initialization invariants."""
import re
from types import SimpleNamespace

import pytest
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from claim_checker.data.joint_context import encode_context, source_context, supervised_context
from claim_checker.models.joint_verifier import JointReader, rationale_loss


class Encoding(dict):
    def sequence_ids(self):
        return self.sequences


class OffsetTokenizer:
    def __call__(self, premise, claim, **kwargs):
        first = [(m.start(), m.end()) for m in re.finditer(r"\S+", premise)]
        second = [(m.start(), m.end()) for m in re.finditer(r"\S+", claim)]
        if kwargs.get("truncation"):
            first = first[:max(0, kwargs["max_length"] - len(second) - 3)]
        offsets = [(0, 0), *first, (0, 0), *second, (0, 0)]
        result = Encoding(input_ids=list(range(len(offsets))), attention_mask=[1] * len(offsets))
        result.sequences = [None, *([0] * len(first)), None, *([1] * len(second)), None]
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


@pytest.fixture
def doc():
    return SimpleNamespace(doc_id="7", title="Repeated words", sentences=["Repeated words.", "Uncited finding.", "Late result."])


def test_original_sentence_ids_and_source_only_targets(doc):
    row = encode_context(OffsetTokenizer(), "Repeated words.", doc, [0, 2], positives={2})
    assert set(row["token_source_ids"]) == {-1, 0, 2}
    targets = row["inputs"]["rationale_targets"]
    for sid, target in zip(row["token_source_ids"], targets, strict=True):
        assert target == (-100 if sid == -1 else int(sid == 2))
    assert row["token_source_ids"][-4:] == [-1] * 4  # separator, claim, separator
    assert row["token_source_ids"][:4] == [-1] * 4  # special token and title


@pytest.mark.parametrize("ids", [[], [2, 0], [0, 0], [-1], [3]])
def test_invalid_source_positions_are_rejected(doc, ids):
    with pytest.raises(ValueError, match="original sentence IDs"):
        source_context(doc, ids)


def training_row(doc):
    return {"claim_id": "1", "claim": "Late result.", "label": "SUPPORT", "view": "reference",
            "context_sentence_ids": [0, 2], "rationale_sentence_ids": [2],
            "premise": source_context(doc, [0, 2])[0]}


def test_positive_source_cannot_be_silently_truncated(doc):
    with pytest.raises(ValueError, match="truncate annotated evidence"):
        supervised_context(OffsetTokenizer(), training_row(doc), doc,
                           SimpleNamespace(evidence_by_doc={"7": [2]}), max_length=8)


def test_changed_premise_and_missing_rationale_rejected(doc):
    row = training_row(doc)
    row["premise"] += " Added text."
    gold = SimpleNamespace(evidence_by_doc={"7": [2]})
    with pytest.raises(ValueError, match="reconstruct exactly"):
        supervised_context(OffsetTokenizer(), row, doc, gold)
    row = training_row(doc)
    row["rationale_sentence_ids"] = [1]
    with pytest.raises(ValueError, match="lost an annotated rationale"):
        supervised_context(OffsetTokenizer(), row, doc, gold)


def test_no_source_fitting_is_an_explicit_error(doc):
    with pytest.raises(ValueError, match="No source token"):
        encode_context(OffsetTokenizer(), "Long claim words", doc, [2], max_length=6)


def test_weak_negative_weight_survives_microbatch_and_masking():
    logits = torch.zeros(2, 3, requires_grad=True)
    targets = torch.tensor([[1, 0, -100], [0, -100, -100]])
    values = rationale_loss(logits, targets)
    expected = torch.tensor([1.625, .25]) * torch.log(torch.tensor(2.))
    torch.testing.assert_close(values, expected)
    torch.testing.assert_close(values, torch.cat([rationale_loss(logits[i:i + 1], targets[i:i + 1]) for i in range(2)]))
    values.sum().backward()
    assert torch.all(logits.grad[targets == -100] == 0)
    assert torch.isfinite(logits.grad).all()


def tiny_model():
    config = DebertaV2Config(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=24, max_position_embeddings=32,
                            num_labels=3, hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
                            pooler_dropout=0., cls_dropout=0.)
    return JointReader(DebertaV2ForSequenceClassification(config), dropout=0.)


def test_initial_logits_match_base_and_joint_heads_receive_gradients():
    model = tiny_model().eval()
    tokens = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    mask = torch.tensor([[0, 1, 1, 0]])
    actual, rationale = model(source_mask=mask, **tokens)
    torch.testing.assert_close(actual, model.backbone(**tokens).logits)
    loss = actual.sum() + rationale_loss(rationale, torch.tensor([[-100, 1, 0, -100]])).sum()
    loss.backward()
    assert model.evidence_projection.weight.grad.abs().sum() > 0
    assert model.rationale_head.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="source token"):
        model(source_mask=torch.zeros_like(mask), **tokens)


def test_joint_heads_roundtrip(tmp_path):
    model = tiny_model().eval()
    with torch.no_grad():
        model.rationale_head.bias.fill_(.75)
        model.evidence_projection.weight.fill_(.01)
    model.save(tmp_path, SimpleNamespace(save_pretrained=lambda folder: None))
    restored = JointReader.load(tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
