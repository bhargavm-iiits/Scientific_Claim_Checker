from types import SimpleNamespace

import pytest

from claim_checker.data.grouped_training import claim_groups, grouped_folds


def gold(cid, text, docs, label="NEI", split="train"):
    return SimpleNamespace(claim_id=cid, claim=text, reference_doc_ids=docs, label=label, split=split, conflict=False)


def test_related_papers_and_normalized_claims_form_transitive_groups():
    claims = [gold("1", "A claim.", ["10"]), gold("2", "Different", ["10", "20"]),
              gold("3", "Third", ["20"]), gold("4", "A CLAIM!", ["99"]), gold("5", "Unrelated", ["50"])]
    groups = claim_groups(claims)
    assert len({groups[c] for c in ("1", "2", "3", "4")}) == 1
    assert groups["5"] != groups["1"]


def test_grouped_protocol_excludes_validation_texts_and_keeps_papers_disjoint():
    train = {str(i): gold(str(i), f"Claim {i}", [str(i // 2)], ["SUPPORT", "CONTRADICT", "NEI"][i // 2 % 3])
             for i in range(36)}
    val = {"v": gold("v", "CLAIM 0!", ["unseen"], split="validation")}
    first = grouped_folds(train, val, folds=3)
    assert first == grouped_folds(train, val, folds=3)
    assert first["excluded_validation_texts"] == ["0"]
    audits = []
    for fold in first["folds"]:
        sets = [{d for cid in fold[key] for d in train[cid].reference_doc_ids} for key in ("train", "stop", "audit")]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert sum(len(fold[key]) for key in ("train", "stop", "audit")) == 35
        audits.extend(fold["audit"])
    assert len(audits) == len(set(audits)) == 35


def test_protocol_rejects_validation_claims_in_training_source():
    with pytest.raises(ValueError, match="original TRAIN"):
        grouped_folds({str(i): gold(str(i), str(i), [str(i)], split="validation") for i in range(9)}, {}, folds=3)
