from claim_checker.agents.examples import ExampleLibrary, build_examples
from claim_checker.schemas import ClaimGold


def gold(claim_id, claim, label, split="train"):
    return ClaimGold(claim_id=claim_id, claim=claim, split=split, cited_doc_ids=["101"],
                     reference_doc_ids=["101"], evidence_by_doc={"101":[2]},
                     label_by_doc={"101":label}, label=label)


def test_examples_exclude_normalized_evaluation_claims(corpus):
    items = [gold("1", "Vitamin D increases density.", "SUPPORT"),
             gold("2", "IL-6 is elevated", "SUPPORT")]
    records = build_examples(items, corpus, {"VITAMIN D increases density!"})
    assert [r["claim_id"] for r in records] == ["2"]
    assert records[0]["evidence"] == corpus["101"].sentences[2]


def test_validation_annotations_cannot_become_examples(corpus):
    import pytest

    with pytest.raises(ValueError, match="training"):
        build_examples([gold("1", "claim", "SUPPORT", split="validation")], corpus, set())


def test_runtime_examples_exclude_same_claim_and_document():
    records = [{"key":str(i),"claim_id":str(i),"claim":text,"doc_id":doc,
                "title":"T","evidence":"E","verdict":label,"source_split":"train"}
               for i,text,doc,label in [(1,"the exact claim","a","SUPPORT"),
                                        (2,"another claim","b","SUPPORT"),
                                        (3,"a different relation","c","NEI")]]
    lib = ExampleLibrary(records)
    assert [r["key"] for r in lib.retrieve("The exact claim!")] == ["2","3"]
    assert all(r["doc_id"] != "b" for r in lib.retrieve("unseen claim", doc_id="b"))
