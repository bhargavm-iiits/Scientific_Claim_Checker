import pytest

from claim_checker.data.prepare import (
    build_corpus,
    build_gold,
    build_pairs,
    claim_label,
    cross_check_loader_rows,
    normalize_evidence,
)
from claim_checker.data.splits import (
    build_splits,
    normalize_claim_text,
    stratified_sample,
    write_split,
)
from claim_checker.errors import DataIntegrityError, FrozenSplitError
from claim_checker.schemas import ClaimGold, normalize_doc_id


def test_integer_and_string_ids_normalize_to_same_identifier():
    assert normalize_doc_id(13734012) == normalize_doc_id("13734012") == normalize_doc_id(" 013734012 ") == "13734012"
    assert normalize_doc_id(4.0) == "4"
    with pytest.raises(ValueError):
        normalize_doc_id("")
    with pytest.raises(ValueError):
        normalize_doc_id(True)


MTEB = [
    {"_id": "1", "title": "Paper one", "text": "First sentence. Second sentence."},
    {"_id": "2", "title": "Paper two", "text": "Alpha. Beta. Gamma."},
]
ORIG = [
    {"doc_id": 1, "title": "Paper one", "abstract": ["First sentence. ", "Second sentence."], "structured": False},
    {"doc_id": 2, "title": "Paper two", "abstract": ["Alpha.", "Beta.", "Gamma."], "structured": False},
]


def test_build_corpus_aligns_and_preserves_sentence_order():
    docs, q = build_corpus(MTEB, ORIG, expected_size=2)
    assert [d.doc_id for d in docs] == ["1", "2"]
    assert docs[0].sentences == ["First sentence.", "Second sentence."]  # stripped, ordered
    assert docs[1].text.startswith("Paper two\n")
    assert q["text_agreement"]["different"] == 0


def test_build_corpus_fails_on_missing_documents_and_title_mismatch():
    with pytest.raises(DataIntegrityError):
        build_corpus(MTEB[:1], ORIG)
    bad = [dict(MTEB[0], title="Another title"), MTEB[1]]
    with pytest.raises(DataIntegrityError):
        build_corpus(bad, ORIG)
    with pytest.raises(DataIntegrityError):
        build_corpus(MTEB, ORIG, expected_size=5183)


@pytest.fixture
def small_corpus():
    docs, _ = build_corpus(MTEB, ORIG)
    return {d.doc_id: d for d in docs}


def test_pairs_reproduce_builder_transformation(small_corpus):
    claims = [{"id": 7, "claim": "c", "cited_doc_ids": [1, 2],
               "evidence": {"1": [{"sentences": [1], "label": "SUPPORT"}, {"sentences": [0], "label": "SUPPORT"}]}}]
    pairs = build_pairs(claims, small_corpus, "train")
    assert [(p.doc_id, p.verdict, p.evidence) for p in pairs] == [("1", "SUPPORT", [1, 0]), ("2", "NEI", [])]
    assert pairs[0].claim_id == "7"


def test_invalid_evidence_position_is_rejected(small_corpus):
    claims = [{"id": 1, "claim": "c", "cited_doc_ids": [2], "evidence": {"2": [{"sentences": [9], "label": "CONTRADICT"}]}}]
    with pytest.raises(DataIntegrityError):
        build_pairs(claims, small_corpus, "train")
    with pytest.raises(DataIntegrityError):
        build_gold(claims, small_corpus, "train")


def test_claim_label_rules():
    assert claim_label({"1": "SUPPORT"}) == ("SUPPORT", False)
    assert claim_label({"1": "CONTRADICT", "2": "CONTRADICT"}) == ("CONTRADICT", False)
    assert claim_label({}) == ("NEI", False)
    assert claim_label({"1": "SUPPORT", "2": "CONTRADICT"}) == (None, True)


def test_claim_with_several_abstracts_is_one_gold_record(small_corpus):
    claims = [{"id": 3, "claim": "c", "cited_doc_ids": [1, 2],
               "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}}]
    gold = build_gold(claims, small_corpus, "validation")
    assert len(gold) == 1
    g = gold[0]
    assert g.label == "SUPPORT" and g.reference_doc_ids == ["1", "2"] and g.evidence_doc_ids == ["1"]


def test_normalize_evidence_handles_arrow_wide_struct():
    wide = {"1": None, "2": [{"sentences": [0, 2], "label": "CONTRADICT"}], "3": None}
    assert normalize_evidence(wide) == {"2": [{"sentences": [0, 2], "label": "CONTRADICT"}]}
    assert normalize_evidence(None) == {}


def test_loader_cross_check_detects_differences():
    a = [{"id": 1, "claim": "x", "cited_doc_ids": [1], "evidence": {"1": None}}]
    b = [{"id": 1, "claim": "x", "cited_doc_ids": [1], "evidence": {}}]
    cross_check_loader_rows(a, b, "train")
    with pytest.raises(DataIntegrityError):
        cross_check_loader_rows(a, [{"id": 1, "claim": "y", "cited_doc_ids": [1], "evidence": {}}], "train")


def _gold(i, label, text=None, conflict=False):
    return ClaimGold(claim_id=str(i), claim=text or f"claim number {i}", split="x", cited_doc_ids=["1"],
                     reference_doc_ids=["1"], evidence_by_doc={}, label_by_doc={},
                     label=None if conflict else label, conflict=conflict)


def test_stratified_sample_is_deterministic_and_proportional():
    items = [_gold(i, "SUPPORT") for i in range(50)] + [_gold(i, "NEI") for i in range(50, 80)] + \
            [_gold(i, "CONTRADICT") for i in range(80, 100)]
    a, alloc = stratified_sample(items, 10, 42)
    b, _ = stratified_sample(list(reversed(items)), 10, 42)
    assert [g.claim_id for g in a] == [g.claim_id for g in b]
    assert alloc == {"CONTRADICT": 2, "NEI": 3, "SUPPORT": 5}


def test_splits_exclude_conflicts_and_remove_overlap():
    val = {str(i): _gold(i, "SUPPORT" if i % 2 else "NEI") for i in range(1, 11)}
    val["99"] = _gold(99, None, conflict=True)
    train = {"500": _gold(500, "SUPPORT", text="Claim Number 3!"), "501": _gold(501, "NEI")}
    out = build_splits(train, val, seed=42, smoke_size=1, final_size=100)
    final = out["final"]
    assert "99" not in final["claim_ids"] and final["excluded_annotation_conflicts"] == ["99"]
    assert final["all_fewer_than_requested"] is True and final["size"] == 10
    assert normalize_claim_text("Claim Number 3!") == normalize_claim_text("claim number 3")
    assert "500" not in out["development"]["claim_ids"]
    assert out["development"]["removed_overlap_with_final"] == [{"train_claim_id": "500", "final_claim_id": "3"}]
    assert set(out["smoke"]["claim_ids"]) <= set(out["development"]["claim_ids"])


def test_frozen_final_is_not_silently_overwritten(tmp_path):
    p = tmp_path / "final.json"
    payload = {"name": "final", "claim_ids": ["1", "2"]}
    assert write_split(p, payload, frozen=True, force=False) == "written"
    assert write_split(p, payload, frozen=True, force=False) == "unchanged"
    with pytest.raises(FrozenSplitError):
        write_split(p, {"name": "final", "claim_ids": ["3"]}, frozen=True, force=False)
    assert write_split(p, {"name": "final", "claim_ids": ["3"]}, frozen=True, force=True) == "written"


def test_smoke_split_records_excluded_conflicts_as_ids_like_final():
    # Regression: smoke stored an int count while final stored the list of ids.
    val = {str(i): _gold(i, "SUPPORT" if i % 2 else "NEI") for i in range(1, 7)}
    train = {"500": _gold(500, "SUPPORT"), "501": _gold(501, "NEI"), "77": _gold(77, None, conflict=True),
             "8": _gold(8, None, conflict=True)}
    out = build_splits(train, val, seed=42, smoke_size=2, final_size=4)
    assert out["smoke"]["excluded_annotation_conflicts"] == ["8", "77"]
    assert isinstance(out["final"]["excluded_annotation_conflicts"], list)
    assert not set(out["smoke"]["claim_ids"]) & {"8", "77"}
