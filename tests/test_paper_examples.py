from claim_checker.agents.paper_examples import PaperExampleLibrary, prepend_paper_examples


def example(key, claim, label, doc="1"):
    return {"key": key, "claim": claim, "verdict": label, "doc_id": doc, "source_split": "train", "evidence": "A stored source passage."}


def test_paper_demonstrations_exclude_live_claim_and_other_documents():
    library = PaperExampleLibrary([
        example("a", "A increases B.", "SUPPORT"),
        example("b", "A decreases B.", "CONTRADICT"),
        example("c", "A increases B elsewhere.", "SUPPORT", "2"),
    ])
    selected = library.retrieve_for_paper("A increases B!", "1")
    assert [e["key"] for e in selected] == ["b"]
    assert library.retrieve_for_paper("Other claim", "missing") == []


def test_paper_demonstrations_keep_one_per_class():
    library = PaperExampleLibrary([
        example("a", "A increases B.", "SUPPORT"),
        example("b", "A can increase B.", "SUPPORT"),
        example("c", "A decreases B.", "CONTRADICT"),
    ])
    selected = library.retrieve_for_paper("A affects B.", "1")
    assert len(selected) == 2
    assert {e["verdict"] for e in selected} == {"SUPPORT", "CONTRADICT"}


def test_demo_labels_are_explicitly_separate_from_target():
    messages = [{"role": "system", "content": "Verify"}, {"role": "user", "content": "A target claim"}]
    result = prepend_paper_examples(messages, [example("a", "Training claim", "SUPPORT")])
    assert result[0] == messages[0]
    assert "Do not copy" in result[1]["content"]
    assert result[1]["content"].endswith("CURRENT TARGET:\n\nA target claim")
    assert prepend_paper_examples(messages, []) == messages
