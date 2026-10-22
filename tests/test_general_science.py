import pytest

from claim_checker.data.general_science import module_documents, parse_module, source_sentences

XML = b'''<document xmlns="http://cnx.rice.edu/cnxml"><title>Water</title>
<metadata><para>This metadata paragraph must never enter the evidence.</para></metadata>
<note class="os-teacher"><para>This teacher instruction must never enter the evidence.</para></note>
<note class="learning-objectives"><para>This learning objective must never enter the evidence.</para></note>
<content><section><title>Solvents</title><para>Water contains <emphasis>hydrogen</emphasis> and oxygen. Water is a polar molecule.</para>
<exercise><problem><para>Choose the false answer: water contains only carbon.</para></problem>
<solution><para>This is the answer key and must be excluded entirely.</para></solution></exercise>
<figure><caption><para>This figure caption requires an image that is unavailable.</para></caption></figure>
<para>Some substances dissolve in water.<footnote>This hidden footnote must not enter evidence.</footnote> Others do not.</para>
</section></content></document>'''


def test_parser_excludes_distractors_answers_and_unavailable_media():
    title, paragraphs = parse_module(XML)
    text = " ".join(p["text"] for p in paragraphs)
    assert title == "Water"
    assert "Water contains hydrogen and oxygen." in text
    assert "Some substances dissolve in water. Others do not." in text
    for excluded in ("metadata", "carbon", "answer key", "caption", "footnote", "teacher", "objective"):
        assert excluded not in text
    assert all(p["headings"] == ["Solvents"] for p in paragraphs)


def test_chunks_preserve_source_sentences_provenance_and_stable_ids():
    kwargs = dict(repository="openstax/osbooks-physics", revision="a" * 40, module="m123",
                  subject="physics", book_title="Physics", book_url="https://openstax.org/details/books/physics", license_name="CC BY")
    docs = module_documents(XML, **kwargs, maximum_sentences=1)
    assert len(docs) == 4
    source = " ".join(p["text"] for p in parse_module(XML)[1])
    assert all(sentence in source for doc in docs for sentence in doc.sentences)
    assert all(doc.source["url"].endswith("/modules/m123/index.cnxml") for doc in docs)
    assert all("OpenStax" in doc.source["attribution"] for doc in docs)
    assert [d.doc_id for d in docs] == [d.doc_id for d in module_documents(XML, **kwargs, maximum_sentences=1)]


def test_source_split_keeps_decimals_and_common_abbreviations():
    assert source_sentences("The value is 3.14. Dr. Smith observed it. Water is polar.") == [
        "The value is 3.14.", "Dr. Smith observed it.", "Water is polar."]


def test_source_xml_entity_declarations_are_rejected():
    with pytest.raises(ValueError, match="DTD/entity"):
        parse_module(b'<!DOCTYPE x [<!ENTITY x "expanded">]><document/>')


def test_general_profile_uses_separate_data_and_generic_model():
    from pathlib import Path

    from claim_checker.config import load_config

    root = Path(__file__).resolve().parents[1]
    normal = load_config(root=root)
    general = load_config(root / "configs/general_science.yaml")
    assert general.knowledge.kind == "general_science"
    assert general.verifier.model_dir == "artifacts/models/nli_large_base"
    for name in ("processed_dir", "index_dir", "runs_dir", "splits_dir", "cache_dir"):
        assert normal.path(name) != general.path(name)


def test_general_runtime_preserves_source_links_quotes_and_local_inference(make_checker):
    from tests.test_trained_evidence_pipeline import FakeVerifier

    checker = make_checker()
    checker.cfg.knowledge.kind = "general_science"
    checker.cfg.product.judge_backend = "nli"
    checker.cfg.product.evidence_mode = "full_abstract"
    checker._verifier = FakeVerifier()
    checker.corpus["101"].source = {"collection": "general_science", "url": "https://openstax.org/example",
                                   "attribution": "OpenStax, Rice University. CC BY."}
    result = checker.check("A source-supported scientific claim.", k=1)
    assert result.status == "ok" and result.verdict == "SUPPORT"
    assert result.documents[0].source["url"] == "https://openstax.org/example"
    assert all(q.quote == checker.corpus[q.doc_id].sentences[q.sentence_id] for q in result.evidence)
    assert result.metadata["knowledge"]["kind"] == "general_science"
    assert "source passages" in result.explanation
    assert not checker.transport.calls


def test_scifact_evaluation_cannot_silently_use_general_corpus(cfg, experiments):
    from claim_checker.errors import ConfigError
    from claim_checker.evaluation.runner import run_evaluation

    cfg.knowledge.kind = "general_science"
    with pytest.raises(ConfigError, match="different evidence collection"):
        run_evaluation(cfg, experiments, None, suite="final")


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///private", "https://user:password@example.org/page"])
def test_source_links_reject_executable_or_credential_urls(url):
    from ui.components import source_citation_html

    assert source_citation_html({"url": url}) == ""


def test_source_attribution_is_escaped():
    from ui.components import source_citation_html

    rendered = source_citation_html({"url": "https://openstax.org/page?a=1&b=2", "attribution": "<script>bad</script>"})
    assert 'rel="noopener noreferrer"' in rendered
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "?a=1&amp;b=2" in rendered


def test_general_search_does_not_spend_top_k_on_duplicate_edition_passages(cfg):
    from claim_checker.retrieval.index import RetrievalOutput, Retriever
    from tests.conftest import make_doc

    cfg.knowledge.kind = "general_science"
    first = make_doc("1", "Book A — Water", ["Water molecules have a polar arrangement."])
    duplicate = make_doc("2", "Book B — Water", ["Water molecules have a polar arrangement."])
    other = make_doc("3", "Book A — Atoms", ["An atom contains electrons and a nucleus."])
    retriever = Retriever.__new__(Retriever)
    retriever.cfg, retriever.fingerprint = cfg, "fixture"
    retriever.corpus = {d.doc_id: d for d in (first, duplicate, other)}
    retriever.candidates = lambda *args: RetrievalOutput(hits=[], candidates={"hybrid": [("1", 3), ("2", 2), ("3", 1)]})
    assert [hit.doc_id for hit in retriever.retrieve("A science claim", rerank=False, keep=3).hits] == ["1", "3"]


def test_perfect_small_smoke_set_does_not_establish_95percent_accuracy():
    from scripts.evaluate_general_science import accuracy_interval

    lower, upper = accuracy_interval(15, 15)
    assert 0.79 < lower < 0.80 and upper == pytest.approx(1)
    assert accuracy_interval(0, 0) is None


def test_benchmark_schema_rejects_unknown_labels_and_hidden_evidence():
    from pydantic import ValidationError

    from scripts.evaluate_general_science import BenchmarkClaim

    with pytest.raises(ValidationError):
        BenchmarkClaim(claim_id="1", claim="A scientific claim", label="TRUE")
    with pytest.raises(ValidationError):
        BenchmarkClaim(claim_id="1", claim="A scientific claim", label="SUPPORT", gold_evidence="hidden")


def test_government_article_parser_excludes_navigation_media_and_duplicate_text():
    from claim_checker.data.government_science import parse_usgs_article, usgs_documents

    content = b'''<html><h1>Water properties</h1><nav><p>This navigation paragraph is not science evidence.</p></nav>
    <article><div class="field--name-body"><p>Water molecules have a polar arrangement of hydrogen and oxygen atoms.</p>
    <figure><figcaption><p>This image caption should not be ingested as evidence.</p></figcaption></figure>
    <p>Water molecules have a polar arrangement of hydrogen and oxygen atoms.</p></div></article></html>'''
    title, paragraphs = parse_usgs_article(content)
    assert title == "Water properties" and len(paragraphs) == 1
    docs = usgs_documents(content, "https://www.usgs.gov/water-science-school/science/example")
    assert len(docs) == 1 and docs[0].source["publisher"] == "U.S. Geological Survey"
    assert "navigation" not in docs[0].text and "caption" not in docs[0].text


def test_government_parser_refuses_pages_without_an_article_body():
    from claim_checker.data.government_science import parse_usgs_article

    with pytest.raises(ValueError, match="body not found"):
        parse_usgs_article(b"<html><nav><p>A long navigation paragraph must not become evidence.</p></nav></html>")
