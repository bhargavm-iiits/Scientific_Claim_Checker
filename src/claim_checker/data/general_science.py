"""Source-preserving OpenStax CNXML ingestion for a separate science collection."""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET

from claim_checker.schemas import CorpusDocument

EXCLUDED = frozenset({"metadata", "exercise", "problem", "solution", "answer", "figure", "table", "footnote", "glossary"})
PEDAGOGY = frozenset({"os-teacher", "teacher-support", "learning-objectives", "review-questions", "critical-thinking", "practice"})
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "Fig.", "Figs.", "Dr.", "Mr.", "Mrs.", "vs.", "U.S.")


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def source_sentences(paragraph):
    """Split normalized source text without rewriting its words or punctuation."""
    text = " ".join(paragraph.split())
    if not text:
        return []
    boundaries = []
    for match in re.finditer(r"[.!?][\"'’”)]*\s+(?=[A-Z0-9\"“])", text):
        prefix = text[:match.start() + 1]
        if any(prefix.endswith(abbreviation) for abbreviation in ABBREVIATIONS):
            continue
        boundaries.append(match.end())
    result, start = [], 0
    for end in boundaries + [len(text)]:
        sentence = text[start:end].strip()
        if sentence:
            assert sentence in text
            result.append(sentence)
        start = end
    return result


def parse_module(data: bytes):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("DTD/entity declarations are not accepted in source XML")
    root = ET.fromstring(data)
    title = next((" ".join("".join(e.itertext()).split()) for e in root
                  if local_name(e.tag) == "title"), "Untitled section")
    paragraphs = []

    def visit(node, headings=()):
        name = local_name(node.tag)
        if name in EXCLUDED or set(node.attrib.get("class", "").split()) & PEDAGOGY:
            return
        if name == "section":
            heading = next((" ".join("".join(e.itertext()).split()) for e in node
                            if local_name(e.tag) == "title"), "")
            headings = (*headings, heading) if heading else headings
        if name == "para":
            # Avoid nested exercise answers, image captions and table cells even when
            # they occur within a paragraph. Keep inline emphasis/math text intact.
            def text_of(element):
                parts = [element.text or ""]
                for child in element:
                    if local_name(child.tag) not in EXCLUDED:
                        parts.append(text_of(child))
                    parts.append(child.tail or "")
                return "".join(parts)

            text = " ".join(text_of(node).split())
            if len(text) >= 35:
                paragraphs.append({"headings": list(headings), "text": text})
            return
        for child in node:
            visit(child, headings)

    visit(root)
    return title, paragraphs


def module_documents(data, *, repository, revision, module, subject, book_title, book_url, license_name,
                     maximum_sentences=8, maximum_chars=2200):
    title, paragraphs = parse_module(data)
    source = {"publisher": "OpenStax, Rice University", "collection": "general_science", "subject": subject,
              "book_title": book_title, "book_url": book_url, "repository": repository, "revision": revision,
              "module": module, "url": f"https://github.com/{repository}/blob/{revision}/modules/{module}/index.cnxml",
              "license": license_name, "attribution": f"{book_title}, OpenStax, Rice University. {license_name}. {book_url}",
              "source_sha256": hashlib.sha256(data).hexdigest()}
    chunks, active, active_heading = [], [], ()

    def flush():
        nonlocal active
        if active:
            chunks.append((active_heading, active))
            active = []

    for paragraph in paragraphs:
        heading = tuple(paragraph["headings"])
        if heading != active_heading:
            flush()
            active_heading = heading
        for sentence in source_sentences(paragraph["text"]):
            if active and (len(active) >= maximum_sentences or sum(map(len, active)) + len(sentence) > maximum_chars):
                flush()
            active.append(sentence)
    flush()
    result = []
    for number, (headings, sentences) in enumerate(chunks):
        doc_title = " — ".join([book_title, title, *headings])
        did = hashlib.sha256(f"{repository}/{revision}/{module}/{number}".encode()).hexdigest()[:20]
        result.append(CorpusDocument(doc_id=f"openstax-{did}", title=doc_title, sentences=sentences,
                                     text=f"{doc_title}\n{' '.join(sentences)}", source={**source, "chunk": number}))
    return result
