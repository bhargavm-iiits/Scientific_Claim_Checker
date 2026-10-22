"""Audit every published passage against cached, attributed source paragraphs."""
from collections import Counter
from pathlib import Path

from claim_checker.data.general_science import parse_module
from claim_checker.data.store import load_corpus, read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    prepared = ROOT / "data/processed/general_science"
    corpus = load_corpus(prepared)
    manifest = read_json(prepared / "source_manifest.json")
    parsed, failures, sources = {}, [], set()
    checked = 0
    for document in corpus.values():
        source = document.source
        if source.get("publisher") == "OpenStax, Rice University":
            path = ROOT / "data/raw/general_science" / source["repository"].split("/")[1] / "modules" / source["module"] / "index.cnxml"
            if path not in parsed:
                _, paragraphs = parse_module(path.read_bytes())
                parsed[path] = (sha256_file(path), [p["text"] for p in paragraphs])
            checksum, paragraphs = parsed[path]
            snapshot = read_json(path.parents[2] / "snapshot.json")
            if snapshot["revision"] != source["revision"]:
                failures.append({"doc_id": document.doc_id, "reason": "revision mismatch"})
        elif source.get("publisher") == "U.S. Geological Survey":
            path = ROOT / "data/sources/usgs_water_verified_excerpt.json"
            excerpt = read_json(path)
            checksum, paragraphs = sha256_file(path), excerpt["paragraphs"]
            if source["url"] != excerpt["url"] or source["acquisition"] != excerpt["acquisition"]:
                failures.append({"doc_id": document.doc_id, "reason": "excerpt provenance mismatch"})
        else:
            failures.append({"doc_id": document.doc_id, "reason": "unexpected publisher"})
            continue
        sources.add(str(path.relative_to(ROOT)))
        if checksum != source["source_sha256"]:
            failures.append({"doc_id": document.doc_id, "reason": "source checksum mismatch"})
        normalized = [" ".join(p.split()) for p in paragraphs]
        for sentence in document.sentences:
            checked += 1
            if not any(sentence in paragraph for paragraph in normalized):
                failures.append({"doc_id": document.doc_id, "reason": "sentence absent from source paragraphs"})
        if not all(source.get(k) for k in ("url", "attribution", "license", "collection")):
            failures.append({"doc_id": document.doc_id, "reason": "missing attribution"})
        if document.text != f"{document.title}\n{' '.join(document.sentences)}":
            failures.append({"doc_id": document.doc_id, "reason": "retrieval text differs from stored sentences"})
    matches = (manifest["corpus_sha256"] == sha256_file(prepared / "corpus.jsonl")
               and manifest["documents"] == len(corpus) and manifest["sentences"] == checked)
    report = {"documents": len(corpus), "sentences_checked": checked, "cached_sources_checked": len(sources),
              "subjects": dict(Counter(d.source["subject"] for d in corpus.values())),
              "manifest_matches": matches, "failures": failures,
              "limitations": "Checks stored-source traceability, not scientific correctness, completeness or reader accuracy."}
    write_json(ROOT / "artifacts/improvement/general_science_source_audit.json", report)
    print(f"{len(corpus)} passages; {checked} sentences; {len(failures)} provenance failures; manifest_matches={matches}")
    assert matches and not failures


if __name__ == "__main__":
    main()
