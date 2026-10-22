"""Append a cached, source-attributed USGS water-science collection to textbooks."""
import argparse
import concurrent.futures
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from claim_checker.data.government_science import usgs_documents, usgs_text_documents
from claim_checker.data.store import load_corpus, read_json, sha256_file, write_json, write_jsonl
from claim_checker.schemas import utc_now
from scripts.prepare_general_science import get

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/general_science/usgs"
PREPARED = ROOT / "data/processed/general_science"
TOPIC = "https://www.usgs.gov/water-science-school/science/water-properties-information-topic"
SEED = "https://www.usgs.gov/water-science-school/science/water-universal-solvent"


def allowed(url):
    parts = urlsplit(url)
    return (parts.scheme == "https" and parts.hostname == "www.usgs.gov" and not parts.query
            and parts.path.startswith("/water-science-school/science/")
            and not any(token in parts.path for token in ("quiz", "gallery", "pictures", "photo")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verified-excerpts", action="store_true", help="use the attributed, source-verified excerpt when direct HTTP responses are empty")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    snapshot = RAW / "snapshot.json"
    if args.verified_excerpts:
        verified = ROOT / "data/sources/usgs_water_verified_excerpt.json"
        excerpt = read_json(verified)
        documents = usgs_text_documents(excerpt["title"], excerpt["paragraphs"], excerpt["url"], sha256_file(verified), excerpt["acquisition"])
        sources = [{"url": excerpt["url"], "verified_excerpt": str(verified.relative_to(ROOT)),
                    "sha256": sha256_file(verified), "passages": len(documents), "acquisition": excerpt["acquisition"]}]
        publish(documents, sources, verified)
        return
    if snapshot.exists():
        frozen = read_json(snapshot)
    else:
        assert args.download, "Use --download once to freeze the source URL list"
        response = get(TOPIC)
        (RAW / "topic.html").write_bytes(response.content)
        links = {urljoin(TOPIC, a["href"]).split("#", 1)[0].rstrip("/") for a in BeautifulSoup(response.content, "html.parser").find_all("a", href=True)}
        urls = [SEED] + [url for url in sorted(links) if allowed(url) and url not in {TOPIC, SEED}][:39]
        frozen = {"created_at": utc_now(), "topic_url": TOPIC, "topic_sha256": sha256_file(RAW / "topic.html"), "urls": urls}
        write_json(snapshot, frozen)

    def load(url):
        assert allowed(url)
        path = RAW / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".html")
        if not path.exists():
            assert args.download, f"Source not cached: {url}"
            response = get(url)
            assert allowed(response.url), "Source redirected outside the supported article collection"
            path.write_bytes(response.content)
        body = path.read_bytes()
        docs = usgs_documents(body, url)
        return docs, {"url": url, "raw_file": path.name, "sha256": sha256_file(path), "passages": len(docs)}

    documents, sources, errors = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(url, pool.submit(load, url)) for url in frozen["urls"]]
        for url, future in futures:
            try:
                docs, source = future.result()
                documents.extend(docs)
                sources.append(source)
            except Exception as exc:
                errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    write_json(RAW / "acquisition_report.json", {"sources": sources, "errors": errors})
    assert not errors, f"USGS sources failed: {errors}; the previous corpus is preserved"
    publish(documents, sources, snapshot)


def publish(documents, sources, snapshot):
    corpus = load_corpus(PREPARED)
    base = [doc for doc in corpus.values() if doc.source.get("publisher") != "U.S. Geological Survey"]
    merged = base + documents
    assert len(merged) == len({doc.doc_id for doc in merged})
    manifest = read_json(PREPARED / "source_manifest.json")
    write_jsonl(PREPARED / "corpus.jsonl", merged)
    manifest.update(documents=len(merged), sentences=sum(len(d.sentences) for d in merged),
                    corpus_sha256=sha256_file(PREPARED / "corpus.jsonl"), updated_at=utc_now(),
                    name="General science: OpenStax + USGS", scope="Cached textbook passages and attributed USGS article text; not all science or live web search")
    manifest["subjects"]["water_science"] = len(documents)
    manifest["government_sources"] = {"snapshot_sha256": sha256_file(snapshot), "sources": sources,
                                       "parser_sha256": sha256_file(ROOT / "src/claim_checker/data/government_science.py"),
                                       "builder_sha256": sha256_file(Path(__file__))}
    write_json(PREPARED / "source_manifest.json", manifest)
    attribution_path = PREPARED / "ATTRIBUTIONS.md"
    original = attribution_path.read_text(encoding="utf-8").split("\n## U.S. Geological Survey", 1)[0]
    added = "\n## U.S. Geological Survey\n\nWater Science School. Each passage retains its original source URL, acquisition route and attribution.\n\n"
    added += "\n".join(f"- {source['url']}" for source in sources) + "\n"
    attribution_path.write_text(original + added, encoding="utf-8")
    print(f"Added {len(documents)} passages from {len(sources)} USGS articles; merged corpus has {len(merged)} passages.")


if __name__ == "__main__":
    main()
