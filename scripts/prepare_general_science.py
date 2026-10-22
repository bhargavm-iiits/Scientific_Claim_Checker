"""Cache pinned official OpenStax text modules and build a separate local corpus.

Raw images, scripts and archives are not downloaded or executed. Each text file
retains its public source permalink, revision, checksum and publisher attribution.
"""
import argparse
import concurrent.futures
import re
import time
from collections import Counter
from pathlib import Path

import requests
import yaml

from claim_checker.data.general_science import module_documents
from claim_checker.data.store import read_json, sha256_file, write_json, write_jsonl
from claim_checker.schemas import utc_now

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/general_science"
PREPARED = ROOT / "data/processed/general_science"
SPEC = ROOT / "configs/general_science_sources.yaml"
MODULE = re.compile(r"modules/([A-Za-z0-9_-]+)/index\.cnxml\Z")


def get(url, *, timeout=40):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "ClaimChecker-EducationalCorpus/1.0"})
            response.raise_for_status()
            if not response.content:
                raise requests.RequestException(f"Empty source response from {url}")
            if len(response.content) > 8_000_000:
                raise ValueError("Unexpectedly large source text response")
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def snapshot(book):
    repository = book["repository"]
    if not re.fullmatch(r"openstax/osbooks-[a-z-]+", repository):
        raise ValueError("Only configured official OpenStax repositories are supported")
    folder = RAW / repository.split("/")[1]
    identity_path = folder / "snapshot.json"
    if identity_path.exists():
        identity = {**read_json(identity_path), **book}
        write_json(identity_path, identity)
        return folder, identity
    metadata = get(f"https://api.github.com/repos/{repository}").json()
    commit = get(f"https://api.github.com/repos/{repository}/commits/{metadata['default_branch']}").json()
    revision = commit["sha"]
    tree = get(f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1").json()
    assert not tree.get("truncated"), "A truncated source tree would silently lose chapters"
    modules = sorted(entry["path"] for entry in tree["tree"] if entry["type"] == "blob" and MODULE.fullmatch(entry["path"]))
    assert modules, "No source modules found"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "source_tree.json", tree)
    license_text = get(f"https://raw.githubusercontent.com/{repository}/{revision}/LICENSE").text
    (folder / "LICENSE.txt").write_text(license_text, encoding="utf-8")
    readme = get(f"https://raw.githubusercontent.com/{repository}/{revision}/README.md").text
    (folder / "README.md").write_text(readme, encoding="utf-8")
    name = ("CC BY-NC-SA" if "NonCommercial" in license_text or "NonCommercial" in readme else
            "CC BY" if "Attribution" in license_text or "Attribution" in readme else "See preserved publisher LICENSE.txt")
    identity = {**book, "revision": revision, "commit_date": commit["commit"]["committer"]["date"],
                "modules": modules, "license": name, "license_sha256": sha256_file(folder / "LICENSE.txt"),
                "snapshotted_at": utc_now()}
    write_json(identity_path, identity)
    return folder, identity


def prepare(*, inspect=False, download=False):
    books = yaml.safe_load(SPEC.read_text(encoding="utf-8"))["books"]
    snapshots, documents, errors, source_files = [], [], [], []
    for book in books:
        cache = RAW / book["repository"].split("/")[1]
        if not (cache / "snapshot.json").exists() and not download:
            raise ValueError("Source snapshot missing; use --download once, then preparation works offline")
        folder, identity = snapshot(book)
        snapshots.append(identity)
        print(f"{identity['repository']} @{identity['revision'][:12]}: {len(identity['modules'])} modules; {identity['license']}", flush=True)
        paths = identity["modules"][:1] if inspect else identity["modules"]

        def load(path, folder=folder, identity=identity):
            target = folder / path
            if not target.is_file():
                if not download:
                    raise ValueError(f"Uncached module: {path}")
                body = get(f"https://raw.githubusercontent.com/{identity['repository']}/{identity['revision']}/{path}").content
                if len(body) > 2_000_000:
                    raise ValueError(f"Unexpectedly large module {path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".download")
                temporary.write_bytes(body)
                temporary.replace(target)
            data = target.read_bytes()
            module = MODULE.fullmatch(path).group(1)
            docs = module_documents(data, repository=identity["repository"], revision=identity["revision"], module=module,
                                     subject=identity["subject"], book_title=identity["title"], book_url=identity["url"],
                                     license_name=identity["license"])
            return docs, {"repository": identity["repository"], "path": path, "sha256": sha256_file(target), "documents": len(docs)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for index, (path, future) in enumerate(zip(paths, [pool.submit(load, p) for p in paths], strict=True), 1):
                try:
                    docs, source = future.result()
                    documents.extend(docs)
                    source_files.append(source)
                except Exception as exc:
                    errors.append({"repository": identity["repository"], "module": path, "error": f"{type(exc).__name__}: {exc}"})
                if index % 50 == 0 or index == len(paths):
                    print(f"{identity['subject']}: {index}/{len(paths)} modules; total passages={len(documents)}; errors={len(errors)}", flush=True)
    if inspect:
        write_json(RAW / "inspection.json", {"snapshots": snapshots, "samples": source_files,
                   "sample_documents": [d.model_dump() for d in documents[:3]], "errors": errors})
        if errors:
            raise ValueError(errors)
        return
    write_json(RAW / "acquisition_report.json", {"snapshots": snapshots, "source_files": source_files, "errors": errors})
    if errors:
        raise ValueError(f"{len(errors)} modules failed; rerun to resume cached downloads; no partial corpus is published")
    assert len(documents) == len({d.doc_id for d in documents})
    PREPARED.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "general_science", "name": "OpenStax general science", "created_at": utc_now(),
                "documents": len(documents), "sentences": sum(len(d.sentences) for d in documents),
                "subjects": dict(Counter(d.source["subject"] for d in documents)), "snapshots": snapshots,
                "source_spec_sha256": sha256_file(SPEC), "parser_sha256": sha256_file(ROOT / "src/claim_checker/data/general_science.py"),
                "builder_sha256": sha256_file(Path(__file__)), "excluded_material": "Exercises, problems, solutions, figures, tables, footnotes and glossaries",
                "scope": "Selected public textbook repositories; not all science and not live web search",
                "evaluation": "No benchmark claims or labels are used in corpus preparation"}
    write_jsonl(PREPARED / "corpus.jsonl", documents)
    manifest["corpus_sha256"] = sha256_file(PREPARED / "corpus.jsonl")
    write_json(PREPARED / "source_manifest.json", manifest)
    attributions = [f"# Source attributions\n\n{manifest['scope']}.\n"]
    for book in snapshots:
        attributions.append(f"## {book['title']}\n\nOpenStax, Rice University. {book['url']}\n\n"
                            f"License: {book['license']}; preserved in data/raw/general_science/{book['repository'].split('/')[1]}/LICENSE.txt.\n\n"
                            f"Source revision: {book['revision']}. Each passage retains its source permalink and attribution.\n")
    (PREPARED / "ATTRIBUTIONS.md").write_text("\n".join(attributions), encoding="utf-8")
    print(f"Published {len(documents)} source passages; {manifest['sentences']} sentences.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="cache source metadata and one module per repository, without publishing a corpus")
    args = parser.parse_args()
    prepare(inspect=args.inspect, download=args.download)


if __name__ == "__main__":
    main()
