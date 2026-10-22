"""Schema checks, ID normalization, corpus alignment, pair reconstruction and claim-level labels.

Pair reconstruction reproduces ``allenai/scifact_entailment``'s builder
(``_generate_examples``): for every claim and every ``cited_doc_id``, the verdict is the label of
that document's evidence sets when the document appears in ``claim["evidence"]`` (all sets of one
document share a label), otherwise ``NEI``; the evidence is the flattened list of sentence
indices; the abstract is the stripped sentence list.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from claim_checker.config import AppConfig
from claim_checker.data.sources import SourceBundle
from claim_checker.data.store import (
    CORPUS_FILE,
    GOLD_FILE,
    PAIRS_FILE,
    QUALITY_FILE,
    write_json,
    write_jsonl,
)
from claim_checker.errors import DataIntegrityError
from claim_checker.schemas import ClaimGold, ClaimPair, CorpusDocument, normalize_doc_id, utc_now

_WS = re.compile(r"\s+")


def norm_text(text: str) -> str:
    """Comparison form: NFKC, casefolded, whitespace collapsed."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


def squash(text: str) -> str:
    """Comparison form ignoring all whitespace (sentence joins may differ in spacing)."""
    return _WS.sub("", unicodedata.normalize("NFKC", text or "")).casefold()


# ---------------------------------------------------------------------------
# Evidence normalization
# ---------------------------------------------------------------------------


def normalize_evidence(evidence: Any) -> dict[str, list[dict]]:
    """Normalize ``claim["evidence"]`` to ``{doc_id: [{"sentences": [...], "label": ...}]}``.

    Works for the raw JSON dict and for Arrow's wide struct produced by ``load_dataset("json")``
    (absent document keys come back as ``None``).
    """
    if not evidence:
        return {}
    out: dict[str, list[dict]] = {}
    for key, sets in evidence.items():
        if sets is None:
            continue
        clean = []
        for entry in sets:
            if entry is None:
                continue
            label = entry["label"]
            if label not in ("SUPPORT", "CONTRADICT"):
                raise DataIntegrityError(f"Unexpected evidence label {label!r}", stage="data")
            clean.append({"sentences": [int(s) for s in entry["sentences"]], "label": label})
        if clean:
            out[normalize_doc_id(key)] = clean
    return out


def normalize_claim_row(row: dict) -> dict:
    return {
        "id": normalize_doc_id(row["id"]),
        "claim": row["claim"],
        "cited_doc_ids": [normalize_doc_id(d) for d in (row.get("cited_doc_ids") or [])],
        "evidence": normalize_evidence(row.get("evidence")),
    }


def cross_check_loader_rows(json_rows: list[dict], raw_rows: list[dict], name: str) -> None:
    """The load_dataset('json') rows must equal a plain JSON parse of the same file."""
    if len(json_rows) != len(raw_rows):
        raise DataIntegrityError(
            f"{name}: load_dataset('json') returned {len(json_rows)} rows, file has {len(raw_rows)}",
            stage="data",
        )
    for a, b in zip(json_rows, raw_rows, strict=True):
        na, nb = normalize_claim_row(a), normalize_claim_row(b)
        if na != nb:
            raise DataIntegrityError(
                f"{name}: row mismatch between load_dataset('json') and raw JSON for claim {nb['id']}",
                stage="data",
            )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def build_corpus(
    mteb_rows: list[dict], original_rows: list[dict], expected_size: int | None = None
) -> tuple[list[CorpusDocument], dict]:
    q: dict[str, Any] = {}
    mteb_ids = [normalize_doc_id(r["_id"]) for r in mteb_rows]
    orig_ids = [normalize_doc_id(r["doc_id"]) for r in original_rows]
    dup_mteb = [d for d, c in Counter(mteb_ids).items() if c > 1]
    dup_orig = [d for d, c in Counter(orig_ids).items() if c > 1]
    if dup_mteb or dup_orig:
        raise DataIntegrityError(
            f"Duplicate document ids (mteb={dup_mteb[:5]}, original={dup_orig[:5]})", stage="data"
        )
    only_mteb = sorted(set(mteb_ids) - set(orig_ids))
    only_orig = sorted(set(orig_ids) - set(mteb_ids))
    q["mteb_documents"] = len(mteb_ids)
    q["original_documents"] = len(orig_ids)
    q["ids_only_in_mteb"] = only_mteb
    q["ids_only_in_original"] = only_orig
    if only_mteb or only_orig:
        raise DataIntegrityError(
            f"MTEB and original SciFact corpus ids differ: {len(only_mteb)} only in MTEB "
            f"(e.g. {only_mteb[:5]}), {len(only_orig)} only in original (e.g. {only_orig[:5]})",
            stage="data",
        )
    if expected_size is not None and len(mteb_ids) != expected_size:
        raise DataIntegrityError(
            f"Corpus has {len(mteb_ids)} documents, expected {expected_size}", stage="data"
        )

    orig_by_id = {normalize_doc_id(r["doc_id"]): r for r in original_rows}
    title_mismatch, text_exact, text_squash, text_mismatch = [], 0, 0, []
    empty_abstracts, n_sents = [], []
    docs: list[CorpusDocument] = []
    for row in sorted(mteb_rows, key=lambda r: _sort_key(normalize_doc_id(r["_id"]))):
        doc_id = normalize_doc_id(row["_id"])
        orig = orig_by_id[doc_id]
        sentences = [s.strip() for s in (orig.get("abstract") or [])]
        if not sentences:
            empty_abstracts.append(doc_id)
        n_sents.append(len(sentences))
        if norm_text(row["title"]) != norm_text(orig["title"]):
            title_mismatch.append({"doc_id": doc_id, "mteb": row["title"], "original": orig["title"]})
        joined = " ".join(sentences)
        if norm_text(joined) == norm_text(row["text"]):
            text_exact += 1
        elif squash(joined) == squash(row["text"]):
            text_squash += 1
        else:
            text_mismatch.append(doc_id)
        title = (row["title"] or orig["title"] or "").strip()
        abstract_text = (row["text"] or joined).strip()
        docs.append(
            CorpusDocument(
                doc_id=doc_id,
                title=title,
                text=f"{title}\n{abstract_text}".strip(),
                sentences=sentences,
                source={
                    "mteb_id": str(row["_id"]),
                    "structured": bool(orig.get("structured", False)),
                    "text_matches_sentences": doc_id not in text_mismatch,
                },
            )
        )
    q["title_mismatches"] = title_mismatch
    q["text_agreement"] = {
        "exact_after_whitespace_normalization": text_exact,
        "equal_ignoring_whitespace": text_squash,
        "different": len(text_mismatch),
        "different_doc_ids_sample": text_mismatch[:20],
    }
    q["empty_abstracts"] = empty_abstracts
    q["sentences_per_document"] = {
        "min": min(n_sents) if n_sents else 0,
        "median": statistics.median(n_sents) if n_sents else 0,
        "max": max(n_sents) if n_sents else 0,
        "total": sum(n_sents),
    }
    if title_mismatch:
        raise DataIntegrityError(
            f"{len(title_mismatch)} titles differ between MTEB and original corpus, "
            f"e.g. {title_mismatch[:3]}",
            stage="data",
        )
    return docs, q


def _sort_key(doc_id: str) -> tuple[int, int | str]:
    return (0, int(doc_id)) if doc_id.isdigit() else (1, doc_id)


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def build_pairs(claims: list[dict], corpus: dict[str, CorpusDocument], split: str) -> list[ClaimPair]:
    """Reproduce the builder's claim-abstract pairs (one row per claim x cited document)."""
    pairs: list[ClaimPair] = []
    for raw in claims:
        claim = normalize_claim_row(raw)
        for doc_id in claim["cited_doc_ids"]:
            if doc_id not in corpus:
                raise DataIntegrityError(
                    f"Claim {claim['id']} cites document {doc_id} missing from the corpus",
                    stage="data",
                )
            doc = corpus[doc_id]
            sets = claim["evidence"].get(doc_id)
            if sets:
                verdict = sets[0]["label"]
                if any(s["label"] != verdict for s in sets):
                    raise DataIntegrityError(
                        f"Claim {claim['id']} doc {doc_id}: evidence sets disagree on label",
                        stage="data",
                    )
                evidence = [i for s in sets for i in s["sentences"]]
            else:
                verdict, evidence = "NEI", []
            bad = [i for i in evidence if not 0 <= i < len(doc.sentences)]
            if bad:
                raise DataIntegrityError(
                    f"Claim {claim['id']} doc {doc_id}: evidence positions {bad} out of range "
                    f"(abstract has {len(doc.sentences)} sentences)",
                    stage="data",
                )
            pairs.append(
                ClaimPair(
                    claim_id=claim["id"],
                    claim=claim["claim"],
                    doc_id=doc_id,
                    title=doc.title,
                    verdict=verdict,
                    evidence=evidence,
                    split=split,
                )
            )
    return pairs


def claim_label(label_by_doc: dict[str, str]) -> tuple[str | None, bool]:
    """Claim-level reference label from per-document evidence labels.

    SUPPORT only -> SUPPORT; CONTRADICT only -> CONTRADICT; none -> NEI;
    both -> (None, conflict=True).
    """
    labels = set(label_by_doc.values())
    if "SUPPORT" in labels and "CONTRADICT" in labels:
        return None, True
    if "SUPPORT" in labels:
        return "SUPPORT", False
    if "CONTRADICT" in labels:
        return "CONTRADICT", False
    return "NEI", False


def build_gold(claims: list[dict], corpus: dict[str, CorpusDocument], split: str) -> list[ClaimGold]:
    out: list[ClaimGold] = []
    seen: set[str] = set()
    for raw in claims:
        claim = normalize_claim_row(raw)
        if claim["id"] in seen:
            raise DataIntegrityError(f"Duplicate claim id {claim['id']} in {split}", stage="data")
        seen.add(claim["id"])
        evidence_by_doc: dict[str, list[int]] = {}
        label_by_doc: dict[str, str] = {}
        for doc_id, sets in claim["evidence"].items():
            if doc_id not in corpus:
                raise DataIntegrityError(
                    f"Claim {claim['id']} has evidence in document {doc_id} missing from the corpus",
                    stage="data",
                )
            n = len(corpus[doc_id].sentences)
            sents = sorted({i for s in sets for i in s["sentences"]})
            if any(not 0 <= i < n for i in sents):
                raise DataIntegrityError(
                    f"Claim {claim['id']} doc {doc_id}: evidence out of range", stage="data"
                )
            evidence_by_doc[doc_id] = sents
            label_by_doc[doc_id] = sets[0]["label"]
        label, conflict = claim_label(label_by_doc)
        out.append(
            ClaimGold(
                claim_id=claim["id"],
                claim=claim["claim"],
                split=split,
                cited_doc_ids=claim["cited_doc_ids"],
                reference_doc_ids=[d for d in claim["cited_doc_ids"] if d in corpus],
                evidence_by_doc=evidence_by_doc,
                label_by_doc=label_by_doc,
                label=label,
                conflict=conflict,
            )
        )
    return out


def compare_with_hf_pairs(pairs: list[ClaimPair], hf_rows: list[dict]) -> dict:
    ours = {(p.claim_id, p.doc_id): (p.verdict, list(p.evidence)) for p in pairs}
    theirs = {
        (normalize_doc_id(r["claim_id"]), normalize_doc_id(r["abstract_id"])): (
            r["verdict"],
            [int(i) for i in r["evidence"]],
        )
        for r in hf_rows
    }
    diff = [k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k)]
    if diff:
        raise DataIntegrityError(
            f"Reconstructed pairs differ from the HF loader on {len(diff)} rows, e.g. {diff[:3]}",
            stage="data",
        )
    return {"rows_compared": len(ours), "differences": 0}


def prepare_data(cfg: AppConfig, bundle: SourceBundle, log=print) -> dict:
    """Validate sources and write the processed corpus, pairs, gold claims and quality report."""
    out_dir = cfg.path("processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    quality: dict[str, Any] = {"created_at": utc_now(), "route": bundle.manifest.get("route")}

    for split, name in (("train", "train"), ("validation", "validation")):
        cross_check_loader_rows(bundle.claims_rows[split], bundle.raw_json_rows[name], split)
    cross_check_corpus = len(bundle.original_corpus_rows) == len(bundle.raw_json_rows["corpus"])
    if not cross_check_corpus:
        raise DataIntegrityError("Original corpus row count differs between loaders", stage="data")

    docs, corpus_q = build_corpus(
        bundle.mteb_rows, bundle.original_corpus_rows, cfg.data.expected_corpus_size
    )
    quality["corpus"] = corpus_q
    corpus = {d.doc_id: d for d in docs}
    write_jsonl(out_dir / CORPUS_FILE, docs)
    log(f"Corpus: {len(docs)} documents aligned (MTEB ids == original ids)")

    quality["splits"] = {}
    for split in ("train", "validation"):
        claims = bundle.claims_rows[split]
        pairs = build_pairs(claims, corpus, split)
        gold = build_gold(claims, corpus, split)
        hf_check = None
        if bundle.hf_pairs and split in bundle.hf_pairs:
            hf_check = compare_with_hf_pairs(pairs, bundle.hf_pairs[split])
        not_cited = []
        for raw in claims:
            c = normalize_claim_row(raw)
            not_cited += [(c["id"], d) for d in c["evidence"] if d not in c["cited_doc_ids"]]
        write_jsonl(out_dir / PAIRS_FILE.format(split=split), pairs)
        write_jsonl(out_dir / GOLD_FILE.format(split=split), gold)
        labels = Counter(g.label if not g.conflict else "CONFLICT" for g in gold)
        quality["splits"][split] = {
            "claims": len(gold),
            "pairs": len(pairs),
            "pair_verdicts": dict(Counter(p.verdict for p in pairs)),
            "claim_labels": dict(labels),
            "annotation_conflicts": [g.claim_id for g in gold if g.conflict],
            "claims_without_cited_docs": [g.claim_id for g in gold if not g.cited_doc_ids],
            "cited_docs_missing_from_corpus": sum(
                len(g.cited_doc_ids) - len(g.reference_doc_ids) for g in gold
            ),
            "evidence_docs_not_in_cited_doc_ids": not_cited,
            "evidence_positions_out_of_range": 0,
            "evidence_sentences": sum(len(v) for g in gold for v in g.evidence_by_doc.values()),
            "hf_loader_cross_check": hf_check,
        }
        log(f"{split}: {len(gold)} claims, {len(pairs)} claim-abstract pairs, labels {dict(labels)}")

    write_json(out_dir / QUALITY_FILE, quality)
    return quality


def processed_paths(cfg: AppConfig) -> list[Path]:
    d = cfg.path("processed_dir")
    return [
        d / CORPUS_FILE,
        *(d / PAIRS_FILE.format(split=s) for s in ("train", "validation")),
        *(d / GOLD_FILE.format(split=s) for s in ("train", "validation")),
    ]
