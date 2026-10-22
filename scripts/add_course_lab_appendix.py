"""Add traceable adaptations without rewriting historical notebook results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Scientific_Claim_Checker.ipynb"
MARKER = "## 12. Course lab adaptations (23 October 2026)"


def main():
    original = NOTEBOOK.read_bytes()
    notebook = json.loads(original)
    if any(MARKER in "".join(cell["source"]) for cell in notebook["cells"]):
        raise SystemExit("Appendix already exists; historical notebook left unchanged.")
    digest = hashlib.sha256(original).hexdigest()
    backup = ROOT / f"artifacts/notebooks/pre_lab_adaptation_{digest[:12]}.ipynb"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        assert backup.read_bytes() == original
    else:
        backup.write_bytes(original)

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
        cell["id"] = "course-" + hashlib.sha256(source.encode()).hexdigest()[:12]
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook["cells"].append(cell)

    intro = "".join(notebook["cells"][0]["source"])
    intro = intro.replace("so nothing here\nis a second implementation.",
                          "so the assessment uses the shared implementation. The explicitly marked\n"
                          "teaching appendix in §12 adapts actual course-lab examples separately.")
    intro += "\n\nSections 0–11 retain the original Qwen assessment and saved results. Later NLI\n"
    intro += "experiments, reused evaluation populations and the current score status are\n"
    intro += "identified in §13; the original 100-claim subset is not a blind test anymore.\n"
    notebook["cells"][0]["source"] = intro.splitlines(keepends=True)
    notebook["cells"][45]["source"] = """## 10. Lab citations

The original production implementation was written independently. After locating the actual
Module 2 and 3 notebooks, §12 was added on 23 October 2026 with traceable adaptations:

| Location | Actual source (one-based notebook cell) | Adaptation |
|---|---|---|
| §12a | Module 2, `4_Structured_Generation_Lab.ipynb`, cells 36–37, section 4.1 | Pydantic schema, Literal enum and JSON postvalidation, adapted from review sentiment to evidence-backed verdicts. Production uses Ollama JSON-schema decoding rather than Outlines. |
| §12b | Module 3, `1_RAG_Pipeline_Lab.ipynb`, cell 14 | Regex tokenization and BM25Okapi search over real SciFact abstracts, with deterministic ties. |
| §12b | Same Module 3 notebook, cells 16 and 24 | Two-stream reciprocal rank fusion and Recall@k, generalized to ranked IDs and multiple annotated reference papers. |
| §5 | Module 4 multi-agent lab | Conceptual topic mapping only: its notebook was not located, so no code adaptation is claimed. |

Exact local source paths, differences and the boundary between new adaptations and the
independent production code are recorded in `Docs/LAB_REUSE.md`. The new appendix does
not retrospectively establish lab reuse in the original build.
""".splitlines(keepends=True)
    add("markdown", MARKER + """

These additions adapt the actual local labs, with exact source-cell citations in §10.
They are teaching demonstrations; they do not replace the frozen evaluation pipeline,
alter saved results or measure claim-checking accuracy.

### 12a. Structured verdict schema — Module 2 cells 36–37

The lab's product-review enum becomes a three-way scientific verdict. Additional checks
require citations to belong to the supplied context. The synthetic records below exercise
validation only. Schema validity does not establish factual correctness. Production sends
its schema to Ollama's `format` argument and validates again; this example makes no model call.
""")
    add("code", '''from typing import Literal
from pydantic import BaseModel, ConfigDict, ValidationError


class CourseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["SUPPORT", "CONTRADICT", "NEI"]
    sentence_ids: list[int]
    reason: str


def validate_course_verdict(raw, available_sentence_ids):
    parsed = CourseVerdict.model_validate_json(raw)
    if not set(parsed.sentence_ids) <= set(available_sentence_ids):
        raise ValueError("Citation outside supplied context")
    if parsed.verdict != "NEI" and not parsed.sentence_ids:
        raise ValueError("Decisive verdict requires a source citation")
    return parsed


schema_demo = validate_course_verdict(
    '{"verdict":"NEI","sentence_ids":[],"reason":"Synthetic schema example."}', [0, 1]
)
print("Allowed verdicts:", CourseVerdict.model_json_schema()["properties"]["verdict"]["enum"])
print("Synthetic record:", schema_demo.model_dump())
for invalid_demo in (
    '{"verdict":"TRUE","sentence_ids":[0],"reason":"Invalid enum."}',
    '{"verdict":"SUPPORT","sentence_ids":[999],"reason":"Invalid source ID."}',
):
    try:
        validate_course_verdict(invalid_demo, [0, 1])
    except (ValueError, ValidationError) as exc:
        print("Rejected:", type(exc).__name__)
    else:
        raise AssertionError("Invalid demonstration unexpectedly accepted")
''')
    add("markdown", """### 12b. BM25, RRF and Recall@k — Module 3 cells 14, 16 and 24

The lab's tokenizer/BM25 score loop is adapted to the real 5,183 stored abstracts.
Its two RRF loops are adapted to ID lists, with deterministic tie-breaking and duplicate
removal. Recall becomes the fraction of reference papers retrieved per claim, so it
supports multiple references. The BM25 illustration uses one fitting TRAIN claim and is
not a benchmark. The RRF illustration uses two genuine sparse ranking streams (title
and full abstract); it demonstrates the fusion rule, not the production dense hybrid.
The production system's BM25/dense/reranker comparison remains in §4.
""")
    add("code", '''import re
from pathlib import Path
from rank_bm25 import BM25Okapi


def course_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def course_bm25_index(source_docs, title_only=False):
    ids = sorted(source_docs)
    texts = [source_docs[i].title + (
        "" if title_only else " " + " ".join(source_docs[i].sentences)
    ) for i in ids]
    return ids, BM25Okapi([course_tokens(text) for text in texts])


def course_bm25_search(query, index, k=10):
    ids, bm25 = index
    scores = bm25.get_scores(course_tokens(query))
    positions = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:k]
    return [ids[i] for i in positions]


def course_rrf(first_ids, second_ids, k_const=60, top_k=10):
    scores = {}
    for rank, doc_id in enumerate(dict.fromkeys(first_ids), start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k_const + rank)
    for rank, doc_id in enumerate(dict.fromkeys(second_ids), start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k_const + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]


def course_reference_recall(ranked_ids, reference_ids, k=3):
    reference_ids = set(reference_ids)
    return len(set(ranked_ids[:k]) & reference_ids) / len(reference_ids) if reference_ids else None


course_plan = read_json(Path(cfg.root) / "artifacts/improvement/lora_science_protocol_v2.json")
course_claim = next(gold_train[i] for i in course_plan["fit_ids"] if gold_train[i].reference_doc_ids)
course_full_index = course_bm25_index(docs)
course_title_index = course_bm25_index(docs, title_only=True)
course_full_ids = course_bm25_search(course_claim.claim, course_full_index)
course_title_ids = course_bm25_search(course_claim.claim, course_title_index)
course_fused_ids = course_rrf(course_full_ids, course_title_ids)
print("TRAIN fitting claim:", course_claim.claim_id, "| corpus papers:", len(docs))
print("Full-abstract BM25:", course_full_ids[:3])
print("Title BM25:", course_title_ids[:3])
print("Two-sparse-stream RRF:", course_fused_ids[:3])
print("Illustrative reference Recall@3:",
      course_reference_recall(course_fused_ids, course_claim.reference_doc_ids))
assert set(course_fused_ids) <= set(course_full_ids) | set(course_title_ids)
del course_full_index, course_title_index
''')
    add("markdown", """## 13. Later experiments and evaluation provenance

The original tables above remain historical results. The original 300 validation claims
have since been reused across experiments, so they are not a blind test. The current best
confirmed result on that complete population is **80.67% accuracy / 0.795974 Macro-F1**,
below the requested >95% / >0.95. See `Docs/ACCURACY_IMPROVEMENT_2026-10-22.md`.

The latest all-layer LoRA candidate uses paper/text-grouped TRAIN fitting (484 claims),
stopping (160) and auditing (163), with fixed settings and exact source quotations.
Its protocol, learning curve, audit, acceptance gate and current status are in
`Docs/LORA_SCIENCE_EXPERIMENT.md`. No target is claimed from training loss or unit tests.
The general-science collection is evaluated separately; its 15-case developer smoke
test does not establish population accuracy. Module 4 and separate judging guidelines
referenced by P08 are still unavailable, so full course-lab compliance is not claimed.
""")
    for i, cell in enumerate(notebook["cells"]):
        cell.setdefault("id", f"original-cell-{i + 1}")
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"original_sha256": digest, "backup": str(backup), "cells": len(notebook["cells"])}))


if __name__ == "__main__":
    main()
