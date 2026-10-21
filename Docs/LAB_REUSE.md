# Course lab reuse

The project brief (section 07, "Labs to Reuse") asks the team to start from three course labs, adapt
them rather than rewrite from scratch, and cite which lab each part of the notebook builds on:

- **Module 2 · Structured_Generation_Lab** — CoT, self-consistency, PoT, ReAct, reflection, outlines
- **Module 3 · RAG_Pipeline_Lab** — chunking, FAISS, BM25, hybrid RRF, cross-encoder re-ranking,
  Recall@k / MRR
- **Module 4 · Multi_Agent_System_End_to_End** — tools, ReAct graph, memory, supervisor,
  human-in-the-loop, MCP

## Materials located during the 22 October 2026 review

Two course notebooks were found and their relevant cells reviewed:

- Module 2: `C:/Users/Bhargav/Downloads/GEN AI/Module 2-20260909T152256Z-1-001/Module 2/Lab/4_Structured_Generation_Lab.ipynb`, section 4.1, cells 36-37: Pydantic schemas, Literal enums and constrained decoding.
- Module 3: `C:/Users/Bhargav/Downloads/GEN AI/Module 3-20260909T152302Z-1-001/Module 3/Lab/1_RAG_Pipeline_Lab.ipynb`, cells 14 (BM25), 16 (RRF), 18 (cross-encoder reranking), 22 (LangGraph) and 24 (Recall@k/MRR).

Cell positions above are one-based JSON notebook positions. Module 4's referenced
notebook was not located. These exact references replace the earlier availability
assumption; they do not establish that the original code was adapted from the labs.
The original production implementation remains an independent build. On 23 October,
the assessment notebook gained explicit adaptations in section 12 and exact
citations in section 10. This addition does not retrospectively establish lab reuse
in the original production code.

## Adaptations added on 23 October 2026

| Assessment notebook | Source cells (one-based) | What changed from the lab |
|---|---|---|
| Section 12a, `CourseVerdict` and `validate_course_verdict` | Module 2, cells 36–37, section 4.1 | Product-review sentiment/rating becomes a SUPPORT/CONTRADICT/NEI enum with sentence IDs and a short reason. The Pydantic schema and JSON postvalidation pattern are retained; context-membership and decisive-citation checks are added. Production uses Ollama JSON-schema generation instead of Outlines. The synthetic validation records make no model calls and measure no factual accuracy. |
| Section 12b, `course_tokens`, `course_bm25_index`, `course_bm25_search` | Module 3, cell 14 | Regex tokenization and BM25Okapi score retrieval are applied to the actual 5,183 SciFact abstracts; source IDs replace LangChain document metadata, and ties are deterministic. |
| Section 12b, `course_rrf` | Module 3, cell 16 | Two rank-accumulation loops, `1/(60+rank)`, adapted to ranked source IDs, duplicate removal and deterministic ties. The executable example fuses title/full-abstract sparse rankings, explicitly a teaching example rather than the production dense hybrid. |
| Section 12b, `course_reference_recall` | Module 3, cell 24 | Single-gold hit rate becomes per-claim reference-paper recall for potentially multiple reference papers. The executable illustration is one fitting TRAIN claim, not an evaluation benchmark. |

The original notebook bytes are preserved as
`artifacts/notebooks/pre_lab_adaptation_0a47b9335811.ipynb`, SHA-256
`0a47b93358119ae287a86f1b7015ed047124541b934bf672006b7b44598303cf`.
The update script is `scripts/add_course_lab_appendix.py`; it refuses a duplicate
appendix. Historical execution outputs and evaluation tables are preserved.
New appendix checks are executed separately from the historical model cells.
Module 4 remains unavailable, so full compliance with the brief's reuse requirement
is still incomplete.

## Original build status

**The lab notebooks were unavailable during the original build, and no lab code was copied or adapted then.** Every
file in `src/claim_checker/`, `ui/`, `prompts/` and `tests/` was written for this project from the
brief, `Docs/Master_Prompt.md` and the public documentation of the libraries used (Hugging Face
Datasets, Sentence Transformers, rank-bm25, Ollama, Pydantic, LangGraph, Streamlit).

The table below is therefore a **conceptual correspondence only**: it shows which lab topic each
project component covers, so that markers can see where the course material is applied. It is not a
claim of code reuse.

The mapping below continues to describe independent production code. The actual
Module 2/3 adaptations and exact source citations are listed above and in the notebook.
Module 4's graph/supervisor mapping remains conceptual until its source is available.

## Mapping of project components to lab topics

### Module 2 — Structured_Generation_Lab

| Project component | File(s) | Lab topic it corresponds to | Notes / differences |
|---|---|---|---|
| JSON-schema constrained model output validated by Pydantic | `models/ollama_client.py` (`structured`, `schema_for`), `schemas.py` (`EvidenceOutput`, `JudgeOutput`, `JudgeOnlyOutput`, `RetrieveJudgeOutput`) | Structured generation / "outlines"-style constrained decoding | Uses Ollama's `format` (JSON schema → grammar) instead of the `outlines` library; output is validated again with Pydantic. |
| One bounded repair turn after invalid output | `ollama_client.py` (repair loop, `max_schema_repair_attempts: 1`) | Reflection / self-correction | Single, bounded correction message listing the concrete problem (schema error or out-of-range sentence IDs); no open-ended reflection loop. |
| Supervisor feedback to the judge for an uncited decisive verdict | `pipeline/supervisor.py` (`CITATION_FEEDBACK`), `pipeline/graph.py` | Reflection | One re-ask, then deterministic downgrade to NEI. |
| Hidden reasoning disabled; short reasons only | `config.py` (`think: false`), prompts | Chain-of-thought (deliberately **not** used) | The design stores concise reasons and stage events instead of chain-of-thought; `<think>` blocks are stripped if present. Self-consistency and PoT were not used (temperature 0, one sample). |
| Prompt templates with system/user parts | `agents/prompting.py`, `prompts/*.jinja2` | Prompt construction for structured tasks | Template hashes are recorded for reproducibility. |

### Module 3 — RAG_Pipeline_Lab

| Project component | File(s) | Lab topic it corresponds to | Notes / differences |
|---|---|---|---|
| Sentence-aware, tokenizer-aware windows with overlap | `retrieval/dense.py` (`make_windows`) | Chunking | Windows never split a sentence and record their zero-based sentence IDs; title repeated in each window. |
| Dense retrieval with BGE-small | `retrieval/dense.py` (`SentenceEncoder`, `DenseIndex`) | Embedding search (lab uses FAISS) | Exact inner product over a NumPy matrix instead of FAISS (5,670 vectors make an index library unnecessary); document score = best window. |
| BM25 keyword search | `retrieval/bm25.py` | BM25 | Custom deterministic scientific tokenizer (keeps numbers and hyphenated terms), stable tie-breaking by ID. |
| Hybrid reciprocal rank fusion | `retrieval/hybrid.py` (`rrf`, k = 60) | Hybrid RRF | Duplicate removal, deterministic ties. |
| Cross-encoder re-ranking | `retrieval/rerank.py` | Cross-encoder re-ranking | Long abstracts windowed and max-aggregated rather than truncated. |
| Recall@k evaluation | `evaluation/metrics.py` (`recall_at_k`, `retrieval_metrics`), `evaluation/runner.py` (`run_retrieval_eval`) | Recall@k / MRR | Recall@1/3/5/10 and evidence-bearing recall; MRR was not implemented. |
| Index fingerprinting and stale-index detection | `retrieval/index.py` | (extension beyond the lab topics) | — |

### Module 4 — Multi_Agent_System_End_to_End

| Project component | File(s) | Lab topic it corresponds to | Notes / differences |
|---|---|---|---|
| Explicit agent graph | `pipeline/graph.py`, `pipeline/state.py` | Graph of agents (lab: ReAct graph) | A fixed LangGraph pipeline (retrieve → evidence → judge → aggregate → supervise → explain), not a free ReAct tool-calling loop: the brief asks for a pipeline with a supervisor. |
| Supervisor with bounded retry and terminal error state | `pipeline/supervisor.py`, `graph.py` (`after_supervise`, `handle_error`) | Supervisor | Validates citations, routes back to the judge once, downgrades, never loops unboundedly (`recursion_limit` 25 as a second bound). |
| Role-specific agents sharing one model | `agents/evidence.py`, `agents/judge.py`, `agents/explainer.py` | Multi-agent roles / tools | Evidence and judge are model calls; the explainer is deterministic code. |
| Shared runtime used by CLI, UI and notebook | `runtime.py` | End-to-end system integration | Lazy resource loading and reuse. |
| Session history in the UI | `ui/app.py` | Memory | Only the current Streamlit session; no long-term memory. |
| Human-in-the-loop | Streamlit review of quotes; human-written `failure_analysis.md` | Human-in-the-loop | No interactive approval step inside the graph. MCP was not used. |

## Components with no lab counterpart

Data acquisition with the recorded compatibility fallback (`data/sources.py`), pair and claim-label
reconstruction (`data/prepare.py`), frozen splits and overlap checks (`data/splits.py`), resumable
evaluation with fingerprints (`evaluation/runner.py`) and the Streamlit interface (`ui/`) were written
specifically for this project.
