# Master_Prompt.md — Scientific Claim Checker Prototype

## 1. Document purpose, product goal, and fixed decisions

**Target file:** `D:\Additional Project\Gen_AI\Docs\Master_Prompt.md`

**Current status:** This document contains the agreed implementation plan. Saving this document does not mean the prototype has been implemented.

The document serves two purposes:

1. Explain the project in simple English so you understand what each component does.
2. Act as a detailed implementation prompt that another engineer or coding agent can follow.

> Act as a senior GenAI engineer building a reproducible Scientific Claim Checker. Follow the architecture, file responsibilities, implementation order, evaluation rules, and acceptance checks in this document. Build working functionality in small milestones. Report actual results, preserve source evidence, and explain implementation choices in simple language. Complete the required system before starting optional extensions.

**Product objective**

Build a local application where someone enters a scientific claim and receives:

- A verdict: `SUPPORT`, `CONTRADICT`, or `NEI`.
- Exact evidence quotations.
- The source abstract identifiers and titles.
- A short explanation.
- A visible indication when the available evidence is insufficient or conflicting.
- A downloadable JSON result.

**Decisions already established**

| Area | Decision | Reason |
|---|---|---|
| Product interface | Streamlit | Provides a complete Python prototype without maintaining a separate frontend and API. |
| Execution environment | Laptop only | Your selected preference. |
| Available hardware | Approximately 16 GB RAM and RTX 4050 Laptop GPU with 6 GB VRAM | Discovered from the current laptop. |
| Main language model | `qwen3:4b` through local Ollama | Explicitly suggested by the project brief. |
| Model fallback | Same model on CPU | Preserves model identity when GPU execution is unavailable; speed may be substantially lower. |
| Embeddings | `BAAI/bge-small-en-v1.5` | Specified in the brief. |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | Full repository identifier for the re-ranker referenced in the brief. |
| Agent orchestration | LangGraph with bounded supervisor logic | Makes stages and their transitions explicit. |
| Search | BM25, dense retrieval, and hybrid RRF | Required retrieval comparison. |
| Dense storage | Local NumPy embedding matrix | Exact similarity search is sufficient for approximately 5,183 abstracts and avoids a separate database service. |
| Assessment deliverable | One notebook importing the reusable project package | Keeps the notebook readable while preserving reproducibility. |
| Optional work | LoRA and climate evaluation after the core prototype | Your selected preference. |
| Hosting | Local application bound to localhost | Public deployment is outside the initial build. |
| Persistence | Local datasets, indexes, and experiment files | No database is needed for the first prototype. |
| User history | Current Streamlit session only | Avoids adding account and database work to the prototype. |

The prototype will need internet access for initial installation and downloads. After all required assets are cached, ordinary claim checking should work locally.

The plan distinguishes three categories throughout:

- **Coursework requirement:** explicitly required by the supplied brief.
- **Engineering decision:** selected here to make the implementation concrete.
- **Optional extension:** implemented after the core acceptance checks pass.

The original project requirements come from [P08_Scientific_Claim_Checker.pdf](C:/Users/Bhargav/Downloads/P08_Scientific_Claim_Checker.pdf).

---

## 2. Architecture, data contracts, and evaluation rules

### Product workflow

```text
User enters a claim
        |
        v
Validate the input and check local resources
        |
        v
Search abstracts using BM25 and dense retrieval
        |
        v
Combine the search rankings using RRF
        |
        v
Re-rank the candidate abstracts
        |
        v
Select the top-k abstracts
        |
        v
Evidence agent selects sentence IDs for each abstract
        |
        v
Judge evaluates each claim–abstract evidence pair
        |
        v
Aggregator combines document decisions
        |
        v
Supervisor verifies the final result
        |
        v
Explainer produces a readable explanation
        |
        v
Streamlit displays verdict, quotations, and sources
```

The agents share one local language model. They are separate responsibilities, prompts, and graph nodes; they do not require separately trained models.

### Preparation workflow

```text
Download source datasets
        |
        v
Validate schemas and record source versions
        |
        v
Recover and preserve original sentence boundaries
        |
        v
Normalize document identifiers
        |
        v
Build claim–abstract examples and claim-level reference labels
        |
        v
Freeze development and evaluation manifests
        |
        v
Build BM25 and embedding indexes
        |
        v
Record index fingerprints
```

**Source loading**

Attempt the dataset loading route specified in the brief:

```python
from datasets import load_dataset

corpus = load_dataset("mteb/scifact", "corpus")["corpus"]
claims = load_dataset("allenai/scifact_entailment")
```

Treat this as an early compatibility check, not as a guaranteed working instruction across all library versions.

The claims repository contains a Python dataset builder. Its actual features represent evidence as integer sentence positions. The implementation must inspect downloaded features rather than assume evidence contains quotation strings. [Official dataset builder](https://huggingface.co/datasets/allenai/scifact_entailment/blob/main/scifact_entailment.py)

If the named claims loader is unsupported by the selected Datasets version:

1. Record the precise compatibility failure.
2. Download the official SciFact release archive referenced by the builder.
3. Read the original JSONL files using `load_dataset("json", ...)`.
4. Reproduce the inspected claim–abstract transformation in ordinary project code.
5. Verify reconstructed columns, label values, evidence positions, and split counts.
6. Record the fallback route, download URL, and content hashes.
7. Explain this compatibility adaptation in the notebook.

Do not silently substitute an unrelated dataset or enable arbitrary remote code execution.

The official archive also supplies the original sentence lists needed for evidence selection across the complete corpus. Use the MTEB download for the retrieval corpus and verify that its document IDs align with the original SciFact corpus.

### Core internal records

These are the minimum contracts needed to prevent evidence and evaluation mistakes.

| Record | Required information | Purpose |
|---|---|---|
| `CorpusDocument` | Document ID, title, retrieval text, original sentence list, source metadata | Preserves searchable text and quotable source sentences. |
| `ClaimPair` | Claim ID, claim text, document ID, reference verdict, reference evidence positions, split | Supports claim–abstract diagnostics. |
| `ClaimGold` | Claim ID, text, reference document IDs, evidence by document, claim-level label | Supports whole-pipeline evaluation. |
| `RetrievalHit` | Document ID, rank, retrieval method, score, index fingerprint | Makes search results traceable. |
| `EvidenceSelection` | Document ID, selected sentence IDs, concise selection note | Separates selecting evidence from writing quotations. |
| `DocumentJudgement` | Document ID, verdict, cited sentence IDs, short reason, self-reported confidence | Represents a judgement against one paper. |
| `ClaimResult` | Status, claim, final verdict, evidence, explanation, conflict flag, timings, run metadata | Shared result returned to the UI, notebook, and CLI. |

**Evidence quotations must be constructed from stored source sentences.** The model selects IDs; application code obtains the actual quotations.

Use zero-based sentence positions internally. The interface may display sentence numbers starting from one, but that conversion must happen only when rendering.

### Claim-level versus pair-level labels

A claim–abstract label describes one particular paper’s relationship to a claim. It must not automatically become a label about the whole corpus.

For example, one abstract can be `NEI` while another supports the same claim. This is visible in the official dataset examples. [Claim–abstract dataset](https://huggingface.co/datasets/allenai/scifact_entailment)

Use the original SciFact claim annotations to construct claim-level reference labels:

- Supporting evidence and no contradicting annotation → `SUPPORT`.
- Contradicting evidence and no supporting annotation → `CONTRADICT`.
- No annotated supporting or contradicting evidence → `NEI`.
- Both supporting and contradicting reference annotations → flag as an annotation conflict and report separately.

Record any excluded annotation conflicts before sampling the final evaluation subset.

Unannotated documents are not automatically proven irrelevant. State this benchmark limitation when reporting evidence precision.

### Aggregating predicted document decisions

Use a deterministic policy:

| Document decisions | Final decision |
|---|---|
| At least one `SUPPORT`, no `CONTRADICT` | `SUPPORT` |
| At least one `CONTRADICT`, no `SUPPORT` | `CONTRADICT` |
| All usable decisions are `NEI` | `NEI` |
| Both `SUPPORT` and `CONTRADICT` appear | `NEI`, with `conflict=true` |
| A stage fails operationally | `status=error`, verdict unset |

Mapping conflicting predicted evidence to `NEI` is a documented prototype convention. The explanation must explicitly say that papers produced conflicting evidence.

A missing model, invalid index, or failed request must not be disguised as scientific uncertainty.

### Model execution defaults

Use these initial settings, then record their actual behavior during the compatibility milestone:

```yaml
model: qwen3:4b
temperature: 0
think: false
context_tokens: 8192
max_output_tokens: 768
request_timeout_seconds: 180
max_schema_repair_attempts: 1
max_concurrent_generations: 1

embedding_device: cpu
reranker_device: cpu
random_seed: 42
```

Keep the embedding model and re-ranker on CPU initially to leave GPU memory available for Ollama.

Check the installed model’s supported options before using them. Ollama supports JSON-schema output through its `format` field; validate the returned data again with Pydantic. [Structured output documentation](https://docs.ollama.com/capabilities/structured-outputs)

The prototype must not claim a guaranteed accuracy, response time, or GPU memory footprint before measuring them.

### Retrieval defaults

- Search the complete downloaded corpus.
- Build one BM25 representation per abstract.
- For dense retrieval, create tokenizer-aware windows for abstracts that exceed the embedding model’s input limit.
- Preserve document IDs and source sentence IDs in every window.
- Rank documents using their highest-scoring dense window.
- Retrieve up to 30 document candidates from each search method.
- Combine rankings using reciprocal rank fusion with constant `60`.
- Re-rank the first 20 unique hybrid candidates.
- Retain the first 10 for retrieval measurements.
- Send the first 3 abstracts to the normal checking pipeline.
- Support `k = 1, 3, 5` for experiments.

Re-ranking scores are relevance scores, not probabilities that a claim is true.

If an abstract exceeds the re-ranker’s input limit, score its windows and aggregate by maximum score. Do not silently score only the beginning of a long abstract.

### Evaluation defaults

Use official training claims for development and prompt changes.

Use a fixed subset of **100 unique eligible validation claims** for the final comparison. Select it with seed `42`, stratified by claim-level label. If fewer than 100 eligible claims exist, use all eligible claims and record that fact.

Freeze the selected IDs before model development is finalized. Retain every associated claim–abstract pair for pair-level diagnostics.

Check for repeated normalized claim text across development and final evaluation. Remove overlapping examples from development and document the overlap.

Run these final system variants on the same frozen claims:

| Variant | Purpose |
|---|---|
| Judge only | Baseline without retrieval. |
| Oracle abstracts | Measures performance when reference abstracts are supplied. |
| Full pipeline | Main product configuration. |
| Evidence agent off | Measures the effect of evidence selection. |
| Re-ranker off | Measures the effect of re-ranking. |
| Full pipeline, `k=1` | Measures a smaller evidence context. |
| Full pipeline, `k=5` | Measures a larger evidence context. |

For the oracle variant, supply the reference abstracts, not the reference verdicts or annotated evidence positions. Keep oracle evidence-sentence experiments separately named if added later.

Measure:

- Reference-abstract Recall@3 and Recall@10.
- Evidence-bearing document recall for claims with annotated evidence.
- Verdict accuracy and macro-F1.
- Per-class verdict performance.
- Evidence sentence precision and recall.
- Operational failure rate.
- Stage latency and total latency.
- Quote validity.
- Conflict frequency.

Report the number and unit behind every metric: claims, claim–abstract pairs, documents, or evidence sentences.

Keep operational errors in the evaluation population. Report verdict coverage and accuracy over all requested claims; do not remove failed cases to improve scores.

---

## 3. Complete file creation map

All paths below are relative to `D:\Additional Project\Gen_AI`.

**Create files as their milestone begins. Do not create dozens of empty placeholder files first.** Each file should contain a working implementation or a useful completed document.

### Project setup and documentation

| File | What it contains | Why it is needed |
|---|---|---|
| `README.md` | Project purpose, installation, quickstart, example output, links to detailed documentation | Gives a new user a clear entry point. |
| `pyproject.toml` | Package metadata, dependencies, developer tools, CLI entry point, test configuration | Makes installation and execution consistent. |
| `uv.lock` | Dependency versions generated after compatibility checks | Allows the tested environment to be recreated. |
| `.python-version` | Python 3.11 selection | Keeps developers on the same initial Python version. |
| `.gitignore` | Rules for virtual environments, model caches, downloaded data, indexes, logs, and temporary files | Prevents large generated files and local state from entering source control. |
| `configs/default.yaml` | Model, retrieval, evidence, runtime, and local path settings | Centralizes the normal prototype configuration. |
| `configs/experiments.yaml` | Named evaluation variants and their explicit overrides | Makes comparisons repeatable. |
| `Docs/Master_Prompt.md` | The complete implementation brief described here | Becomes the main human and coding-agent build guide. |
| `Docs/ARCHITECTURE.md` | System diagram, component responsibilities, data flow, graph transitions | Explains how the product fits together. |
| `Docs/DATA_CONTRACTS.md` | Record schemas, source mappings, sentence indexing, ID normalization, compatibility fallback | Prevents different components from interpreting data differently. |
| `Docs/EVALUATION.md` | Frozen split rules, metric definitions, aggregation policy, experiment matrix | Prevents misleading or incomparable results. |
| `Docs/RUNBOOK.md` | Installation, preparation, launch, offline operation, resume, and troubleshooting | Helps someone operate the prototype without reading implementation code. |
| `Docs/LAB_REUSE.md` | Course lab references and an honest mapping to adapted project components | Meets the brief’s requirement to cite reused labs. |
| `Docs/LIMITATIONS.md` | Dataset coverage, abstract-only evidence, uncalibrated confidence, model limitations, upstream attribution and usage terms | Makes the prototype’s boundaries clear. |
| `Docs/PRESENTATION.md` | Ten-minute presentation outline, demo sequence, likely questions | Connects engineering work to the assessment deliverable. |
| `notebooks/Scientific_Claim_Checker.ipynb` | Dataset loading, exploration, experiments, tables, error analysis, conclusions | Supplies the required single notebook. |

Do not invent exact dependency versions in the plan. Resolve and lock versions after the data-loading, graph, model, and structured-output smoke checks pass.

### Shared package and data preparation

| File | What it contains | Why it is needed |
|---|---|---|
| `src/claim_checker/__init__.py` | Package identity and version | Makes the shared implementation importable. |
| `src/claim_checker/config.py` | YAML loading, default merging, path resolution, configuration validation | Ensures CLI, notebook, and app use the same settings. |
| `src/claim_checker/schemas.py` | Pydantic records for documents, claims, evidence, judgements, results, and errors | Establishes shared contracts. |
| `src/claim_checker/runtime.py` | Lazy loading and reuse of embedding, re-ranking, and model clients | Prevents repeated initialization and unnecessary memory use. |
| `src/claim_checker/cli.py` | Commands for diagnosis, preparation, indexing, checking, evaluation, and reporting | Provides a scriptable entry point independent of Streamlit. |
| `src/claim_checker/data/__init__.py` | Data subpackage initialization | Keeps data responsibilities organized. |
| `src/claim_checker/data/sources.py` | Dataset acquisition, compatibility handling, archive reading, source fingerprints | Makes downloads reproducible and traceable. |
| `src/claim_checker/data/prepare.py` | Schema checks, ID normalization, pair reconstruction, corpus alignment, claim-label construction | Converts upstream data into reliable internal records. |
| `src/claim_checker/data/splits.py` | Development IDs, smoke IDs, frozen final IDs, overlap checks | Prevents evaluation leakage. |
| `src/claim_checker/data/store.py` | Loading and writing normalized JSONL and manifest files | Avoids duplicating storage logic across commands. |

### Retrieval and local inference

| File | What it contains | Why it is needed |
|---|---|---|
| `src/claim_checker/retrieval/__init__.py` | Retrieval subpackage initialization | Organizes search components. |
| `src/claim_checker/retrieval/index.py` | Index construction, document order, window metadata, fingerprints | Keeps embeddings aligned with the correct source documents. |
| `src/claim_checker/retrieval/bm25.py` | Deterministic tokenization and BM25 search | Implements the keyword retrieval baseline. |
| `src/claim_checker/retrieval/dense.py` | BGE encoding, normalized embeddings, NumPy similarity search, document aggregation | Implements meaning-based retrieval. |
| `src/claim_checker/retrieval/hybrid.py` | Reciprocal rank fusion and duplicate removal | Combines keyword and dense rankings. |
| `src/claim_checker/retrieval/rerank.py` | Claim–window cross-encoder scoring and document ranking | Implements the more careful second ranking stage. |
| `src/claim_checker/models/__init__.py` | Model subpackage initialization | Separates model integration from agent behavior. |
| `src/claim_checker/models/ollama_client.py` | Local endpoint calls, model availability checks, schema submission, response validation, bounded repair | Centralizes inference behavior and failure handling. |

The model wrapper must use a local endpoint by default. It must not silently switch to a paid API or cloud-hosted model.

### Agents, graph, and aggregation

| File | What it contains | Why it is needed |
|---|---|---|
| `src/claim_checker/agents/__init__.py` | Agent subpackage initialization | Organizes the role-specific implementations. |
| `src/claim_checker/agents/evidence.py` | Sentence-numbered contexts, evidence selection, sentence ID validation | Finds relevant evidence without inventing quotations. |
| `src/claim_checker/agents/judge.py` | Evidence-based and full-abstract judgement modes | Supports the main system and evidence-off ablation. |
| `src/claim_checker/agents/explainer.py` | Deterministic plain-English rendering of validated decisions and short judgement reasons | Produces readable output without adding another model call. |
| `src/claim_checker/pipeline/__init__.py` | Pipeline subpackage initialization | Groups orchestration components. |
| `src/claim_checker/pipeline/state.py` | Typed graph state and stage status | Defines what information passes between stages. |
| `src/claim_checker/pipeline/graph.py` | LangGraph nodes, transitions, and end conditions | Connects the agents into an explicit workflow. |
| `src/claim_checker/pipeline/supervisor.py` | Readiness checks, output validation, bounded retry routing, terminal error handling | Keeps the workflow controlled and prevents infinite loops. |
| `src/claim_checker/pipeline/aggregate.py` | Document-to-claim verdict combination and conflict handling | Produces consistent claim-level decisions. |
| `prompts/evidence.jinja2` | Evidence selection instructions and required output format | Makes evidence behavior inspectable and versioned. |
| `prompts/judge.jinja2` | Judgement instructions, allowed labels, grounding rules, concise reason requirement | Makes verdict behavior consistent. |
| `prompts/judge_only.jinja2` | Claim-only baseline instructions | Keeps the baseline explicit and reproducible. |

The full-abstract ablation uses the same judgement template and output schema, with the permitted context changed from selected sentences to all available sentences.

Store concise decision reasons and stage events. Do not require or display hidden chain-of-thought reasoning.

### Evaluation and product interface

| File | What it contains | Why it is needed |
|---|---|---|
| `src/claim_checker/evaluation/__init__.py` | Evaluation subpackage initialization | Organizes experiment code. |
| `src/claim_checker/evaluation/metrics.py` | Retrieval, verdict, evidence, coverage, validity, and latency calculations | Keeps metric definitions consistent. |
| `src/claim_checker/evaluation/runner.py` | Variant execution, checkpointing, exact-manifest resume, prediction recording | Makes long laptop experiments manageable. |
| `src/claim_checker/evaluation/analysis.py` | Results tables, confusion matrices, failure selection, analysis report generation | Turns predictions into interpretable findings. |
| `ui/app.py` | Streamlit application entry point and session behavior | Provides the product prototype. |
| `ui/views.py` | Claim form, result rendering, evidence panels, status messages, report display | Keeps UI rendering separate from orchestration. |

The UI calls the shared pipeline. It must not contain a second implementation of retrieval or judgement.

### Tests and automation

| File | What it verifies | Why it is needed |
|---|---|---|
| `tests/conftest.py` | Small synthetic corpus and reusable fixtures | Allows reliable tests without downloads or a running model. |
| `tests/test_data.py` | ID matching, sentence positions, label construction, split isolation | Catches data mistakes before expensive inference. |
| `tests/test_retrieval.py` | Ranking behavior, RRF, duplicate removal, window-to-document mapping | Protects retrieval correctness. |
| `tests/test_evidence.py` | Exact quotations, invalid sentence IDs, empty evidence | Protects the grounding contract. |
| `tests/test_model.py` | Structured response validation, timeout handling, retry limits | Protects the model boundary. |
| `tests/test_pipeline.py` | Stage transitions, aggregation, conflict handling, operational errors | Protects end-to-end behavior. |
| `tests/test_evaluation.py` | Hand-calculated metrics, error denominators, variant isolation, resume compatibility | Prevents incorrect research conclusions. |
| `tests/test_ui.py` | Input handling, ready/error/result states, JSON download | Verifies meaningful product interactions. |
| `.github/workflows/ci.yml` | Lightweight linting and offline tests | Makes future changes easier to check consistently. |

CI must not require downloading model weights, accessing private credentials, or running lengthy benchmark inference.

### Generated files

These are produced by commands. They are not manually authored source files.

| Generated path | Contents and purpose |
|---|---|
| `data/raw/source_manifest.json` | URLs, resolved revisions, checksums, loading route, and download metadata. |
| `data/raw/scifact/data.tar.gz` | Cached official source archive when required for sentence lists or compatibility reconstruction. |
| `data/raw/scifact/corpus.jsonl` | Extracted original corpus with sentence boundaries. |
| `data/raw/scifact/claims_train.jsonl` | Original development-source claim annotations. |
| `data/raw/scifact/claims_dev.jsonl` | Original validation-source claim annotations. |
| `data/processed/corpus.jsonl` | Validated internal corpus records. |
| `data/processed/claim_pairs_train.jsonl` | Normalized training claim–abstract rows. |
| `data/processed/claim_pairs_validation.jsonl` | Normalized validation claim–abstract rows. |
| `data/processed/claim_gold_train.jsonl` | Grouped training claim references. |
| `data/processed/claim_gold_validation.jsonl` | Grouped validation claim references. |
| `data/processed/data_quality.json` | Counts, missing matches, overlaps, label distributions, and alignment problems. |
| `data/splits/development.json` | Development claim IDs and selection metadata. |
| `data/splits/smoke.json` | Twenty development claims for early real-model checks. |
| `data/splits/final.json` | Frozen final claim IDs, seed, class counts, and source fingerprint. |
| `artifacts/index/manifest.json` | Corpus, tokenizer, embedding model, windowing, and index configuration fingerprints. |
| `artifacts/index/documents.jsonl` | Document order and BM25 search text. |
| `artifacts/index/windows.jsonl` | Dense window text and source mappings. |
| `artifacts/index/embeddings.npy` | Normalized dense vectors aligned with the window records. |
| `artifacts/runs/<run_id>/manifest.json` | Exact experiment identity, configurations, model digest, prompt hashes, and environment. |
| `artifacts/runs/<run_id>/predictions.jsonl` | Final claim predictions for each variant. |
| `artifacts/runs/<run_id>/pair_predictions.jsonl` | Document-level judgement diagnostics. |
| `artifacts/runs/<run_id>/retrieval.jsonl` | Ranked document results and reference-document comparisons. |
| `artifacts/runs/<run_id>/events.jsonl` | Stage outcomes, timings, retries, and structured errors. |
| `artifacts/runs/<run_id>/metrics.json` | Machine-readable metric outputs with denominators. |
| `artifacts/runs/<run_id>/results.csv` | Combined comparison table. |
| `artifacts/runs/<run_id>/confusion_matrices.json` | Per-variant verdict confusion matrices. |
| `artifacts/runs/<run_id>/failure_cases.csv` | Failed examples selected for inspection. |
| `artifacts/runs/<run_id>/failure_analysis.md` | Human-reviewed analysis of at least ten failures. |
| `artifacts/runs/<run_id>/report.md` | Readable experiment report. |
| `artifacts/notebooks/Scientific_Claim_Checker.executed.ipynb` | Clean executed notebook generated during final verification. |

For source control, retain code, configuration, documentation, tests, the notebook, and lightweight approved reports. Keep large downloads, caches, and indexes ignored.

### Optional extension files

Create these only after the core prototype passes its acceptance checks.

| File | Purpose |
|---|---|
| `configs/finetune.yaml` | Training model, adapter settings, data boundaries, and training budget. |
| `configs/climate.yaml` | Climate dataset mapping and separate evaluation settings. |
| `src/claim_checker/extensions/__init__.py` | Extension package initialization. |
| `src/claim_checker/extensions/lora.py` | Train and evaluate a small LoRA Judge through a separate Transformers/PEFT path. |
| `src/claim_checker/extensions/climate.py` | Prepare climate evidence and run an explicitly separate domain-transfer experiment. |
| `tests/test_extensions.py` | Label mapping and training/evaluation isolation checks for extensions. |

Do not mix the optional Transformers fine-tuning dependencies into the basic inference environment unnecessarily. Declare a separate dependency extra.

---

## 4. Step-by-step implementation milestones

### Milestone 1 — Establish the development environment

**Create:** root setup files, default configuration, package initialization, and initial runbook.

Steps:

1. Select Python 3.11.
2. Initialize the package using a `src` layout.
3. Define core dependencies: Datasets, Hugging Face Hub, NumPy, pandas, scikit-learn, Sentence Transformers, rank-bm25, Ollama client, Pydantic, PyYAML, Jinja2, LangGraph, and Streamlit.
4. Define developer dependencies: pytest, Ruff, Jupyter tooling, and notebook execution tools.
5. Install Ollama separately.
6. Download `qwen3:4b`.
7. Implement `claim-checker doctor`.
8. Check model availability, local endpoint access, package imports, writable paths, and a small structured-output request.
9. Record actual installed versions and model metadata.
10. Generate the lockfile only after compatibility checks succeed.

**Completion check:** A local structured response is generated, parsed, and validated successfully.

### Milestone 2 — Download and validate the data

**Create:** data source, preparation, storage, schema, and data-contract files.

Steps:

1. Run the named dataset loading calls.
2. Apply the documented compatibility fallback only if necessary.
3. Obtain the original sentence-segmented corpus.
4. Normalize all document IDs to strings.
5. Verify corpus uniqueness and expected scale.
6. Compare MTEB documents against original document IDs.
7. Preserve original sentence ordering.
8. Verify every reference evidence position is in range.
9. Construct both pair-level and claim-level records.
10. Write a data quality report.
11. Fail on unexplained missing documents or sentence mismatches rather than silently dropping them.

**Completion check:** The normalized corpus, reference examples, sentence mappings, and source manifest are internally consistent.

### Milestone 3 — Freeze experiment populations

**Create:** split logic and initial evaluation documentation.

Steps:

1. Group rows by claim ID.
2. Check repeated normalized text across splits.
3. Reserve training claims for development.
4. Select twenty development claims for smoke experiments.
5. Select the fixed final validation subset.
6. Save IDs and class distributions.
7. Calculate and store manifest hashes.
8. Prevent normal evaluation commands from overwriting the frozen final manifest.

**Completion check:** Every comparison can identify the exact same final claim population.

### Milestone 4 — Build BM25 retrieval

**Create:** BM25 and index foundations.

Steps:

1. Use a deterministic tokenizer.
2. Preserve scientific numbers and meaningful term structure.
3. Build BM25 from title and abstract text.
4. Implement ranked retrieval with document IDs.
5. Add stable tie-breaking by document ID.
6. Inspect a few development queries.
7. Measure reference-document Recall@3 and Recall@10.

**Completion check:** Search returns real corpus documents in a reproducible order.

### Milestone 5 — Build dense and hybrid retrieval

**Create:** dense and hybrid modules.

Steps:

1. Load the BGE embedding model.
2. Follow its documented query formatting.
3. Count tokens before embedding.
4. Split long abstracts into sentence-aware windows.
5. Encode documents in small CPU batches.
6. Normalize document and query embeddings.
7. Store embeddings and window metadata.
8. Implement exact inner-product search.
9. Collapse windows back to documents.
10. Fuse dense and BM25 rankings with RRF.
11. Measure all three retrieval approaches on the same examples.

**Completion check:** BM25, dense, and hybrid results are available with matching document identities.

### Milestone 6 — Add re-ranking

**Create:** re-ranking implementation.

Steps:

1. Load the specified cross-encoder.
2. Score the configured shortlist.
3. Window overlong abstracts instead of silently discarding their endings.
4. Aggregate window scores back to documents.
5. Compare retrieval with and without re-ranking.
6. Record both recall changes and added latency.

**Completion check:** A six-way retrieval comparison exists: three search methods, each with re-ranking on and off.

### Milestone 7 — Implement structured model calls and oracle judging

**Create:** model wrapper, Judge, judgement prompt, and baseline prompt.

Steps:

1. Submit a JSON schema to Ollama.
2. Use numbered sentences with source document IDs.
3. Parse the response with Pydantic.
4. Reject unsupported verdict labels.
5. Validate all cited IDs against supplied context.
6. Allow one repair attempt for invalid output.
7. Return a structured error after repeated failure.
8. Run the Judge on reference abstracts from development examples.
9. Inspect SUPPORT, CONTRADICT, and NEI behavior separately.
10. Keep all reference labels outside model inputs.

**Completion check:** Oracle predictions can be scored without retrieval being involved.

### Milestone 8 — Implement the Evidence agent

**Create:** evidence implementation and prompt.

Steps:

1. Supply the claim and sentence-numbered abstract.
2. Ask for evidence sentence IDs.
3. Allow an empty list.
4. Reject IDs outside the supplied source.
5. Recover exact quotation text from the corpus.
6. Deduplicate selected IDs.
7. Preserve document identity.
8. Handle large contexts using bounded sentence windows.
9. Flag any unsupported oversized input explicitly.
10. Evaluate selected sentences against the reference annotations.

**Completion check:** Every displayed quotation maps to a real source sentence.

### Milestone 9 — Connect the supervised graph

**Create:** graph state, graph, supervisor, aggregation, explainer, and runtime integration.

Steps:

1. Define the graph’s stage states.
2. Connect retrieval to evidence selection.
3. Connect evidence selection to per-document judgement.
4. Combine document verdicts.
5. Check quotation and output validity.
6. Generate the plain-English explanation.
7. Record timings and stage events.
8. Implement explicit terminal success and error states.
9. Bound retries and graph execution.
10. Expose the same pipeline through CLI and Python.

**Completion check:** A claim can move from input to validated final result through one shared workflow.

### Milestone 10 — Build the Streamlit prototype

**Create:** `ui/app.py` and `ui/views.py`.

The interface will contain three tabs:

| Tab | Contents |
|---|---|
| **Check a claim** | Claim form, example claims, progress, verdict, explanation, evidence, and JSON download. |
| **How it works** | Plain-language workflow and the meaning of the three verdicts. |
| **Evaluation results** | Saved comparison table, class performance, and failure examples. |

Product behavior:

- Use a form so typing does not trigger inference.
- Reject blank input.
- Limit input to 2,000 characters and a tokenizer-aware claim budget.
- Disable repeated submission while a request is running.
- Reuse model resources.
- Show actionable messages when preparation or model setup is missing.
- Keep technical settings in an advanced area.
- Keep source titles, document IDs, and quotations visible.
- Label confidence as model-reported and uncalibrated if displayed.
- Treat dataset IDs as identifiers; do not invent paper URLs or DOIs.
- Build example claims from real dataset records.
- Clearly distinguish saved examples from freshly generated results.

**Completion check:** A user can enter a claim and understand the verdict and its evidence without reading logs.

### Milestone 11 — Run the comparison suite

**Create:** metric, runner, analysis, and experiment configuration files.

Steps:

1. Finish the smoke suite first.
2. Freeze prompts and configurations.
3. Run the retrieval comparisons.
4. Run every named final system variant.
5. Save predictions after each claim.
6. Resume only when source, model, prompt, and configuration fingerprints match.
7. Calculate verdict, retrieval, and evidence metrics.
8. Record errors, coverage, conflicts, and latency.
9. Produce the combined table.
10. Keep pair-level diagnostics separate from claim-level scores.

Long experiments should run in resumable batches. Do not promise that the complete matrix will finish quickly on a laptop.

**Completion check:** All reported comparisons use the same frozen claims and traceable configurations.

### Milestone 12 — Analyse at least ten failures

Steps:

1. Select failures across the three classes.
2. Give special attention to CONTRADICT errors.
3. Include retrieval misses, evidence misses, and judgement mistakes where present.
4. Inspect source abstracts manually.
5. Record the earliest stage responsible for each failure.
6. Separate operational errors from scientific judgement errors.
7. Suggest a concrete improvement for each category.
8. Do not change prompts after seeing final failures and then present the same final subset as untouched evaluation.

If additional development follows, identify the later results as a new experiment.

**Completion check:** The report explains why failures happened, rather than only listing incorrect predictions.

### Milestone 13 — Complete the notebook and presentation

Steps:

1. Put data acquisition near the beginning of the notebook.
2. Explain every major stage before running its code.
3. Import the shared package.
4. Show source versions, selected IDs, and class counts.
5. Include retrieval and oracle experiments.
6. Include the full comparison table.
7. Include at least ten failure analyses.
8. Cite reused course labs accurately.
9. State limitations and optional future work.
10. Execute the notebook from a clean kernel.
11. Prepare the ten-minute presentation.
12. Rehearse a complete local demo.

The presentation will cover: problem, data, architecture, retrieval, evidence, judgement, results, failures, demo, and limitations.

**Completion check:** The notebook runs in order and the presentation is supported by actual artifacts.

### Milestone 14 — Verify local delivery

The planned CLI commands are:

```powershell
uv sync --locked
ollama pull qwen3:4b

uv run claim-checker doctor
uv run claim-checker prepare-data
uv run claim-checker build-index

uv run claim-checker check --claim "A claim entered by the user"

uv run claim-checker evaluate --suite smoke
uv run claim-checker evaluate --suite final
uv run claim-checker report --run-id <actual-run-id>

uv run streamlit run ui/app.py --server.address 127.0.0.1
```

These are target commands for the implementation, not commands that already exist.

Verify:

- A new environment can install the locked dependencies.
- Preparation can be repeated without corrupting artifacts.
- A stale index is detected.
- A downloaded and prepared installation can check claims offline.
- Missing resources produce clear setup instructions.
- The executed notebook and saved reports correspond to the recorded experiment.

### Milestone 15 — Optional extensions

**LoRA**

- Use only the training split.
- Start with the brief’s small Qwen Judge option.
- Use a separate Transformers/PEFT environment extra.
- Compare the trained Judge with an appropriate untuned version of the same base model.
- Preserve the final evaluation boundary.
- Record training time, memory, adapter settings, and actual results.
- Do not present model-size changes as improvements caused only by fine-tuning.

**Climate claims**

- Inspect the dataset’s label and evidence structure first.
- Create a separate climate evidence collection.
- Preserve any additional source labels rather than forcing them into an undocumented mapping.
- Report supported label mappings explicitly.
- Evaluate separately from SciFact.
- Do not search biomedical abstracts and interpret their lack of climate evidence as a meaningful domain-transfer benchmark.

---

## 5. Verification, acceptance criteria, assumptions, and direct links

### Required test scenarios

| Scenario | Expected outcome |
|---|---|
| Integer and string document IDs refer to the same paper | They normalize to the same identifier. |
| Evidence contains an invalid sentence position | Validation rejects it. |
| A model invents quotation text | The text is not accepted; quotations come from stored source sentences. |
| Evidence contains no decisive sentence | A valid NEI decision is possible. |
| Two documents produce opposing decisive verdicts | Final result carries the documented conflict flag. |
| Ollama is unavailable | Operational error with setup guidance. |
| Model response is invalid twice | Terminal structured error, without an unbounded retry loop. |
| Corpus or embedding configuration changes | Existing index is treated as stale. |
| One claim has several paired abstracts | It remains one claim for claim-level metrics. |
| An evaluation prediction fails | Failure remains visible in coverage and all-request accuracy. |
| An experiment is resumed with changed prompts | Resume is rejected or a new run is created. |
| A long abstract contains relevant evidence near the end | Retrieval windows preserve access to that content. |
| UI receives blank or oversized input | Validation occurs before expensive inference. |
| User downloads a result | JSON contains the same validated result shown on screen. |

Use synthetic fixtures for logic tests and a small real-data smoke suite for integration. Do not require an LLM to produce a particular scientific label in every automated unit test.

### Definition of done

The core prototype is complete when:

- The complete corpus is downloaded and indexed.
- Source and sentence mappings are validated.
- The application runs locally.
- The graph has distinct retrieval, evidence, judgement, supervision, and explanation responsibilities.
- Every successful decisive verdict contains valid source evidence.
- Operational errors remain distinguishable from NEI.
- Retrieval methods and re-ranking have been compared.
- Oracle and retrieved-evidence performance are reported.
- The baseline, full system, and ablations use the same fixed evaluation population.
- Results are measured rather than fabricated.
- At least ten failures are analysed.
- The required notebook executes from a clean kernel.
- Setup and troubleshooting instructions are complete.
- The presentation outline and local demo are ready.

No arbitrary accuracy threshold will be invented. Correct evaluation, evidence integrity, reproducibility, and honest reporting are mandatory even if the small model’s accuracy is modest.

### Assumptions and unresolved external material

- The separate **Group Project Guidelines & Judging** document has not been supplied.
- The referenced course lab notebooks are not present.
- Their specific marking rules and actual code reuse cannot be claimed as verified.
- `Docs/LAB_REUSE.md` must initially identify these missing materials honestly.
- Upstream dataset and model usage terms must be recorded before distributing bundled assets.
- The plan targets an educational prototype, with evidence limited to the downloaded collection.
- This master prompt is a saved plan. Prototype implementation is a separate task.

### Official links

| Resource | Direct website link |
|---|---|
| SciFact retrieval corpus | [MTEB SciFact](https://huggingface.co/datasets/mteb/scifact) |
| Claim–abstract dataset | [SciFact Entailment](https://huggingface.co/datasets/allenai/scifact_entailment) |
| Dataset transformation reference | [Official entailment builder](https://huggingface.co/datasets/allenai/scifact_entailment/blob/main/scifact_entailment.py) |
| Original project and data format | [SciFact repository](https://github.com/allenai/scifact) · [Data documentation](https://github.com/allenai/scifact/blob/master/doc/data.md) |
| Original research paper | [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/) |
| Dataset loading | [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets/en/loading) |
| Ollama installation | [Download Ollama](https://ollama.com/download) |
| Main model | [Qwen3 4B on Ollama](https://ollama.com/library/qwen3:4b) |
| Local chat API | [Ollama chat endpoint](https://docs.ollama.com/api/chat) |
| Structured model output | [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs) |
| Embedding model | [BGE small English v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| Re-ranking model | [MS MARCO MiniLM cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) |
| Embedding and re-ranking library | [Sentence Transformers documentation](https://www.sbert.net/docs/quickstart.html) |
| BM25 implementation | [rank-bm25 repository](https://github.com/dorianbrown/rank_bm25) |
| Agent graph | [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) |
| Product interface | [Streamlit documentation](https://docs.streamlit.io/) |
| Data validation | [Pydantic documentation](https://docs.pydantic.dev/latest/) |
| Evaluation metrics | [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) |
| Environment management | [uv documentation](https://docs.astral.sh/uv/) |
| PyTorch installation | [Official installation selector](https://pytorch.org/get-started/locally/) |
| Optional LoRA tooling | [Hugging Face PEFT](https://huggingface.co/docs/peft/en/index) |
| Optional climate dataset | [Climate-FEVER](https://huggingface.co/datasets/tdiggelm/climate_fever) |
