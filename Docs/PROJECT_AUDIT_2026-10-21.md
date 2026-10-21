**Scientific Claim Checker — project audit, 21 October 2026**

**Assessment: 85/100 for completion and alignment with the supplied P08 PDF.** The core application is substantially implemented. It is in the integration, evaluation and submission-finishing stage. Its current measured classification accuracy is **68/100**, which is a different measure from project completion.

This is my evidence-based assessment, not an official course mark. The separate “Group Project Guidelines & Judging” document referenced by the PDF is absent, so its additional requirements and weighting cannot be verified. Optional LoRA and Climate-FEVER work is excluded from the completion denominator.

The specification reviewed was the complete four-page file `C:\Users\Bhargav\Downloads\P08_Scientific_Claim_Checker.pdf`. I extracted the original PDF text and inspected the four existing page renders. The primary evidence was executable source, prompts, configurations, tests, notebooks, raw/processed data and saved experiment records; documentation claims were cross-checked against those records.

The review covers the implementation and test files individually. Generated data and caches were checked programmatically across their records; I do not claim to have semantically read every word of every abstract, cached response, dependency or binary. Python bytecode, tool caches and generated package metadata are not additional implementation. The existing `.pytest_cache` was inaccessible; tests used a fresh audit cache under `tmp`.

**How the score is allocated**

| Area | Available | Earned | Reason |
|---|---:|---:|---|
| Dataset acquisition, joins, evidence mapping and frozen splits | 15 | 15 | Complete data and reconstructed records verified. |
| Required models, full-corpus indexing and retrieval comparisons | 20 | 20 | Correct configured model families; BM25, dense, hybrid and six-way comparison present and checked. |
| Evidence, judge, supervisor, aggregation, explanation and shared application | 20 | 19 | Real working pipeline and UI; display integration defects remain. |
| Baselines, ablations, metrics and error analysis | 20 | 17 | Complete v1 matrix; complete v2 product/oracle predictions, but incomplete v2 suite, reporting and current-model ablations. |
| Assessment notebook | 10 | 8 | Complete executed v1 notebook exists; narrative and live default are now out of sync. |
| Required adaptation and citation of course labs | 5 | 0 | Explicitly not performed. Conceptual correspondence is not code adaptation. |
| Ten-minute talk and five-minute Q&A readiness | 5 | 3 | Timed plan, demo and Q&A notes exist; current demo is stale and final delivery is unverified. |
| Installation and reproducibility | 5 | 3 | Strong artifact tracing, but current missing dependencies, version drift and incomplete environment locking. |
| **Total** | **100** | **85** | |

The weighting is explicit so the grade can be challenged or adjusted. A low prediction score does not mean most of the software is missing. Conversely, exact quotations and passing unit tests do not establish that scientific verdicts are correct.

**What has been built**

The implemented flow is:

`claim → input/resource validation → BM25 + dense retrieval → RRF → cross-encoder → evidence sentence selection → per-document judgement → deterministic aggregation → supervisor → explanation → validated JSON/UI result`

Evidence selection and judgement use the same local `qwen3:4b` model with different prompts and schemas. Retrieval and explanation are implemented as code. This is a valid separation of responsibilities for the PDF's pipeline; the brief does not require a different trained model for each agent or an LLM-based explainer.

The default product uses `prompts/v2/judge.jinja2`, with checklist and reason generated before the verdict, thinking disabled, and a consistency supervisor. The named `full` evaluation variant deliberately remains v1. `full_v2_nothink` is the current product variant. This distinction explains many of the inconsistencies in the current documentation.

The Streamlit application, named ClaimLens, has claim entry, examples, settings, readiness checks, streamed stage progress, verdict/evidence/source cards, session history and matching JSON download. Its other tabs explain the workflow and load saved evaluation artifacts. It shares the same `ClaimChecker` used by the CLI and notebook. The saved UI screenshots show a developed interface; they are historical screenshots, not proof that today's default model behaves identically.

**Requirement-by-requirement assessment**

| P08 requirement | Observed implementation | Status |
|---|---|---|
| Index all 5,183 abstracts | 5,183 processed documents; 5,670 dense windows | Complete |
| Download using `load_dataset` near the notebook top | Notebook explicitly calls both named datasets; compatibility fallback is disclosed | Complete with justified loader deviation |
| Join claims and abstracts by ID | Canonical string IDs, pair reconstruction, gold labels and sentence validation | Complete |
| Fixed-seed subset with declared size | Frozen 100 validation claims, seed 42, stratified; overlap removed from development | Complete |
| Suggested local model, embeddings and reranker | Qwen3 4B, BGE-small-en-v1.5, MS MARCO MiniLM-L6-v2 | Complete |
| Free T4 or laptop CPU feasibility | CPU retrieval and local quantised Ollama demonstrated; saved latency used laptop GPU for the LLM | Partially verified: no clean T4 or CPU-only end-to-end reproduction in this audit |
| BM25 vs dense vs hybrid, each with/without reranking | All six saved rankings and metrics exist; metrics recompute correctly | Complete |
| Oracle abstracts to isolate downstream errors | Both v1 and v2 oracle results exist | Complete; includes evidence selection as well as judgement |
| Retriever → Evidence → Judge → Explainer, with supervisor | Explicit LangGraph nodes, bounded retry, terminal error route | Complete |
| SUPPORT / CONTRADICT / NEI with quoted evidence | Typed verdicts, valid source sentence IDs, exact corpus text | Complete mechanically; verdict accuracy remains imperfect |
| Accuracy, macro-F1, Recall@3/@10, evidence precision/recall | Implemented with explicit units and denominators | Complete for completed runs |
| Judge-only and single-prompt baselines | Both implemented and evaluated on 100 final claims | Complete in v1 |
| Evidence on/off, k={1,3,5}, reranker on/off | Full v1 matrix of eight variants | Complete in v1; equivalent v2 ablations not evaluated |
| One fixed-population result table | `final_v1/results.csv` and notebook table | Complete in v1 |
| At least ten failures, emphasis on CONTRADICT | 19 detailed cases, including 9 CONTRADICT; summary of all 44 full-v1 failures | Complete in v1; current v2 needs its own analysis |
| One notebook | Authoring notebook and matching executed copy | Present; needs current-version alignment |
| Adapt the three course labs and cite adapted sections | `LAB_REUSE.md` explicitly says none was copied or adapted | Missing |
| Ten-minute talk + five-minute Q&A | Timed presentation script and prepared answers | Prepared in outline; needs current results and demo rehearsal |
| LoRA and Climate-FEVER | Optional dependency extra and future-work discussion only | Not implemented; optional, no deduction in core score |

The raw-data fallback is not a fabricated substitute dataset: the downloaded upstream builder points to the official SciFact archive, and its claim–abstract transformation agrees with the project code. It is reasonable engineering adaptation. Exact acceptance of the fallback under the course rules remains a marking decision.

**Independent verification results**

| Check | Observed result |
|---|---|
| Raw/processed abstract sentence alignment | No mismatches across 5,183 documents |
| Training records | 809 claims, 919 claim–abstract pairs; rebuilt gold and pairs exactly match stored records |
| Validation records | 300 claims, 340 pairs; rebuilt gold and pairs exactly match stored records |
| Final subset | 100 unique IDs: SUPPORT 41, CONTRADICT 21, NEI 38 |
| Development/smoke/tuning overlap with final claim text | Zero normalized-text overlap |
| Frozen split ID hashes | All four checked hashes valid |
| Index embedding array | 5,670 × 384, finite values, unit-normalised within 1.2e-7 |
| Index checksum | Matches saved embedding checksum |
| Index sentence coverage | Every source sentence covered; no incorrect window text or truncated windows |
| Official archive and extracted source checksums | Match the saved source manifest |
| Six-way retrieval metric recomputation | All six match saved metrics |
| Saved verdict metric recomputation | Matches every available saved run metric checked |
| v1 final matrix | Eight variants × 100 claims; 800 saved predictions, zero operational errors |
| Current v2 product | 100/100 final predictions; zero operational errors; 170/170 exact valid quotes |
| LLM disk-cache syntax | 2,870 entries inspected; entry JSON and stored response JSON all parse |
| Existing tests | 272 passed, 13 deprecation warnings, 16.04 seconds |
| Static lint | Ruff passes for `src`, `tests`, `ui` |
| Executed notebook artifact | 23/23 code cells have execution counts; zero error outputs; cell source matches authoring notebook |

Tests initially could not start in the existing conda environment because `sniffio` was missing. For audit verification only, `sniffio` was installed into `tmp/audit_deps` and added through process-local `PYTHONPATH`. Automatic third-party pytest plugin loading was disabled. The application environment was not repaired globally. Thus “272 passed” must be read with this explicit environment qualification.

`pip check` reports missing `sniffio` and `httpx2` dependencies of the installed `langsmith`. The missing `sniffio` also blocks the actual `langgraph → langchain_core` runtime import without the audit override. The current installed NumPy, pandas and ipykernel versions differ from `requirements.lock.txt`:

| Package | File pin | Installed at audit |
|---|---|---|
| numpy | 2.5.2 | 2.4.6 |
| pandas | 3.0.5 | 3.0.3 |
| ipykernel | 7.3.0 | 6.31.0 |

With the temporary dependency path, `doctor --skip-model-test --json` passed its listed checks: data, split, fresh index, imports, local Ollama 0.40.6 and the expected qwen3:4b digest. Two real live checks then exercised structured generation. I did not rerun the final evaluation, modify its predictions, or re-execute the full notebook.

**Actual prediction quality**

The following values were recomputed from saved predictions, not copied uncritically from prose:

| Run / variant | Claims | Accuracy | Macro-F1 | CONTRADICT F1 |
|---|---:|---:|---:|---:|
| final_v1 / judge_only | 100 | 50% | 0.504 | 0.529 |
| final_v1 / retrieve_judge_single_prompt | 100 | 56% | 0.450 | 0.174 |
| final_v1 / full | 100 | 56% | 0.487 | 0.296 |
| final_v1 / oracle | 100 | 67% | 0.571 | 0.240 |
| final_v1 / evidence_off | 100 | 53% | 0.446 | 0.320 |
| final_v1 / rerank_off | 100 | 54% | 0.468 | 0.296 |
| final_v1 / full_k1 | 100 | 56% | 0.473 | 0.214 |
| final_v1 / full_k5 | 100 | 57% | 0.493 | 0.296 |
| final_v2 / full_v2_nothink — current product | 100 | **68%** | **0.662** | **0.558** |
| final_v2 / oracle_v2 | 100 | **70%** | **0.671** | **0.541** |

The additional `final_v2/full_v2` thinking variant has only **53/100** predictions at the audit snapshot. Its partial-prefix score must not be represented as a complete final score. The entire v2 suite has 253/300 planned predictions and no `metrics.json`, `results.csv`, confusion matrices, failure-case CSV or combined report yet. The complete product and oracle subsets can still be validly scored individually, as above.

The current product's confusion matrix is:

| Gold \ predicted | SUPPORT | CONTRADICT | NEI |
|---|---:|---:|---:|
| SUPPORT, n=41 | 28 | 5 | 8 |
| CONTRADICT, n=21 | 2 | 12 | 7 |
| NEI, n=38 | 5 | 5 | 28 |

It gets 68 right and 32 wrong. CONTRADICT recall improved from 4/21 in v1 to 12/21 in v2, but 9 CONTRADICT claims are still missed. SUPPORT recall declined from 37/41 to 28/41. This is a real tradeoff: fewer false SUPPORT decisions, more conservative or false CONTRADICT decisions.

Retrieval is considerably stronger than verdict classification. Hybrid + reranking yields reference-document Recall@3 **0.676** and Recall@10 **0.824**, with evidence-bearing document Recall@3 **0.910** and Recall@10 **0.942**. These are means of per-claim document recall, not a literal statement that 91% of all claims are correctly classified. The evidence-bearing denominator is 62 claims; ordinary reference recall uses all 100.

For current v2, evidence selection precision/recall is **0.632 / 0.653** and judge-cited evidence precision/recall is **0.808 / 0.488**. Precision excludes selections in unannotated documents, as explicitly documented; it is not overall precision over every displayed sentence. All 170 quotations are exact source text, but exact text can still be irrelevant or misinterpreted.

The paired saved v1/v2 comparison shows a +12 percentage-point accuracy gain. Its exact McNemar p-value is **0.073**; the recorded macro-F1 bootstrap interval supports improvement, but the accuracy comparison does not pass a conventional 0.05 McNemar threshold. The 100-claim final subset was already inspected during v1 failure analysis. `Docs/LIMITATIONS.md` correctly acknowledges v2 as a second evaluation on an already-inspected set, even though actual v2 iterations used training claims. Do not call v2 a fresh blind test.

**Live behavior checked during this audit**

| Claim | Reference label | Current default output | Observation |
|---|---|---|---|
| Train claim 34: “A deficiency of folate increases blood levels of homocysteine.” | SUPPORT | **CONTRADICT** | README/presentation's supported demo no longer matches the default. Repeated once to inspect the full result; same wrong label. |
| Train claim 2: “1 in 5 million in UK have abnormal PrP positivity.” | CONTRADICT | **CONTRADICT** | Numeric case handled correctly by v2 in this check. |

Both checks used real local model calls, not scripted transports, with LLM disk caching disabled and zero LLM cache hits. Quotes matched the corpus. The first process check took about 29.5 seconds including initial model/resource loading, and the next about 9.6 seconds. These two timings are observations, not a latency benchmark or an accuracy estimate.

For claim 34 the retrieved reference document was present. The judge compared deficiency with supplementation incorrectly and described the evidence as opposite to the claim. Both decisive document judgements became CONTRADICT; final model-reported confidence was 0.95. The supervisor accepted them because checklist, reason and label were internally consistent. This demonstrates the supervisor's actual limit: it checks grounding and internal consistency, not scientific truth.

**Findings that prevent a 100/100 assessment**

1. **Required lab adaptation is absent.** `Docs/LAB_REUSE.md:14` and notebook §10 explicitly state that no course lab code was adapted. The PDF asks the team to start from and adapt those labs. A topic-to-component table does not meet that requirement. The actual labs must be examined, appropriate reuse performed, and exact section/cell citations added; do not invent citations.

2. **The current environment does not reproduce the documented ready state without assistance.** Normal imports/test collection fail for missing `sniffio`; `pip check` also reports `httpx2`. Several installed versions differ from the purported lock. `requirements.lock.txt` pins only direct dependencies, and `pyproject.toml` uses broad ranges; CI installs those ranges rather than the pins. The notebook imports `matplotlib`, which is not declared as a direct project/dev dependency. Use one verified environment specification and verify clean installation and notebook execution there. This audit left the application environment unchanged.

3. **The notebook and presentation describe v1 while the live default is v2.** Notebook cell 3 sets `RUN_ID="final_v1"`; its agent narrative shows the v1 judge prompt, but its live demo calls `checker.check(...)`, which now selects v2. The presentation's supported folate demo is wrong under the current default. Either present a clearly labelled v1 experiment and explicitly select v1 for its demo, or align the notebook, reports and demo to v2. Historical v1 scores should remain labelled and preserved.

4. **V2 delivery is unfinished.** The product and oracle each have all 100 predictions, but the thinking variant is incomplete and no combined v2 report/failure analysis exists. The original full matrix is v1; there are no completed v2 evidence-off, reranker-off or k=1/k=5 ablations. Finish the intended evaluation scope or explicitly narrow and label the v2 results. Keep partial results visibly partial.

5. **The UI does not use the prepared one-based reason text.** `runtime.py:596` creates `reason_display`, and the explainer uses it, but `ui/components.py:519` still renders `j.reason`. A synthetic v2 example containing `Sentence [0]` displays that raw reference while the abstract is labelled `S1`. This was reproduced using the actual component function. Render the prepared display reason consistently and cover that integration in tests.

6. **The UI mislabels consistency downgrades.** `ui/components.py:452` and `ui/views.py:240` label every downgraded judgement as “had no valid citation”. V2 can downgrade a judgement with valid citations because its verdict disagrees with its checklist/reason. The backend explanation distinguishes them correctly; the source-card note does not. This was reproduced with valid citations plus a consistency flag.

7. **Partial-run reporting does not enforce the advertised full denominator.** `evaluation/analysis.py:60` scores only records supplied, and `analyse_run` at line 182 does not validate completion against the manifest population. Therefore a report made during a 53/100 run can show coverage 1.0 for 53 observed records while the report text says all requested claims. The notebook independently labels partial runs, but the CLI/UI report path lacks that protection. This does not invalidate the completed v1/product-v2 numbers verified here.

8. **Resume identity is strong but incomplete.** `evaluation/runner.py:78` fingerprints data, population, index identity, model digest, prompts and configuration. It does not hash application code or compare the recorded dependency environment. Changing aggregation, consistency rules or other Python behavior can therefore leave the resume identity unchanged and mix implementation versions in one run. No such mixed-run corruption was established in this audit; the finding concerns the guarantee made by the mechanism.

9. **Index integrity checks are weaker than the manifest suggests.** `retrieval/index.py:115` records an embedding checksum and model revision, but `check_index` at line 129 compares corpus/config identity and file presence; it does not verify the embedding checksum, and the saved model revision is not part of the expected identity. The current embeddings are valid and their checksum matches. The gap is detection of future corruption or changed upstream model weights under the same model name.

10. **Documentation and submission housekeeping remain.** README lists 142 functions / 193 tests; the audit finds 204 test functions and 272 collected passing cases. `Docs/EVALUATION.md` ends at §10 while the newer limitations text references v2 results in §11. Presentation checkboxes still mark v1 reporting and notebook execution incomplete despite existing artifacts. These are stale status records. No Git repository metadata was present in the workspace, so commit history, remote backups and actual CI execution could not be verified.

These findings do not mean a rewrite is needed. Data handling, retrieval, graph composition, bounded model calls, exact quote extraction, operational error handling and the original comparison suite are already substantial working components.

**File-by-file implementation assessment**

Paths below are relative to the project root. “Implemented” refers to inspected code and the checks described above; it is not a promise of complete test coverage or perfect model behavior.

| File | Role and assessment |
|---|---|
| `src/claim_checker/__init__.py` | Package metadata and public entry-point documentation; minimal and complete. |
| `src/claim_checker/__main__.py` | Routes `python -m claim_checker` to the CLI; implemented. |
| `src/claim_checker/config.py` | Typed configuration, YAML merging, project paths, loopback-only model host, suite/variant validation and hashes. Good foundation; numeric constraints are uneven across configurable settings. |
| `src/claim_checker/schemas.py` | Typed corpus, gold, selection, judgement, quote, event, error and result models; v2 checklist schema preserves generation order. Individual fields are validated; result-wide success/evidence invariants rely on pipeline code rather than schema-level checks. |
| `src/claim_checker/errors.py` | Typed failures and setup hints; supports the important NEI-vs-operational-error distinction. |
| `src/claim_checker/cli.py` | Data/index preparation, checks, doctor, retrieval evaluation, resumable evaluation, reporting, dev split, comparisons and latency. Implemented; report completion guard and environment consistency need attention. |
| `src/claim_checker/runtime.py` | Shared lazy checker, resource reuse, lock, LRU retrieval cache, baseline dispatch, graph streaming and result assembly. Real v2 checks succeeded; display fields are not consumed uniformly by UI. |
| `src/claim_checker/data/__init__.py` | Data package marker/docstring; no missing functionality hidden here. |
| `src/claim_checker/data/sources.py` | Both named loader attempts, official archive compatibility route, plain-text builder inspection and source provenance. Implemented; source revisions are recorded but not pinned for future downloads. |
| `src/claim_checker/data/prepare.py` | ID/text alignment, evidence validation, pair reconstruction, claim labels and data-quality report. Rebuilt full gold/pair files match stored records. |
| `src/claim_checker/data/splits.py` | Deterministic stratification, final freezing, duplicate-text removal, smoke and v2 tuning population. All saved split hashes and final overlap checks passed. |
| `src/claim_checker/data/store.py` | UTF-8 JSON/JSONL, atomic writes, flushed checkpoint appends and typed loaders. Implemented. A torn final JSONL line is not automatically recovered by `iter_jsonl`; diagnostic reconciliation addresses a different interruption boundary. |
| `src/claim_checker/retrieval/__init__.py` | Retrieval package marker/docstring. |
| `src/claim_checker/retrieval/bm25.py` | Scientific term/number tokenizer, BM25 and stable ordering. Implemented and tested; lexical relevance alone does not encode scientific polarity. |
| `src/claim_checker/retrieval/dense.py` | BGE loader, lightweight tokenizer, sentence-aware windows, normalised encoding and exact max-window document scoring. Current index covers every sentence without flagged truncation. |
| `src/claim_checker/retrieval/hybrid.py` | Duplicate-safe reciprocal rank fusion; concise, deterministic and tested. |
| `src/claim_checker/retrieval/rerank.py` | Claim-aware token budget, windowed cross-encoder scoring and max-window aggregation. Implemented; relevance scores are correctly distinguished from truth probabilities. |
| `src/claim_checker/retrieval/index.py` | Index construction, artifacts, identity, composed retriever and stale-index handling. Actual artifacts validate; checksum/model-revision enforcement gap described above. |
| `src/claim_checker/models/__init__.py` | Model package marker/docstring. |
| `src/claim_checker/models/ollama_client.py` | Local transport, model digest, self-contained JSON schema, repair, cache, timeout/error mapping and optional thinking continuation. Extensive tests. Context checks use a chars/3 estimate, and enlarged repair/continuation prompts are not independently rechecked; this is not an exact tokenizer guarantee. |
| `src/claim_checker/agents/__init__.py` | Agent package marker/docstring. |
| `src/claim_checker/agents/prompting.py` | Strict Jinja rendering, separate system/user parts, v1/v2 template hashes. Implemented. |
| `src/claim_checker/agents/evidence.py` | Bounded sentence windows, model-selected IDs, ID filtering and exact quote building. Source-grounding design is strong; selected evidence recall is about 65%, so omissions remain consequential. |
| `src/claim_checker/agents/judge.py` | Evidence/full-abstract judge, v1/v2 schemas, claim-only and single-prompt baselines, normalised citation IDs. Implemented; scientific relation interpretation remains the major accuracy limitation. |
| `src/claim_checker/agents/explainer.py` | Deterministic wording for decisions, errors, conflicts, single-prompt results and one-based v2 reason references. Implemented; accurately surfaces backend decisions without independently checking their truth. |
| `src/claim_checker/pipeline/__init__.py` | Pipeline package marker/docstring. |
| `src/claim_checker/pipeline/state.py` | Typed options/state, timing reducer and stages. Implemented. |
| `src/claim_checker/pipeline/graph.py` | Real LangGraph workflow, oracle routing, evidence/judge stages, bounded repair, error terminal and v2 supervision. Working and exercised by tests/live checks. |
| `src/claim_checker/pipeline/supervisor.py` | Input limits, citation checks, uncited-decisive detection and downgrade. Implemented; establishes citation validity, not semantic entailment. |
| `src/claim_checker/pipeline/aggregate.py` | Deterministic document-to-claim rule; conflicting decisive documents become NEI with a flag. Implemented and documented. |
| `src/claim_checker/pipeline/consistency.py` | V2 checklist/label and reason-phrase checks; one shared repair round. Useful heuristic, but consistent misreadings pass, as in the live folate example. |
| `src/claim_checker/evaluation/__init__.py` | Evaluation package marker/docstring. |
| `src/claim_checker/evaluation/metrics.py` | Verdict/error metrics, retrieval, evidence, quote integrity, operational rates, cache provenance and latency. Completed-run calculations verify. The caller must supply the full intended population. |
| `src/claim_checker/evaluation/runner.py` | Fixed population, variant execution, checkpointing, diagnostic reconciliation and run identity. Implemented; source-code/environment fingerprints are missing. |
| `src/claim_checker/evaluation/analysis.py` | Recomputes metric artifacts, tables, confusion matrices and failure-case selection. Implemented; does not establish run completeness before reporting. |
| `src/claim_checker/evaluation/compare.py` | Paired bootstrap and exact McNemar comparisons, error-pattern counts. Implemented and used by saved comparisons; compares shared IDs, so population size still needs checking. |
| `src/claim_checker/evaluation/latency.py` | Warm-up-separated cold/cached benchmarks and environment/hardware context. Implemented; existing saved benchmark is historical and must be matched to the product version being described. |
| `prompts/evidence.jinja2` | Clear evidence-ID selection instructions, allows empty evidence; implemented. |
| `prompts/judge.jinja2` | Frozen v1 per-document judging prompt; preserved for historical evaluation. |
| `prompts/judge_only.jinja2` | Claim-only baseline; deliberately uses model knowledge without source evidence. |
| `prompts/retrieve_judge.jinja2` | One-prompt retrieval baseline with document/sentence citations; implemented. |
| `prompts/v2/judge.jinja2` | Current product checklist/reason-before-label prompt and comparison rules. Improves benchmark point estimates; cannot be treated as a reliable scientific reasoning engine. |
| `ui/app.py` | Streamlit session flow, form, settings, examples, readiness, history, live execution and shared runtime. Implemented and exercised by AppTest. |
| `ui/views.py` | Validation adapters, result formatting, saved artifacts and tab rendering. Implemented; downgrade wording and partial-run presentation need correction. |
| `ui/components.py` | Escaped HTML cards, quote/source display, timelines, confidence and history. Implemented; raw reason rendering and downgraded-reason labels reproduced as defects. |
| `ui/theme.py` | Stylesheet, optional font import and static live character counter. Implemented; system fonts support offline fallback. |
| `ui/assets/theme.css` | Comprehensive styling, responsive rules, focus visibility and reduced-motion support. Current selectors depend on Streamlit DOM structure; future version changes require UI regression review. |
| `configs/default.yaml` | Complete paths/model/retrieval/runtime defaults, v2 configuration and product selection. Current product explicitly v2/no-thinking. |
| `configs/experiments.yaml` | Eight original variants, dev experiments and three v2 variants. `full` description still calls it the main product although it is v1. |
| `.streamlit/config.toml` | Local server, usage-stat setting, watcher setting and theme; implemented. |
| `.github/workflows/ci.yml` | Offline mocked test/lint workflow for Python 3.11. Configuration exists; no successful remote CI run was verified. Does not install the pinned lock file or execute the notebook. |
| `pyproject.toml` | Packaging, CLI entry point, broad dependency ranges, dev/LoRA extras, pytest and Ruff. Missing direct notebook `matplotlib` declaration and full environment lock. |
| `requirements.lock.txt` | Direct package pins, not a transitive dependency lock; three pins drift from current installed versions. |
| `.python-version` | Records intended Python tooling version; not proof of which interpreter commands actually use. |
| `.gitignore` | Separates regenerated data/index/cache from selected reproducibility artifacts. Comparisons/dev-tuning artifacts remain excluded by broad artifact rules; verify submission packaging includes needed evidence. |
| `README.md` | Useful architecture, setup, commands and v1 results. Test count and supported demo are stale for current code/default. |
| `resume_1.md` | Historical handoff, not current status authority; predates much of v2 and the current 272-test suite. |

**Tests, individually**

All counts below are test function definitions; parametrisation produces 272 executed cases from 204 functions overall.

| File | Functions | Coverage observed |
|---|---:|---|
| `tests/conftest.py` | Fixtures | Small synthetic corpus, scripted Ollama and fake retriever. Makes offline tests reliable but does not simulate model scientific competence. |
| `tests/test_data.py` | 13 | IDs, corpus alignment, pair transformation, label rules, bad evidence, stratification, freezing and duplicate-text removal. |
| `tests/test_retrieval.py` | 8 | Scientific tokens, BM25 ties, RRF, long-text windows, dense collapse, reranking tail evidence and stale corpus/config. |
| `tests/test_evidence.py` | 6 | Exact quote extraction, ID filtering, bounded windows, repair and empty selections. |
| `tests/test_model.py` | 21 | Schema parsing, repair fallback, model/host matching, errors, context guard, cache reuse/corruption and atomic writes. |
| `tests/test_pipeline.py` | 35 | End-to-end synthetic graph, conflicts, uncited repair, error states, baselines, input rejection, oracle checks and progress/cache provenance. |
| `tests/test_evaluation.py` | 21 | Hand-checked metric denominators, checkpoints/resume, failure heuristics, diagnostics, downgrade counts, path validation and latency output. |
| `tests/test_ui.py` | 42 | Pure view helpers and Streamlit AppTest flows for submission, history, validation, errors, conflicts, examples, progress, stored results and downloads. |
| `tests/test_ui_components.py` | 25 | HTML escaping, verdict/error cards, source highlights, confidence, timelines and theme assets. Missing v2 display integration regressions identified by this audit. |
| `tests/test_v2.py` | 26 | Frozen v1 behavior, schema order, thinking budget/cache, consistency repair, product wiring, tuning split and paired comparisons. |
| `tests/test_result_extras.py` | 7 | Added confidence, embedded sentences, one-based reason field, punctuation and warm-state helpers. Verifies the reason field but not its actual consumption in source-card HTML. |

**Notebook, documentation and generated artifacts**

| File / artifact | Assessment |
|---|---|
| `notebooks/Scientific_Claim_Checker.ipynb` | 47 cells, including 23 code cells. Explicit data loading, compatibility explanation, retrieval, graph, v1 matrix, live demo, failures and limitations. Source notebook has no saved outputs; matching executed copy does. Needs current-version alignment. |
| `artifacts/notebooks/Scientific_Claim_Checker.executed.ipynb` | 23 executed code cells, no error outputs, same cell source. Historical successful execution, not proof that current broken environment or changed default will reproduce its displayed results. |
| `Docs/Master_Prompt.md` | Detailed implementation plan. Useful background; completion is judged against actual files and P08 requirements, not against its aspirations or handoff claims. |
| `Docs/ARCHITECTURE.md` | Thorough architecture and additive v2 notes; substantially matches code. Claims about prepared display reasons exceed their current UI integration. |
| `Docs/DATA_CONTRACTS.md` | Explains record formats, sentence positions, aggregation and provenance. Useful; some invariants described as contracts are enforced by pipeline behavior rather than every schema constructor. |
| `Docs/EVALUATION.md` | Detailed v1 denominators, full matrix and interpretation. V2 final results are not incorporated despite references elsewhere. |
| `Docs/LIMITATIONS.md` | Meaningful limitations, including the already-inspected v2 test population and heuristic consistency check. Its v2 section references a nonexistent evaluation §11. |
| `Docs/LAB_REUSE.md` | Honest gap disclosure and conceptual mapping. Required actual adaptation/citations are absent. |
| `Docs/PRESENTATION.md` | Timed ten-minute script, Q&A and demo. Historical v1 narrative and now-wrong supported demo need revision; checkbox status is stale. A PowerPoint file is not independently required by the supplied PDF. |
| `Docs/RUNBOOK.md` | Detailed install, operation, offline use, evaluation and troubleshooting. Needs verification against a clean reproducible environment, rather than reliance on shared conda base. |
| `data/raw/source_manifest.json` | Real provenance record; archive/extracted checksums verify. |
| `data/raw/scifact/corpus.jsonl`, `claims_train.jsonl`, `claims_dev.jsonl` | Full stored source records used for audit reconstruction; source checksum checks pass. |
| `data/raw/scifact/data.tar.gz` | Source archive checksum agrees with manifest. |
| `data/raw/scifact/builder/scifact_entailment.py` | Upstream builder read as source text; fallback transformation agrees with its relevant logic. Not executed. |
| `data/processed/corpus.jsonl` | All 5,183 sentence lists agree with raw corpus. |
| `data/processed/claim_gold_train.jsonl`, `claim_gold_validation.jsonl` | All reconstructed claim labels and evidence mappings exactly match stored records. |
| `data/processed/claim_pairs_train.jsonl`, `claim_pairs_validation.jsonl` | All 919/340 pairs exactly match reconstruction. |
| `data/processed/data_quality.json` | Quality/provenance summary, supported by the independent checks above. |
| `data/splits/final.json`, `development.json`, `smoke.json`, `dev_tune.json` | All unique IDs/hash checks pass; train-based populations have no normalized-text overlap with final. |
| `artifacts/index/manifest.json`, `documents.jsonl`, `windows.jsonl`, `embeddings.npy` | Complete real index; coverage, window text, numerical properties and checksum verified. |
| `artifacts/retrieval/retrieval_eval_final.json`, `.csv`, `retrieval_per_claim_final.jsonl` | Six-way comparison on fixed final claims; saved metric values recompute correctly. |
| `artifacts/runs/final_v1/` | Complete eight-variant experiment, metrics, report, trace files and 19-case failure analysis. |
| `artifacts/runs/final_v2/` | Complete product/oracle predictions; incomplete thinking variant; combined reports absent at snapshot. |
| `artifacts/runs/dev_v1/`, `dev_v2_it1/`, `dev_v2_it2/`, `dev_v2_frozen/` | Actual tuning runs. Iteration-2 thinking was deliberately stopped after three claims and is not a full experiment. Frozen product run matches saved metrics. |
| `artifacts/runs/smoke_v1/`, `smoke_integration/` | Real 20-claim and 5-claim integration evidence; not final performance estimates. |
| `artifacts/comparisons/` | Paired v1/v2 and tuning comparisons; current product/oracle comparisons are present even though v2 suite report is absent. |
| `artifacts/dev_tuning/` | Prompt/config snapshots, tuning notes and exploratory scripts, not another production implementation. Frozen configuration is explained. |
| `artifacts/latency/` | Saved cold/cached measurements with explicit warm-up/cache context; present files should not automatically be attributed to a newer default. |
| `artifacts/ui_screenshots/` | Twelve historical UI captures. Empty, support and operational-error images visually inspected during this audit. |
| `artifacts/cache/llm/` | 2,870 syntax-valid cached responses; inspected as generated data, not manually adjudicated scientific evidence. |
| `artifacts/logs/` and top-level artifact logs | Historical progress/debug evidence. File presence or a success flag alone was not accepted as proof of completed evaluation. |
| `src/claim_checker.egg-info/` | Generated editable-install metadata; not independent proof that dependencies are currently healthy. |
| `tmp/`, bytecode and tool caches | Temporary/generated working files, not required product features. Audit helpers/evidence are under `tmp` and should not be confused with project test additions. |

**Remaining work, in order**

1. Restore a clean, complete environment specification; verify ordinary imports, `pip check`, tests, doctor and notebook without audit-specific path overrides.
2. Choose and label the submission's primary version. Align runtime options, notebook narrative, figures, reports and demo to that choice.
3. Resolve the required lab adaptation and exact citations using the actual lab material, and review the missing general marking guidelines.
4. Finish or explicitly scope the v2 evaluation; produce its current results/failure analysis and any ablations used to justify the current product.
5. Fix the two reproduced UI display defects; add meaningful integration regression coverage for those specific user-visible mistakes.
6. Add completion-aware partial reports and stronger source/environment/index provenance checks before relying on exact-resume claims.
7. Update stale counts/checklists and rehearse the current talk/demo. Treat LoRA and Climate-FEVER as later stretch work after these core delivery gaps.

The most useful next phase is finishing and reconciling the existing system. A new data pipeline, new UI or wholesale rewrite would discard work that is already implemented and verified.

**Audit evidence files created**

`tmp/project_audit_evidence.json` contains record reconstruction, split/index verification, per-run recomputed metrics and an implementation inventory with line counts and hashes. `tmp/audit_runtime_evidence.json` contains the two uncached live checks, dependency drift and reproduced UI issues. `tmp/audit_folate_full.json` contains the repeated folate check with its full evidence and reasons. `tmp/audit_integrity_extra.json` contains cache syntax and source checksum checks; `tmp/audit_retrieval_metrics.json` contains the six-way metric recomputation.

This audit added this report and temporary audit dependencies/helpers/evidence. It did not edit application source, model prompts, default settings, datasets, the index, or frozen experiment predictions.
