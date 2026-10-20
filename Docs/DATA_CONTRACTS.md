# Data contracts

This document defines the records every component relies on, where each field comes from, how
sentence positions and document IDs are represented, and how the claims data is obtained when the
brief's loader no longer works. All records are Pydantic models in `src/claim_checker/schemas.py`
with `extra="forbid"` (unknown fields are rejected).

## 1. Sources and the loading route

| Source | How it is loaded | Used for |
|---|---|---|
| `mteb/scifact`, config `corpus`, split `corpus` (revision `cf10ab6856b15b0e670ef8ae5dae4e266c12d035`) | `load_dataset("mteb/scifact", "corpus")["corpus"]` — works | Retrieval text (`_id`, `title`, `text`) for all 5,183 abstracts |
| `allenai/scifact_entailment` | `load_dataset("allenai/scifact_entailment")` — **fails** with datasets 5.1.0 | (intended) claim–abstract pairs |
| Official SciFact release archive `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` | Downloaded once, members extracted, files read with `load_dataset("json", data_files=...)` | `corpus.jsonl` (original sentence lists), `claims_train.jsonl` (809 claims), `claims_dev.jsonl` (300 claims) |

### Compatibility fallback (`route = fallback_official_archive_json`)

The brief's claims loader is a script-based dataset. Current `datasets` releases no longer run dataset
scripts. The exact failure recorded in `data/raw/source_manifest.json`:

```text
builtins.RuntimeError: Dataset scripts are no longer supported, but found scifact_entailment.py
```

What `data/sources.py` does instead (remote code is never enabled; `trust_remote_code` is not used):

1. Records the exception verbatim (`entailment_loader.success = false`).
2. Downloads the builder script `scifact_entailment.py` **as text only** (repo revision
   `621da8dd0621d91b276ab5a98e2cd28bf3f90d38`, sha256 `1dc86899…c22a52b`, `executed: false`) and
   reads its `_URL` constant with a regular expression (`archive_url_source: builder_script`).
   The configured URL is used only if the script cannot be read.
3. Downloads the archive (sha256 `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`,
   3,115,079 bytes) to `data/raw/scifact/data.tar.gz`, reusing a cached copy unless
   `--force-download` is given.
4. Extracts only the three expected regular members (no `extractall`, no path traversal) and records
   each file's sha256 and line count (corpus 5,183; train 809; dev 300).
5. Reads each file with `load_dataset("json", ...)` and, independently, with a plain JSON parser,
   and requires the two to agree row by row after normalisation.
6. Reproduces the builder's `_generate_examples` transformation in `data/prepare.py` (section 4).
7. Writes `data/raw/source_manifest.json`: environment versions, MTEB revision, columns, features, row
   count and content hash, the loader attempt and its exception, builder-script details, archive URL
   and hash, extracted file hashes, the route name and a plain-English route description.

If the official loader works in a future `datasets` version, its rows are compared with the
reconstruction (`compare_with_hf_pairs`) and any difference stops preparation.

## 2. Corpus alignment

`build_corpus` (`data/prepare.py`) joins the MTEB rows to the original corpus rows by normalised ID
and fails loudly on any inconsistency:

| Check | Result for the current data |
|---|---|
| Duplicate IDs in either source | none |
| IDs only in MTEB / only in original | 0 / 0 (exact match, 5,183 documents) |
| Expected corpus size | 5,183 ✔ |
| Title mismatches (after NFKC, casefold, whitespace collapse) | 0 (any mismatch would abort) |
| MTEB `text` vs. original sentences joined with spaces | 5,183 equal after whitespace normalisation, 0 different |
| Empty abstracts | 0 |
| Sentences per document | min 3, median 8, max 367; 45,952 in total |

Decision: **MTEB supplies the retrieval text; the original archive supplies the sentence list.** Since
the two agree exactly, quotations taken from the sentence list are the same text that was searched.

## 3. Record schemas

### `CorpusDocument` (`data/processed/corpus.jsonl`)

| Field | Type | Meaning |
|---|---|---|
| `doc_id` | str | Normalised document ID (section 6) |
| `title` | str | Title (MTEB, falling back to original) |
| `text` | str | Retrieval text: `title + "\n" + abstract` (used by BM25) |
| `sentences` | list[str] | Original SciFact sentence list, stripped; position = zero-based sentence ID |
| `source` | dict | `mteb_id` (as given upstream), `structured` (original flag for structured abstracts), `text_matches_sentences` |

### `ClaimPair` (`data/processed/claim_pairs_{train,validation}.jsonl`)

One row per claim × cited document — the shape of `allenai/scifact_entailment`.

| Field | Type | Meaning |
|---|---|---|
| `claim_id` | str | Normalised claim ID |
| `claim` | str | Claim text |
| `doc_id` | str | Cited document |
| `title` | str | Title of that document |
| `verdict` | `SUPPORT`/`CONTRADICT`/`NEI` | Label of this paper's relation to the claim |
| `evidence` | list[int] | Flattened zero-based evidence sentence positions (empty for NEI) |
| `split` | str | `train` or `validation` |

Counts: train 919 pairs (SUPPORT 370, CONTRADICT 194, NEI 355); validation 340 pairs (SUPPORT 138,
CONTRADICT 71, NEI 131).

### `ClaimGold` (`data/processed/claim_gold_{train,validation}.jsonl`)

One row per claim — used for whole-pipeline evaluation.

| Field | Type | Meaning |
|---|---|---|
| `claim_id`, `claim`, `split` | str | Identity |
| `cited_doc_ids` | list[str] | All documents cited by the annotators for this claim |
| `reference_doc_ids` | list[str] | Cited documents present in the corpus (all of them, for SciFact) — the target for retrieval Recall@k and the oracle documents |
| `evidence_by_doc` | dict[str, list[int]] | Sorted unique evidence sentence IDs per evidence-bearing document |
| `label_by_doc` | dict[str, verdict] | Label of each evidence-bearing document |
| `label` | verdict or null | Claim-level reference label (section 5); null only for an annotation conflict |
| `conflict` | bool | Annotators gave both SUPPORT and CONTRADICT documents |
| `evidence_doc_ids` (property) | list[str] | Documents with annotated evidence — the target for evidence-bearing recall |

Counts: train 809 claims (SUPPORT 332, CONTRADICT 173, NEI 304); validation 300 claims (SUPPORT 124,
CONTRADICT 64, NEI 112); 0 annotation conflicts in either split; 1,025 and 366 reference evidence
sentences respectively; every cited document exists in the corpus; every evidence position is in range.

### Pipeline records

| Record | Key fields | Notes |
|---|---|---|
| `RetrievalHit` | `doc_id`, `rank` (**one-based**), `method` (`bm25`, `dense`, `hybrid`, `…+rerank`, `oracle`), `score`, `index_fingerprint`, `title`, `stage_scores` | Re-ranker scores are relevance logits, not probabilities of truth. `stage_scores` keeps the first-stage score and the number of re-rank windows. |
| `EvidenceSelection` | `doc_id`, `sentence_ids` (zero-based, unique, sorted), `note`, `windowed`, `dropped_invalid_ids`, `cached` | The model selects IDs only. |
| `DocumentJudgement` | `doc_id`, `verdict`, `cited_sentence_ids`, `reason`, `confidence` (model-reported, uncalibrated, may be null), `mode` (`evidence`/`full_abstract`/`single_prompt`), `context_sentence_ids`, `skipped_model`, `repaired`, `downgraded`, `dropped_invalid_ids`, `cached` | Cited IDs ⊆ context IDs, guaranteed. |
| `EvidenceQuote` | `doc_id`, `title`, `sentence_id` (zero-based), `quote`, `doc_verdict` | `quote` is always `corpus[doc_id].sentences[sentence_id]`. |
| `StageEvent` | `stage`, `status` (`ok`/`warning`/`error`/`info`), `message`, `timestamp`, `data` | Stage outcomes, repairs, downgrades, windowing. A schema-invalid repair that kept the first schema-valid answer is a `warning` event with `data.flag = "repair_fallback"` (plus `doc_id` and the remaining `problems`); there is no separate record field for it. |
| `DocumentReport` | `doc_id`, `title`, `rank`, `retrieval_score`, `retrieval_method`, `selected_sentence_ids`, `judgement` | One per examined abstract. |
| `ErrorInfo` | `error_type`, `message`, `stage`, `hint` | Typed operational errors, e.g. `model_unavailable`, `stale_index`, `context_overflow`, `model_invalid_output`, `invalid_input`. |
| `ClaimResult` | `status` (`ok`/`error`), `claim`, `claim_id`, `verdict` (null when `status="error"`), `conflict`, `evidence`, `documents`, `explanation`, `timings` (seconds per stage and `total`), `metadata`, `events`, `error` | The shared result for CLI, UI download and notebook. `explanation` is deterministic text: pipeline results name the source as "the retrieved abstracts" or, in oracle mode, "the supplied reference abstracts"; the single-prompt baseline has its own claim-level wording (`explain_single_prompt`). `metadata.progress_callback_errors` appears only if a live `progress` callback raised. |

### Structured model outputs (submitted to Ollama as JSON schemas)

| Model | Fields | Used by |
|---|---|---|
| `EvidenceOutput` | `sentence_ids: list[int]`, `note: str` | Evidence agent |
| `JudgeOutput` | `verdict`, `cited_sentence_ids: list[int]`, `reason`, `confidence ∈ [0,1]` | Judge (evidence and full-abstract modes) |
| `JudgeOnlyOutput` | `verdict`, `reason`, `confidence` | Claim-only baseline |
| `RetrieveJudgeOutput` | `verdict`, `citations: [{doc_id, sentence_id}]`, `reason`, `confidence` | Single-prompt baseline |
| `DoctorProbe` | `answer`, `sentence_ids`, `reason` | `doctor` structured-output check |

Reference labels and annotated evidence positions are never placed in any prompt. The oracle variant
receives the reference **document IDs** only. An oracle request with no ids, or with an id that is
not in the corpus, is an operational error (`invalid_input` at `validate_input`, respectively
`data_integrity` at `retrieve`), never an NEI over zero or a subset of documents.

## 4. Pair reconstruction (reproducing the builder)

For every claim and every `cited_doc_id` (in order):

- if the document appears in the claim's `evidence` map: `verdict` = the label of its evidence sets
  (all sets of one document must agree, otherwise preparation stops) and `evidence` = the flattened
  sentence indices of all its sets;
- otherwise `verdict = NEI` and `evidence = []`.

Arrow's wide-struct representation of `evidence` from `load_dataset("json")` (absent keys become
`None`) and the raw JSON dict are normalised to the same `{doc_id: [{sentences, label}]}` form before
use. Evidence labels other than SUPPORT/CONTRADICT abort preparation.

## 5. Claim-level reference labels

A pair label describes one paper; it is not automatically a statement about the whole corpus (one
abstract can be NEI while another supports the claim). Claim labels are therefore built from the
evidence-bearing documents only:

| Annotated document labels for the claim | Claim label |
|---|---|
| SUPPORT only | SUPPORT |
| CONTRADICT only | CONTRADICT |
| none (no evidence-bearing document) | NEI |
| both SUPPORT and CONTRADICT | `label = null`, `conflict = true`; excluded from the final split and listed |

Unannotated documents are **not proven irrelevant**: SciFact annotators only examined cited documents,
so a retrieved, unannotated abstract may contain real evidence. This matters for evidence precision
(see `EVALUATION.md`).

## 6. Identifier normalisation

`normalize_doc_id` is applied to every document and claim ID at every boundary (MTEB `_id`, original
`doc_id`, `cited_doc_ids`, evidence keys, model citations, pydantic validators):

| Input | Output |
|---|---|
| `13734012` (int) | `"13734012"` |
| `"13734012"` | `"13734012"` |
| `" 013734012 "` | `"13734012"` (stripped, leading zeros removed for all-digit IDs) |
| `13734012.0` (float) | `"13734012"`; non-integer floats are rejected |
| `None`, `""`, booleans | rejected with an error |
| non-numeric text | stripped text |

Claims and documents are sorted numerically by ID when order matters (splits, corpus file).

## 7. Sentence indexing

- **Zero-based everywhere internally**: `CorpusDocument.sentences[i]`, `ClaimPair.evidence`,
  `ClaimGold.evidence_by_doc`, prompts (`[0] …`, `[1] …`), model outputs, `EvidenceSelection`,
  `DocumentJudgement`, `EvidenceQuote.sentence_id`, the downloaded JSON and all metrics.
- **One-based only when rendering for people**: the explanation text ("sentence 5"), the CLI's
  `s5` marker and the Streamlit evidence panels add 1 at display time.
- Retrieval ranks (`RetrievalHit.rank`, `DocumentReport.rank`) are one-based by definition.
- Dense and re-ranking windows record the zero-based `sentence_ids` they cover, so evidence near the
  end of a long abstract stays reachable.

## 8. Frozen split files (`data/splits/`)

| File | Contents |
|---|---|
| `final.json` | 100 claim IDs from the 300 eligible validation claims, seed 42, stratified by claim label (SUPPORT 41, CONTRADICT 21, NEI 38), allocation, excluded conflicts (none), `ids_sha256`, validation gold-file fingerprint. Refuses to be overwritten with different content unless `prepare-data --force-splits`. |
| `development.json` | 808 training claims (all 809 minus train claim 871, whose normalised text duplicates final claim 870); also lists train claim 1291, which overlaps a validation claim **not** in the final subset and is kept. |
| `smoke.json` | 20 development claims, seed 42, stratified (SUPPORT 8, CONTRADICT 4, NEI 8). The split writer now records `excluded_annotation_conflicts` as the sorted list of excluded claim IDs, like `final.json`; the existing `smoke.json` (created 2026-10-20T14:12:30Z) was left unchanged to preserve `smoke_v1` provenance and still stores the older count form (`0`). There are no annotation conflicts, so both forms mean "none excluded". |

Claim text is normalised for overlap checks with NFKC, casefolding, punctuation removed and whitespace
collapsed.

## 9. Index files (`artifacts/index/`)

| File | Contents |
|---|---|
| `manifest.json` | Identity (corpus sha256, tokenizer version, embedding model, query instruction, window settings), fingerprint `3fa9e53a7857605c`, counts, embedding dimension 384, embeddings sha256, BGE model revision, build timings, library versions |
| `documents.jsonl` | Document order with `doc_id`, `title`, `text` (BM25 input) |
| `windows.jsonl` | `window_id`, `doc_id`, `window_index`, `sentence_ids`, `text` (`title\nsentences`), `tokens`, `truncated` |
| `embeddings.npy` | float32 matrix (5,670 × 384), L2-normalised, row-aligned with `windows.jsonl` |

## 10. Run files (`artifacts/runs/<run_id>/`)

`manifest.json` (identity, variants, configuration, environment, claim IDs, gold label counts),
`predictions.jsonl` (one record per variant × claim, written last so it defines completion),
`pair_predictions.jsonl`, `retrieval.jsonl`, `events.jsonl`, and after analysis `metrics.json`,
`results.csv`, `confusion_matrices.json`, `failure_cases.csv`, `report.md`. `failure_analysis.md` is
written by a person after reading the failure cases; it is not generated. On resume, diagnostic rows
(`retrieval.jsonl`, `pair_predictions.jsonl`, `events.jsonl`) without a matching prediction, and stale
duplicate blocks, are removed before the run continues (a file is rewritten atomically only if
something is removed). In `metrics.json`, each evidence block (`evidence_selected`,
`evidence_cited`) carries its denominators explicitly: `precision_denominator`,
`recall_denominator`, `claims_in_recall_denominator`, `errored_claims_included`,
`errored_annotated_pairs_included` and `errored_reference_sentences_included` (see
`EVALUATION.md` §4). Each prediction record also carries `retrieval_cache_hit` (`true` = served
from the in-process retrieval cache, `false` = computed, `null` = no retrieval; absent in runs saved
before the flag existed = unknown) and `retrieve_original_compute_s` (on a hit: the cached
computation's sub-step seconds; never copied into `timings`). Each variant in `metrics.json` has a
`cache` block: `llm` (`model_calls`, `cache_hits`, `cache_hit_rate` = hits / (calls + hits)) and
`retrieval` (`claims_with_retrieval`, `cache_hits`, `cache_misses`, `unknown`, `cache_hit_rate` =
hits / (hits + misses), `unknown_with_copied_subtimings`). A prediction record's `downgraded` is true
when a decisive answer was downgraded to NEI for lack of a valid citation — a per-document judgement
(pipeline) or the claim-level single-prompt answer; single-prompt records also carry `model_verdict`
(the model's answer before any downgrade; `null` for other strategies and absent in runs saved
before the field existed, such as `final_v1`). `metrics.json` reports `downgraded_claims` and
`downgraded_claim_ids` per variant, counting a claim as downgraded if its record says so **or** its
saved events contain a `downgraded_no_citation` flag (this recovers the 4 single-prompt downgrades
of `final_v1`, whose records predate the field). Run ids are validated: letters, digits, `.`, `_`
and `-` only, no `..`, no path separators or absolute paths, so a run folder is always directly
inside `paths.runs_dir`.
