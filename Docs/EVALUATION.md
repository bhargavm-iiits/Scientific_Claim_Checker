# Evaluation

How the Scientific Claim Checker is evaluated: which claims, which variants, how every metric is
computed and what its denominator is, how runs are made reproducible and resumable, and the measured
results. Metric code: `src/claim_checker/evaluation/metrics.py`; runs: `evaluation/runner.py`;
reports: `evaluation/analysis.py`; variants: `configs/experiments.yaml`.

No accuracy threshold is assumed. Correct evaluation, evidence integrity, reproducibility and honest
reporting matter more than the absolute score of a 4-billion-parameter quantised model.

## 1. Frozen populations

| Split | Source | Size | Class counts (claims) | Selection |
|---|---|---|---|---|
| `final` | SciFact validation (`claims_dev.jsonl`) | **100** of 300 eligible | SUPPORT 41, CONTRADICT 21, NEI 38 | Seed 42, stratified by claim-level reference label, largest-remainder allocation; candidates sorted by ID before a seeded shuffle |
| `development` | SciFact train | 808 of 809 | SUPPORT 332, CONTRADICT 173, NEI 303 | All training claims minus overlaps with the final subset |
| `smoke` | `development` | 20 | SUPPORT 8, CONTRADICT 4, NEI 8 | Seed 42, stratified |

Rules:

1. **Eligibility.** Validation claims with an annotation conflict (both SUPPORT and CONTRADICT
   evidence) would be excluded and listed; there are none, so all 300 are eligible.
2. **Freezing.** `data/splits/final.json` stores the IDs, their sha256, seed, allocation and the
   fingerprint of the validation gold file. `prepare-data` refuses to overwrite it with different
   content; `--force-splits` exists only for a deliberately new, separately reported experiment.
3. **Leakage check.** Claim text is normalised (NFKC, casefold, punctuation removed, whitespace
   collapsed) and compared across splits. Train claim **871** has the same text as final claim
   **870** and is removed from development. Train claim **1291** matches a validation claim that is
   *not* in the final subset; it is kept and recorded.
4. **Development use.** Prompt and configuration work used only training claims (smoke and
   development). The final subset is not used to tune anything. If prompts change after final
   failures are inspected, the new results must be reported as a new run, not as untouched
   evaluation.
5. **Every variant runs on the same 100 claims**; all claim–abstract pairs of those claims are kept
   for pair-level diagnostics (`pair_predictions.jsonl`).

## 2. Experiment matrix

Defined in `configs/experiments.yaml`. Defaults: `strategy=pipeline`, `k=3`,
`retrieval_method=hybrid`, `rerank=true`, `evidence_mode=evidence`, `oracle=false`.

| Variant | Kind | What changes | Question it answers |
|---|---|---|---|
| `judge_only` | Baseline | Claim only, no documents (`judge_only.jinja2`) | What does the model "know" without retrieval? |
| `retrieve_judge_single_prompt` | Baseline | Top-3 re-ranked abstracts in **one** prompt that judges and cites `(doc_id, sentence_id)` | Does splitting evidence selection, per-document judgement and aggregation help vs. one prompt? |
| `oracle` | Upper reference | Reference (cited) abstracts supplied instead of retrieval; evidence agent and judge as normal | Judgement quality when retrieval is perfect. Only document IDs are supplied — never gold verdicts or evidence positions. |
| `full` | Main system | Hybrid + re-rank, k = 3, evidence agent on | The product configuration |
| `evidence_off` | Ablation | Judge reads every sentence of each abstract (`full_abstract`) | Effect of the evidence agent |
| `rerank_off` | Ablation | Hybrid RRF order without the cross-encoder | Effect of re-ranking end-to-end |
| `full_k1` | Ablation | Only the top-1 abstract | Smaller context |
| `full_k5` | Ablation | Top-5 abstracts | Larger context |

Suites: `smoke` = `full`, `oracle` on the 20 smoke claims; `final` = all eight variants on the 100
final claims. This covers the brief's minimum standard (a baseline, the full system and an ablation on
one fixed subset in one table, plus ten or more failure analyses) and the full list of baselines and
ablations it names.

## 3. Aggregation policy (document → claim)

| Document decisions among the k examined abstracts | Final verdict |
|---|---|
| ≥ 1 SUPPORT, no CONTRADICT | SUPPORT |
| ≥ 1 CONTRADICT, no SUPPORT | CONTRADICT |
| all NEI | NEI |
| both SUPPORT and CONTRADICT | **NEI with `conflict=true`** |
| a stage failed operationally | `status=error`, no verdict |

Mapping conflicting predicted evidence to NEI is a documented prototype convention (SciFact's own
claim labels never mix SUPPORT and CONTRADICT, and there are no annotation conflicts in this data).
The explanation says explicitly that papers disagreed. Conflict frequency is reported.

A decisive document verdict that cites no valid sentence after one supervisor repair is downgraded to
NEI; the number of claims with a downgraded judgement is reported.

## 4. Metric definitions and denominators

Every reported number states its unit: **claims**, **claim–abstract pairs**, **documents**,
**evidence sentences** or **quotations**.

### Verdict metrics (unit: claims; `verdict_metrics`)

- Population: **all requested claims** of the variant (100 for the final suite).
- An operational error is predicted label `ERROR`, which is always wrong. Errors are never dropped.
- `accuracy = correct / n_claims`.
- `coverage = claims with a verdict / n_claims`; `accuracy_on_covered = correct / covered` is
  reported separately and never replaces accuracy.
- Per-class precision, recall and F1 for SUPPORT, CONTRADICT and NEI (scikit-learn,
  `zero_division=0`). `support` = gold claims of that class. An `ERROR` prediction counts as a miss
  for its gold class.
- `macro_f1` = unweighted mean of the per-class F1 over the classes present in the gold labels (all
  three in the final split).
- Confusion matrix: rows = gold label, columns = SUPPORT, CONTRADICT, NEI, ERROR.

### Retrieval metrics (unit: claims; `retrieval_metrics`)

- `Recall@k` = mean over claims with ≥ 1 reference document of
  |reference documents ∩ top-k retrieved| / |reference documents|. Reference documents = all cited
  documents of the claim (every final claim has at least one, so n = 100).
- `evidence-bearing Recall@k` = the same, but the targets are only documents with annotated evidence,
  and only claims that have such documents are counted (**n = 62** in the final split — the 38 NEI
  claims have none, and 62 = 41 SUPPORT + 21 CONTRADICT).
- In system runs the ranked list is the 10 documents kept by the pipeline (top 20 re-ranked, first 10
  kept). `evidence_doc_recall_examined` = share of evidence-bearing documents among the k abstracts
  actually examined.
- The oracle and judge-only variants have no retrieval metrics.

### Evidence sentence metrics (unit: evidence sentences, micro-averaged; `evidence_metrics`)

Computed twice: on sentences **selected** by the evidence agent and on sentences **cited** by the
judge.

- `precision = true-positive sentences / predicted sentences inside documents that carry reference
  evidence for that claim`.
- `recall = true-positive sentences / all reference evidence sentences of the evaluated claims`.
  This is end-to-end: a retrieval miss lowers recall.
- Predicted sentences in **unannotated** documents are counted and reported separately and are
  **excluded from precision**, because SciFact only annotated cited documents and an unannotated
  abstract is not proven irrelevant.
- Population: claims whose record has the field (variants with an evidence agent for "selected";
  pipeline and single-prompt variants for "cited"), **plus every claim that ended in an operational
  error**: such a claim predicts nothing, so it adds its reference sentences to the recall
  denominator with zero true positives (operational errors stay in the evaluation population; they
  cannot raise recall by being dropped). Precision is unaffected because errored claims predict no
  sentences. A successful record without the field (the stage does not apply, e.g. judge-only) is not
  evaluated.
- Every evidence block in `metrics.json` states its denominators: `precision_denominator`
  (= predicted sentences in annotated documents), `recall_denominator` (= reference sentences),
  `claims_in_recall_denominator`, and how much of that comes from operational errors
  (`errored_claims_included`, `errored_annotated_pairs_included`,
  `errored_reference_sentences_included`). In `final_v1` no claim failed, so all three are 0 and, for
  example, `full` selected-evidence recall is 0.653 = 79 / 121 sentences over 100 claims.

### Other metrics

| Metric | Definition | Unit |
|---|---|---|
| Quote validity | quotations whose text equals `corpus[doc_id].sentences[sentence_id]` / all quotations | quotations |
| Operational failure rate | claims with `status=error` / requested claims, plus counts by `error_type` | claims |
| Conflict rate | claims with `conflict=true` / requested claims (also / successful claims) | claims |
| Downgraded claims | claims with ≥ 1 document judgement downgraded to NEI (pipeline variants) or whose claim-level single-prompt answer was downgraded to NEI; for records saved before this was propagated, recovered from the `downgraded_no_citation` events in `events.jsonl` (ids listed in `metrics.json` → `downgraded_claim_ids`) | claims |
| Latency | per-claim `total` seconds: n, mean, median, p95; the same per stage (`validate_input`, `retrieve`, `retrieve.bm25`, `retrieve.dense`, `retrieve.fuse`, `retrieve.rerank`, `select_evidence`, `judge`, `aggregate`, `supervise`, `explain`) | seconds per claim |

Latency caveats:

- **LLM cache.** Identical model requests are served from `artifacts/cache/llm/`. Variants run later in
  a suite reuse calls made by earlier variants (for example the same evidence call in `full`,
  `full_k5`, `rerank_off` and `oracle`), so their latency can be far lower than a cold run. Each
  prediction stores `llm_calls` and `llm_cache_hits`; only cold-cache numbers describe real speed.
- **Retrieval cache.** Retrieval outputs are also cached in-process per `(claim, method, rerank, keep)`,
  so a later variant with the same retrieval settings skips the ~1.2 s search. Such a prediction has
  `retrieval_cache_hit: true`, its `retrieve` time is the real near-zero lookup (its `total` excludes
  the search), it has no `retrieve.*` sub-timings, and the original sub-step seconds are stored in
  `retrieve_original_compute_s`. `metrics.json` (`cache`) and `report.md` ("Cache usage") give
  per-variant LLM and retrieval cache-hit rates. Runs saved before this flag existed (e.g. `final_v1`)
  show the status as *unknown*; the report still recognises records whose `retrieve.*` sub-timings
  exceed their own `retrieve` stage time (impossible for a fresh search) as copied cache hits.
- **Warm-up.** The first claim of a process includes loading the corpus, BM25, BGE, the cross-encoder
  and the model check inside `validate_input`; the latency benchmark below measures it separately.
- **Saved cold-latency benchmark** (`python -m claim_checker latency --suite smoke --limit 10
  --no-cache`; raw data `artifacts/latency/cold_full_k3.json`, summary `cold_full_k3.md`). Full
  pipeline (hybrid + re-rank, evidence agent, k = 3) on the first 10 smoke (train) claims in a fresh
  process, LLM disk cache disabled and a new in-process retrieval cache; the artifact verifies 0 LLM
  and 0 retrieval cache hits (52 model calls) and 0 errors. Warm-up is excluded from the statistics:

  | stage | mean s | median s | p95 s |
  |---|---|---|---|
  | retrieve (BM25 + dense + fuse + re-rank) | 1.20 | 1.07 | 1.60 |
  | &nbsp;&nbsp;of which re-rank (CPU cross-encoder) | 1.15 | 1.02 | 1.56 |
  | select_evidence | 3.49 | 3.51 | 3.95 |
  | judge | 3.22 | 3.49 | 5.43 |
  | **total per claim** | **7.92** | **8.42** | **10.07** |

  Warm-up per process: 19.95 s loading resources (`ClaimChecker.warmup()`) + 4.44 s for a first,
  unmeasured claim = 24.4 s. The Ollama model was already resident in GPU memory (`/api/ps`, recorded
  in the artifact), so loading the model weights from disk is not included. The GPU was idle at the
  start (0 % utilisation in `nvidia-smi`); desktop applications (browser, game launcher) were open,
  and a run with other GPU work would be slower. An earlier run of the same command gave 7.97 s mean
  and 17.5 s resource warm-up, and the cached benchmark below needed 30.1 s for the same resource
  warm-up, so expect roughly 20–30 s. With the LLM cache on and every answer already cached
  (`--repeat`, `artifacts/latency/cached_full_k3.json`) a new claim costs 1.25 s (the retrieval still
  runs) and repeating a claim in the same process costs 0.012 s. These numbers replace an earlier
  5-claim development measurement (8.78 s) that was not saved as an artifact.

### Pair-level diagnostics (unit: claim–abstract pairs)

`pair_predictions.jsonl` keeps, for every examined document: whether it is a reference document, the
gold pair label (`unannotated` for non-cited documents), gold evidence, selected and cited sentences,
predicted verdict, reason, confidence, `skipped_model` and `downgraded`. These are diagnostics only and
are never mixed into claim-level scores.

### Failure-stage heuristic (`failure_stage`)

For each wrong prediction, the earliest responsible stage:

1. `operational` — status error;
2. judge-only baseline — `judgement error (no retrieval)`;
3. gold SUPPORT/CONTRADICT: `retrieval miss` if no evidence-bearing document was examined; else
   `evidence miss` if the evidence agent selected none of the gold evidence sentences in those
   documents; else `judgement error`;
4. gold NEI: `judgement error (false decisive verdict)` or `judgement error (conflict)`.

`select_failures` picks failures from the `full` variant first, round-robin across classes with
CONTRADICT given a double share, and adds other variants until at least ten are available. The
heuristic is a starting point; `failure_analysis.md` records the human reading of each case against
the source abstracts.

## 5. Run identity, checkpointing and resume

`python -m claim_checker evaluate --suite final --run-id final_v1` writes to
`artifacts/runs/final_v1/`.

- **Checkpointing:** after each claim, `retrieval.jsonl`, `pair_predictions.jsonl` and `events.jsonl`
  are appended and flushed, then `predictions.jsonl` is appended last. A `(variant, claim_id)` is
  complete only when its prediction line exists.
- **Resume:** re-running the same command skips completed `(variant, claim_id)` pairs. Before
  continuing it reconciles the diagnostic files: rows in `retrieval.jsonl`, `pair_predictions.jsonl`
  and `events.jsonl` whose `(variant, claim_id)` has no prediction (a claim interrupted between the
  diagnostic appends and the prediction append) are removed, as are stale duplicate blocks, and the
  number removed is logged; a file is rewritten atomically only if something is removed, so resuming
  a complete run changes nothing. The run is
  resumed only if the stored identity matches exactly:
  suite, split, sha256 of the population IDs and its size, sha256 of the processed corpus and of the
  gold file, split fingerprint, index fingerprint (when a variant needs retrieval), model name **and
  digest**, prompt-template hashes, and a fingerprint of the `model`, `retrieval`, `evidence`, `judge`
  and `runtime` configuration sections. Each variant's definition must also be unchanged. Otherwise
  `RunMismatchError` is raised: use a new `--run-id`.
- **Model loss:** if a claim fails with `model_unavailable` the run stops (completed claims are kept)
  instead of filling the population with errors; rerun the same command after fixing Ollama. Other
  operational errors are recorded as error predictions and stay in the denominators.
- **Analysis:** `report --run-id <id>` (also run automatically at the end of `evaluate`) uses the
  latest prediction per `(variant, claim_id)` and writes `metrics.json`, `results.csv`,
  `confusion_matrices.json`, `failure_cases.csv` and `report.md`. Orphaned or duplicate diagnostic rows
  left by an interruption are removed on the next resume (above); in any case claim-level metrics only
  read the latest prediction per `(variant, claim_id)`.

## 6. Six-way retrieval comparison (measured)

Command: `python -m claim_checker retrieval-eval --split final`. Output:
`artifacts/retrieval/retrieval_eval_final.json` / `.csv` and per-claim rankings in
`retrieval_per_claim_final.jsonl`. Index fingerprint `3fa9e53a7857605c`. Final split: **100 claims**
for reference-document recall; **62 claims** for evidence-bearing recall. Latency = mean seconds per
claim on CPU for search (+ re-ranking of the top 20 when on).

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Evid. R@1 | Evid. R@3 | Evid. R@5 | Evid. R@10 | Mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.506 | 0.680 | 0.733 | 0.789 | 0.687 | 0.833 | 0.914 | 0.934 | 0.024 | 0.038 |
| BM25 + re-rank | 0.542 | 0.666 | 0.693 | 0.769 | 0.788 | 0.878 | 0.898 | 0.934 | 1.065 | 1.432 |
| Dense (BGE) | 0.559 | **0.718** | **0.789** | 0.865 | 0.743 | 0.902 | 0.923 | 0.944 | 0.026 | 0.032 |
| Dense + re-rank | **0.562** | 0.668 | 0.736 | 0.867 | 0.796 | 0.890 | 0.918 | 0.938 | 1.107 | 1.641 |
| Hybrid (RRF) | 0.516 | 0.685 | 0.752 | **0.868** | 0.703 | 0.874 | 0.903 | **0.964** | 0.050 | 0.067 |
| Hybrid + re-rank (product) | 0.557 | 0.676 | 0.738 | 0.824 | **0.812** | **0.910** | **0.930** | 0.942 | 1.118 | 1.573 |

Reading the table:

- Dense retrieval has the best reference-document Recall@3 (0.718) and Recall@5; hybrid RRF has the
  best Recall@10 (0.868) and evidence-bearing Recall@10 (0.964).
- Re-ranking **does not improve Recall@3** for any method (it slightly lowers it), but it raises
  Recall@1 and evidence-bearing Recall@1 for all three methods, and evidence-bearing Recall@3 for
  BM25 (0.833 → 0.878) and hybrid (0.874 → 0.910; hybrid also gains at @5). For hybrid — the product
  setting — the documents most useful for a verdict move into the top 3 that the pipeline reads.
  For dense retrieval, re-ranking lowers evidence-bearing Recall@3 and @5 slightly.
- Re-ranking costs about 1 s per claim on CPU (20 candidates, windowed), roughly 20–45× the search
  time; first-stage search is 0.02–0.05 s.
- Reference-document recall includes cited NEI documents that annotators judged irrelevant to the
  verdict; evidence-bearing recall is the more relevant number for the downstream judge.
- The product keeps hybrid + re-rank because the pipeline reads only the top 3 and evidence-bearing
  documents matter most; the `rerank_off` variant measures whether that choice helps end to end.
- The re-ranker was trained on MS MARCO web search, not biomedical claims (see `LIMITATIONS.md`),
  which is a plausible reason it does not lift reference-document Recall@3.

## 7. Smoke run (development check, not a final result)

Run `smoke_v1`: 20 smoke claims from the **training** split (SUPPORT 8, CONTRADICT 4, NEI 8), model
digest `359d7dd4…74fae7`, index `3fa9e53a7857605c`. Shown only to document that the pipeline runs end
to end; 20 claims are far too few for conclusions.

| Variant | Coverage | Accuracy | Macro-F1 | F1 SUPPORT | F1 CONTRADICT | F1 NEI | Quote validity | Conflicts | Failures |
|---|---|---|---|---|---|---|---|---|---|
| full | 1.000 | 0.600 (12/20) | 0.432 | 0.762 | 0.000 | 0.533 | 1.000 | 1 | 0 |
| oracle | 1.000 | 0.700 (14/20) | 0.517 | 0.800 | 0.000 | 0.750 | 1.000 | 0 | 0 |

Already visible: no CONTRADICT claim was predicted correctly (the model tends to answer SUPPORT or
NEI), and several gold-NEI claims received a false SUPPORT. Smoke latency is not comparable across the
two variants because `oracle` ran after `full` and reused its cached model calls.

## 8. Final results (100 frozen validation claims)

Run `final_v1` (`artifacts/runs/final_v1/`): suite `final`, 100 claims per variant (gold SUPPORT 41,
CONTRADICT 21, NEI 38), model `qwen3:4b` digest `359d7dd4…74fae7`, index `3fa9e53a7857605c`, prompt
hashes evidence `25376c48b37ac36e`, judge `ae8f024d7631b2f1`, judge_only `dc2ea4990866fadd`,
retrieve_judge `2ea3577c957ee917`, created 2026-10-20T14:45:42Z. Prompts were frozen before the run
and not changed afterwards. All numbers below are read from `metrics.json` / `results.csv` (and
recomputed from `predictions.jsonl` for the confidence intervals). After later code fixes,
`report --run-id final_v1` was re-run from the saved predictions; every number was identical, and
`metrics.json` / `report.md` gained explicit evidence denominators (§4).

**Saved explanation text predates two explainer fixes.** The `explanation` strings stored in
`final_v1/predictions.jsonl` were generated before (a) the single-prompt baseline got its own
claim-level explainer (`explain_single_prompt`) and (b) oracle explanations were changed to say
"the supplied reference abstracts". So 81 of the 100 saved `retrieve_judge_single_prompt`
explanations still contain the old placeholder "document ALL: …", and all 100 saved `oracle`
explanations still say "retrieved abstracts" (65 decisive) or "the available collection is
insufficient" (35 NEI). This
affects only that display text: verdicts, citations, quotations and every metric are computed from
the structured fields and are unaffected. New checks and new runs produce the corrected wording.

### 8.1 Main comparison (unit: claims, n = 100 per variant)

| Variant | Coverage | Accuracy (correct/100) | Macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | Evid. docs examined (n = 62) | Ev. P sel. | Ev. R sel. | Ev. P cited | Ev. R cited | Quote validity | Conflict rate | Failure rate | Mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| judge_only | 1.000 | 0.500 (50) | **0.504** | 0.545 | **0.529** | 0.436 | – | – | – | – | – | – | – | – (no quotes) | 0.00 | 0.00 | 1.396 | 1.902 |
| retrieve_judge_single_prompt | 1.000 | 0.560 (56) | 0.450 | 0.650 | 0.174 | 0.526 | 0.676 | 0.824 | 0.910 | – | – | 0.793 | 0.537 | 1.000 | 0.00 | 0.00 | 4.754 | 10.320 |
| oracle | 1.000 | **0.670 (67)** | **0.571** | **0.706** | 0.240 | **0.767** | – | – | – | 0.614 | **0.711** | 0.811 | **0.603** | 1.000 | 0.00 | 0.00 | 2.598 | 4.790 |
| **full** | 1.000 | 0.560 (56) | 0.487 | 0.638 | 0.296 | 0.526 | 0.676 | 0.824 | 0.910 | 0.632 | 0.653 | 0.825 | 0.545 | 1.000 | 0.03 | 0.00 | 4.831 | 9.416 |
| evidence_off | 1.000 | 0.530 (53) | 0.446 | 0.639 | 0.320 | 0.377 | 0.676 | 0.824 | 0.910 | – | – | 0.740 | 0.587 | 1.000 | 0.03 | 0.00 | 5.239 | 8.105 |
| rerank_off | 1.000 | 0.540 (54) | 0.468 | 0.626 | 0.296 | 0.483 | 0.685 | 0.868 | 0.874 | 0.633 | 0.628 | 0.815 | 0.545 | 1.000 | 0.04 | 0.00 | 1.992 | 4.426 |
| full_k1 | 1.000 | 0.560 (56) | 0.473 | 0.660 | 0.214 | 0.545 | 0.676 | 0.824 | 0.812 | **0.660** | 0.579 | **0.855** | 0.488 | 1.000 | 0.00 | 0.00 | 0.007 ⚠ | 0.019 ⚠ |
| full_k5 | 1.000 | 0.570 (57) | 0.493 | 0.655 | 0.296 | 0.526 | 0.676 | 0.824 | 0.930 | 0.623 | 0.669 | 0.800 | 0.562 | 1.000 | 0.04 | 0.00 | 2.395 | 5.114 |

Units and notes: Recall@3 / Recall@10 = reference-document recall over the 100 claims, computed on the
pipeline's kept top-10 list (so `full_k1` and `full_k5` show the same list as `full`; what changes
with k is the number of evidence-bearing documents *examined*, column "Evid. docs examined", over the
62 claims with evidence). Evidence P/R are micro-averaged over evidence sentences (denominators in
§8.4). Quote validity is over quotations. **Latency is not comparable across variants** (cache, see
§8.5); the ⚠ values for `full_k1` are a cache artefact (every model request was served from the
cache), not a speed result.

**Approximate uncertainty.** Percentile bootstrap over claims (numpy `default_rng(42)`, 1000
resamples of the 100 claims, the same resamples for every variant; macro-F1 over the classes present
in each resample). This treats the 100 claims as a sample and ignores model non-determinism, so the
intervals are approximate.

| Variant | Accuracy [95 % CI] | Macro-F1 [95 % CI] |
|---|---|---|
| judge_only | 0.500 [0.410, 0.590] | 0.504 [0.395, 0.599] |
| retrieve_judge_single_prompt | 0.560 [0.470, 0.660] | 0.450 [0.350, 0.556] |
| oracle | 0.670 [0.570, 0.760] | 0.571 [0.477, 0.670] |
| full | 0.560 [0.460, 0.660] | 0.487 [0.373, 0.597] |
| evidence_off | 0.530 [0.430, 0.630] | 0.446 [0.332, 0.553] |
| rerank_off | 0.540 [0.440, 0.640] | 0.468 [0.357, 0.575] |
| full_k1 | 0.560 [0.470, 0.660] | 0.473 [0.373, 0.579] |
| full_k5 | 0.570 [0.470, 0.670] | 0.493 [0.380, 0.603] |

Paired comparisons on the same claims (difference A − B with paired-bootstrap 95 % CI; "discordant" =
claims A got right and B wrong / B right and A wrong, with an exact two-sided McNemar p-value):

| A − B | Δ accuracy [95 % CI] | Δ macro-F1 [95 % CI] | Discordant (A✓B✗ / B✓A✗) | McNemar p |
|---|---|---|---|---|
| oracle − full | +0.110 [+0.010, +0.190] | +0.084 [−0.031, +0.194] | 17 / 6 | 0.035 |
| full − judge_only | +0.060 [−0.060, +0.180] | −0.017 [−0.146, +0.116] | 23 / 17 | 0.43 |
| full − retrieve_judge_single_prompt | +0.000 [−0.090, +0.090] | +0.037 [−0.085, +0.162] | 10 / 10 | 1.00 |
| full − evidence_off | +0.030 [−0.040, +0.100] | +0.041 [−0.054, +0.137] | 9 / 6 | 0.61 |
| full − rerank_off | +0.020 [−0.020, +0.070] | +0.018 [−0.027, +0.070] | 4 / 2 | 0.69 |
| full − full_k1 | +0.000 [−0.070, +0.070] | +0.013 [−0.087, +0.114] | 6 / 6 | 1.00 |
| full_k5 − full | +0.010 [+0.000, +0.030] | +0.006 [−0.003, +0.022] | 1 / 0 | 1.00 |

Only the oracle–full accuracy gap is clearly larger than the resampling noise; every baseline and
ablation difference involving `full` is within it.

### 8.2 Per-class precision / recall / F1 (unit: claims; gold support S 41 / C 21 / N 38)

| Variant | SUPPORT P / R / F1 | CONTRADICT P / R / F1 | NEI P / R / F1 | Claims predicted C (correct) |
|---|---|---|---|---|
| judge_only | 0.511 / 0.585 / 0.545 | 0.692 / **0.429** / **0.529** | 0.425 / 0.447 / 0.436 | 13 (9) |
| retrieve_judge_single_prompt | 0.494 / 0.951 / 0.650 | 1.000 / 0.095 / 0.174 | 0.789 / 0.395 / 0.526 | 2 (2) |
| oracle | 0.590 / 0.878 / 0.706 | 0.750 / 0.143 / 0.240 | 0.800 / 0.737 / 0.767 | 4 (3) |
| full | 0.493 / 0.902 / 0.638 | 0.667 / 0.190 / 0.296 | 0.789 / 0.395 / 0.526 | 6 (4) |
| evidence_off | 0.481 / 0.951 / 0.639 | 1.000 / 0.190 / 0.320 | 0.667 / 0.263 / 0.377 | 4 (4) |
| rerank_off | 0.486 / 0.878 / 0.626 | 0.667 / 0.190 / 0.296 | 0.700 / 0.368 / 0.483 | 6 (4) |
| full_k1 | 0.538 / 0.854 / 0.660 | 0.429 / 0.143 / 0.214 | 0.643 / 0.474 / 0.545 | 7 (3) |
| full_k5 | 0.507 / 0.927 / 0.655 | 0.667 / 0.190 / 0.296 | 0.789 / 0.395 / 0.526 | 6 (4) |

Every document-reading variant has SUPPORT recall ≥ 0.85 with SUPPORT precision ≈ 0.48–0.59: the
system says SUPPORT far too often. CONTRADICT recall is 0.095–0.190 for all of them, including the
oracle.

### 8.3 Confusion matrices (unit: claims; rows = gold, columns = predicted)

| Gold ↓ / predicted → | judge_only S / C / N / ERR | oracle S / C / N / ERR | full S / C / N / ERR |
|---|---|---|---|
| SUPPORT (41) | 24 / 2 / 15 / 0 | 36 / 1 / 4 / 0 | 37 / 2 / 2 / 0 |
| CONTRADICT (21) | 4 / 9 / 8 / 0 | 15 / 3 / 3 / 0 | 15 / 4 / 2 / 0 |
| NEI (38) | 19 / 2 / 17 / 0 | 10 / 0 / 28 / 0 | 23 / 0 / 15 / 0 |

The other five variants are in `report.md` / `confusion_matrices.json`. The dominant `full` errors are
NEI → SUPPORT (23 of 38 NEI claims) and CONTRADICT → SUPPORT (15 of 21 CONTRADICT claims).

### 8.4 Evidence sentences (unit: evidence sentences, micro-averaged; 121 reference sentences in the evidence-bearing documents of the 62 SUPPORT/CONTRADICT claims)

| Variant | Sentences | Precision = TP / predicted in annotated docs | Recall = TP / reference sentences | Predicted in unannotated docs (excluded from precision) |
|---|---|---|---|---|
| oracle | selected | 0.614 = 86 / 140 | 0.711 = 86 / 121 | 38 |
| oracle | cited | 0.811 = 73 / 90 | 0.603 = 73 / 121 | 19 |
| **full** | selected | 0.632 = 79 / 125 | 0.653 = 79 / 121 | 259 |
| **full** | cited | 0.825 = 66 / 80 | 0.545 = 66 / 121 | 128 |
| retrieve_judge_single_prompt | cited | 0.793 = 65 / 82 | 0.537 = 65 / 121 | 81 |
| evidence_off | cited | 0.740 = 71 / 96 | 0.587 = 71 / 121 | 167 |
| rerank_off | selected | 0.633 = 76 / 120 | 0.628 = 76 / 121 | 264 |
| rerank_off | cited | 0.815 = 66 / 81 | 0.545 = 66 / 121 | 124 |
| full_k1 | selected | 0.660 = 70 / 106 | 0.579 = 70 / 121 | 68 |
| full_k1 | cited | 0.855 = 59 / 69 | 0.488 = 59 / 121 | 42 |
| full_k5 | selected | 0.623 = 81 / 130 | 0.669 = 81 / 121 | 361 |
| full_k5 | cited | 0.800 = 68 / 85 | 0.562 = 68 / 121 | 163 |

(For oracle, "unannotated" means sentences in cited abstracts that carry no evidence annotation, e.g.
the cited abstracts of NEI claims.) Recall is end-to-end (a retrieval miss lowers it). Citing is
stricter than selecting: the judge cites about two thirds of the selected annotated sentences with
higher precision. Of 300 abstracts examined by `full`, 142 got an empty selection (judge skipped, NEI
without a model call); 4 of those were evidence-bearing reference abstracts (claims 142, 208, 793,
1020).

### 8.5 Operational summary, cache and latency (unit: claims; latency in seconds per claim)

Variants ran in this order: judge_only → retrieve_judge_single_prompt → oracle → full → evidence_off →
rerank_off → full_k1 → full_k5 (2026-10-20 14:45–15:25 UTC). "Model calls" are requests actually sent
to Ollama (including schema-repair attempts); "cache hits" were answered from `artifacts/cache/llm/`.

| Variant | OK / requested | Errors | Conflicts | Downgraded | Model calls | Cache hits | Hit share | Cache state | Mean s | Median s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| judge_only | 100 / 100 | 0 | 0 | 0 | 100 | 0 | 0 % | cold | 1.396 | 1.175 | 1.902 |
| retrieve_judge_single_prompt | 100 / 100 | 0 | 0 | **4** ¹ | 106 | 0 | 0 % | cold | 4.754 | 4.042 | 10.320 |
| oracle | 100 / 100 | 0 | 0 | 0 | 195 | 0 | 0 % | cold | 2.598 | 2.749 | 4.790 |
| full | 100 / 100 | 0 | 3 | 0 | 325 | 137 | 30 % | partly warm (reuses oracle calls) | 4.831 | 4.003 | 9.416 |
| evidence_off | 100 / 100 | 0 | 3 | 0 | 300 | 0 | 0 % | cold | 5.239 | 4.929 | 8.105 |
| rerank_off | 100 / 100 | 0 | 4 | 0 | 143 | 317 | 69 % | warm | 1.992 | 1.507 | 4.426 |
| full_k1 | 100 / 100 | 0 | 0 | 0 | 0 | 177 | 100 % | fully cached | 0.007 | 0.006 | 0.019 |
| full_k5 | 100 / 100 | 0 | 4 | 0 | 179 | 528 | 75 % | warm | 2.395 | 2.094 | 5.114 |

- **0 operational errors** in 800 claim runs (no `errors_by_type` entries); **no downgraded
  judgement in any pipeline variant**, but **4 downgraded answers in `retrieve_judge_single_prompt`**
  (claims 350, 437, 793, 852 — a citation-parsing bug, see ¹ below); quote validity **1.000** for
  every variant that quotes (1,307 quotations in total; `full` 208 / 208).
- ¹ **Single-prompt citation-parsing bug (found in the final review, fixed for future runs).** The
  single-prompt template introduces each abstract as `Document <id>: <title>`, and in 4 claims the
  model echoed that header in its citations (`"doc_id": "Document 16927286"`). The citation parser
  did not strip the label, so these *valid* citations (every one names an examined abstract and an
  existing sentence) were rejected, and the decisive answers were downgraded to NEI as uncited
  (events `downgraded_no_citation` in `final_v1/events.jsonl`). The metrics code also only counted
  per-document downgrades, so the original report said 0. Model answers before the downgrade: 350
  SUPPORT (gold NEI), 437 SUPPORT (gold NEI), 793 CONTRADICT (gold CONTRADICT), 852 SUPPORT (gold
  CONTRADICT). **The recorded, frozen result stays accuracy 0.560 / macro-F1 0.450.** Re-scoring the
  model's raw pre-downgrade answers gives accuracy **0.550** / macro-F1 **0.458** (F1 S/C/N 0.634 /
  0.250 / 0.491; 3 instead of 2 CONTRADICT predictions): two downgrades happened to turn wrong
  answers into correct NEIs, one cost a correct CONTRADICT. Both versions are within the bootstrap
  noise of §8.3, so no conclusion changes. The parser now strips a leading `Document`/`Doc` label
  (optionally followed by `:` or `#`) before normalising the id, single-prompt downgrades are written
  to `predictions.jsonl` (`downgraded`, plus the raw `model_verdict`) and counted in `metrics.json`
  (regression tests in `tests/test_pipeline.py` and `tests/test_evaluation.py`). A re-run with the
  fixed parser would be a new experiment (`final_v2`); `final_v1` is not re-run.
- Conflicts (claims): `full` 213 (gold NEI), 783 (gold NEI), 1029 (gold CONTRADICT); the two on gold-NEI
  claims score as correct under the conflict → NEI convention, 1029 is wrong.
- **Latency caveats.** Only the cold variants (judge_only, single prompt, oracle, evidence_off) measure
  real model speed, and they differ in the number of documents read. `full_k1`'s 0.007 s is a pure cache
  artefact. In addition, retrieval results are cached in-process per claim: the hybrid + re-rank
  retrieval (mean 1.19 s, measured in `retrieve_judge_single_prompt`, of which re-ranking 1.07 s) is
  paid once, and `full`, `evidence_off`, `full_k1` and `full_k5` record about 0.01 s for `retrieve` —
  their `total` excludes retrieval although their `retrieve.bm25/dense/rerank` sub-stage entries repeat
  the first computation's timings. `full`'s 4.83 s (evidence 3.20 s + judge 1.61 s) is therefore
  neither cold nor complete; the cold product figure is the saved latency benchmark in §4
  (`artifacts/latency/cold_full_k3.json`: 7.92 s mean per claim on 10 smoke claims). The first claim of
  the process (judge_only, claim 1) includes about 14 s of warm-up in `validate_input`.

### 8.6 Interpretation

1. **Oracle vs full (cost of retrieval): +0.11 accuracy, +0.08 macro-F1.** This is the only difference
   clearly beyond noise (paired CI for accuracy [+0.01, +0.19]; 17 vs 6 discordant claims). It is
   almost entirely on **NEI** claims: 16 of the 17 claims the oracle gets right and `full` wrong are
   gold NEI (NEI recall 0.737 vs 0.395). Missing evidence is a small part — evidence-bearing documents
   are among the examined top 3 for 0.910 of the 62 claims with evidence — the larger cost is
   *distractors*: retrieval brings in topically related, unannotated abstracts and the judge calls
   them SUPPORT (18 of the 23 NEI → SUPPORT errors come only from such abstracts). The oracle also
   reads fewer abstracts (114 for 100 claims vs 300), so it has fewer chances to over-call. On
   SUPPORT/CONTRADICT claims the two are close; the oracle's CONTRADICT F1 is even lower (0.240 vs
   0.296).
2. **The judge-only baseline has a higher macro-F1 (0.504) than the full pipeline (0.487) despite
   lower accuracy (0.50 vs 0.56).** The per-class numbers explain it: without documents the model gets
   9 of 21 CONTRADICT claims right (F1 0.529; P 0.692, R 0.429), because many SciFact CONTRADICT claims
   are inverted versions of well-known findings that its world knowledge rejects. Shown an abstract on
   exactly that topic, the 4B judge tends to answer SUPPORT (gold-CONTRADICT abstracts judged SUPPORT in
   14 of 20 examined pairs in `full`, 15 of 24 in `oracle`), so `full` reaches only 4 / 21 (F1 0.296).
   `full` is much better on SUPPORT (recall 0.902 vs 0.585; F1 0.638 vs 0.545) and NEI (F1 0.526 vs
   0.436), which is where its accuracy advantage comes from. Macro-F1 weights the 21-claim CONTRADICT
   class equally, so the CONTRADICT gap outweighs the rest. Both differences are within noise
   (Δ accuracy +0.06 [−0.06, +0.18]; Δ macro-F1 −0.02 [−0.15, +0.12]), and judge-only answers have no
   quotable evidence, so they cannot be verified.
3. **Single prompt vs pipeline:** identical accuracy (0.56), macro-F1 0.450 vs 0.487. The single prompt
   almost never says CONTRADICT (2 claims predicted, F1 0.174); the separate evidence / judge / aggregate
   stages produce slightly more correct CONTRADICT verdicts (4) and higher cited-evidence precision
   (0.825 vs 0.793). The difference is within noise. (Four single-prompt answers were downgraded by a
   citation-parsing bug, §8.5 ¹; the raw answers score 0.550 / 0.458 — also within noise of `full`.)
4. **Evidence agent helps a little: +0.03 accuracy, +0.04 macro-F1** over `evidence_off` (whole
   abstracts). The gain is on NEI (F1 0.526 vs 0.377; NEI → SUPPORT 23 vs 28): with the full abstract the
   judge finds more to call SUPPORT. `evidence_off` is marginally better on CONTRADICT (F1 0.320 vs
   0.296). Directionally positive, within noise (9 vs 6 discordant claims).
5. **Re-ranker: small help** (+0.02 accuracy, +0.02 macro-F1 vs `rerank_off`; 4 vs 2 discordant). It
   raises evidence-bearing documents among the examined top 3 from 0.874 to 0.910 while reference
   Recall@3 is unchanged (0.676 vs 0.685), consistent with §6.
6. **k = 5 ≈ k = 3** (0.57 vs 0.56; one claim differs). **k = 1** has the same accuracy but a different
   error mix: fewer distractors (NEI F1 0.545, NEI → SUPPORT 18) but fewer evidence-bearing documents
   examined (0.812) and the weakest CONTRADICT F1 of the pipeline variants (0.214).
7. **CONTRADICT is the bottleneck, and it is a judgement problem.** Every document-reading variant
   finds at most 4 of the 21 CONTRADICT claims; the oracle finds 3. In `full`, 14 of the 15
   CONTRADICT → SUPPORT errors examined the gold-CONTRADICT abstract and judged it SUPPORT, often with
   a reason that itself states the opposite direction (see §9 and `failure_analysis.md`).
8. **Operationally the system is solid:** 0 errors, coverage 1.000, quote validity 1.000, no downgraded
   judgement in the pipeline variants (4 downgraded single-prompt answers, caused by the since-fixed
   citation-parsing bug in §8.5 ¹), conflicts in 3–4 % of claims.

With 100 claims (21 CONTRADICT), differences of a few claims are not meaningful; none of the ablation
or baseline comparisons with `full` would survive a significance test.

## 9. Failure analysis

`failure_cases.csv` lists the automatically selected failures with the heuristic stage;
`failure_analysis.md` (human-written) reads at least ten of them against the source abstracts,
separates operational errors from judgement errors, gives special attention to CONTRADICT and
proposes one concrete improvement per failure category.

An example already seen during development (training claim 2, not in the final split): *"1 in 5
million in UK have abnormal PrP positivity."* — gold CONTRADICT. The model answered SUPPORT while its
own reason stated that the abstract's figure (493 per million) is much higher than 1 in 5 million: a
numerical-comparison judgement error where the reason contradicts the verdict.

**Final run (`full`, 44 failures of 100 claims; human reading in
[`artifacts/runs/final_v1/failure_analysis.md`](../artifacts/runs/final_v1/failure_analysis.md), 19
detailed cases including 9 CONTRADICT).**

| Earliest stage (unit: claims) | Heuristic (`failure_cases.csv`) | Human reading | of which gold S / C / N |
|---|---|---|---|
| operational | 0 | 0 | 0 / 0 / 0 |
| retrieval miss | 1 | 1 | 0 / 1 / 0 |
| evidence miss | 5 | 5 | 2 / 3 / 0 |
| judgement error | 38 | 31 | 1 / 13 / 17 |
| annotation-ambiguity | – | 7 | 1 / 0 / 6 |
| **Total failures** | 44 / 100 | 44 / 100 | 4 / 41, 17 / 21, 23 / 38 |

Gold → predicted cells of the 44 failures: NEI → SUPPORT 23 (of 38 NEI claims), CONTRADICT → SUPPORT
15 (of 21), CONTRADICT → NEI 2, SUPPORT → NEI 2, SUPPORT → CONTRADICT 2 (of 41). Recurring patterns:
polarity / direction inversion (12), entity or population substitution on NEI claims (9),
over-inference from related mechanisms (7), annotation ambiguity (7), empty or partial evidence
selection (5), numeric / comparative (2), negation (1), retrieval miss (1). In 14 of the 44 failures
the model's own reason argues for a different label than the verdict it emitted. Proposed fixes
(reason-before-verdict decoding, a verdict–reason consistency check, entity-overlap checks for
decisive citations, an empty-selection re-ask, conclusion-sentence inclusion) are for a new experiment
`final_v2`; the prompts were not changed after these failures were inspected.

## 10. Reproducing

```powershell
$py = "C:\Users\Bhargav\anaconda3\python.exe"
& $py -m claim_checker prepare-data          # frozen splits are unchanged on re-run
& $py -m claim_checker build-index
& $py -m claim_checker retrieval-eval --split final
& $py -m claim_checker evaluate --suite smoke --run-id smoke_v1
& $py -m claim_checker evaluate --suite final --run-id final_v1   # resumable: rerun the same command
& $py -m claim_checker report --run-id final_v1
```

Model generation uses temperature 0 and seed 42 with `think=false`; with the same model digest and
prompts, results are expected to repeat, but GPU numerical non-determinism in Ollama is possible, so
the cache (keyed by digest and exact request) is what guarantees identical answers on re-runs.
