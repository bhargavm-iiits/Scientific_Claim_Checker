# Architecture

This document explains how the Scientific Claim Checker fits together: the components, how data flows
between them, the LangGraph workflow and its transitions, the supervisor's policies, the LLM response
cache and how heavy resources are loaded and reused. Everything described here is implemented in
`src/claim_checker/`; file references are given so each statement can be checked against the code.

## 1. System diagram

```text
                         +---------------------------------------------+
   CLI (cli.py)  ------->|                                             |
   Streamlit (ui/) ----->|   ClaimChecker  (runtime.py)                |
   Notebook  ----------->|   one shared, lazily initialised pipeline   |
                         +----------------------+----------------------+
                                                |
                       LangGraph StateGraph (pipeline/graph.py)
                                                |
   +--------------+   +-----------+   +-----------------+   +-------+   +-----------+   +-----------+   +---------+
   |validate_input|-->| retrieve  |-->| select_evidence |-->| judge |-->| aggregate |-->| supervise |-->| explain |--> END
   +------+-------+   +-----+-----+   +--------+--------+   +---+---+   +-----+-----+   +-----+-----+   +----+----+
          |                 |                  |               ^  |            |               |              |
          |                 |                  |               |  |            |  repair once  |              |
          |                 |                  |               +--+------------+---------------+              |
          |                 |                  |                  |            |               |              |
          +-----------------+------------------+------------------+------------+---------------+--------------+
                                   any operational failure  --->  handle_error ---> END (status=error, verdict=None)

  Resources used by the nodes
  ---------------------------
  retrieve         -> Retriever (retrieval/index.py): BM25Index (bm25.py) + DenseIndex (dense.py, BGE windows)
                      -> rrf (hybrid.py) -> CrossEncoderScorer + rerank_documents (rerank.py)
  select_evidence  -> select_evidence (agents/evidence.py)  --+
  judge            -> judge_document  (agents/judge.py)     --+--> OllamaClient (models/ollama_client.py)
                                                                    JSON-schema format, Pydantic validation,
                                                                    1 repair attempt, on-disk response cache
                                                                    -> local Ollama (127.0.0.1:11434), qwen3:4b
  supervise        -> supervisor.py (citation checks, downgrade)
  aggregate        -> aggregate.py (deterministic rule)
  explain          -> agents/explainer.py (deterministic text, no model call)

  Prepared files
  --------------
  data/processed/corpus.jsonl ........ CorpusDocument records (5,183)
  artifacts/index/ ................... manifest.json, documents.jsonl, windows.jsonl, embeddings.npy
  artifacts/cache/llm/ ............... cached structured responses keyed by request hash
  prompts/*.jinja2 ................... versioned prompts (hashes recorded in results and run manifests)
```

The brief's system design (`claim -> Retriever -> Evidence -> Judge -> Explainer`) maps onto the
graph one-to-one, with two additions: an explicit **aggregate** step (several abstracts are judged
separately and must be combined) and a **supervisor** that checks the output and controls the bounded
repair loop.

The "agents" share one local model. They are separate responsibilities, prompts and graph nodes, not
separately trained models.

## 2. Components

| Component | File(s) | Responsibility |
|---|---|---|
| Configuration | `config.py`, `configs/default.yaml`, `configs/experiments.yaml` | Typed settings (Pydantic) merged over YAML; path resolution against the project root; loopback-only model host check (`local_host_problem`); variant and suite definitions; configuration fingerprints. |
| Schemas | `schemas.py` | Shared records (`CorpusDocument`, `ClaimPair`, `ClaimGold`, `RetrievalHit`, `EvidenceSelection`, `DocumentJudgement`, `ClaimResult`, …) and the structured LLM output models. See `DATA_CONTRACTS.md`. |
| Errors | `errors.py` | Typed operational errors with a stage and a setup hint (`ModelUnavailableError`, `StaleIndexError`, `ContextOverflowError`, …). Only `InputValidationError` is non-operational. |
| Data | `data/sources.py`, `data/prepare.py`, `data/splits.py`, `data/store.py` | Download with recorded fallback route, validation, ID normalisation, pair and claim-label construction, frozen splits, atomic JSON/JSONL storage. |
| Retrieval | `retrieval/bm25.py`, `dense.py`, `hybrid.py`, `rerank.py`, `index.py` | Scientific tokenizer + BM25; tokenizer-aware sentence windows + BGE embeddings + exact inner product with max-window document scores; RRF; windowed cross-encoder re-ranking; index build, fingerprint and stale-index detection. |
| Model client | `models/ollama_client.py` | Local Ollama transport, model availability and digest (`model_name_matches`, shared with `check_readiness`), JSON-schema structured output, Pydantic validation, semantic validators, at most one repair (never discarding a schema-valid answer), context-size guard, response cache. |
| Agents | `agents/evidence.py`, `judge.py`, `explainer.py`, `prompting.py` | Evidence selection (IDs only), per-document judgement (evidence or full-abstract mode), claim-only and single-prompt baselines, deterministic explanation, Jinja2 prompt rendering and hashing. |
| Pipeline | `pipeline/state.py`, `graph.py`, `supervisor.py`, `aggregate.py` | Typed graph state, LangGraph nodes and routing, supervisor policies, aggregation rule. |
| Runtime | `runtime.py` | `ClaimChecker` (shared entry point; lazy resources; variant dispatch; result assembly; optional live `progress` callback; `validate_claim` / `count_claim_tokens` that need only the tokenizer) and `check_readiness` (cheap, loads no model). |
| Evaluation | `evaluation/runner.py`, `metrics.py`, `analysis.py`, `compare.py`, `latency.py` | Resumable variant runs with identity checks and diagnostic-file reconciliation on resume, six-way retrieval evaluation, metric definitions with denominators, reports and failure selection. |
| CLI | `cli.py`, `__main__.py` | `doctor`, `prepare-data`, `make-dev-split`, `build-index`, `check`, `retrieval-eval`, `evaluate`, `report`, `compare`, `latency`. |
| UI | `ui/app.py`, `ui/views.py` | Streamlit session handling and rendering; calls `ClaimChecker.check` — no second implementation of retrieval or judgement. |

## 3. Data flow for one claim

1. **Input.** `ClaimChecker.check(claim, k=3, retrieval_method="hybrid", rerank=True,
   evidence_mode="evidence", progress=None)` builds `PipelineOptions` and invokes the compiled graph
   with `recursion_limit = 25`. If a `progress(stage, events)` callable is passed, the graph is
   streamed (`stream_mode=["updates", "values"]`) instead and the callable is called after each node
   finishes, with that node's stage events; the returned `ClaimResult` is the same as with `invoke`.
   Exceptions raised by the callback are swallowed and counted in
   `metadata["progress_callback_errors"]`, so a display problem cannot fail a check. The Streamlit
   stage board uses this.
2. **validate_input.** Whitespace is collapsed; blank claims, claims over 2,000 characters and claims
   over 256 BGE tokens are rejected (`InputValidationError`). The order is: character checks, then the
   token budget using only the embedding model's tokenizer (`tokenizer.json`, about 0.2 s; no
   embedding model, index, re-ranker or Ollama), then the expensive readiness checks — so an
   over-long claim is rejected before heavy resources load. (If the tokenizer itself cannot be loaded
   because setup is incomplete, readiness runs first so the user gets the setup hint.) The same checks
   are public as `ClaimChecker.validate_claim(claim)` and `ClaimChecker.count_claim_tokens(text)`.
   `k` must be between 1 and `keep_top`
   (10). Readiness is enforced: prepared corpus present, index present and fresh (fingerprint), model
   downloaded in Ollama. On the first call of a process this step also loads the corpus, BM25, the
   BGE encoder and checks the model, so its timing includes that warm-up (the cross-encoder then
   loads during the first re-rank). The saved latency benchmark measures the whole warm-up separately:
   19.95 s for `ClaimChecker.warmup()` plus 4.44 s for the first claim
   (`artifacts/latency/cold_full_k3.json`).
3. **retrieve.**
   - BM25 over `title + newline + abstract` with a deterministic scientific tokenizer
     (`sci-tok-v1`: NFKC, casefold, keeps numbers and hyphenated terms plus their parts, removes a
     short stopword list) → top 30.
   - Dense: the query is prefixed with the BGE instruction
     `"Represent this sentence for searching relevant passages: "`, encoded and normalised; exact
     inner product against 5,670 normalised window embeddings (384-d); each document's score is its
     best window → top 30.
   - Hybrid: RRF, `score(d) = Σ 1/(60 + rank)` over the two lists; ties broken by document ID.
   - Re-rank (default on): the first 20 hybrid candidates are scored by the cross-encoder; abstracts
     longer than the 512-token pair budget are split into overlapping sentence windows and the
     maximum window score is used. The first 10 are kept for metrics and display; the first `k` go on.
   - Each hit is a `RetrievalHit` carrying method label (e.g. `hybrid+rerank`), score, stage scores
     and the index fingerprint. In oracle mode the reference abstracts are supplied instead (method
     `oracle`); no retrieval is run.
4. **select_evidence** (skipped when `evidence_mode="full_abstract"`). For each of the `k` abstracts
   the evidence prompt lists sentences as `[id] text` with zero-based IDs. Abstracts with more than 40
   sentences or 12,000 characters are split into bounded windows, one model call per window; results
   are merged. The model returns `{"sentence_ids": [...], "note": "..."}`. IDs outside the supplied
   window fail validation (triggering the repair attempt) and are dropped if they persist. An empty
   list is valid.
5. **judge.** For each abstract the judge sees only the selected sentences (evidence mode) or every
   sentence (full-abstract ablation), with the same template and schema. Output:
   `{"verdict", "cited_sentence_ids", "reason", "confidence"}`. Cited IDs must be among the
   sentences supplied to the judge. If no sentence was selected, the abstract is judged NEI without a
   model call (`skipped_model=true`).
6. **aggregate.** Fixed rule (`pipeline/aggregate.py`):

   | Document decisions | Final |
   |---|---|
   | ≥ 1 SUPPORT, no CONTRADICT | SUPPORT |
   | ≥ 1 CONTRADICT, no SUPPORT | CONTRADICT |
   | all NEI (or none) | NEI |
   | both SUPPORT and CONTRADICT | NEI, `conflict = true` |
   | a stage failed operationally | never reaches aggregation: `status = error`, verdict unset |

7. **supervise.** Validates every cited ID (in range, and inside the judge's context) and applies the
   decisive-without-evidence policy (section 5).
8. **explain.** `agents/explainer.py` renders a plain-English explanation from the validated decisions:
   verdict wording, how many of the `k` abstracts decided it, each deciding abstract's title, ID,
   one-based sentence numbers and short reason; an explicit conflict sentence when `conflict=true`;
   "insufficient collection, not false" wording for NEI; the maximum model-reported confidence
   labelled uncalibrated. No model call. `explain(..., context="retrieved" | "supplied")` names the
   source correctly: in oracle mode the abstracts are "the supplied reference abstracts" (and an NEI
   says the supplied abstracts are insufficient), not "retrieved abstracts".
9. **Result assembly.** `ClaimChecker._result_from_state` builds the `ClaimResult`: quotations are
   created from `corpus[doc_id].sentences[sentence_id]` for every cited ID, each tagged with that
   document's verdict; documents keep rank, retrieval score, selected IDs and the full judgement;
   metadata records package version, options, model name and digest, model settings, index
   fingerprint, prompt hashes, LLM call/cache/repair counts, the full top-10 retrieval list and the
   evidence notes; timings are per stage (plus `retrieve.bm25`, `retrieve.dense`, `retrieve.fuse`,
   `retrieve.rerank`) and `total` in seconds.

Baselines that are not graphs: `check_judge_only` (claim only, `judge_only.jinja2`) and
`check_single_prompt` (top-k abstracts in one prompt that judges and cites `(doc_id, sentence_id)`
pairs, `retrieve_judge.jinja2`). Both use the same validation, readiness checks, error handling and
`ClaimResult` record. The single-prompt baseline applies the same grounding rule: a decisive verdict
with no valid citation after the repair attempt is downgraded to NEI. Because it produces one
claim-level verdict (no per-document verdicts), it has its own explainer, `explain_single_prompt`,
which says that one model call read all k abstracts together, lists the cited `(document, sentence)`
pairs and the model's reason, and states a downgrade when one happened; it does not invent
per-document decisions.

## 4. LangGraph nodes and transitions

Defined in `pipeline/graph.py` (`build_graph`); state type `GraphState` in `pipeline/state.py`.

| Node | Reads | Writes | Next on success | On failure |
|---|---|---|---|---|
| `validate_input` | claim, options | claim (normalised), status=running | `retrieve` | `handle_error` |
| `retrieve` | claim, options | hits, doc_ids (top k), retrieval sub-timings | `select_evidence` | `handle_error` |
| `select_evidence` | doc_ids | selections (or `{}` when evidence agent off) | `judge` | `handle_error` |
| `judge` | doc_ids or pending_repair, selections | judgements, pending_repair=[] | `aggregate` | `handle_error` |
| `aggregate` | judgements | verdict, conflict | `supervise` | `handle_error` |
| `supervise` | judgements, citation_repairs | pending_repair / downgraded judgements, verdict, conflict | `judge` if repair pending, else `explain` | `handle_error` |
| `explain` | verdict, conflict, judgements | explanation, status=ok | `END` | `handle_error` |
| `handle_error` | error | status=error, verdict=None, conflict=False, explanation | `END` | — |

Every node is wrapped by `_node()`, which:

- catches `ClaimCheckerError` (typed, with stage and hint) and any unexpected exception
  (`unexpected_error`) and turns them into `status="error"` plus an `ErrorInfo`;
- appends a `StageEvent` (`ok` with seconds, or `error` with the error type);
- adds the node's wall time to `timings` (the `timings` channel sums values, so a judge repair
  accumulates).

Conditional edges route to `handle_error` whenever `status == "error"`. The graph therefore has
exactly two terminal outcomes: `status="ok"` with a verdict, or `status="error"` with `verdict=None`
and a hint. `ClaimChecker.check` additionally catches anything raised by `graph.invoke` itself (for
example a recursion-limit error) and returns `error_type="graph_error"`; it never raises for
operational problems.

**Bounded execution.** The only cycle is `supervise -> judge -> aggregate -> supervise`, and it can
run at most `judge.max_citation_repairs` (1) times because `citation_repairs` is incremented before
routing back. LangGraph's `recursion_limit` (25) is a second, independent bound.

## 5. Supervisor policies

Implemented in `pipeline/supervisor.py` and the `supervise` node.

1. **Input and readiness first.** Blank, oversized or over-token-budget claims are rejected before any
   model or index work. Missing data, a missing or stale index, or an unreachable/missing model raise
   typed operational errors with a setup hint (for example "Run `claim-checker build-index`").
2. **Citation validity.** Every cited sentence ID must exist in the document and must have been
   supplied to the judge (`context_sentence_ids`). The agents already drop invalid IDs after their
   repair attempt, so a violation here indicates a bug and ends the run with an error rather than
   producing an unverifiable quotation.
3. **Bounded retry for decisive verdicts without evidence.** If a document judgement is SUPPORT or
   CONTRADICT but cites no valid sentence, the supervisor records a `citation_repair` warning event
   and routes only those documents back to the judge with explicit feedback ("Your previous verdict
   was X but it cited no valid sentence id. Either cite … or answer NEI."). This happens at most once
   (`max_citation_repairs: 1`).
4. **Decisive-without-evidence downgrade.** If after the repair the judgement still cites nothing, it is
   replaced by NEI with `downgraded=true`, its reason is prefixed with
   `[downgraded from X: no valid cited sentence]`, a `downgraded_no_citation` event is recorded, the
   claim verdict is re-aggregated and the explanation states that a downgrade happened. Consequence:
   **every successful decisive verdict carries at least one valid, quotable source sentence.**
5. **Errors are not NEI.** Model, index, data, timeout, context-overflow and invalid-output failures
   end in `status="error"` with `verdict=None`. Evaluation counts them as wrong and reports them in
   coverage and failure rate.

Schema-level repair is handled one layer lower, in `OllamaClient.structured`: when the output is not
valid JSON for the schema, or a semantic validator reports problems (for example sentence IDs outside
the numbered list), the client appends the invalid answer and a correction message and asks once
more (`max_schema_repair_attempts: 1`, so at most one repair). A repair can only improve the
outcome:

- the most recent **schema-valid** answer is the result (a schema-valid repair replaces the first
  answer, because it was produced with feedback about the first answer's problems);
- a **schema-invalid repair never discards a schema-valid first answer**: that answer is returned with
  its semantic `problems` and `repair_fallback=True` (also stored in the cache entry), the agent
  records a warning stage event with `data.flag = "repair_fallback"`, and the normal semantic handling
  then applies — invalid IDs are filtered and the supervisor's re-ask / NEI downgrade still runs;
- only when no attempt was schema-valid is `ModelOutputError` raised (an operational error, never NEI).

## 6. Model calls

- Endpoint: `http://127.0.0.1:11434`. `ModelConfig` parses `model.host` and accepts it only if the
  scheme is `http`/`https`, there are no credentials in the URL and the hostname is exactly
  `localhost`, `127.0.0.1` or `::1` (written `[::1]`); look-alikes such as
  `http://localhost.example.com` or `http://127.0.0.1.evil.net` are rejected, so the client cannot be
  pointed at a hosted model by configuration.
- Model matching: one function, `model_name_matches(installed, configured)`, is used by both
  `check_readiness` and `OllamaClient`, so they always agree. An untagged name means `:latest` on
  either side (`qwen3` = `qwen3:latest`); any other tag must match exactly (`qwen3:4b` never matches
  `qwen3:8b`).
- Request: `ollama.Client.chat(model="qwen3:4b", messages=[system, user], format=<JSON schema>,
  options={temperature: 0, seed: 42, num_ctx: 8192, num_predict: 768}, think=False,
  keep_alive="30m", stream=False)`. `$ref`s in the Pydantic schema are inlined before submission.
- Response handling: `<think>` blocks and Markdown fences are stripped, then Pydantic validates.
- **Context guard:** before each request the prompt size is estimated conservatively
  (characters / 3 + schema size). If it exceeds `num_ctx − num_predict` (7,424 tokens), a
  `ContextOverflowError` is raised instead of letting Ollama silently truncate the start of the
  prompt. The evidence agent avoids this with bounded windows; the full-abstract judge and the
  single-prompt baseline can hit it in principle (the longest SciFact abstract, doc 10749308, is
  10,127 characters as stored, 9,943 across its 58 sentences, so a single abstract fits).
- Concurrency: one generation at a time (`max_concurrent_generations: 1`).

## 7. LLM response cache

`OllamaClient` keeps a deterministic on-disk cache in `artifacts/cache/llm/<2 hex>/<sha256>.json`.

- **Key:** SHA-256 of the canonical JSON of model name, **model digest**, the exact messages, the
  submitted schema, the generation options, the `think` flag and the repair budget. Any change to a
  prompt template, the claim, the selected sentences, the model build or the options produces a new
  key, so stale answers cannot be reused.
- **Value:** the final raw JSON answer, number of attempts, any remaining semantic problems and, when
  the repair was schema-invalid and the first answer was kept, `repair_fallback: true`.
  Corrupt or incompatible entries are ignored and regenerated.
- **Why:** with temperature 0 and a fixed seed, repeating the same request is wasted GPU time. The
  cache lets an interrupted evaluation resume cheaply, lets variants that share identical calls (for
  example the evidence call for the same claim–abstract pair in `full`, `full_k5` and `oracle`) reuse
  them, and makes repeated demo claims instant.
- **Consequences for reporting:** results record `llm_calls` and `llm_cache_hits` per claim and each
  `DocumentJudgement`/`EvidenceSelection` has a `cached` flag. Latency of a variant that ran after
  another variant can be dominated by cache hits, so latency comparisons must state whether the cache
  was cold (see `EVALUATION.md`). Computing the cache key requires the model digest, so Ollama must be
  reachable even when every answer is cached.
- Disable with `model.cache_enabled: false`; clear by deleting `artifacts/cache/llm/`.

## 8. Resource lifecycle

- `ClaimChecker()` loads nothing heavy. It builds the prompt library and compiles the graph.
- On first use (lazy properties): corpus (`data/processed/corpus.jsonl`, about 16 MB) → BGE encoder
  (from the local Hugging Face cache first, `local_files_only=True`, falling back to a download only
  if not cached) → `Retriever.load` (checks the index manifest against the current corpus hash and
  retrieval settings, loads windows and embeddings, rebuilds BM25 in memory) → cross-encoder on first
  re-rank → Ollama client and model digest.
- `ClaimChecker.warmup()` performs all of this up front (the UI's "Load models" button); measured at
  19.95 s on this laptop in the saved cold benchmark (`artifacts/latency/cold_full_k3.json`; 30.1 s in
  a second process, `cached_full_k3.json`), with the Ollama model already resident in GPU memory.
- Resources are reused for every later call in the same process. The Streamlit app holds one
  `ClaimChecker` per server process via `st.cache_resource`; the CLI creates one per command.
- A small in-process LRU cache (512 entries) stores retrieval outputs per `(claim, method, rerank,
  keep)`, so variants sharing retrieval do not repeat it. A cache hit is recorded honestly:
  `timings["retrieve"]` is the real near-zero lookup, no `retrieve.*` sub-timings are written,
  `metadata["retrieval_cache_hit"]` is `true` (`false` when computed) and the earlier computation's
  sub-step seconds are kept separately in `metadata["retrieve_original_compute_s"]`; the retrieve
  stage also emits an info event with `flag: retrieval_cache_hit`. Predictions copy both keys, and
  `metrics.json` / `report.md` report the per-variant retrieval-cache hit rate next to the LLM one.
- `ClaimChecker.check` holds a re-entrant lock, so concurrent UI sessions are serialised; the model
  client also limits concurrent generations to one.
- Embedding model and re-ranker run on CPU to leave the 6 GB GPU to Ollama; Ollama keeps the model
  loaded for 30 minutes after the last call (`keep_alive`).
- `check_readiness()` is cheap: it checks files, the index manifest and Ollama's `/api/version` and
  `/api/tags` over HTTP without loading any model. The UI calls it at most once a minute per session.

## 9. Index identity and staleness

`artifacts/index/manifest.json` stores an identity record: corpus SHA-256, BM25 tokenizer version, BM25
text definition, embedding model, query instruction, window size (512 tokens), window overlap (1
sentence), window format and normalisation. Its hash prefix is the index fingerprint
(`3fa9e53a7857605c` for the current build: 5,183 documents, 5,670 windows, 455 documents with more
than one window, 0 truncated single-sentence windows, 384-d embeddings). Any mismatch with the current
configuration or corpus raises `StaleIndexError` with the hint to rebuild. The fingerprint is copied
into every `RetrievalHit`, every result's metadata and every run manifest.

## 10. Interfaces

- **CLI** (`python -m claim_checker …`): see `RUNBOOK.md`.
- **Python / notebook:** `from claim_checker.runtime import ClaimChecker; ClaimChecker().check("…")`;
  `check(..., progress=lambda stage, events: ...)` reports each finished stage live;
  `check_variant(claim, "full_k5")` runs any named variant (forwarding `progress` for pipeline
  variants); `validate_claim(claim)` and `count_claim_tokens(text)` check input without loading the
  index or contacting the model.
- **Streamlit:** tabs "Check a claim", "How it works", "Evaluation results". The form prevents typing
  from triggering inference, locks while a check runs, shows sentence numbers starting at 1, labels
  confidence as uncalibrated, separates fresh results from earlier session results and saved
  evaluation numbers, and offers the `ClaimResult` JSON for download.

## 11. v2 judge (added after `final_v1`)

The v1 pipeline above is unchanged and reproducible (`--variant full`, suite `final`). v2 changes only
the **judge step** and the **supervisor**; retrieval, the evidence agent (same v1 prompt, so its cached
answers are shared), aggregation and the explainer are the same. Selection is per call:
`PipelineOptions.prompt_set` (`v1` | `v2`) and `PipelineOptions.judge_think`, set from the variant
(`configs/experiments.yaml`: `full_v2`, `full_v2_nothink`, `oracle_v2`) or, for
`ClaimChecker.check()` without explicit arguments, from the `product:` section of
`configs/default.yaml`. v2 settings live in the separate `judge_v2:` section, so the v1 configuration
fingerprint and the v1 prompt-hash record are byte-identical to `final_v1`.

**Prompt and schema.** `prompts/v2/judge.jinja2` (new file; the four v1 templates are untouched) with
`JudgeOutputV2` (`schemas.py`). The JSON property order is the generation order (Ollama's grammar
follows the schema's property order; `inline_refs` and the client keep insertion order, checked by a
test and by a live call), and it is

`claim_assertion → evidence_finding → same_subject → evidence_direction → reason → verdict → cited_sentence_ids → confidence`

so the label is decoded **after** the model has written what the claim asserts, what the sentences
report, a subject match (`yes | partly | no`) and a direction (`same_as_claim | opposite_to_claim |
no_effect_found | not_stated`). v1 decoded `verdict` first, which is the likely cause of the 14
`final_v1` failures whose reason argued for a different label. The prompt states decision rules
(SUPPORT only for a direct statement of the claim's relation, population and direction; CONTRADICT
for the opposite direction or no effect; NEI for merely related text, a different population /
intervention / outcome, or insufficient text) and number-reading rules (HR / OR / RR below 1 =
lower risk or slower progression; a CI containing 1 = no significant difference). Citations are still
sentence IDs from the supplied context, validated as in v1; quotations are still built by code.

**Thinking (judge step only).** With `judge_think=true` the judge request uses qwen3 thinking
(`think=True`); the evidence agent stays `think=False`. Ollama returns thinking separately
(`message.thinking`); only `message.content` is parsed, and only the **length** of the thinking text
is kept (`DocumentJudgement.thinking`, `metadata.judge_thinking`, the cache entry) — never the text.
Unbounded qwen3:4b thinking ran 1,000–3,000+ tokens per judge call on development claims (30–60 s,
sometimes exhausting a 3,072-token `num_predict` without any answer), and brevity instructions did
not bound it, so thinking is **budget-forced**: the first request runs with
`num_predict = judge_v2.think_budget_tokens` (512); if it stops for length, the thinking so far is
closed with a fixed sentence and `</think>` and the answer is decoded in a raw continuation
(`/api/generate`, `raw=True`, qwen3 ChatML template built by `qwen3_raw_prompt`) under the same JSON
schema and options. If raw continuation is not available (another model family, or a transport
without it), the client answers once without thinking (`think_fallback`). Both are extra to the
schema-repair budget and are recorded (`budget_forced`, `think_fallback`, warning events).

**Cache key.** For requests without per-request overrides (all v1 calls) the cache identity is
exactly the v1 one, so v1 cached answers keep hitting. A v2 request differs anyway (different
messages and schema); in addition the effective `think` flag is always in the key and the thinking
budget is added when thinking is on, so think / no-think / different-budget answers never collide.

**Consistency check (supervisor, deterministic).** `pipeline/consistency.py` compares each v2
decisive verdict with the judge's own checklist and reason: SUPPORT is inconsistent with a direction
other than `same_as_claim`, with `same_subject` other than `yes`, or with a reason that literally
argues against support ("contradicts", "does not support", "not enough evidence", ...); CONTRADICT is
inconsistent with `same_as_claim` / `not_stated`, a different subject, or a reason saying the
evidence "supports the claim". NEI is never repaired (an NEI whose checklist reads as decisive is
only reported, `nei_checklist_decisive`). Inconsistent documents are re-asked with feedback naming
the problem. **v2 change to the repair rule:** the missing-citation re-ask and the consistency re-ask
share the single repair round (`judge.max_citation_repairs = 1`), so a document is still re-asked at
most once; whatever is still uncited or inconsistent afterwards is downgraded to NEI
(`downgraded_no_citation` / `downgraded_inconsistent`, reason prefixed, `consistency_flags`
recorded). Run files record the v2 diagnostics per document (`checklist`, `thinking`,
`consistency_flags`) in `pair_predictions.jsonl`.

**Which v2 is the product.** Tuning on the `dev_tune` split (120 training claims; log in
`artifacts/dev_tuning/iterations.md`) showed thinking to be slower and *less* accurate than the same
v2 prompt without thinking, so the frozen product configuration is the v2 judge **without thinking**
(`product: {prompt_set: v2, judge_think: false}`; variant `full_v2_nothink`). Thinking remains
available (`full_v2`, `judge_think=True`, budget-forced as above) and is reported as an ablation.
`ClaimChecker.check()` without explicit `prompt_set` / `judge_think` uses the product setting;
`check_variant(..., "full")` and `--variant full` always run v1.

**Additive result fields (v1 and v2).** `metadata["confidence"]` is the claim-level model-reported
confidence (uncalibrated): the highest confidence among examined documents whose verdict equals the
claim verdict and that went to the model; `None` for a conflict, an error, or an NEI reached without
a model judgement (judge-only and single-prompt baselines report their single answer's confidence,
`None` when the single-prompt answer was downgraded). `DocumentReport.sentences` carries each
examined abstract's stored sentence list (zero-based positions) so saved results render without the
corpus. For v2 judgements `DocumentJudgement.reason_display` (also used in the explanation) rewrites
the model's zero-based references ("Sentence [0]", "[3]", "sentence 2") to the one-based numbers the
UI and quotes use; only IDs that were supplied to the judge are rewritten, and `reason` keeps the
model's text. `ClaimChecker.is_warm` / `resources_loaded()` report whether heavy resources are
loaded (loading nothing). Error explanations end with exactly one full stop.

**Run identity.** A run containing v2 variants adds `prompt_hashes_v2` and `config_fingerprint_v2`
(hash of `judge_v2`) to its identity; v1-only runs keep the exact v1 identity, and v1 variant
definitions are recorded without the v2 fields (`variant_dump`).
