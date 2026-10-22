# Limitations, scope and attribution

This prototype is an educational claim checker. Its answers describe what a small local model concludes
from a fixed collection of research abstracts. They are not medical, scientific or editorial advice.

## 1. Evidence coverage

- **Only 5,183 abstracts.** The knowledge base is the SciFact corpus (biomedical abstracts drawn from
  Semantic Scholar). Claims about anything outside it — most of science, recent work, other domains —
  will usually get NEI, which then means "this collection is insufficient", not "the claim is false".
- **Abstracts only.** Full texts, tables, figures and supplementary material are not available. An
  abstract may summarise a result loosely or omit the conditions under which it holds.
- **No source quality judgement.** The system does not weigh study design, sample size, retractions or
  publication date; one abstract can decide a verdict.
- **Document IDs are dataset identifiers.** No paper URLs or DOIs are generated.

## 2. Benchmark limitations

- **Unannotated documents are not proven irrelevant.** SciFact annotators only labelled the documents
  cited for each claim. A retrieved, uncited abstract can contain genuine evidence; counting it as a
  false positive would understate precision. Evidence precision therefore only counts sentences in
  annotated documents, and selections in unannotated documents are reported separately (see
  `EVALUATION.md`).
- **Small final population.** 100 validation claims (CONTRADICT only 21) give wide uncertainty; a
  difference of a few claims between variants is not meaningful on its own. Only approximate
  percentile-bootstrap intervals over claims are reported (`EVALUATION.md` §8.1; e.g. `full` accuracy
  0.560 [0.46, 0.66]); they ignore model non-determinism, and only the oracle–full gap exceeds them.
- **Saved `final_v1` explanation strings use the earlier explainer wording** (single-prompt
  "document ALL: …", oracle "retrieved abstracts"). The explainer was fixed after the run; verdicts and
  metrics are unaffected (`EVALUATION.md` §8).
- **Claim-level labels are derived.** They follow the rule in `DATA_CONTRACTS.md` (no annotation
  conflicts exist in this data).
- **Smoke and development claims come from the training split**, and the brief's dataset is public, so
  the model may have seen SciFact during pre-training; this cannot be ruled out.

## 3. Model limitations

- **Small, quantised model.** `qwen3:4b` in Ollama's default 4-bit quantisation (Q4_K_M, 4.0 B
  parameters), `think=false`, temperature 0. It follows the JSON schema reliably but makes
  reasoning errors, especially with numbers, negation and direction of effect. Example from
  development: for *"1 in 5 million in UK have abnormal PrP positivity"* (gold CONTRADICT) it
  answered SUPPORT while its own reason said the abstract's 493 per million is much higher than
  1 in 5 million.
- **CONTRADICT is the weak class.** In the 20-claim smoke run neither `full` nor `oracle` predicted any
  CONTRADICT claim correctly. Confirmed on the 100 final claims (`final_v1`, 21 CONTRADICT): `full`
  CONTRADICT F1 **0.296** (P 0.667, R 0.190 = 4 / 21) and `oracle` **0.240** (P 0.750, R 0.143 =
  3 / 21), against 0.529 (9 / 21) for the judge-only baseline. With the right abstract in hand the judge
  still labels 15 of the 21 CONTRADICT claims SUPPORT, often while its own reason states the opposite
  direction — a judgement limitation, not a retrieval one (see
  `artifacts/runs/final_v1/failure_analysis.md`).
- **Model-written reasons can contain slips** (one run wrote "60:00 microg" for "600 microg"). Quotes
  are always exact because they are copied from the corpus; reasons are not. In `final_v1`, 117 of
  974 judge reasons contain a stray ":" inside a word (e.g. "IL-:2"); the raw cached model responses
  contain the same text, so it is model output, not post-processing.
- **Uncalibrated confidence.** `confidence` is the model's self-reported number. It is shown as
  "model-reported (uncalibrated)" and is not used for any decision.
- **Same model for all agents.** Evidence selection and judgement share one model, so their errors are
  correlated.
- **Hidden reasoning disabled.** `think=false` keeps calls fast and outputs short; enabling Qwen3's
  thinking mode might improve hard cases at a large latency cost (not evaluated for v1; evaluated for
  the v2 judge in §10 — it did not help on the development claims).
- **Determinism.** Temperature 0 and seed 42 make outputs repeatable in practice; the on-disk cache
  makes repeated requests identical. GPU numerical differences across machines or Ollama versions can
  still change answers.

## 4. Design conventions with consequences

- **Conflict → NEI.** When one examined abstract supports and another contradicts the claim, the final
  verdict is NEI with `conflict=true` and the explanation says so. This is a prototype convention; a
  user should read both sides. SciFact gold labels never contain such conflicts, so every predicted
  conflict counts as wrong unless the gold label is NEI.
- **Decisive verdicts need citations.** A SUPPORT/CONTRADICT judgement without a valid cited sentence
  is re-asked once and then downgraded to NEI. This guarantees grounded verdicts but can turn a correct
  but uncited answer into NEI.
- **No evidence selected → NEI without a model call.** If the evidence agent selects nothing in an
  abstract, the judge does not see that abstract. Evidence-agent misses therefore become NEI; the
  `evidence_off` ablation measures this effect.
- **Top-k only.** The pipeline reads 3 abstracts by default. Evidence ranked 4th or lower is invisible
  to the judge (k = 5 is available as an option and an ablation).

## 5. Retrieval limitations

- **Re-ranker domain mismatch.** `cross-encoder/ms-marco-MiniLM-L6-v2` was trained on MS MARCO web
  search queries, not biomedical claims. On the final split it does not improve Recall@3 for any first
  stage (hybrid 0.685 → 0.676) although it improves Recall@1 and hybrid evidence-bearing Recall@3
  (0.874 → 0.910). A biomedical re-ranker was not tried.
- **Re-ranking cost.** About 1 s per claim on CPU for 20 candidates.
- **Embedding model size.** BGE-small (384-d) was chosen for CPU speed; larger or biomedical embedding
  models may retrieve better.
- **Relevance is not truth.** Retrieval and re-ranking scores measure topical relevance only.
- **Windowing.** Long abstracts are split into 512-token windows with one sentence of overlap
  (455 documents need more than one window); a document's dense score is its best window, which can
  favour documents with one strongly matching passage.

## 6. Context-length handling

- Model context is 8,192 tokens with 768 reserved for output. Before every request the prompt size is
  estimated conservatively (characters / 3). If it would not fit, the request is refused with
  `context_overflow` (an operational error with a hint) rather than letting Ollama silently drop the
  start of the prompt.
- The evidence agent splits abstracts with more than 40 sentences or 12,000 characters into bounded
  windows, so it never overflows. The full-abstract judge and the single-prompt baseline do not
  window; they could overflow on unusually long inputs (the longest SciFact abstract, doc 10749308, is
  10,127 characters as stored, 9,943 across its 58 sentences, which still fits on its own).
- Claims are limited to 2,000 characters and 256 BGE tokens; the re-ranker budget is 512 tokens per
  claim–window pair. Both claim limits are checked before any heavy resource loads (the token count
  uses only the BGE tokenizer file, about 0.2 s).

## 7. Engineering deviations from the plan

- **Python 3.14 instead of 3.11.** The plan specified Python 3.11 with `uv` and a `uv.lock`. The
  verified environment is the Anaconda base interpreter, Python 3.14.6, with pip; exact direct
  dependency versions are recorded in `requirements.lock.txt` instead of `uv.lock`. `pyproject.toml`
  still declares `requires-python >= 3.11` and CI tests on 3.11, but only 3.14.6 was verified end to
  end locally.
- **Installed into conda base.** This upgraded shared packages and created pip conflicts with Spyder,
  python-lsp-server, spyder-kernels and numba (details in `RUNBOOK.md`). A dedicated environment is
  recommended for anyone reproducing the project.
- **Brief loader replaced by a fallback.** `load_dataset("allenai/scifact_entailment")` no longer works
  with datasets 5.1.0 (script datasets are unsupported). The fallback reads the official archive and
  reproduces the builder's transformation; it is recorded and explained, not hidden.
- **CPU retrieval.** Embeddings and the re-ranker run on CPU to leave the 6 GB GPU to Ollama.
- **Optional extensions not done.** LoRA fine-tuning of a small Qwen judge and the Climate-FEVER
  out-of-domain test were not implemented (see future work in the notebook).

## 8. Upstream data, models and terms

Verified on 2026-10-20 from the upstream repositories and the Hugging Face Hub API. Licences can
change; **verify before redistribution.** This repository does not redistribute data or model weights:
`data/`, `artifacts/index/` and model caches are git-ignored and are downloaded by each user.

| Asset | Source | Licence / terms found | Notes |
|---|---|---|---|
| SciFact claims and evidence annotations (`claims_*.jsonl`) | allenai/scifact `LICENSE.md` | **CC BY 4.0** | Attribution required: Wadden et al., "Fact or Fiction: Verifying Scientific Claims", EMNLP 2020. |
| SciFact abstracts (`corpus.jsonl`) | allenai/scifact `LICENSE.md` | **ODC-By 1.0** (part of Semantic Scholar S2ORC) | Attribution required. |
| SciFact code | allenai/scifact | Apache-2.0 | Not used; only data. |
| `allenai/scifact_entailment` (HF dataset card) | Hugging Face Hub, revision `621da8dd…` | Tagged **CC BY-NC 2.0** | The card's tag (non-commercial) differs from the original repository's CC BY 4.0 / ODC-By. The data actually used comes from the official archive referenced by this card's builder. Treat use as **non-commercial and attributed** unless the discrepancy is clarified with the authors. |
| `mteb/scifact` | Hugging Face Hub, revision `cf10ab68…` | Tagged **unknown** | Repackaging of the SciFact corpus; follow the original SciFact terms above. |
| `BAAI/bge-small-en-v1.5` | Hugging Face Hub, revision `5c38ec7c…` | **MIT** | Embedding model. |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Hugging Face Hub | **Apache-2.0** | Trained on MS MARCO, whose published terms describe the dataset as intended for non-commercial research — verify before any commercial use. |
| `qwen3:4b` (Ollama) / Qwen3-4B | Ollama model metadata; Hugging Face `Qwen/Qwen3-4B` | **Apache-2.0** | Local Ollama build: GGUF, Q4_K_M, digest `359d7dd4…74fae7`. |
| Ollama, LangGraph, Streamlit, Sentence Transformers, rank-bm25, Hugging Face Datasets | respective projects | Open-source licences (MIT / Apache-2.0) | Installed from PyPI / ollama.com; see each package's metadata. |

Required attribution when presenting results:

> Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020).
> *Fact or Fiction: Verifying Scientific Claims.* EMNLP 2020.
> https://aclanthology.org/2020.emnlp-main.609/

## 9. Not in scope

Public deployment, user accounts, persistent history, paid or cloud models, full-text retrieval, web
search, multilingual claims, and any guarantee of accuracy or response time.

## 10. v2 judge (added after `final_v1`)

- **The final split is not untouched for v2.** The 100 final claims were inspected in the
  human-reviewed `final_v1` failure analysis, and the v2 design (reason and checklist before the
  verdict, a consistency check, number-reading rules) addresses the failure *categories* found
  there. No v2 prompt, rule or setting was tuned on final claims — all iteration used the 120-claim
  `dev_tune` split from the training data, and `final_v2` was run once with the frozen configuration
  — but `final_v2` is a second look at an already-inspected test set, not a blind evaluation.
- **Small samples, wide intervals.** 100 final and 120 development claims give 95 % bootstrap
  intervals about ±0.09 wide for accuracy. On development v2 gained only +0.017 accuracy and +0.058
  macro-F1 over v1 (not significant); the larger gain on the final split (+0.120 accuracy, +0.175
  macro-F1, McNemar p = 0.073) may partly be sampling variation. Expect the true improvement to lie
  somewhere in between.
- **Trade-off, not a free gain.** v2 removes most false SUPPORT verdicts but is more conservative:
  more gold-SUPPORT claims become NEI or CONTRADICT, and false CONTRADICT verdicts rise
  (`EVALUATION.md` §11). A main cause on development was the `same_subject = partly` label applied to
  paraphrased subjects; an attempt to fix it (iteration 2) made results worse and was discarded.
- **Thinking did not pay off at this model size and budget.** Unbounded qwen3:4b thinking ran
  1,000–3,000+ tokens per judge call and sometimes never answered; it is therefore budget-forced at
  512 tokens (every judge call reached the budget on dev). With the same v2 prompt it was slower and
  less accurate than no thinking on the development claims, so the product judge does not think.
  A larger budget was not evaluated because it would exceed the latency target.
- **Budget forcing depends on the qwen3 chat template.** The continuation after a truncated thinking
  block is built with qwen3's ChatML template in raw mode. For other model families the client falls
  back to answering without thinking.
- **The consistency check is heuristic.** It compares the verdict with the judge's own checklist and
  a short list of reason phrases. It cannot detect a checklist that is itself wrong (for example a
  misread direction written consistently in the checklist, reason and verdict).
- **Display rewrite of sentence numbers.** v2 reasons are shown with one-based sentence numbers by a
  regular-expression rewrite of references the model wrote ("Sentence [0]", "[3]", "sentence 2");
  unusual phrasings are left as written (`reason` keeps the raw model text).
