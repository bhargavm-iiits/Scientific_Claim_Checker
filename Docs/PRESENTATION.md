# Presentation plan — 10 minutes + 5 minutes Q&A

Deliverable from the brief: one notebook, a 10-minute talk and 5 minutes of questions. The separate
"Group Project Guidelines & Judging" document was not available when this plan was written; check its
required structure and adjust the slide order if it differs.

Every number on a slide must come from a saved artifact (`artifacts/retrieval/…`,
`artifacts/runs/final_v1/…`). The result and failure slides below are filled from the completed
`final_v1` run (`results.csv`, `metrics.json`, `failure_analysis.md`).

## Running order and timing

| # | Time | Slide | Content (say) | Show |
|---|---|---|---|---|
| 1 | 0:00–0:45 | **Problem** | Health and science misinformation spreads faster than experts can check it. Task: given a claim, find relevant abstracts among 5,183 papers, quote the evidence sentences and answer SUPPORT, CONTRADICT or NOT ENOUGH INFO. Constraints: laptop only, model ≤ 4B, no paid APIs. | One-line claim → verdict + quote example |
| 2 | 0:45–1:45 | **Data** | MTEB SciFact corpus (5,183 abstracts) + SciFact claims. The brief's `load_dataset("allenai/scifact_entailment")` fails in datasets 5.1.0 (script datasets unsupported) → recorded fallback to the official archive, builder logic reproduced, all hashes logged. MTEB and original IDs match exactly; 0 title/text mismatches; sentences per abstract 3 / 8 / 367 (min/median/max). Final test set: 100 validation claims, seed 42, stratified (S 41, C 21, N 38); one duplicate claim removed from development. | Source manifest excerpt; class-count table |
| 3 | 1:45–2:45 | **Architecture** | LangGraph pipeline with a supervisor: validate → retrieve → evidence agent → judge (per abstract) → aggregate → supervise → explain. One local model (qwen3:4b via Ollama) with separate prompts and JSON schemas. Errors end in `status=error`, never NEI. | ASCII diagram from `ARCHITECTURE.md` |
| 4 | 2:45–3:45 | **Retrieval** | BM25 vs dense (BGE windows) vs hybrid RRF, each ± MiniLM cross-encoder. Dense best Recall@3 (0.718); hybrid best Recall@10 (0.868) and evidence-bearing Recall@10 (0.964). Re-ranking does not raise Recall@3, but lifts Recall@1 and hybrid evidence-bearing Recall@3 0.874 → 0.910, at ~1 s/claim. | Six-way table (EVALUATION.md §6) |
| 5 | 3:45–4:30 | **Evidence** | The model returns sentence IDs only; the code copies the quotation from the stored sentence — quotations cannot be invented. Long abstracts are windowed. Evidence precision excludes unannotated documents (not proven irrelevant). | Example: selected IDs → quotes; evidence P/R with denominators |
| 6 | 4:30–5:15 | **Judgement** | Per-document verdict with cited IDs, short reason, uncalibrated confidence. Supervisor: decisive verdict without citation → one re-ask → NEI. Deterministic aggregation; conflict → NEI + flag. | Aggregation table |
| 7 | 5:15–6:30 | **Results** | One table, same 100 claims: judge-only and single-prompt baselines, oracle, full system, ablations (evidence off, re-rank off, k = 1, k = 5). Accuracy over **all** requested claims; coverage and failure rate shown. | Run `final_v1`, 100 claims per variant, coverage 1.000 and 0 errors everywhere. Accuracy / macro-F1 / F1 S·C·N: judge_only 0.500 / 0.504 / 0.545·0.529·0.436; single prompt 0.560 / 0.450 / 0.650·0.174·0.526; **oracle 0.670 / 0.571** / 0.706·0.240·0.767; **full 0.560 / 0.487** / 0.638·0.296·0.526; evidence_off 0.530 / 0.446; rerank_off 0.540 / 0.468; k = 1 0.560 / 0.473; k = 5 0.570 / 0.493. Approx. bootstrap 95 % CI for full accuracy [0.46, 0.66]: only oracle − full (+0.11) is beyond noise. Say: judge-only has the higher macro-F1 because without documents it rejects 9 / 21 CONTRADICT claims vs 4 / 21 for full. Latency: cold judge-only 1.4 s, cold evidence_off 5.2 s; later variants reused cached calls (k = 1 0.007 s is a cache artefact) — quote the saved cold benchmark for the product: 7.9 s per claim (median 8.4 s, p95 10.1 s; 10 smoke claims, LLM cache off, `artifacts/latency/cold_full_k3.json`). |
| 8 | 6:30–7:30 | **Failures** | ≥ 10 analysed failures by earliest stage: retrieval miss / evidence miss / judgement error / operational. CONTRADICT is hardest: numeric comparisons, negation, direction of effect. Example: PrP claim — reason says 493 per million ≫ 1 in 5 million, verdict still SUPPORT. | `full`: 44 / 100 wrong, earliest stage (human reading): retrieval miss 1, evidence miss 5, judgement error 31, annotation ambiguity 7, operational 0. 38 of 44 are false SUPPORT (NEI → S 23 / 38, CONTRADICT → S 15 / 21). Cases: **516** CRP — reason says "the claim … is contradicted", verdict SUPPORT; **922** stable partnership — HR 0.79 read as "faster progression"; **268** cold exposure — a bile-acid abstract accepted as evidence ("CDCA mimics cold exposure"). Fix for `final_v2` (prompts unchanged in `final_v1`): reason before verdict + consistency check. Source: `artifacts/runs/final_v1/failure_analysis.md`. |
| 9 | 7:30–9:15 | **Live demo** | Script below. | Streamlit app |
| 10 | 9:15–10:00 | **Limitations & next steps** | 5,183 abstracts only; abstract-only evidence; small 4-bit model; uncalibrated confidence; MS MARCO re-ranker mismatch; conflict → NEI convention. Next: LoRA judge on the training split, biomedical re-ranker, Climate-FEVER out-of-domain test. | Bullet list |

Speaker split for a group: slides 1–2, 3–4, 5–7, 8–10 (demo driven by whoever presents slide 9).

## Demo script (about 1 min 45 s)

Preparation (before the talk, not on the clock):

1. Start Ollama; run `C:\Users\Bhargav\anaconda3\python.exe -m claim_checker doctor` — all critical
   checks OK.
2. `C:\Users\Bhargav\anaconda3\python.exe -m streamlit run ui/app.py --server.address 127.0.0.1`.
3. Press **Load models** in the sidebar (about 20–30 s; `ClaimChecker.warmup()` measured 19.95 s and 30.1 s in `artifacts/latency/`).
4. Run both demo claims once so they are cached (each then returns in well under a second). Mention
   honestly that a cold check takes about 8 s on this laptop (7.9 s mean, p95 10.1 s,
   `artifacts/latency/cold_full_k3.json`).
5. Have `artifacts/runs/final_v1/report.md` and the notebook open as a fallback.

Live steps:

1. **Supported claim** (training claim 34): type
   *"A deficiency of folate increases blood levels of homocysteine."* → **SUPPORT**. Point out:
   2 of 3 examined abstracts support it; the reference abstract 11705328 ("Randomized trial of folic
   acid supplementation and serum homocysteine levels.") is ranked 2; the quote is the exact sentence
   ("Median serum homocysteine level decreased with increasing folic acid dosage…"), shown as
   sentence 5 on screen (zero-based ID 4 in the JSON); confidence is labelled uncalibrated. The third
   abstract had no selected evidence → NEI without a model call.
2. **Honest failure** (training claim 2): *"1 in 5 million in UK have abnormal PrP positivity."* (gold
   CONTRADICT). Show the verdict and the judge's reason; explain why a small model fails numeric
   comparisons and why the quote lets a human catch it.
3. **Out-of-collection claim**: type a claim the corpus is unlikely to cover (for example a recent
   non-biomedical claim) → expect NEI, and stress "NEI means the collection is insufficient, not that
   the claim is false".
4. **Error handling** (optional, 15 s): show the "How it works" tab's NEI-vs-error explanation, or
   mention that stopping Ollama produces "Check failed — no verdict" with a fix, not NEI.
5. **Download JSON** — the same validated record shown on screen.
6. Switch to the **Evaluation results** tab: saved table from `final_v1` and the retrieval comparison
   ("read from disk; nothing here runs the model").

Fallbacks: if the model is slow or unavailable, use the cached claims (instant), or show the saved
results tab and the notebook's recorded live-check output.

## Likely questions and prepared answers

| Question | Answer |
|---|---|
| Why does `load_dataset("allenai/scifact_entailment")` not appear to work? | It is a script-based dataset; datasets 5.1.0 refuses scripts (`RuntimeError: Dataset scripts are no longer supported`). We did not enable remote code. We read the builder as text, downloaded the official archive it points to (sha256 recorded), reproduced its transformation and validated it (IDs, labels, evidence positions, counts). The notebook shows the error and the fallback. |
| Why NumPy instead of FAISS? | 5,670 vectors × 384 dimensions: exact search takes ~0.03 s; an index library adds complexity without benefit at this size. |
| Why does re-ranking not help Recall@3? | The cross-encoder is trained on MS MARCO web queries, not biomedical claims. It does improve Recall@1 and the rank of evidence-bearing documents (hybrid evidence-bearing Recall@3 0.874 → 0.910), which is what the top-3 pipeline reads. `rerank_off` measures the end-to-end effect. |
| How do you know quotes are not hallucinated? | The model only outputs sentence IDs; IDs are validated against the supplied sentences; the text is copied from the stored corpus. Quote validity is measured (1.000 in every saved development run that produced quotes). Confirmed on `final_v1`: 1.000 in all seven variants that quote — 1,307 of 1,307 quotations (`full` 208 / 208). |
| Why is a conflict reported as NEI? | Documented prototype convention: the evidence disagrees, so the system does not pick a side; the flag and the explanation show both sides. SciFact gold labels have no such conflicts. |
| What is the difference between NEI and an error? | NEI is a scientific answer ("the abstracts examined do not settle it"). An error (Ollama down, stale index, invalid output) has `status=error`, no verdict, a fix hint, and counts as wrong in accuracy. |
| How was the test set protected from tuning? | Frozen 100-claim validation subset (IDs hashed, overwrite refused); prompts developed on training claims only; train/test duplicate (871 vs 870) removed from development. |
| Why is the oracle not 100 %? | The oracle supplies the right abstracts, not the answers; the remaining errors are evidence selection and judgement errors of the 4B model. On `final_v1` the oracle reaches 0.670 accuracy / 0.571 macro-F1; its main error is CONTRADICT → SUPPORT (15 of its 33 errors — the judge reads the contradicting abstract and still says SUPPORT), then NEI → SUPPORT (10). |
| Is the confidence meaningful? | No — it is self-reported and uncalibrated, displayed as such and never used for decisions. Calibration would need a held-out reliability analysis. |
| How fast is it? | 7.9 s per claim with the LLM cache off on this laptop (median 8.4 s, p95 10.1 s; retrieval 1.2 s, evidence 3.5 s, judge 3.2 s; 10 smoke claims, `artifacts/latency/cold_full_k3.json`), plus a one-off warm-up of about 20–30 s per process; a repeated claim is served from the caches in about 0.01 s (`artifacts/latency/cached_full_k3.json`). |
| What about the evidence precision being low/high? | Precision only counts annotated documents, because SciFact did not annotate other documents; selections in unannotated documents are reported separately. |
| Why 4B and not a bigger model? | The brief limits models to ≤ 4B for laptop/Colab compute; qwen3:4b is the suggested option and fits the 6 GB GPU in Q4_K_M. |
| Which course labs did you reuse? | The lab notebooks were not available to us during the build, so no lab code was copied; `Docs/LAB_REUSE.md` maps components to Module 2/3/4 topics. <!-- TODO (student): replace with actual lab citations if the labs are obtained --> |
| What would you do next? | LoRA-tune a small Qwen judge on the training split (compare against the same untuned base), try a biomedical re-ranker, add Climate-FEVER as a separate out-of-domain test, calibrate confidence. |
| Is it safe to use for health decisions? | No. It is an educational prototype over 5,183 abstracts with a small model; it shows its evidence so a human can check it. |

## Checklist before presenting

- [ ] `final_v1` complete; `report --run-id final_v1` re-run; `failure_analysis.md` written (≥ 10 cases).
- [x] All `final_v1` result placeholders in `Docs/`, the README, the notebook and this file replaced.
- [ ] Notebook executed from a clean kernel; executed copy saved in `artifacts/notebooks/`.
- [ ] Lab citations checked (`LAB_REUSE.md` TODO).
- [ ] Demo claims cached; Ollama running; UI warmed up; fallback tabs open.
