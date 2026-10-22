# All-layer LoRA scientific verifier

The experiment is complete and **rejected by its frozen audit gate**. Its
163-claim TRAIN audit scored **121/163, 74.23% accuracy /0.730693 Macro-F1**.
No final refit or validation-300 evaluation followed.
The best saved original-300 result remains **80.67% accuracy /0.795974 Macro-F1**,
below the user's targets. The P08 brief does not specify a 95% threshold.

## Target and structural corrections

The acceptance report checks strict >0.95 for full-pipeline accuracy, Macro-F1,
each class's precision/recall/F1, retrieval Recall@3/@10, evidence sentence
precision/recall and quotation validity. Oracle evidence remains a separate
diagnostic. Baselines and ablations retain their real measured results.

An aggregate check of the original 300 labels found 366 annotated evidence
sentences. The original reader's two-quotes-per-paper limit permits at most
**83.8798%** pooled evidence recall, even with perfect selection and unlimited
papers. Thus >95% across all requested quality metrics was structurally
impossible with that limit. The new candidate can quote all qualifying source
sentences, including evidence outside the verdict's best context window.

With the quote limit removed, the ideal pooled evidence-recall bound under a
three-document budget is **98.3607%**. The ideal reference Recall@3 bound is
**99.4833%**. These are mathematical bounds, not achieved model scores.
The aggregate audit is `artifacts/improvement/metric_limits_20261023.json`;
it exports no evaluation examples to the learner.
This aggregate validation diagnostic informed the citation-capacity correction;
validation annotations were not used in gradient fitting or epoch selection.
The validation population has already been reused and is not blind.

## Combined candidate

- Start from the pinned generic 435M-parameter DeBERTa NLI base, rather than
  any published SciFact-fitted model.
- Adapt query/value attention projections in all 24 layers with rank-16,
  alpha-32 LoRA, plus the existing classifier/pooler. The pilot measured
  2,625,539 trainable parameters. Pretrained encoder weights stay frozen.
- Mine one hard nonreference TRAIN paper from real hybrid/reranked retrieval,
  excluding stopping/audit papers. Such negative labels are weak, weight 0.25.
  Add one lexical non-rationale sentence per positive fitting claim/paper with
  weak NEI weight 0.1, after excluding the union of annotated evidence positions.
- Widen BM25/dense candidates to 100 each, rerank 50 and retain 20. This can
  change the top-ten candidate set; merely reranking the old ten cannot improve
  its Recall@10.
- Initially judge the top three above the original -2 reranker gate, with
  SUPPORT/CONTRADICT cutoffs 0.5/0.9. If no initial verdict or window conflict
  exists, inspect deeper candidates. Retain the strongest source per direction
  at >=0.9, including opposing evidence or strong window conflicts, within three
  reported papers. Every actually scored candidate is logged separately.
- For decisive papers, score all original sentences and cite every sentence
  whose predicted verdict probability is >=0.5; retain the original best-source
  fallback when none qualify. Quotations always copy exact source text.
  Repeated identical NLI inputs are cached only within the current claim.

This uses ideas from [LoRA](https://arxiv.org/abs/2106.09685) and
[verification-feedback retrieval](https://arxiv.org/abs/2508.11122), but is not
a reproduction of either paper's full experimental setup. Multiple changes
form one fixed candidate; this comparison cannot isolate a single causal gain.

## Protocol and compute

Version 1's CPU negative mining was interrupted after 43 TRAIN queries, before
model fitting or auditing. Those records and its successful hardware pilot
are preserved. Version 2 uses GPU retrieval, supports configured CPU inference
and uses FP32 when BF16 is unavailable, including a T4-compatible precision
fallback. T4 execution has not been measured. The original Qwen product and
coursework baseline/ablation results remain available.

Version 2 is frozen in `artifacts/improvement/lora_science_protocol_v2.json`.
Fitting/stopping/auditing use the existing paper/text-connected TRAIN groups:
484 fitting claims, 160 stopping claims and 163 audit claims. Validation claim
texts are excluded before fitting. The exposure manifest guards fitting and
stopping IDs and normalized texts. Negative mining cannot use stop/audit papers.

The fixed recipe is LR 5e-5, weight decay 0.05, backbone dropout 0.2, LoRA dropout
0.1, physical batch two/effective batch 16, max 512 tokens, max four epochs,
patience one and minimum stopping-loss gain 0.002. Class weights use square-root
inverse frequency from fitting rows only. Stopping selection uses unweighted
reference-pair cross-entropy, not audit or validation scores. Training and
stopping objective magnitudes have different weights.

The audit must improve both accuracy and Macro-F1 by at least 0.02 over its
saved 163-claim baseline, with no decrease in citation precision/recall or
retrieval Recall@3/@10, zero operational errors and exact quotations. Only a
passing candidate may be freshly refitted on 807 eligible TRAIN claims for
the selected epoch count, then tested on all original 300 validation claims
and the separate original 100-claim oracle diagnostic.

The TRAIN audit and original validation claims have been reused in earlier
experiments. They are not blind tests. LoRA, grouping and early stopping do
not prove absence of overfitting. Weak negatives can contain unannotated
evidence; model probabilities are uncalibrated.

## Measured audit results

Both columns below use the same 163 TRAIN audit claims, with no omitted errors
or classes. The baseline is the saved title-aware NLI control; this candidate
combines a new fitted adapter, wider retrieval, feedback and citation changes.

| Metric | Saved baseline | LoRA candidate |
|---|---:|---:|
| Correct claims | 118/163 | 121/163 |
| Accuracy | 72.3926% | 74.2331% |
| Three-class Macro-F1 | 0.725585 | 0.730693 |
| Reference-paper Recall@3 | 71.8814% | 71.5746% |
| Reference-paper Recall@10 | 88.0368% | 83.1288% |
| Evidence-sentence precision | 53.7879% | 25.3219% |
| Evidence-sentence recall | 36.5979% | 60.8247% |
| Operational errors | 0 | 0 |

The candidate's **843/843 quotations are exact**, but exact copying does not
establish relevance or entailment. It cites 466 sentences in annotated papers,
of which 118 match annotated evidence; another 377 in unannotated papers are
excluded from precision, as in the baseline. Recall uses all 194 annotated
sentences. Evidence-paper Recall@10 is 96.0784% on the 102 evidence-bearing
claims; that different metric must not replace the all-163 reference recall.

Candidate per-class results:

| Class | Precision | Recall | F1 | Claims |
|---|---:|---:|---:|---:|
| SUPPORT | 0.818182 | 0.805970 | 0.812030 | 67 |
| CONTRADICT | 0.733333 | 0.628571 | 0.676923 | 35 |
| NEI | 0.671642 | 0.737705 | 0.703125 | 61 |

The frozen gate fails accuracy gain, Macro-F1 gain, citation precision and both
reference-retrieval recall checks. Citation recall, zero errors and exact
quotations pass. Neither the two primary >0.95 targets nor all-score targets
are met. The 1.8405-percentage-point accuracy gain and 0.005108 Macro-F1 gain
are below the required 0.02 improvements; thresholds were not relaxed.

The candidate corrects 13 baseline errors and introduces 10 regressions. A
10,000-draw paired bootstrap over 103 connected paper/text groups gives 95%
candidate-minus-baseline intervals of **[-0.042556, 0.078316]** for accuracy
and **[-0.059221, 0.069887]** for Macro-F1. Both include zero; these intervals
also do not correct for earlier selection on reused data.

All 42 errors divide into 22 wrong verdicts despite a retained annotated paper,
16 decisive answers on gold-NEI claims, and four annotated papers scored but
not retained. Thus simply finding more papers does not resolve most errors.
Removing the quote cap improves recall but, with this sentence scorer, admits
too many irrelevant sentences. Nine claims promoted deeper sources; this is
not a count of nine newly correct answers. The candidate scored a mean of
10.4049 papers and maximum 20, with at most three reported papers.

Raw results, paired uncertainty, ten failure cases and full target checks are in
`artifacts/improvement/lora_science_audit163_v2/`. The human review is
`failure_analysis.md`; all labels remain unchanged. The original best
300-claim model/configuration remains available. A new 300-claim score is not
claimed from this rejected TRAIN-audit candidate.

## Learning curve and verification

All four permitted epochs completed. Epoch 3 was selected before the audit:

| Epoch | Weighted fitting loss | Unweighted stopping CE | Selected improvement |
|---|---:|---:|---|
| 1 | 0.577802 | 0.418812 | Yes |
| 2 | 0.390737 | 0.369632 | Yes |
| 3 | 0.344845 | 0.347961 | Yes; retained |
| 4 | 0.318238 | 0.348230 | No |

The final epoch lowers the fitting objective while the separate stopping
objective plateaus/slightly worsens. The frozen rule retains epoch 3; this
small change alone does not establish the degree of overfitting. The two
objective magnitudes have different weights and should not be subtracted as
a generalization gap. The [learning curve](figures/lora_science_learning_curve.png)
shows them in separate panels. Peak allocated fitting memory was 1.9613 GiB.

The final complete regression suite passed **434 tests**, with 14 warnings,
in 48.03 seconds; Ruff passed. This includes source-only promotion,
preservation of opposing evidence, adapter gradients/merge parity, CPU-device
handling, uncapped citations and cache ordering/reset. The test log is
`artifacts/improvement/lora_science_full_tests_v3.log`.

The initial 512-token hardware pilot passed all-layer backpropagation and used
1.948 GiB allocated GPU memory, saving no checkpoint. The final version's
hardware pilot and controlled workflow are recorded in `lora_science_*_v2`
artifacts. This is implementation verification, not measured claim accuracy.

An independent audit of the 1,707 fitting rows and 292 stopping rows found
zero source reconstruction/positive-label mismatches, zero positive-evidence
truncations and zero fitting/stopping/audit source-paper or claim-text overlap.
All validation fitting-text overlaps were excluded. All 43 available CPU/GPU
mining-prefix queries agree in top-20 order and chosen negative; this is a
prefix check, not a full randomized parity benchmark. The exposure guard covers
all 644 fitting/stopping claims. Frozen source hashes match the protocol.

The assessment notebook now contains traceable Module 2/3 lab adaptations,
validated in isolation without model calls. Its historical outputs are preserved
byte-for-byte in a backup and unchanged in the assessment notebook. See
`Docs/LAB_REUSE.md` and `artifacts/notebooks/course_lab_appendix_verification.json`.
Module 4 and the separate judging document referenced by P08 remain unavailable.

## Reproduction in Anaconda Prompt / CMD

```bat
cd /d "D:\Additional Project\Gen_AI"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=D:\Additional Project\Gen_AI\tmp\lora_deps;D:\Additional Project\Gen_AI\tmp\gpu_deps;D:\Additional Project\Gen_AI\tmp\audit_deps"
set "HF_HUB_OFFLINE=1"
"C:\Users\Bhargav\anaconda3\python.exe" -m scripts.run_lora_scientific_verifier_v2 --stage all
```

Run model jobs sequentially. Frozen identities refuse changed experiments;
interrupted fitting cannot silently resume under modified code. Historical
weights, datasets, raw predictions and reports remain preserved.
