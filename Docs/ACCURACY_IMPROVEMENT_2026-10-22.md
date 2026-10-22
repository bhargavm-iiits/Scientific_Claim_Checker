# Accuracy improvement experiments — 22 October 2026

Current target: **accuracy > 0.95 and three-class Macro-F1 > 0.95** on the original
300 SciFact validation claims. Earlier experiments targeted >0.85. Neither current
target is achieved. A target is not a measured result.
The historical reports and prediction files have not been changed.

The latest [all-layer LoRA scientific verifier](LORA_SCIENCE_EXPERIMENT.md)
completed 484-claim fitting, 160-claim stopping and the full 163-claim TRAIN audit.
It selected epoch 3 before auditing and scored **121/163, 74.23% /0.730693**,
against the same audit baseline's **118/163, 72.39% /0.725585**. It fixed 13
errors and introduced 10; paired group-bootstrap difference intervals include zero.
Evidence recall improved from 0.365979 to 0.608247, but precision fell from
0.537879 to 0.253219. Reference Recall@3/@10 also worsened. All **843/843**
quotations were source-exact and operational errors were zero. The candidate
failed its frozen multi-metric gate, so **no final refit or validation-300 run**
followed. The best confirmed original-300 result remains **80.67% /0.795974**;
the requested >95% scores remain unmet. The final suite passed **434 tests**
and Ruff passed. These are implementation checks, not quality scores.

This experiment also removed a structural two-quotes-per-paper recall cap in
the experimental adapter and expanded real retrieval. The original 300 labels
permit at most 83.8798% evidence recall under that old cap even with unlimited
papers. Correcting capacity alone did not yield selective evidence. Aggregate
validation cardinalities informed this diagnostic; evaluation labels, cases and
denominators were not changed. Both the audit and validation are reused data,
not blind tests. The assessment notebook now has actual Module 2/3 lab
adaptations with isolated verification and preserved historical outputs;
Module 4's source and the separate judging document remain unavailable.

The earlier [joint verdict/rationale reader](JOINT_READER_EXPERIMENT.md) completed
484-claim fitting, 160-claim stopping and a separate 163-claim TRAIN audit. It
retained epoch 2 after stopping loss plateaued. Audit results were
**122/163, 74.85% /0.743697**, versus baseline **118/163, 72.39% /0.725585**.
The 0.018113 Macro-F1 gain missed the frozen minimum of 0.02, so no refit or
validation-300 evaluation followed. All 277 quotations were source-exact, but
annotated-document citation precision and recall worsened to 0.306748/0.257732
from 0.537879/0.365979. Paired group-bootstrap difference intervals include zero.
The candidate was rejected and the existing stronger profile preserved.
Independent saved-token/source/overlap checks passed; all 410 test cases passed
across the main and optional runs, and Ruff passed. Neither repeated-audit
selection nor overfitting absence is resolved by these controls.

The earlier [PDF review and new reasoning experiments](P08_REVIEW_AND_NEW_APPROACH.md)
are complete. A training-free Qwen3.5 native reader failed its fixed 20-claim TRAIN
screen: **65.00% /0.510379**, with two operational errors included. A separate
logical-polarity probability-pooling method passed its initial screen but failed
the full 163-claim TRAIN audit: **71.17% /0.709147**, versus its paired same-weight,
same-window control **72.39% /0.725585**, zero errors. Its 193/193 quotations were
source-exact. Both candidates were rejected before validation 300. No thresholds
or weights were revised from those failures. All **398 regression tests** passed;
Ruff passed. The strongest 300-claim score remains **80.67% /0.795974**.

Two further [evidence-ranking experiments](EVIDENCE_RANKING_EXPERIMENT.md) are
complete. On the same 163 reserved TRAIN claims, the fixed same-reader baseline
was **72.39% / 0.725585**; the fitted ranker scored **69.33% / 0.694290** and the
independently calibrated variant scored **71.17% / 0.710092**. Both failed the
predeclared improvement gate, so neither was refitted or evaluated on validation
300. The calibrated run had zero operational errors and 223/223 exact source
quotations. This audit was reused internal development evidence, not a blind test.
The strongest confirmed validation result is still **242/300, 80.67% / 0.795974**.
At that stage the full regression suite passed **375 tests**; Ruff passed.

The latest [controlled RoBERTa experiment](SCIFACT95_EXPERIMENT.md) completed a
647/160 paper-separated fitting/stopping protocol, selected epoch one before
validation, refitted once on 807 eligible TRAIN claims and evaluated all 300
validation claims. It scored **70.00% / 0.686956**, with zero operational errors,
and was rejected. The stronger title-aware profile remains **80.67% / 0.795974**.
The separate reference-abstract diagnostic scored 83% /0.825000 and is not pooled.
The second epoch worsened stopping loss despite lower training loss; early stopping
selected the first. All 361 tests and Ruff pass. Source-exact quotations and CPU
parity were checked separately from prediction accuracy. See the linked report for
the frozen recipe, paired comparison and limitations.

## Starting point

The existing product (`qwen3:4b`, v2 judge, selected evidence, three retrieved documents)
scored **68/100 accuracy and 0.662113 Macro-F1** on the original final-100 population.
On the internal dev_tune population its completed frozen run scored **67.5% / 0.648602**.
The internal development population is training-derived; it is used for model selection,
not as an independently held-out claim of final performance.

## Completed initial comparisons

| Experiment | Population | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Existing v2 product | Development, 120 | 67.50% | 0.648602 |
| Relationship prompt, existing Qwen model, full abstracts | Development, 40 | 72.50% | 0.701314 |
| Dedicated Qwen instruction model, full abstracts and training examples | Same development 40 | 67.50% | 0.670238 |
| Generic DeBERTa NLI, full-abstract windows, top 3 | Development, 120 | 56.67% | 0.568363 |
| First scientific NLI training run, epoch 3, raw top-3 full-abstract decisions | Development, 120 | 65.83% | 0.642224 |
| Revised scientific NLI, epoch 1, development-selected top-1 and class cutoffs | Development, 120 | 74.17% | 0.711374 |
| Dedicated instruction model, v2 judge on saved selected evidence (raw judge component) | Development, 40 | 62.50% | 0.611471 |

The 74.17% revised-NLI row is a **tuning result**. Its settings were chosen on those same 120 claims,
so it must not be described as held-out performance. Later checkpoint selection and final
confirmation results are recorded below when complete. Full precision is retained in
`artifacts/improvement/*/metrics.json` and the calibration JSON files.

## Implementation

An optional NLI judge backend now uses the same retrieval, LangGraph orchestration,
claim aggregation, citation validation and saved-result interfaces as the existing application.
It loads a local, discriminative DeBERTa model, scores claim/abstract windows, applies explicit
confidence cutoffs, and selects original sentence IDs for verbatim source quotations.
It consumes document text and the live claim; it does not read evaluation annotations.
The full-abstract backend does not make an Ollama judge call. This is an architectural
extension to the original LLM judge, and should be identified as such in a project submission.
The existing LLM evaluation variants remain available.

The instruction-model and training-example branch is an experimental profile, not evidence
that a newer model improved the application. The completed comparison rejected that combination.

## Training and evaluation integrity

The starting NLI checkpoint is
[`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli),
pinned to revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.
Its label mapping is read from the model configuration, rather than assuming an order.
All inference weights and tokenizers are local. The scientific classifier has approximately
184 million parameters, separate from the existing 4B Ollama model.

For checkpoint selection, the saved revised training artifact contains **2,556 pairs from
666 training claims**. It excludes normalized texts of every public validation claim,
every dev_tune claim, and every smoke claim. An integrity check found **zero forbidden
text overlaps and zero mismatches between annotated pair labels and the training source**.
Two BM25 hard negatives per eligible training claim are added as weak NEI supervision;
these are non-reference documents, not newly annotated gold negatives. Original full abstracts
and annotated rationale passages provide the positive/reference examples.

The revised run trains the 12 transformer layers and classification layers, freezes the word
embeddings, uses seed 42, AdamW at 1e-5, maximum input length 512, physical batch size 2,
gradient accumulation 4, and four epochs. Full settings and training-data hashes are saved
with the checkpoints. Training uses the RTX 4050; CPU inference remains supported.

After choosing a configuration on internal development data, a final refit may use the
remaining original training claims, including internal development. All validation texts
remain excluded. The evaluator explicitly refuses to report training-derived split scores
from such a final-fit checkpoint.

The 200 validation claims outside the earlier final-100 were frozen in
`data/splits/accuracy_holdout.json` before confirmation testing. Its population hash is
`8035112ee57f114506ec4f4b0efcd49cccaa4b35e0d5cf786e4a441907b87633`.
The original 100 claims are a historical recheck, since their earlier errors were inspected.
No validation claim labels are used in training, prompt demonstrations, or inference.
Labels are never changed to improve reported metrics, and failures count against coverage.

The published MultiVerS SciFact checkpoint was excluded before inference. Its
[paper](https://arxiv.org/html/2112.01640) reports a training population of 1,109 SciFact
claims, which raises overlap concerns with this project's 809/300 train/validation split.
It is unsuitable as an independently held-out score for this project without stronger
provenance. The incomplete download was removed and was never used for predictions.

## Reproduction

Use the project Python environment. For GPU training in this session, GPU PyTorch was
installed separately under `tmp/gpu_deps`; the existing CPU PyTorch installation was preserved.

```powershell
python scripts/train_verifier.py --name nli_scifact_train_v2 --epochs 4 --layers 12 --batch-size 2 --max-length 512 --device cuda --hard-negatives 2 --accumulate 4 --learning-rate 0.00001
python scripts/verify_training_artifact.py artifacts/models/nli_scifact_train_v2
python scripts/score_verifier_checkpoints.py
python scripts/analyze_verifier.py artifacts/improvement/nli_scifact_train_v2_epoch4_dev120 --asymmetric
python scripts/train_verifier.py --name nli_scifact_final_v2 --epochs 3 --layers 12 --batch-size 2 --max-length 512 --device cuda --hard-negatives 2 --accumulate 4 --learning-rate 0.00001 --final-fit
python scripts/verify_training_artifact.py artifacts/models/nli_scifact_final_v2
```

The candidate configuration is `configs/verifier_candidate.yaml`. Checkpoint, thresholds
and document count must be frozen before running `verifier_holdout`. Evaluation manifests
include weight, tokenizer, model-configuration and verifier-code hashes. Changing these
inputs refuses to resume an existing run.

## Final selection and confirmation

The selected development configuration is revised training run v2, epoch 3, full abstracts,
top-3 documents, and confidence cutoffs of 0.5 for both decisive classes. It scored
71.67% accuracy and 0.715993 Macro-F1 on development. Its balanced F1 exceeds the
higher-accuracy epoch-1/top-1 candidate. This choice is frozen in
`artifacts/improvement/frozen_selection_v2.json` before validation evaluation.

The final refit uses 807 original training claims, excluding both normalized-text overlaps
with public validation. It uses the selected settings and three epochs. Its internal
development scores must not be presented as held-out metrics.

The final refit contains 3,095 training rows: 1,481 annotated/reference views and 1,614
weak hard negatives. Its saved-data audit again found zero validation-text overlap.

### Completed confirmation results

| Configuration | Population | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Original Qwen v2 product | Historical final-100 | 68.00% | 0.662113 |
| Scientific NLI, retrieved top-3 full abstracts | Same historical final-100 | **78.00%** | **0.783333** |
| Scientific NLI, retrieved top-3 full abstracts | Frozen additional 200 | **73.50%** | **0.716879** |
| Scientific NLI, retrieved top-3 full abstracts | All 300 validation claims pooled | **75.00%** | **0.739464** |
| Scientific NLI, reference abstracts supplied (oracle upper bound) | Historical final-100 | **81.00%** | **0.797984** |

**The requested >85% accuracy and >0.85 Macro-F1 have not been achieved.**
All 300 retrieved-evidence cases completed without operational failures. The pooled
Macro-F1 was recomputed from all predictions, not averaged across the two subsets.
The 200-claim result is separate from the historically inspected 100-claim benchmark.

The paired final-100 comparison improves accuracy by 10 percentage points and Macro-F1
by 0.121220. With 10,000 paired bootstrap resamples (seed 42), the 95% interval for the
accuracy difference is [0.000, 0.200], and for the Macro-F1 difference [0.021, 0.227].
McNemar's exact two-sided p-value is 0.0755: the accuracy difference does not establish
significance at the conventional 0.05 level on this small benchmark.
See `artifacts/comparisons/verifier_v2_vs_product.md`.

On the historical final-100, CONTRADICT F1 rises from 0.558140 to 0.800000.
On the additional 200 claims it is only 0.623377 (recall 0.558140), showing that
contradiction-versus-NEI errors remain a substantial generalization problem.
Oracle accuracy is also below target, so retrieval improvements alone cannot close the gap.

### Evidence quality and deployment status

All 152 quotations in the new final-100 run are exact stored source sentences. This is
quotation validity, not proof that every quoted sentence is the correct rationale.
Gold-rationale citation precision falls from **0.808219 to 0.652174**, while recall changes
from **0.487603 to 0.495868**. The label improvement therefore comes with less precise
rationale selection. The original LLM/evidence-agent default is preserved. The trained
NLI alternative is available through `configs/verifier_candidate.yaml`; it is not silently
presented as the original full multi-agent configuration.

CPU inference was checked using the first five fixed historical cases with their saved,
genuine retrieval inputs. CPU PyTorch 2.14.1 reproduced all GPU document/claim labels
and exact quotations. This is a compatibility check, not an additional accuracy estimate.
The full confirmation runs used GPU PyTorch 2.11.0 on the local RTX 4050, with CPU retrieval.

```powershell
python -m claim_checker --config configs/verifier_candidate.yaml check --claim "Your scientific claim"
```

The original reports and predictions remain intact. New predictions, complete metrics,
failure analyses and reports are under `artifacts/runs/verifier_holdout_v2`,
`artifacts/runs/verifier_recheck_v2`, and `artifacts/runs/verifier_oracle_v2`.
The combined result is `artifacts/improvement/final_verifier_results.json`.
The verifier implementation passed the 285-test regression suite. Subsequent judge
experiments are recorded below; they must not be confused with the validation results.

### Further development-only judge experiments

All experiments below use the same stratified 40-claim subset of `dev_tune`, with
saved genuine top-3 retrieval results. No reference IDs or gold labels enter prompts.
These are judge-component results, not full product validation results.

| Judge experiment | Accuracy | Macro-F1 |
|---|---:|---:|
| Existing Qwen3 4B, v3 full-abstract prompt | 72.50% | 0.701314 |
| Qwen3 4B Instruct 2507, v3 full abstract + training examples | 67.50% | 0.670238 |
| Qwen3 4B Instruct 2507, v2 selected evidence | 62.50% | 0.611471 |
| Qwen3 4B Instruct 2507, v4 full abstract | 70.00% | 0.691244 |
| Qwen3 4B Instruct 2507, v5 full abstract + bounded consistency supervisor | 62.50% | 0.567669 |

The later prompts did not improve this development benchmark and are not promoted.
Interrupted experiments have no complete metrics and are excluded from this table.
Further work is testing LoRA adaptation of the brief's Qwen2.5-1.5B-Instruct judge.
Training and evaluation utilities are `scripts/train_lora_judge.py`,
`scripts/audit_lora_training.py`, and `scripts/benchmark_lora_judge.py`.
Selection training excludes every public validation text and both internal tuning
and smoke populations. Inference receives only claims and original numbered source
sentences, with citation validity enforced before scoring. A successful training run
alone will not establish that either requested score has been reached.

The LoRA selection run is `artifacts/models/qwen25_scifact_lora_train_v1`.
Its 1,927 examples come from 666 original training claims: 1,242 reference views
and 685 weak-negative windows. The saved-data audit found zero excluded-text
overlaps and zero prompt/target/source mismatches. Class counts are 988 NEI,
614 SUPPORT and 325 CONTRADICT. It uses the shortest original annotated rationale
per positive document; all source sentence numbers remain zero-based and unchanged.

The base revision is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, with weights SHA-256
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
Runtime versions are PyTorch 2.11.0+cu128, Transformers 5.20.2, PEFT 0.21.2 and
Accelerate 1.15.0. Libraries are isolated in `tmp/lora_deps` and `tmp/gpu_deps`;
the user's original CPU PyTorch installation is preserved.

The recipe uses BF16 base weights, rank-8 LoRA over attention and MLP projections,
gradient checkpointing, 1,024-token contexts, physical batches of four, accumulation
of two and a learning rate of 2e-4. Optimizer groups contain eight examples; sorting
within a group reduces padding without changing group membership. Only answer
tokens contribute to the loss. Numerical tests compare both unpadded and padded
training losses with standard masked causal loss, and verify that LoRA parameters
receive gradients. The pilot's peak allocated GPU memory was 4.37 GiB.

```powershell
$env:PYTHONPATH = "$PWD\tmp\lora_deps;$PWD\tmp\gpu_deps;$PWD\tmp\audit_deps"
$env:HF_HUB_OFFLINE = "1"
& "C:\Users\Bhargav\anaconda3\python.exe" scripts/train_lora_judge.py --name YOUR_NEW_RUN_NAME
& "C:\Users\Bhargav\anaconda3\python.exe" scripts/score_lora_checkpoints.py --training-name YOUR_NEW_RUN_NAME
```

`artifacts/improvement/lora_selection_plan.json` records the selection rule before
training. Checkpoint evaluation is complete, with the selection saved in
`artifacts/improvement/frozen_selection_lora_v1.json`. No LoRA validation accuracy
is claimed from these development results.

| LoRA checkpoint | Development claims | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Epoch 1 | 120 | 66.67% | 0.652677 |
| Epoch 2 (selected) | 120 | 71.67% | 0.706957 |
| Epoch 3 | 120 | 70.83% | 0.704421 |

All checkpoint evaluations completed with zero operational failures. Training took
3,963 seconds (about 66 minutes). The selected LoRA checkpoint does not exceed the
earlier NLI development Macro-F1, and it is not promoted. Saved-output replay through
the application matched all 120 labels/statuses, with 122/122 exact source quotations.
Its gold-rationale citation precision is 0.684211 and recall is 0.348993. This replay
checks integration and citations; it is not another independent accuracy evaluation.

A five-fold numeric ensemble diagnostic combining this LoRA checkpoint and the
earlier development-excluding NLI checkpoint scored 76.67% accuracy and 0.752008
Macro-F1 (logistic regression, C=0.1). This is out-of-fold scoring on reused internal
development data, not validation performance. It also remains below target.

The optional application backend is selected with `configs/lora_candidate.yaml`.
It uses the same trained prompt and generator as the component benchmark, validates
sentence citations, propagates window conflicts, and works through the existing
retrieval/aggregation/supervisor/explanation workflow. This candidate reads full
abstracts; the separate LLM evidence-selection call is disabled and this must be
reported as the full-abstract variant. The original product default is preserved.
Readiness and warmup for this local backend do not require Ollama. CPU is the
candidate configuration's default; GPU evaluation uses an explicit CUDA override.
The optional backend's application tests use synthetic data and do not establish
model accuracy. Saved retrieval-input provenance is recorded in
`artifacts/improvement/judge_retrieval_provenance.json`.

### Larger-verifier follow-up

The next experimental base is the independent
[DeBERTa-v3-large NLI model](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli),
revision `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`. Its documented supervised
datasets are MNLI, FEVER-NLI, ANLI, LingNLI and WANLI; no SciFact-trained checkpoint
is imported. The downloaded weights passed their published size/SHA256 checks.

`scripts/train_verifier.py` now offers explicitly recorded BF16 autocast, gradient
checkpointing and global class balancing. Historical runs retain their original
loss recipe by default. The new global loss fixes a physical-batch-size issue:
weighted mean cross-entropy divides by the sum of target-class weights, so a single
example's class weight cancels. The new path multiplies individual unweighted
losses by class weights and normalizes across the complete optimizer group. A
numerical test verifies invariance to physical batching.

The selection plan is `artifacts/improvement/nli_large_selection_plan.json`.
Two pilots used the same first 32 shuffled training examples and effective batch
size eight; they did not inspect evaluation labels. Physical batches of two with
four accumulation steps were faster (0.180 seconds/example), with peak allocated
GPU memory 3.92 GiB and reserved memory 4.21 GiB. The full candidate trains the last
12 transformer layers plus the classifier/pooler (152,207,363 trainable parameters),
for four epochs on the development-excluding training pool. Complete checkpoint
scoring and calibration are restricted to the frozen internal development set.
Validation will require a separately frozen configuration and refit.

The completed development selection is epoch 2, retrieved top-3 full abstracts,
SUPPORT cutoff 0.5 and CONTRADICT cutoff 0.98: **79.17% accuracy and 0.784379
Macro-F1** on all 120 development claims. It is recorded in
`artifacts/improvement/frozen_selection_nli_large_v1.json`. The actual saved-data
audit again found zero evaluation-text overlaps and zero annotation mismatches.
An additional 18-candidate class-score adjustment diagnostic reached only
0.784788 Macro-F1 at the same accuracy; this negligible difference is not promoted.
The frozen primary threshold recipe is preserved.

The final refit is `nli_large_scifact_final_v1`, with 3,095 pairs from 807 eligible
original training claims and all validation texts excluded. It runs two epochs,
retaining the four-epoch learning-rate horizon under which the selected checkpoint
was obtained. Its pre-confirmation plan is
`artifacts/improvement/nli_large_refit_plan.json`. Confirmation is now complete:

| Frozen larger verifier v1 | Claims | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Historical final cohort | 100 | 75.00% | 0.751405 |
| Additional validation cohort | 200 | 81.00% | 0.800394 |
| Pooled validation | 300 | 79.00% | 0.784732 |
| Oracle reference-abstract diagnostic | 100 | 81.00% | 0.802304 |

All four populations have complete operational coverage. These are reused
validation cohorts, not new blind tests. The oracle supplies reference document
IDs and must not be presented as normal retrieval accuracy. Full results are in
`artifacts/improvement/final_large_verifier_results.json`. The historical 100
have citation precision 0.670455, recall 0.487603 and 122/122 source-exact quotes;
the precision remains below the original LLM product's 0.808219. A five-claim
CPU/GPU label, document-decision and quotation parity check passed with the
original CPU PyTorch installation; this does not establish full CPU accuracy or
latency. No final-refit development score is represented as held out.

`configs/verifier_large_candidate.yaml` exposes this optional backend; its CPU
configuration is separate from the CUDA configuration used for confirmation.
Pure NLI evaluation identities and runtime metadata now identify the actual
classifier and do not require an unrelated Ollama model/server. The regression
test for this specifically fails if evaluation touches an LLM client.

### Evidence-complete, title-aware larger verifier v2

An audit of the saved v1 positive training views found 86 truncated abstract views,
including 21 that did not retain a complete annotated rationale. The revised
TRAIN-only context builder chooses a complete original rationale and fills the
remaining token budget with neighbouring source sentences. Sparse rationales
retain their original sentence indices. Article titles are added to training and
inference contexts. Runtime windowing receives no annotations.

The saved pilot data contain 934 positive views with **zero truncated positive
views and zero missing complete rationales**. Source/label and excluded-text
audits passed. The predeclared plan is
`artifacts/improvement/nli_title_v2_selection_plan.json`. The candidate trains all
24 encoder layers plus the head (303,362,051 parameters), with word embeddings
frozen, four epochs, learning rate 5e-6, effective batch eight, BF16 activations,
gradient checkpointing and globally normalized class weighting. Model weights
remain FP32; bitsandbytes 0.50.2 compresses optimizer states. This optional
training dependency is separate from CPU inference.

Hardware-only pilots compared physical batches one, two and four on the same
32 training examples. Batch two was fastest; peak allocated memory was 4.12 GiB.
The current run is `nli_large_title_train_v2`. Development selection compares
all four epochs, retrieved top-1/3/5 abstracts, predeclared class cutoffs and
optional cross-encoder relevance gates. Inference windows overlap by two
sentences. Gates never remove claims from evaluation; a claim with no accepted
documents receives NEI. Oracle reference-document evaluation bypasses the gate.

This recipe has now completed development selection: epoch 1, top-3 abstracts,
retrieval score gate -2, SUPPORT cutoff 0.5 and CONTRADICT cutoff 0.9 scored
**80.83% accuracy and 0.807335 Macro-F1**. Its complete frozen artifact is
`artifacts/improvement/frozen_selection_nli_title_v2.json`. A saved-window-score
replay through the real application reproduced all **120/120** decisions. That
replay uses placeholder sentence-ranking scores: it establishes integration
parity, not fresh predictive accuracy or citation precision/recall.

Before any title-family validation, one controlled development comparison was
added: `nli_large_title_clean_train_v3` omits the 1,332 weak BM25 negative pairs.
The actual saved pilot contains 1,224 original reference/rationale pairs from
the same 666 eligible claims, with zero excluded-text overlaps, zero source/label
mismatches and complete positive rationale coverage. The statistical recipe and
development grid remain the same. The explicit extension plan is
`artifacts/improvement/nli_title_family_selection_plan.json`; it records that the
80.83% weak-recipe development result was already known. Only the better recipe
by development Macro-F1, then accuracy, will be refitted and confirmed. Validation
does not choose between these recipes.

The complete family workflow is reproducible with
`python scripts/run_title_verifier_family.py` in the GPU training environment.
It preserves existing frozen selections and refuses to replace a changed final
recipe. In addition to the verifier inference extra, 8-bit optimizer training
requires the optional `verifier-training` extra; it is not needed to run the
saved CPU classifier.

A final development-only complementarity diagnostic is predeclared in
`artifacts/improvement/nli_title_ensemble_diagnostic_plan.json`. It averages the
two selected checkpoints' scores only when every source window aligns. A more
complex two-model runtime will be considered only if a genuine mixture exceeds
both development targets and improves Macro-F1 by at least 0.02 over the frozen
single-model winner. Otherwise the single-model recipe is retained. This plan
was written before the annotated-only checkpoints were development-scored.

The controlled comparison is complete. The annotated-only recipe selected epoch
1 and scored **77.50% / 0.773333** on development. The best score mixture reached
**81.67% / 0.814278**, which failed the predeclared implementation criterion; no
two-model runtime was added. The weak-supervision title recipe remains the family
winner (**80.83% / 0.807335**). The frozen family artifact is
`artifacts/improvement/frozen_selection_nli_title_family_v1.json`; the final refit
is `nli_large_titlefamily_final_v1`, one epoch with the original four-epoch LR
horizon. Its development selection was frozen before validation began.

### Separate newer language-model check

The approved `qwen3.5:2b` download completed. The [Ollama model
page](https://ollama.com/library/qwen3.5:2b) lists a 1.88B text tower and 331M
vision component. The actual local GGUF/Q8 artifact is 2,683,396,812 bytes, with
digest `0689d44085e06d165161a8a9a1731344278cfb5aade63a3c3dbdb48ab54b130a`.
It fits the assignment's parameter budget. The [official model
card](https://huggingface.co/Qwen/Qwen3.5-2B) identifies the source model; local
metadata are saved in `artifacts/improvement/qwen35_2b_model_identity.json`.

The predeclared procedure is
`artifacts/improvement/qwen35_judge_development_plan.json`: after NLI confirmation,
try the existing v4 full-abstract prompt, no examples, thinking disabled, on the
same fixed development-40. Advance to all development-120 only if both initial
metrics are at least 0.80. Consider integration only if the complete 120 clears
both requested targets. All prompts use genuine retrieved text; no validation
labels or reference IDs are supplied. The GPU experiment is skipped if the NLI
confirmation already clears both targets. Downloading does not load the model
on the GPU, and generation does not run concurrently with training.

The completed fixed-40 result was **37.50% accuracy / 0.312088 Macro-F1**. It failed
the advance gate, so development-120 and validation were not run. The model was
unloaded after the check. This rejects the tested greedy, non-thinking prompt
configuration for this task; it is not a general claim about the model's quality.

`scripts/refit_and_confirm_title_verifier.py` freezes the selected settings,
refits the frozen family winner on the validation-excluding original training pool with the same
four-epoch learning-rate horizon, audits the actual saved data, then evaluates
the separate historical-100, additional-200 and oracle cohorts. Confirmation is complete:

| Title-aware NLI family winner | Claims | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Historical final cohort | 100 | 75.00% | 0.743712 |
| Additional validation cohort | 200 | 83.50% | 0.822082 |
| Pooled validation | 300 | **80.67%** | **0.795974** |
| Oracle reference-abstract diagnostic | 100 | 81.00% | 0.795946 |

There were zero operational errors. These reused validation cohorts are not new
blind tests. Both requested targets remain unmet. Results are in
`artifacts/improvement/final_title_verifier_results.json`, and the optional CPU
profile is `configs/verifier_title_candidate.yaml`. Five fixed CPU/GPU claim and
document decisions matched with the original CPU PyTorch 2.14.1 installation;
all CPU quotations were source-exact. This is a device-parity check, not a full
CPU accuracy or latency evaluation.

The historical-100 citation precision is **0.571429**, recall **0.396694**, below
the original LLM product's 0.808219 / 0.487603. All **345/345** quotations across
the retrieved 300 are verbatim source text; exactness does not establish that a
quotation is the right supporting evidence. The paired historical-100 accuracy
gain is seven percentage points, with a 95% bootstrap interval of -3 to +17
points and McNemar p=0.2649; this small cohort does not establish a statistically
clear accuracy advantage. See `artifacts/comparisons/title_family_vs_product.md`.

A reporting fix retains claims with zero accepted documents in the citation
recall denominator. The earlier record writer used null citations in that case,
which was mistaken for a stage that did not apply. New records use an empty
mapping, and the reader normalizes old empty-retrieval records. Only the three
new title-family derived reports were regenerated. Their prediction hashes and
all verdict metrics are unchanged; the audit is
`artifacts/improvement/citation_reporting_correction.json`.

The complete regression suite passes **333 tests**, and Ruff passes across source,
UI, scripts and tests. The CLI
regression verifies that an omitted `--evidence-mode` follows the selected
configuration, so a full-abstract NLI profile does not silently invoke the LLM
evidence agent. The default application configuration remains the original LLM
product while these alternatives are assessed.

### Further evidence-selection diagnostics

The annotation-free single/adjacent-pair context diagnostic selected adjacent
pairs, top-3 retrieval, gate -2 and both cutoffs 0.98. It scored **82.50% /
0.818824** on development-120 and failed its integration gate, so it was not
promoted. Raw scores, the frozen grid and the exact executed source archive are
in `artifacts/improvement/nli_short_passages_dev120`.

A separate **non-deployable gold-rationale diagnostic** reached 95.00% on all
development reference evidence, 93.33% with genuine top-3 retrieval and 94.17%
with top-5 retrieval. It supplies annotated sentence IDs and empty evidence for
NEI claims, so it is explicitly an oracle upper bound, never an achieved system
score. It uses the development-excluding reader, not the final refit. This
diagnosis motivates learning rationale selection; it does not meet the target.

The first selector fine-tunes the already cached independent MS MARCO MiniLM
checkpoint, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, with binary
TRAIN sentence-rationale labels. Its actual data audit has **6,884 sentences /
666 claims**, zero forbidden-text overlaps, zero source/annotation mismatches
and no omitted sentences. Six epochs took about 140 seconds, peaking at 1.16
GiB. Epoch 6 had the best fixed-threshold reference-supplied rationale F1,
0.626140. This is a component F1, not claim accuracy.

The retrieved development grid reached **83.33% / 0.825898**. Its best fusion
weight ignored the small selector, so no trained-selector runtime was added.
The original plan and pre-scoring fusion extension are
`artifacts/improvement/evidence_selector_selection_plan.json` and
`artifacts/improvement/evidence_fusion_development_plan.json`. An independently
predeclared stronger-selector attempt reuses the existing DeBERTa base NLI
encoder with a new binary head, frozen embeddings, six epochs, candidate-only
source contexts and premise-first input. It retains all exclusion and audit
rules. Its plan is `artifacts/improvement/evidence_deberta_selection_plan.json`;
its completed retrieved development results are described below.

The stronger selector again audited **6,884 sentences from 666 claims**, with
zero forbidden-text overlaps, source/annotation mismatches or omissions. It uses
physical batches of four and accumulation four, preserving effective batch 16;
the weighted binary loss is normalized over each optimizer group. Its binary
head is newly initialized while all pretrained encoder weights are retained.
Source construction, input order, complete-candidate token limits and original
sentence-ID selection have regression coverage.

Epoch 4 had the strongest reference-supplied binary rationale F1, **0.714754**.
The coarse retrieved development grid reached **83.33% / 0.830662**. Explicitly
recorded development-only reader and selector cutoff refinements reached
**84.17% accuracy / 0.840411 Macro-F1** on all120 internal development claims.
The chosen configuration uses top5 retrieved abstracts, retrieval gate -2,
selector probability cutoff0.475, at most one sentence per abstract, no NLI
fusion, and reader support/contradiction cutoffs0.98/0.97.

This failed the initial >0.85 development integration gate. An exploratory
confirmation exception was recorded **before validation**, in
`artifacts/improvement/trained_evidence_confirmation_plan.json`, to measure the
strongest remaining candidate after refitting on all eligible TRAIN data. The
failed gate remains recorded as failed. Neither this exception nor development
accuracy establishes either requested target.

The final selector refit completed four epochs using the selected six-epoch LR
horizon on **8,409 sentences / 807 TRAIN claims**, excluding every normalized
validation text. Its audit has zero source/annotation mismatches, forbidden-text
overlaps or omissions. The profile is `configs/trained_evidence_candidate.yaml`.
The real application now supports local trained evidence selection, local
readiness/warmup, model/code identities, CLI doctor and UI configuration profiles.
Cached development-output replay matched120/120 decisions with97/97 exact source
quotations and no Ollama calls. This replay verifies integration, not new
predictive accuracy.

The completed all-TRAIN-refitted application scored:

| Cohort | Claims | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Additional validation | 200 | 77.00% | 0.768276 |
| Historical validation | 100 | 74.00% | 0.751211 |
| Pooled retrieved validation | 300 | **76.00%** | **0.762882** |
| Oracle reference-abstract diagnostic | 100 | 81.00% | 0.811694 |

All400 application checks completed without operational errors. Both requested
targets are unmet, and this refitted selector does not replace the previous
title-aware profile. The frozen settings and full results are in
`frozen_trained_evidence_candidate.json` and `final_trained_evidence_results.json`
under `artifacts/improvement`.

Its pooled cited-sentence precision is **0.917910**, recall **0.336066**; all
**199/199** quotations are source-exact. Historical100 precision/recall are
0.926829/0.314050, and additional200 are0.913978/0.346939. Precision excludes
unannotated document selections, which are not proven irrelevant; recall keeps
all applicable claims. The higher citation precision comes with lower recall
and lower verdict accuracy. See `trained_evidence_quality.json` and the paired
historical comparisons in `artifacts/comparisons/trained_evidence_vs_*.md`.
Five fixed CPU checks also matched GPU claim/document verdicts with exact
quotations using the original CPU Torch2.14.1 installation
(`cpu_parity_trained_evidence.json`). This checks device parity only.

A fixed follow-up assesses the **exact development-selected checkpoints**
without all-TRAIN refitting, retaining every previously selected threshold and
top5 retrieval setting. Its plan is
`artifacts/improvement/selection_checkpoint_confirmation_plan.json`, and its
profile is `configs/trained_evidence_selection_candidate.yaml`. The refit's
aggregate validation regression informed this follow-up choice; individual
validation error facts are not used to construct rules or training examples.
This is iterative work on reused validation, not a new blind test. Its actual
application follow-up completed: additional200 **76.00% / 0.758397**,
historical100 **72.00% / 0.727995**, pooled300 **74.67% / 0.748624**, and separate
oracle100 **75.00% / 0.745643**. All400 checks completed without operational
errors. Pooled citation precision/recall are0.881481/0.325137, with207/207 exact
quotations. Reverting to the selected weights did not improve the validation
result. See `final_selection_evidence_results.json`, `selection_evidence_quality.json`
and `artifacts/comparisons/selection_evidence_vs_*.md`.

### Overfitting investigation and revised protocol

The saved curves show late-epoch overfitting: selector rationale F1 falls from
0.714754 at epoch4 to0.698962 at epoch6 while training loss keeps falling. With
fixed inference settings, NLI development claim Macro-F1 falls from0.807335 at
epoch1 to0.757560 at epoch4 while training loss falls from0.614251 to0.056276.
The three selector searches alone tested52,288 correlated configurations on
120 development claims. This creates a serious selection-overfitting risk; the
best development score is not a dependable validation estimate. Exact
normalized text overlaps remain zero in the actual fitted data.

[The detailed overfitting audit](OVERFITTING_AUDIT_2026-10-22.md) separates these
findings from refit effects and distribution uncertainty. A new frozen
TRAIN-only protocol groups shared reference papers and normalized duplicate
claims, isolates fitting/stopping/auditing, fixes one regularized recipe and
allows no threshold grid. It contains807 claims in490 independent groups.
The new trainer freezes checkpoint selection before real application audit
inference. Grouped TRAIN audit results are never substituted for validation300,
and all original runs/defaults remain preserved. No >85% claim is made.

The first controlled grouped run completed on484 fitting claims and160 stopping
claims, selecting epoch4 because stopping loss improved throughout the four-epoch
cap. Its later real application audit scored **72.39% / 0.725585** on163 reserved
TRAIN claims, with zero errors and219/219 source-exact quotations. These are
different claims and a different training size from validation300; the scores
cannot be directly ranked. Only one of five folds is complete, with no final
refit or new external-validation claim. The better confirmed80.67% /0.795974
validation profile remains available, and both requested targets remain unmet.

### Prepared annotation-assisted LLM diagnostic

An optional component experiment (`benchmark_judge.py --paper-examples`) supplies
historical labeled TRAIN claims about a genuinely retrieved paper. This is explicitly
**TRAIN-annotation-assisted prompting**, distinct from corpus-only RAG. It excludes
the current normalized claim and uses no validation labels or reference IDs.
`scripts/audit_paper_examples.py` checked all 744 example claims, document/label
annotations and original passages: zero excluded-text overlaps or source mismatches.
Such examples are available for at least one retrieved paper in 72 of 120 development
claims. No accuracy improvement is claimed until this separate experiment is run.

The completed fixed-40 diagnostic scored **67.50% accuracy and 0.663273 Macro-F1**,
so it is not promoted. Some unchanged inputs reused the existing response cache;
its timings must not be presented as a cold-latency benchmark. Labels and source
passages in the demonstrations came only from the audited training artifact.

### Updated request: >95% accuracy, >0.95 Macro-F1 and general science

The revised numerical targets remain unmet. The strongest confirmed retrieved
SciFact validation result remains **80.67% accuracy / 0.795974 Macro-F1**, on a
reused 300-claim population. No general-science score can replace that result.

A [separate general-science extension](GENERAL_SCIENCE.md) now contains 14,056
source-attributed passages from OpenStax textbooks and one verified USGS excerpt.
It uses an unmodified generic pretrained NLI reader and fixed inference settings,
without fitting to its developer-written demonstration labels. Source provenance,
separate evaluation identities and explicit benchmark-reuse tracking are included.
This increases evidence coverage; it is not proof of 95% accuracy or of absence
of overfitting. A representative, independently labeled holdout remains necessary.

The first frozen general-science smoke run completed **14/15 correct:93.33%
accuracy /0.932660 Macro-F1**, with zero operational errors and46/46 source-exact
quotes. It is a developer-written demo set, not an independent general-science
benchmark, and it does not meet either target. The Moon/star case returned NEI
because of conflicting model decisions. No settings were retuned against that
failure. Separate CPU wording checks returned SUPPORT for "Water is a Universal
Solvent", CONTRADICT for "Water dissolves every substance" and SUPPORT for its
negation. The original SciFact metrics remain unchanged. All350 tests and Ruff
pass; all91,207 general-source sentences passed the provenance audit.
