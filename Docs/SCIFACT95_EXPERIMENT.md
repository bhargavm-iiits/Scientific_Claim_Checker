# SciFact 300: requested accuracy and Macro-F1 above 95%

The target applies to the original 300 SciFact validation claims, not general-science
demos. It requires **at least 286/300 correct** and separately **three-class Macro-F1
above 0.95**. No target score or absence of overfitting is guaranteed.

## Completed result: candidate rejected

**Neither requested target was achieved.** The original best profile remains
`configs/verifier_title_candidate.yaml`: 242/300 correct, 80.67% accuracy and
0.795974 Macro-F1. The new model is preserved as a rejected experiment.

| New RoBERTa evaluation | Claims | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Additional validation |200|68.50%|0.664998|
| Historical validation |100|73.00%|0.729001|
| **Pooled original validation** |**300**|**70.00%**|**0.686956**|
| Separate reference-abstract diagnostic |100|83.00%|0.825000|

All 400 checks completed without operational errors. The pooled score includes
all three classes and all 300 retrieved-evidence cases; the reference-abstract
diagnostic is excluded. Validation labels, claim populations and metric definitions
were unchanged. They were never used as training targets or checkpoint-selection
inputs. These cohorts remain reused validation, not blind evaluation.

The paired comparison found 55 claims correct only for the baseline and 23
correct only for the candidate. Its accuracy change is -10.67 percentage points;
the paired 10,000-resample 95% interval is [-16.33, -5.00] points. Macro-F1 changes
by -0.109018, with interval [-0.167544, -0.051438]. McNemar's exact p=0.000378.
These results reject this candidate. Even an ideal selector restricted to choosing
one of these two models' existing verdicts could select a correct verdict on only
265/300 claims (88.33%); this is a diagnostic, not an implemented ensemble result.

All 570 quotations match stored source sentences. Annotated-document citation
precision/recall are 0.479452/0.382514; 278 quotations in unannotated documents are
excluded from precision because their relevance is not established. Exact quotation
matching did not ensure correct verdicts. Five fixed CPU checks matched GPU
document/claim verdicts and source-exact quotations.

The stopping rule selected epoch one. Epoch two lowered weighted training loss
from 0.709087 to 0.349930, but increased unweighted stopping loss from 0.482163
to 0.617122. Stopping-pair Macro-F1 fell from 0.819997 to 0.817457. The losses have
different weighting, so their absolute gap is not directly comparable. The final
refit used one epoch on 807 TRAIN claims/2,288 pairs, with zero validation-text
overlap or source/label mismatches. This does not establish absence of overfitting.

Final verification: **361 tests pass**, including actual fitted-ID/text overlap
rejection for selection checkpoints as well as final refits. Ruff passes. The
evaluation runner and standalone verifier benchmark now enforce those population
checks when manifests record fitted IDs/text hashes.

Artifacts: `final_roberta_verifier_results.json`, `roberta_paired_quality.json`,
`cpu_parity_roberta.json` and `roberta_final_full_tests.log` under
`artifacts/improvement`; source predictions/reports under `artifacts/runs/roberta_scifact_*_v1`.
No model was promoted and no >95% performance claim is made.

## Unchanged measured baseline

The title-aware DeBERTa profile scored 242/300 correct: **80.67% accuracy and 0.795974
Macro-F1**. The same 300 claims are repeatedly reused validation, not a new blind test.
An aggregate audit of its 58 errors found:

| Error category | Claims |
|---|---:|
| Wrong decisive claim despite examining an annotated evidence paper |35|
| Wrong decisive claim without an annotated evidence paper in retrieved top3 |10|
| Annotated evidence paper removed by the relevance gate |3|
| Gold NEI predicted as decisive |10|

Among 188 gold decisive claims, an annotated evidence paper appeared in top 3 for 175,
top 10 for 182, and passed the examination gate for 172. Missing annotated evidence
does not prove that no other valid evidence exists. The categories are diagnostic,
not a new accuracy denominator. No examples or labels were exported for training.
The existing oracle 100 diagnostic scored 81%, so improving retrieval alone cannot
resolve the reader's errors. Reaching the accuracy target requires 44 net additional
correct predictions while also meeting the separate Macro-F1 target.

## One different reader, fixed before evaluation

The new candidate uses the author's generic
[RoBERTa-large NLI model](https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli),
whose documented training sources are SNLI, MNLI, FEVER-NLI and ANLI R1–R3.
It is not a SciFact-finetuned public checkpoint. Revision
`5b605abab9b75bc87ab66cfc049ef58d9d64b8ed` and published weights SHA-256
`a51c55cb7dc41c4c862f06a3d9c413a8aa0619222bc56f06cb6f2f5b2d3ee14a`
are recorded locally. The downloaded weights passed that checksum check.
Loading uses PyTorch's restricted `weights_only=True` path; new trained checkpoints
are saved as safetensors.

`artifacts/improvement/scifact95_roberta_protocol_v1.json` freezes one recipe:

- 647 TRAIN fitting claims and 160 stopping claims, separated by reference-paper
  groups and normalized claim text; all public validation texts excluded.
- Full abstracts and annotated rationales as source-preserving training views,
  plus one BM25 nonreference weak NEI example per claim where available. Papers
  belonging to stopping groups are excluded from fitting negatives as well.
- Frozen token embeddings; 24 encoder layers and the classifier trainable.
  Dropout 0.2, weight decay 0.05, learning rate 1e-5, effective batch 16, seed 42.
- Up to 4 epochs, stopping after one epoch without at least 0.002 improvement in
  mean unweighted annotated stopping-pair cross-entropy.
- Fixed inference: top 3 retrieved abstracts, relevance gate -2, support/contradiction
  cutoffs 0.5/0.5, titles included, window overlap 2. No threshold grid.
- One final refit from the generic base on 807 eligible TRAIN claims, for the chosen
  epoch count and original four-epoch learning-rate horizon, then one confirmation
  on validation 300. A separate oracle 100 diagnostic is never pooled into that score.

Actual selection-data audit: 647 fitting claims/1,826 pairs and 160 stopping
claims/292 pairs, 637 weak-negative pairs, zero validation-text overlaps,
zero fit/stop paper overlaps and zero source/annotation mismatches. The longest-input
GPU pilot completed eight batches and an optimizer step with finite loss, using
3.17 GiB peak allocated GPU memory. Software guards explicitly reject validation
text overlap, shared fitting/stopping papers, incorrect labels, fabricated premises
and silently missing fitting claims.

This new architecture uses one grouped stopping split. It does not complete or
replace the older five-fold DeBERTa protocol. Stopping-pair metrics use supplied
reference documents; they are not end-to-end validation accuracy. Public pretraining
data contamination cannot be ruled out by the local text-overlap audit alone.

## Reproduce on this computer (Anaconda Prompt / CMD)

```bat
cd /d "D:\Additional Project\Gen_AI"
set "TEST_PYTHON=C:\Users\Bhargav\anaconda3\python.exe"
set "PYTHONPATH=D:\Additional Project\Gen_AI\tmp\lora_deps;D:\Additional Project\Gen_AI\tmp\gpu_deps;D:\Additional Project\Gen_AI\tmp\audit_deps"
set "HF_HUB_OFFLINE=1"
"%TEST_PYTHON%" -m scripts.run_roberta_workflow
```

Completed stages are reused only when their immutable identity matches. Interrupted
training is preserved and rejected rather than mixed with a different run. The log is
`artifacts/improvement/roberta_scifact_workflow_v1.log`; final measured results are
written to `artifacts/improvement/final_roberta_verifier_results.json`. Original
predictions, labels, metric definitions and default configuration remain preserved.

The workflow also registers actual fitted claim IDs and normalized text hashes
for checkpoint provenance and evaluation leakage checks. The selected checkpoint
is not held out on all legacy development claims; only its separate stopping
population was excluded from fitting. The final refit may be evaluated on validation,
not as a held-out result on TRAIN-derived development splits.
