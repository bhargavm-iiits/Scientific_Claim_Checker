# Evidence-aware ranking experiment

The target remains accuracy above 95% and three-class Macro-F1 above 0.95 on
the unchanged 300 SciFact validation claims. The best confirmed profile remains
80.67% /0.795974. This experiment has not established either target.

## First result and separate calibration follow-up

The first ranker failed the audit gate: **113/163 correct, 69.33% accuracy and
0.694290 Macro-F1**, versus 118/163, 72.39% /0.725585 for the same-reader baseline.
It was rejected without a final refit or any validation-300 run.

An annotated evidence paper appeared in the top three for 92/102 decisive audit
claims, close to the baseline's 93/102. After filtering, evidence remained for
only 74/102 versus 91/102. These are counts of claims with at least one annotated
evidence paper, not scientific-truth certification. The aggregate evidence loss
motivated a separate calibration experiment rather than tuning against individual
audit errors or validation labels.

The separate plan `evidence_rank_calibrated_protocol_v2.json` keeps 484 fitting
claims, splits the old stopping population into **80 stopping and 80 calibration
claims by paper group**, and keeps the 163-claim audit outside all parameter
learning. A fresh generic model is fitted; calibration claims do not select its
checkpoint. Epoch three was selected on the 80 stopping claims.

After checkpoint selection, a fixed lower-tail order-statistic rule with alpha
0.05 uses the minimum annotated-positive document score per calibration paper
group. It produced raw-logit cutoff **-6.066863**, using 34 positive groups and
56 claim/document pairs. This targets retention of annotated source evidence
when retrieved. It is **not 95% claim accuracy**, proof of coverage on validation,
or a probability that a scientific claim is true.

Unlike the first protocol, this follow-up keeps the selected ranker weights
unchanged after calibration. There is no 807-claim ranker refit that would change
the calibrated predictor. Evaluation guards include all 644 claims used for
gradients, checkpoint selection or calibration. The old 163-claim TRAIN audit is
reused internal development evidence, not a fresh blind test. The same fixed
two-percentage-point audit gate controls access to validation 300.

The calibrated experiment is run with
`python -m scripts.train_calibrated_evidence_ranker --stage all`. Its log is
`artifacts/improvement/evidence_rank_calibrated_v2.log` and its audit folder is
`artifacts/improvement/evidence_rank_calibrated_audit_v2`. All **375 regression
tests** pass, including the finite-score and fixed-order calibration checks.

## Completed results: both candidates rejected

| System | Correct / audit claims | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Same-reader baseline |118/163|72.39%|0.725585|
| Fitted ranker, fixed gate zero |113/163|69.33%|0.694290|
| Fresh ranker, independent calibration |116/163|71.17%|0.710092|

These are **TRAIN-audit results**, not scores on the requested validation 300.
Both candidates failed the predeclared requirement to improve both metrics by at
least 0.02. Neither triggered a final refit or a validation-300 run. The calibrated
run completed all 163 claims with zero operational errors and **223/223 quotations
matching stored source sentences exactly**. A correct quotation does not establish
a correct verdict.

The calibrated training run selected epoch three: training/stopping BCE was
0.369741/0.380238, 0.303459/0.283048, 0.279293/0.266739, then
0.279686/0.272527. Epoch four worsened stopping loss and was rejected at the
four-epoch cap. This controls checkpoint choice; it does not prove that overfitting
is absent. Calibration restored much of the evidence retention lost by the first
gate, but did not improve the actual claim metric.

Mean annotated-evidence document recall among the 102 decisive audit claims
actually examined was 0.709150 for V1, 0.880719 for calibrated V2 and 0.874183
for the baseline. Better evidence retention alone was insufficient. In the paired
V2 comparison, three claims changed from correct to wrong and one from wrong to
correct. Accuracy changed by -1.23 percentage points (paired claim-bootstrap 95%
interval -3.68 to +1.23); Macro-F1 changed by -0.015492 (interval -0.043595 to
+0.007313). These exploratory intervals resample claims, not paper groups, and
do not establish an improvement on an independent population. The saved paired
report is `artifacts/improvement/evidence_rank_calibrated_paired_audit_v2.json`.

Exact metrics and predictions are preserved in
`artifacts/improvement/evidence_rank_audit_v1` and
`artifacts/improvement/evidence_rank_calibrated_audit_v2`. The original stronger
profile and its **80.67% / 0.795974 on validation 300** remain unchanged. Accuracy
strictly above 95% requires at least **286/300 correct**; the current best has 242.
Macro-F1 must independently exceed 0.95. Neither target has been achieved.

## Original V1 fixed design

The existing MiniLM cross-encoder is adapted to rank original abstract windows
by the presence of annotated claim evidence. SUPPORT and CONTRADICT rationales
both count as evidence; the existing NLI reader still decides their direction.
This experiment changes document ranking and rejection, not the stored claims,
labels, source quotations or accuracy denominator.

The source model is the locally cached generic
`cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. No new network download was needed.
The immutable plan is `artifacts/improvement/evidence_rank_protocol_v1.json`.

| Role | TRAIN claims | Use |
|---|---:|---|
| Fitting |484|Learn ranking weights|
| Stopping |160|Select one checkpoint by window BCE|
| Application audit |163|Accept/reject the fixed ranking recipe|

Groups follow the existing reference-paper/duplicate-text partition. Audit and
stopping reference papers are excluded from fitting examples. Stopping source
documents also exclude every fitted source document and audit reference paper.
Validation texts are excluded from all three TRAIN-derived populations.

Each fitting claim supplies its reference documents and up to four nonreference
documents from genuine BM25+dense RRF top-20 candidates. Retrieval itself receives
only the claim text. Nonreference documents are **weak negatives**: absent
annotations do not prove that a paper lacks relevant scientific evidence.

Training uses the same claim-first, title/newline/source-sentence windows as live
reranking. A window is positive only when it contains a complete original annotated
rationale. Partial rationale windows and oversized training inputs are recorded
and omitted rather than given misleading labels. Evaluation keeps every claim.

Actual selection data contains 2,873 fitting windows: 392 annotated positives,
235 annotated NEI windows, 2,204 weak nonreference windows and 42 weak nonrationale
windows. Two partial-rationale fitting windows were omitted. All 484 fitting
claims remain represented. Stopping uses 968 windows from all 160 claims.
Source reconstruction, target/annotation checks, fitting/stopping source-paper
overlap and validation-text overlap checks passed.

The single frozen recipe uses six encoder layers and the classifier, frozen token
embeddings, dropout 0.2, weight decay 0.05, AdamW at 5e-6, physical/effective
batch sizes 8/32, and seed 42. Maximum length is 512 tokens. Selection uses mean
unweighted stopping-window binary cross-entropy, with at most four epochs,
patience one and a minimum improvement of 0.001. No threshold grid is run.

| Epoch | Training BCE | Stopping BCE | Checkpoint improved |
|---|---:|---:|---|
|1|0.369794|0.397866|Yes|
|2|0.303397|0.297508|Yes|
|3|0.279277|0.280948|Yes|
|4|0.279772|0.286106|No|

Epoch three was frozen before the application audit. The run reached the
four-epoch cap; epoch four also met the patience rejection criterion. These are
window-level losses with partly weak targets, not end-to-end claim scores.

## Original V1 application audit and conditional validation gate

The audit runs the real pipeline on all 163 reserved TRAIN claims, using a fixed
NLI reader trained on the separate 484 fitting claims. Its frozen earlier baseline
is 118/163 correct: 72.39% accuracy /0.725585 Macro-F1. The new ranker must improve
**both metrics by at least 0.02**, with zero operational errors, to qualify for
any final refit or validation-300 evaluation. Runtime ranking uses raw logits,
verified against the saved activation configuration, with gate zero and top three
documents. Reader cutoffs remain 0.5/0.9. No audit errors choose new thresholds.

If that gate fails, preserve the rejected experiment and stop before validation.
If it passes, refit the ranker from the generic base on 807 eligible TRAIN claims
for the selected three epochs, retaining the original four-epoch learning-rate
horizon. Final validation combines it with the existing stronger 807-TRAIN NLI
reader. This reader transfer is predeclared: the 163-claim audit is not an estimate
of the final combination's accuracy. All 300 validation claims and all three
labels remain included; operational errors count as wrong.

The audit is internal development evidence. Validation 300 is repeatedly reused;
neither is a new blind test or proof that overfitting is absent. The original
application, best profile and historical predictions remain preserved.

Local trained reranker weights, tokenizer, model configuration and training
manifest are now included in evaluation identities. Fitted claim IDs/text hashes
are checked for the ranker as well as the NLI reader. Tests verify that changing
local weights changes the identity and that fitting/evaluation overlap is rejected.
The initial regression suite passed 368 tests; the completed calibrated follow-up
brings this to **375 passing tests**. Ruff passes.

## Reproduce in Anaconda Prompt / CMD

```bat
cd /d "D:\Additional Project\Gen_AI"
set "TEST_PYTHON=C:\Users\Bhargav\anaconda3\python.exe"
set "PYTHONPATH=D:\Additional Project\Gen_AI\tmp\gpu_deps;D:\Additional Project\Gen_AI\tmp\audit_deps"
set "HF_HUB_OFFLINE=1"
"%TEST_PYTHON%" -m scripts.prepare_evidence_rank_protocol
"%TEST_PYTHON%" -m scripts.train_evidence_ranker --stage all
```

Training artifacts: `artifacts/models/evidence_rank_select_v1`. The application
audit and its frozen identity/results go to
`artifacts/improvement/evidence_rank_audit_v1`. Logs are
`artifacts/improvement/evidence_rank_selection_v1.log` and
`artifacts/improvement/evidence_rank_workflow_v1.log`. Changed identities refuse
resume; a changed recipe needs a new version.
