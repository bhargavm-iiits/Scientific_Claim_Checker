# P08 brief review and new reasoning experiment

The supplied four-page PDF was read in full on 22 October 2026. It asks for a
scientific claim checker with retrieval, evidence selection, judgment, supervision
and quotations. It does **not** prescribe a 95% accuracy or Macro-F1 threshold.
The user's stricter target remains **accuracy >95% AND Macro-F1 >0.95 on the same
300 SciFact validation claims**. Best confirmed so far: **242/300, 80.67% /0.795974**.

## What the brief requires and where it is implemented

| PDF requirement | Existing project implementation / evidence | Assessment |
|---|---|---|
| Index 5,183 downloaded abstracts; join claims by abstract ID | `src/claim_checker/data/`, `data/raw/source_manifest.json`, `artifacts/index/` | Implemented; original sentence boundaries and source IDs are retained. |
| Notebook starts with `load_dataset(...)`; fixed seed and stated subset size | `notebooks/Scientific_Claim_Checker.ipynb` data-acquisition and split sections | Present. The unsupported claims-builder route is recorded with an official-archive/JSON-loader fallback. A fresh end-to-end notebook rerun is not claimed in this review. |
| BM25, dense, hybrid, reranking; Recall@3/@10 | `src/claim_checker/retrieval/`, saved retrieval results and notebook comparisons | Implemented. A reader improvement does not itself establish retrieval improvement. |
| Multi-agent pipeline with supervisor and source quotations | `pipeline/graph.py`, `agents/evidence.py`, `agents/judge.py`, `pipeline/supervisor.py` | Original Ollama product retains these stages. Quotes are constructed from stored sentences, not generated text. |
| Oracle versus retrieved judgment; accuracy and Macro-F1 | `configs/experiments.yaml`, `evaluation/runner.py`, saved evaluations | Implemented. Oracle and retrieved populations/results must remain separate. |
| Evidence precision/recall; CONTRADICT error analysis | `evaluation/metrics.py`, notebook and accuracy/overfitting reports | Implemented. Unannotated papers are not automatically false evidence. |
| Baseline, full pipeline, one ablation on the same fixed subset; at least ten failures | `final` suite includes claim-only, single prompt, full pipeline, evidence off, reranker off and k=1/5; notebook failure section | Present in the original evaluation; the newer NLI alternative must not replace the required comparisons without explanation. |
| Local/open small model, laptop CPU or free Colab T4; no paid API required | Original Qwen3 4B Ollama product; separate local NLI/LoRA alternatives | Implemented locally. Actual runtime and memory must be measured for each new candidate. |
| One notebook, 10-minute talk and 5-minute Q&A; adapt and cite reused labs | Existing notebook includes Module 2/3/4 topic references and `Docs/LAB_REUSE.md` | Notebook is present and reports an independent original build. Module 2 and 3 labs were located and reviewed in Downloads during this review; actual code adaptation and updated exact notebook citations remain outstanding. Module 4 was not located. A delivered talk or rehearsal is not established by source-code review. |
| Optional LoRA and Climate-FEVER stretch | LoRA experiment exists; general-science collection extends another scope | Stretch work is separate from core completion. General-science smoke results are not Climate-FEVER results. |

## Why change the reader

One coursework gap is independent of accuracy: the brief asks students to adapt
the actual course labs. The notebook currently provides an honest conceptual
mapping, not evidence of that adaptation. Module 2's structured-generation lab
was located under Downloads/GEN AI and checked at section 4.1, cells 36-37;
Module 3's RAG lab was checked at cells 14 (BM25), 16 (RRF), 18 (cross-encoder),
22 (LangGraph) and 24 (evaluation). These are one-based notebook cell positions.
Reviewing them now does not establish that the original implementation adapted
their code. Higher scores cannot close this gap. Exact local sources are recorded
in `Docs/LAB_REUSE.md`.

The preserved 300-claim baseline error audit contains 58 wrong verdicts. Of these,
35 decisive claims were wrong even after an annotated evidence paper was examined;
10 missed annotated evidence in the top three, three lost it at the relevance gate,
and ten gold-NEI claims received decisive verdicts. This is an aggregate reused
validation diagnostic, not a new training set or a source of prompt examples.
It suggests that retrieval-only experiments cannot address most current errors.

Earlier experiments also showed that prolonged fine-tuning can worsen held-out
performance while training loss falls. The trained reranker and its independent
calibration follow-up both failed their 163-claim TRAIN audit. Those artifacts and
the stronger original profile remain preserved.

## New approach: native reasoning with per-paper grounding

This candidate uses `qwen3.5:4b` through local Ollama, with **no project fine-tuning,
no in-context SciFact demonstrations and no threshold search**. Its public sources
are the [Ollama model page](https://ollama.com/library/qwen3.5:4b) and the
[author model card](https://huggingface.co/Qwen/Qwen3.5-4B). Their general benchmark
results do not establish accuracy on this project. Actual local model digest,
quantization, configuration and server version are frozen in the experiment plan.

The downloaded artifact is 3,324,173,934 bytes, Q4_K_M, with digest
`d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.
Ollama reports **4.2B parameters despite the 4b tag**. This slightly exceeds the
brief's literal four-billion-parameter limit and must be disclosed as an
experimental extension. The original Qwen3 4B coursework product is retained.

The candidate retrieves the same hybrid/reranked top ten papers and examines the
same top three after the fixed -2 relevance gate. One native-reasoning call sees
these source papers and returns a separate structured comparison for each. It
compares the intervention, outcome, scope, direction and quantities before making
a decision. This combines evidence selection and judgment in a joint reader;
it is an **experimental architectural alternative**, not the original separate
evidence-agent implementation required for the core coursework comparison.

Code requires exactly one output per supplied paper, validates zero-based sentence
IDs and requires citations for decisive paper judgments. Invalid output receives
the existing bounded repair; persistent invalid output is an operational error,
not a scientific NEI. Quotations are copied from stored sentences. The existing
deterministic aggregator maps opposing paper judgments to NEI with a conflict flag.
Model confidence does not control labels. Native thinking uses Ollama's supported
interface, without a hand-written Qwen3 chat-template continuation.

The generation budget is 2,048 tokens in a 16,384-token context, temperature zero,
seed 42, one concurrent request. If thinking exhausts its output budget without an
answer, the client performs its pre-existing bounded no-thinking fallback and
records that fact. There is no silent source truncation or dropping of hard claims.

## Evaluation and overfitting controls

The synthetic runtime preflight exhausted the native thinking budget and required
the no-thinking fallback. Before the long 163-claim audit, an additional frozen
viability screen uses 20 stratified TRAIN stopping claims (seed 2026), disjoint
from the 163 audit IDs. It must score at least 0.80 on both metrics with zero
errors to advance. Its plan is `reasoned_reader_screen_protocol_v1.json` and
outputs are `reasoned_reader_screen20_audit_v1`. It is a small TRAIN screen,
**not a replacement for validation 300**. No prompt/settings are changed after
screen predictions begin.

The prompt, code, model digest, index and populations are frozen before prediction.
First evaluate all 163 claims from the existing TRAIN-only audit. It is reused
development evidence, not a fresh blind test. The candidate can proceed to one
unchanged 300-claim confirmation only if audit accuracy and Macro-F1 are both at
least 0.80 with zero operational errors. This gate tests basic viability against
the earlier grouped-reader audit (72.39% /0.725585); it does not establish that the
candidate beats the stronger 807-TRAIN model on validation.

All three classes and every requested claim remain in the denominator. Errors
count as wrong. No validation labels enter prompts, fitting, examples or parameter
selection. No project fine-tuning avoids additional weight-fitting overfit, but
does not prove that the pretrained model never encountered public SciFact text.
Repeatedly reused evaluation data also prevents a claim of a fresh blind result.

The requested 95% target remains unproven until measured: at least 286/300 correct
are necessary for strict accuracy >95%, and Macro-F1 must independently exceed
0.95. Rejected or incomplete experiments do not replace the best profile.

Source: `src/claim_checker/agents/reasoned_reader.py`; prompt:
`prompts/reasoned_reader_v1.txt`; runner: `scripts/run_reasoned_reader.py`.
Frozen plan: `artifacts/improvement/reasoned_reader_protocol_v1.json`.
Audit outputs: `artifacts/improvement/reasoned_reader_audit_v1`.

Validation of the implementation: **398 regression tests passed**, including 23
new reader/adapter and polarity tests. Ruff passed. These are software checks, not measurements
of scientific claim accuracy.

## Second fixed approach: polarity consistency

`scripts/run_polarity_consistency.py` freezes a separate training-free experiment.
For each source premise P, score the original claim H and the full-scope negation
"It is not the case that H" using the same NLI weights. Swap SUPPORT/CONTRADICT
for the negated hypothesis, leave NEI unchanged, then take an equal-weight
geometric mean and normalize. There is no learned mixing weight. Probabilities
are floored at 1e-9 only to handle softmax underflow.

The paired control uses exactly the same model and source windows, budgeted for
the longer hypothesis, but does not pool scores. Thresholds stay at SUPPORT 0.5,
CONTRADICT 0.9; retrieval stays top three with gate -2. This isolates the pooling
step from a change in weights or source coverage. Ordinary-score caching reuses
identical inputs across modes; it never caches a pooled score as an ordinary one.

The same fixed 20 TRAIN stopping claims provide an initial screen: neither metric
may decline against the paired control, and both systems must have zero errors.
Only then may the full 163-claim TRAIN audit run; both metrics must improve by at
least 0.02 before any validation-300 run. The audit reader is the existing model
fitted on 484 separate TRAIN claims. If the audit passes, the unchanged formula
transfers to the stronger existing 807-TRAIN reader for one 300-claim confirmation.
That transfer is predeclared, and the 163-claim score is not an estimate of the
stronger combination. Original best-validation predictions remain preserved.

The plan is `artifacts/improvement/polarity_consistency_protocol_v1.json`. Ten
focused tests verify negation scope, label-swap symmetry, finite normalized scores,
shared input windows and cache separation. A model can still misunderstand a
negation; the formula provides no scientific-truth or accuracy guarantee.

## Completed native-reader screen

The fixed 20-claim TRAIN screen finished at **13/20 correct, 65.00% accuracy and
0.510379 Macro-F1**, with two operational errors. One error was invalid JSON after
bounded repair; another was a RAM-allocation failure while a CPU NLI screen ran
concurrently. That concurrency was a mistake: separate processors did not provide
enough total system memory. The CPU experiment was interrupted and its partial
artifacts preserved. Subsequent model jobs run sequentially.

The native reader fails the 0.80 viability gate even if both operational failures
are replaced with correct predictions: 15/20 =75%. Replaying unchanged successful
cached outputs and retrying only the two failures therefore cannot qualify it.
No isolated native repeat, full 163-claim audit or validation-300 run was started.
The native model was unloaded before the GPU polarity experiment. No prompt or
settings were revised using these errors. This is a rejected small TRAIN screen,
not a new score on the requested 300 claims.

Reports: `artifacts/improvement/reasoned_reader_screen_decision_v1.json` and
`reasoned_reader_screen20_audit_v1/metrics.json`. The interrupted CPU polarity
screen is preserved in `polarity_screen_v1`. Its isolated GPU repeat has its own
folder `polarity_screen_isolated_gpu_v1` and frozen runtime protocol
`isolated_polarity_protocol_v1.json`. Native and GPU-NLI dependency versions are
recorded separately. The optional isolation runner contains a native-repeat branch
for reproducibility, but that branch was not executed because of the bound above.

## Polarity screen and paired checks

The isolated 20-claim GPU screen passed its predeclared non-regression gate:
control **75.00% /0.714986**, pooled candidate **75.00% /0.732026**, with zero
operational errors. Both systems got exactly the same 15 claims correct. The
Macro-F1 change came from a different distribution of wrong predictions, not an
accuracy gain. Paired bootstrapping of the 17 reference-paper/text groups gives
an accuracy-difference interval of [0,0] and a Macro-F1-difference interval of
[-0.029855,+0.087934]. This small screen does not establish an improvement.

All twelve available verdict comparisons from the six-claim interrupted CPU
prefix agree with GPU results. That prefix is not a random parity sample or a
complete CPU benchmark. The 163-claim audit is still required. Paired control and
candidate runs reuse retrieval and ordinary-score caches; their saved timings
must not be presented as standalone cold-latency measurements.

## Completed polarity audit: rejected

| Paired 163-claim TRAIN audit | Correct | Accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|---:|
| Same-weight, same-window control |118|72.3926%|0.725585|0|
| Polarity pooling |116|71.1656%|0.709147|0|

The candidate failed the predeclared two-metric improvement gate. There was **no
validation-300 run, weight refit, cutoff revision or product promotion**. All
193 candidate quotations matched stored source sentences exactly (control
219/219). Eight claims changed from correct to wrong and six from wrong to
correct. A paired bootstrap over 103 paper/text groups gives a 95% interval of
[-6.2893,+3.8462] percentage points for the accuracy difference and
[-0.067785,+0.034331] for Macro-F1. The audit does not establish improvement.

Exact outputs: `artifacts/improvement/polarity_audit_v1/metrics.json` and
`paired_quality.json`. All model jobs are finished. The best original 300-claim
result remains **242/300, 80.67% /0.795974**. Both requested >95% targets are
still unmet. The new experiments add no weight fitting, but their failed results
and reused populations do not establish that overfitting is absent.

## Reproduction in Anaconda Prompt / CMD

A later [joint verdict/rationale reader](JOINT_READER_EXPERIMENT.md) also
completed its fixed TRAIN audit. It improved accuracy to 74.85% and Macro-F1
to 0.743697 on the same 163 claims, but missed the minimum Macro-F1 gain and
worsened evidence relevance. It was rejected before refitting or validation
300. The best original-300 result remains 80.67% /0.795974.

```bat
cd /d "D:\Additional Project\Gen_AI"
set "PYTHONPATH=D:\Additional Project\Gen_AI\tmp\audit_deps"
set "HF_HUB_OFFLINE=1"
"C:\Users\Bhargav\anaconda3\python.exe" -m scripts.screen_reasoned_reader
```

The model must already be downloaded with `ollama pull qwen3.5:4b`. The screen's
recorded failure stops this candidate; do not bypass it to run validation.

For the completed, isolated polarity workflow, unload this experiment's Qwen
model and run the next command in a separate process:

```bat
set "PYTHONPATH=D:\Additional Project\Gen_AI\tmp\gpu_deps;D:\Additional Project\Gen_AI\tmp\audit_deps"
"C:\Users\Bhargav\anaconda3\python.exe" -m scripts.run_isolated_reader_checks --stage polarity
"C:\Users\Bhargav\anaconda3\python.exe" -m scripts.summarize_polarity --stage audit
```

All model jobs must run sequentially on this laptop, including CPU jobs. The
isolation workflow resumes the completed fixed screen/audit and stops at the
failed gate. Changed frozen identities refuse resume; changing a recipe requires
a new named experiment rather than overwriting old predictions.
