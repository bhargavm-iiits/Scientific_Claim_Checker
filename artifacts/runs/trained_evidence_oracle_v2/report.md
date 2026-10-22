# Evaluation report - run `trained_evidence_oracle_v2`

- Suite: **trained_oracle** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `Scientific NLI verifier` digest `70f1a6de62293c74712804c58a12ce407b21df4f40cd3107d51978eaa0e5f646`
- Index fingerprint: `None`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T09:18:42+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trained_evidence_oracle | 100 | 1.000 | 0.810 | 0.812 | 0.829 | 0.811 | 0.796 | - | - | 0.900 | 0.446 | 1.000 | 0.000 | 0.000 | 0.132 | 0.261 |

## Confusion matrices (rows = gold, columns = predicted)

**trained_evidence_oracle**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 29 | 0 | 12 | 0 |
| CONTRADICT | 0 | 15 | 6 | 0 |
| NEI | 0 | 1 | 37 | 0 |

## Evidence metrics (denominators)

- trained_evidence_oracle / evidence_selected: precision 0.900 = 54/60 sentences; recall 0.446 = 54/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 4 predicted sentences in unannotated documents (excluded from precision).
- trained_evidence_oracle / evidence_cited: precision 0.936 = 44/47 sentences; recall 0.364 = 44/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 1 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- trained_evidence_oracle: 100/100 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| trained_evidence_oracle | 0 | 0 | - | 0 | - | - | - | n/a (no retrieval) |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (19 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| trained_evidence_oracle | 185: Breast cancer development is determined exclusively by genetic factors. | CONTRADICT | NEI | judgement error |
| trained_evidence_oracle | 513: High cardiopulmonary fitness causes increased mortality rate. | CONTRADICT | NEI | evidence miss |
| trained_evidence_oracle | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | NEI | judgement error |
| trained_evidence_oracle | 783: Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| trained_evidence_oracle | 532: Hyperfibrinogenemia decreases rates of femoropopliteal bypass thrombosis. | CONTRADICT | NEI | evidence miss |
| trained_evidence_oracle | 578: In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis. | CONTRADICT | NEI | judgement error |
| trained_evidence_oracle | 148: Autophagy declines in aged organisms. | SUPPORT | NEI | evidence miss |
| trained_evidence_oracle | 619: Increased vessel density along with a reduction in fibrosis decreases the efficacy of chem | CONTRADICT | NEI | evidence miss |
| trained_evidence_oracle | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | NEI | evidence miss |
| trained_evidence_oracle | 314: Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophi | SUPPORT | NEI | evidence miss |
| trained_evidence_oracle | 436: Free histones are degraded by a Rad53-dependent mechanism once DNA has been replicated. | SUPPORT | NEI | judgement error |
| trained_evidence_oracle | 636: Inositol lipid 3-phosphatase PTEN converts Ptdlns(3,4)P 2 into phosphatidylinositol 4-phos | SUPPORT | NEI | judgement error |
| trained_evidence_oracle | 742: Macrolides have no protective effect against myocardial infarction. | SUPPORT | NEI | evidence miss |
| trained_evidence_oracle | 957: Podocytes are motile and migrate in the presence of injury. | SUPPORT | NEI | evidence miss |
| trained_evidence_oracle | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | NEI | evidence miss |
| trained_evidence_oracle | 1187: The YAP1 and TEAD complex tanslocates into the nucleus where it interacts with transcripti | SUPPORT | NEI | judgement error |
| trained_evidence_oracle | 1241: The myocardial lineage develops from cardiac progenitors of mesodermal origin. | SUPPORT | NEI | evidence miss |
| trained_evidence_oracle | 1274: The tip of the inner tube of the toxic type VI secretion system (T6SS) antibacterial effec | SUPPORT | NEI | judgement error |
| trained_evidence_oracle | 1368: Vitamin D deficiency effects the term of delivery. | SUPPORT | NEI | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
