# Evaluation report - run `nli_titlefamily_oracle_v1`

- Suite: **title_oracle** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `Scientific NLI verifier` digest `70f1a6de62293c74712804c58a12ce407b21df4f40cd3107d51978eaa0e5f646`
- Index fingerprint: `None`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T05:31:15+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_title_oracle | 100 | 1.000 | 0.810 | 0.796 | 0.865 | 0.727 | 0.796 | - | - | - | - | 1.000 | 0.020 | 0.000 | 0.170 | 0.392 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_title_oracle**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 32 | 0 | 9 | 0 |
| CONTRADICT | 0 | 12 | 9 | 0 |
| NEI | 1 | 0 | 37 | 0 |

## Evidence metrics (denominators)

- nli_title_oracle / evidence_cited: precision 0.558 = 53/95 sentences; recall 0.438 = 53/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 2 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_title_oracle: 100/100 ok, 0 errors {}, 2 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_title_oracle | 0 | 0 | - | 0 | - | - | - | n/a (no retrieval) |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (19 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_title_oracle | 185: Breast cancer development is determined exclusively by genetic factors. | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 338: Dexamethasone decreases risk of postoperative bleeding. | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1099: Statins decrease blood cholesterol. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_oracle | 513: High cardiopulmonary fitness causes increased mortality rate. | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 578: In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis. | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 431: FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS). | SUPPORT | NEI | judgement error |
| nli_title_oracle | 619: Increased vessel density along with a reduction in fibrosis decreases the efficacy of chem | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 759: Mathematical models predict that using Artemisinin-based combination therapy over nongamet | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 636: Inositol lipid 3-phosphatase PTEN converts Ptdlns(3,4)P 2 into phosphatidylinositol 4-phos | SUPPORT | NEI | judgement error |
| nli_title_oracle | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 1041: Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nu | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1336: UCB T cells reduce TCR diversity after transplantation. | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 1187: The YAP1 and TEAD complex tanslocates into the nucleus where it interacts with transcripti | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1216: The extracellular domain of TMEM27 is cleaved in human beta cells. | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1241: The myocardial lineage develops from cardiac progenitors of mesodermal origin. | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1337: Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164. | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1368: Vitamin D deficiency effects the term of delivery. | SUPPORT | NEI | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
