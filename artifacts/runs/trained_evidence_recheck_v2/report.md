# Evaluation report - run `trained_evidence_recheck_v2`

- Suite: **trained_recheck** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `Scientific NLI verifier` digest `70f1a6de62293c74712804c58a12ce407b21df4f40cd3107d51978eaa0e5f646`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T09:14:49+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trained_evidence | 100 | 1.000 | 0.740 | 0.751 | 0.763 | 0.800 | 0.690 | 0.676 | 0.824 | 0.889 | 0.397 | 1.000 | 0.000 | 0.000 | 2.280 | 3.537 |

## Confusion matrices (rows = gold, columns = predicted)

**trained_evidence**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 29 | 0 | 12 | 0 |
| CONTRADICT | 0 | 16 | 5 | 0 |
| NEI | 6 | 3 | 29 | 0 |

## Evidence metrics (denominators)

- trained_evidence / evidence_selected: precision 0.889 = 48/54 sentences; recall 0.397 = 48/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 76 predicted sentences in unannotated documents (excluded from precision).
- trained_evidence / evidence_cited: precision 0.927 = 38/41 sentences; recall 0.314 = 38/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 29 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- trained_evidence: 100/100 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| trained_evidence | 0 | 0 | - | 100 | 0 | 100 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (26 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| trained_evidence | 532: Hyperfibrinogenemia decreases rates of femoropopliteal bypass thrombosis. | CONTRADICT | NEI | evidence miss |
| trained_evidence | 578: In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis. | CONTRADICT | NEI | judgement error |
| trained_evidence | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | NEI | judgement error |
| trained_evidence | 213: CRP is not predictive of postoperative mortality following Coronary Artery Bypass Graft (C | NEI | CONTRADICT | judgement error (false decisive verdict) |
| trained_evidence | 619: Increased vessel density along with a reduction in fibrosis decreases the efficacy of chem | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | NEI | evidence miss |
| trained_evidence | 314: Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophi | SUPPORT | NEI | retrieval miss |
| trained_evidence | 507: Helminths interfere with immune system control of macrophages activated by IL-4 favor Myco | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 1041: Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nu | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 431: FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS). | SUPPORT | NEI | retrieval miss |
| trained_evidence | 727: Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterpart | NEI | CONTRADICT | judgement error (false decisive verdict) |
| trained_evidence | 436: Free histones are degraded by a Rad53-dependent mechanism once DNA has been replicated. | SUPPORT | NEI | judgement error |
| trained_evidence | 775: Mice defective for deoxyribonucleic acid (DNA) polymerase I (polI) reveal increased sensit | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 636: Inositol lipid 3-phosphatase PTEN converts Ptdlns(3,4)P 2 into phosphatidylinositol 4-phos | SUPPORT | NEI | judgement error |
| trained_evidence | 783: Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| trained_evidence | 742: Macrolides have no protective effect against myocardial infarction. | SUPPORT | NEI | evidence miss |
| trained_evidence | 870: Obesity decreases life quality. | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 957: Podocytes are motile and migrate in the presence of injury. | SUPPORT | NEI | evidence miss |
| trained_evidence | 1099: Statins decrease blood cholesterol. | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | NEI | evidence miss |
| trained_evidence | 1213: The deregulated and prolonged activation of monocytes has deleterious effects in inflammat | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 1187: The YAP1 and TEAD complex tanslocates into the nucleus where it interacts with transcripti | SUPPORT | NEI | judgement error |
| trained_evidence | 1279: The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune event | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 1241: The myocardial lineage develops from cardiac progenitors of mesodermal origin. | SUPPORT | NEI | evidence miss |
| trained_evidence | 1274: The tip of the inner tube of the toxic type VI secretion system (T6SS) antibacterial effec | SUPPORT | NEI | judgement error |
| trained_evidence | 1368: Vitamin D deficiency effects the term of delivery. | SUPPORT | NEI | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
