# Evaluation report - run `nli_large_recheck_v1`

- Suite: **verifier_recheck** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `Scientific NLI verifier` digest `4cb0d9805a8b7799ff57bad758d4089fc829ae3782bd0716101a443834c489d7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-21T23:58:25+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_full | 100 | 1.000 | 0.750 | 0.751 | 0.761 | 0.756 | 0.738 | 0.676 | 0.824 | - | - | 1.000 | 0.000 | 0.000 | 1.547 | 2.196 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 27 | 3 | 11 | 0 |
| CONTRADICT | 0 | 17 | 4 | 0 |
| NEI | 3 | 4 | 31 | 0 |

## Evidence metrics (denominators)

- nli_full / evidence_cited: precision 0.670 = 59/88 sentences; recall 0.488 = 59/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 34 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_full: 100/100 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_full | 0 | 0 | - | 100 | 0 | 100 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (25 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_full | 619: Increased vessel density along with a reduction in fibrosis decreases the efficacy of chem | CONTRADICT | NEI | judgement error |
| nli_full | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | NEI | judgement error |
| nli_full | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | CONTRADICT | judgement error |
| nli_full | 213: CRP is not predictive of postoperative mortality following Coronary Artery Bypass Graft (C | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 1041: Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nu | CONTRADICT | NEI | retrieval miss |
| nli_full | 1336: UCB T cells reduce TCR diversity after transplantation. | CONTRADICT | NEI | judgement error |
| nli_full | 208: CHEK2 is not associated with breast cancer. | SUPPORT | NEI | judgement error |
| nli_full | 507: Helminths interfere with immune system control of macrophages activated by IL-4 favor Myco | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 230: Carriers of the alcohol aldehyde dehydrogenase deficiency mutation drink less that non-car | SUPPORT | CONTRADICT | judgement error |
| nli_full | 727: Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterpart | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 314: Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophi | SUPPORT | NEI | retrieval miss |
| nli_full | 783: Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 431: FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS). | SUPPORT | NEI | retrieval miss |
| nli_full | 1099: Statins decrease blood cholesterol. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 636: Inositol lipid 3-phosphatase PTEN converts Ptdlns(3,4)P 2 into phosphatidylinositol 4-phos | SUPPORT | NEI | judgement error |
| nli_full | 1278: The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 880: Occupancy of ribosomes by IncRNAs mirror 5 0-UTRs | SUPPORT | NEI | judgement error |
| nli_full | 1279: The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune event | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | NEI | judgement error |
| nli_full | 1187: The YAP1 and TEAD complex tanslocates into the nucleus where it interacts with transcripti | SUPPORT | NEI | judgement error |
| nli_full | 1216: The extracellular domain of TMEM27 is cleaved in human beta cells. | SUPPORT | NEI | judgement error |
| nli_full | 1241: The myocardial lineage develops from cardiac progenitors of mesodermal origin. | SUPPORT | NEI | judgement error |
| nli_full | 1274: The tip of the inner tube of the toxic type VI secretion system (T6SS) antibacterial effec | SUPPORT | NEI | judgement error |
| nli_full | 1337: Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164. | SUPPORT | NEI | judgement error |
| nli_full | 1368: Vitamin D deficiency effects the term of delivery. | SUPPORT | CONTRADICT | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
