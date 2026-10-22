# Evaluation report - run `selection_evidence_recheck_v1`

- Suite: **selection_recheck** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `Scientific NLI verifier` digest `37ec9d414d82188543815463770880d4005d1eb8763b47b30a1c2fe9505fea7e`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T09:27:42+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| selection_evidence | 100 | 1.000 | 0.720 | 0.728 | 0.740 | 0.762 | 0.682 | 0.676 | 0.824 | 0.882 | 0.372 | 1.000 | 0.020 | 0.000 | 15.524 | 5.105 |

## Confusion matrices (rows = gold, columns = predicted)

**selection_evidence**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 27 | 1 | 13 | 0 |
| CONTRADICT | 0 | 16 | 5 | 0 |
| NEI | 5 | 4 | 29 | 0 |

## Evidence metrics (denominators)

- selection_evidence / evidence_selected: precision 0.882 = 45/51 sentences; recall 0.372 = 45/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 101 predicted sentences in unannotated documents (excluded from precision).
- selection_evidence / evidence_cited: precision 0.900 = 36/40 sentences; recall 0.298 = 36/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 34 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- selection_evidence: 100/100 ok, 0 errors {}, 2 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| selection_evidence | 0 | 0 | - | 100 | 0 | 100 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (28 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| selection_evidence | 532: Hyperfibrinogenemia decreases rates of femoropopliteal bypass thrombosis. | CONTRADICT | NEI | evidence miss |
| selection_evidence | 619: Increased vessel density along with a reduction in fibrosis decreases the efficacy of chem | CONTRADICT | NEI | retrieval miss |
| selection_evidence | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | NEI | judgement error |
| selection_evidence | 213: CRP is not predictive of postoperative mortality following Coronary Artery Bypass Graft (C | NEI | CONTRADICT | judgement error (false decisive verdict) |
| selection_evidence | 759: Mathematical models predict that using Artemisinin-based combination therapy over nongamet | CONTRADICT | NEI | judgement error |
| selection_evidence | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | NEI | evidence miss |
| selection_evidence | 314: Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophi | SUPPORT | NEI | retrieval miss |
| selection_evidence | 507: Helminths interfere with immune system control of macrophages activated by IL-4 favor Myco | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 1041: Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nu | CONTRADICT | NEI | retrieval miss |
| selection_evidence | 431: FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS). | SUPPORT | NEI | retrieval miss |
| selection_evidence | 644: Insulin increases risk of severe kidney failure. | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 436: Free histones are degraded by a Rad53-dependent mechanism once DNA has been replicated. | SUPPORT | NEI | judgement error |
| selection_evidence | 727: Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterpart | NEI | CONTRADICT | judgement error (false decisive verdict) |
| selection_evidence | 636: Inositol lipid 3-phosphatase PTEN converts Ptdlns(3,4)P 2 into phosphatidylinositol 4-phos | SUPPORT | NEI | judgement error |
| selection_evidence | 775: Mice defective for deoxyribonucleic acid (DNA) polymerase I (polI) reveal increased sensit | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 742: Macrolides have no protective effect against myocardial infarction. | SUPPORT | CONTRADICT | evidence miss |
| selection_evidence | 783: Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| selection_evidence | 880: Occupancy of ribosomes by IncRNAs mirror 5 0-UTRs | SUPPORT | NEI | evidence miss |
| selection_evidence | 870: Obesity decreases life quality. | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 957: Podocytes are motile and migrate in the presence of injury. | SUPPORT | NEI | evidence miss |
| selection_evidence | 913: PPAR-RXRs are inhibited by PPAR ligands. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| selection_evidence | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | NEI | evidence miss |
| selection_evidence | 1278: The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 1187: The YAP1 and TEAD complex tanslocates into the nucleus where it interacts with transcripti | SUPPORT | NEI | evidence miss |
| selection_evidence | 1241: The myocardial lineage develops from cardiac progenitors of mesodermal origin. | SUPPORT | NEI | evidence miss |
| selection_evidence | 1274: The tip of the inner tube of the toxic type VI secretion system (T6SS) antibacterial effec | SUPPORT | NEI | judgement error |
| selection_evidence | 1337: Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164. | SUPPORT | NEI | judgement error |
| selection_evidence | 1368: Vitamin D deficiency effects the term of delivery. | SUPPORT | NEI | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
