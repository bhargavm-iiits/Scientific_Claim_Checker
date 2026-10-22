# Evaluation report - run `final_v1`

- Suite: **final** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-20T14:45:42+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| judge_only | 100 | 1.000 | 0.500 | 0.504 | 0.545 | 0.529 | 0.436 | - | - | - | - | - | 0.000 | 0.000 | 1.396 | 1.902 |
| retrieve_judge_single_prompt | 100 | 1.000 | 0.560 | 0.450 | 0.650 | 0.174 | 0.526 | 0.676 | 0.824 | - | - | 1.000 | 0.000 | 0.000 | 4.754 | 10.320 |
| oracle | 100 | 1.000 | 0.670 | 0.571 | 0.706 | 0.240 | 0.767 | - | - | 0.614 | 0.711 | 1.000 | 0.000 | 0.000 | 2.598 | 4.790 |
| full | 100 | 1.000 | 0.560 | 0.487 | 0.638 | 0.296 | 0.526 | 0.676 | 0.824 | 0.632 | 0.653 | 1.000 | 0.030 | 0.000 | 4.831 | 9.416 |
| evidence_off | 100 | 1.000 | 0.530 | 0.446 | 0.639 | 0.320 | 0.377 | 0.676 | 0.824 | - | - | 1.000 | 0.030 | 0.000 | 5.239 | 8.105 |
| rerank_off | 100 | 1.000 | 0.540 | 0.468 | 0.626 | 0.296 | 0.483 | 0.685 | 0.868 | 0.633 | 0.628 | 1.000 | 0.040 | 0.000 | 1.992 | 4.426 |
| full_k1 | 100 | 1.000 | 0.560 | 0.473 | 0.660 | 0.214 | 0.545 | 0.676 | 0.824 | 0.660 | 0.579 | 1.000 | 0.000 | 0.000 | 0.007 | 0.019 |
| full_k5 | 100 | 1.000 | 0.570 | 0.493 | 0.655 | 0.296 | 0.526 | 0.676 | 0.824 | 0.623 | 0.669 | 1.000 | 0.040 | 0.000 | 2.395 | 5.114 |

## Confusion matrices (rows = gold, columns = predicted)

**judge_only**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 24 | 2 | 15 | 0 |
| CONTRADICT | 4 | 9 | 8 | 0 |
| NEI | 19 | 2 | 17 | 0 |

**retrieve_judge_single_prompt**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 39 | 0 | 2 | 0 |
| CONTRADICT | 17 | 2 | 2 | 0 |
| NEI | 23 | 0 | 15 | 0 |

**oracle**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 36 | 1 | 4 | 0 |
| CONTRADICT | 15 | 3 | 3 | 0 |
| NEI | 10 | 0 | 28 | 0 |

**full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 37 | 2 | 2 | 0 |
| CONTRADICT | 15 | 4 | 2 | 0 |
| NEI | 23 | 0 | 15 | 0 |

**evidence_off**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 39 | 0 | 2 | 0 |
| CONTRADICT | 14 | 4 | 3 | 0 |
| NEI | 28 | 0 | 10 | 0 |

**rerank_off**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 36 | 2 | 3 | 0 |
| CONTRADICT | 14 | 4 | 3 | 0 |
| NEI | 24 | 0 | 14 | 0 |

**full_k1**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 35 | 2 | 4 | 0 |
| CONTRADICT | 12 | 3 | 6 | 0 |
| NEI | 18 | 2 | 18 | 0 |

**full_k5**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 38 | 2 | 1 | 0 |
| CONTRADICT | 14 | 4 | 3 | 0 |
| NEI | 23 | 0 | 15 | 0 |

## Evidence metrics (denominators)

- retrieve_judge_single_prompt / evidence_cited: precision 0.793 = 65/82 sentences; recall 0.537 = 65/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 81 predicted sentences in unannotated documents (excluded from precision).
- oracle / evidence_selected: precision 0.614 = 86/140 sentences; recall 0.711 = 86/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 38 predicted sentences in unannotated documents (excluded from precision).
- oracle / evidence_cited: precision 0.811 = 73/90 sentences; recall 0.603 = 73/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 19 predicted sentences in unannotated documents (excluded from precision).
- full / evidence_selected: precision 0.632 = 79/125 sentences; recall 0.653 = 79/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 259 predicted sentences in unannotated documents (excluded from precision).
- full / evidence_cited: precision 0.825 = 66/80 sentences; recall 0.545 = 66/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 128 predicted sentences in unannotated documents (excluded from precision).
- evidence_off / evidence_cited: precision 0.740 = 71/96 sentences; recall 0.587 = 71/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 167 predicted sentences in unannotated documents (excluded from precision).
- rerank_off / evidence_selected: precision 0.633 = 76/120 sentences; recall 0.628 = 76/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 264 predicted sentences in unannotated documents (excluded from precision).
- rerank_off / evidence_cited: precision 0.815 = 66/81 sentences; recall 0.545 = 66/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 124 predicted sentences in unannotated documents (excluded from precision).
- full_k1 / evidence_selected: precision 0.660 = 70/106 sentences; recall 0.579 = 70/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 68 predicted sentences in unannotated documents (excluded from precision).
- full_k1 / evidence_cited: precision 0.855 = 59/69 sentences; recall 0.488 = 59/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 42 predicted sentences in unannotated documents (excluded from precision).
- full_k5 / evidence_selected: precision 0.623 = 81/130 sentences; recall 0.669 = 81/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 361 predicted sentences in unannotated documents (excluded from precision).
- full_k5 / evidence_cited: precision 0.800 = 68/85 sentences; recall 0.562 = 68/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 163 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- judge_only: 100/100 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.
- retrieve_judge_single_prompt: 100/100 ok, 0 errors {}, 0 conflicts, 4 claims with a downgraded judgement (decisive answer without a valid citation -> NEI; claims 350, 437, 793, 852).
- oracle: 100/100 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.
- full: 100/100 ok, 0 errors {}, 3 conflicts, 0 claims with a downgraded judgement.
- evidence_off: 100/100 ok, 0 errors {}, 3 conflicts, 0 claims with a downgraded judgement.
- rerank_off: 100/100 ok, 0 errors {}, 4 conflicts, 0 claims with a downgraded judgement.
- full_k1: 100/100 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.
- full_k5: 100/100 ok, 0 errors {}, 4 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| judge_only | 100 | 0 | 0 % | 0 | - | - | - | n/a (no retrieval) |
| retrieve_judge_single_prompt | 106 | 0 | 0 % | 100 | 0 | 0 | 100 | - |
| oracle | 195 | 0 | 0 % | 0 | - | - | - | n/a (no retrieval) |
| full | 325 | 137 | 30 % | 100 | 0 | 0 | 100 | - |
| evidence_off | 300 | 0 | 0 % | 100 | 0 | 0 | 100 | - |
| rerank_off | 143 | 317 | 69 % | 100 | 0 | 0 | 100 | - |
| full_k1 | 0 | 177 | 100 % | 100 | 0 | 0 | 100 | - |
| full_k5 | 179 | 528 | 75 % | 100 | 0 | 0 | 100 | - |

- **Caveat (retrieval cache):** 400 claim(s) without a cache flag have `retrieve.*` sub-timings larger than their own `retrieve` stage, i.e. copied from an earlier cached computation (full 100, evidence_off 100, full_k1 100, full_k5 100). For these claims `retrieve` is the near-zero lookup actually measured and `total` latency excludes the search (about one uncached retrieval per claim); compare latency only between variants with the same cache state.
- Retrieval-cache status is unknown for 600 claim(s) recorded before the `retrieval_cache_hit` flag existed.
- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (44 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| full | 142: Autologous transplantation of mesenchymal stem cells causes a higher rate of opportunistic | CONTRADICT | NEI | evidence miss |
| full | 180: Blocking the interaction between TDP-43 and respiratory complex I proteins ND3 and ND6 lea | CONTRADICT | SUPPORT | judgement error |
| full | 208: CHEK2 is not associated with breast cancer. | SUPPORT | NEI | evidence miss |
| full | 1: 0-dimensional biomaterials show inductive properties. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 249: Chenodeosycholic acid treatment reduces whole-body energy expenditure. | CONTRADICT | SUPPORT | judgement error |
| full | 338: Dexamethasone decreases risk of postoperative bleeding. | CONTRADICT | SUPPORT | judgement error |
| full | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | CONTRADICT | evidence miss |
| full | 36: A deficiency of vitamin B12 increases blood levels of homocysteine. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 452: Gene expression does not vary appreciably across genetically identical cells. | CONTRADICT | SUPPORT | evidence miss |
| full | 516: High levels of CRP reduces the risk of exacerbations in chronic obstructive pulmonary dise | CONTRADICT | SUPPORT | judgement error |
| full | 1298: Thigh-length graduated compression stockings (GCS) did not reduce deep vein thrombosis in  | SUPPORT | CONTRADICT | judgement error |
| full | 237: Cells lacking clpC have a defect in sporulation efficiency in Bacillus subtilis. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 532: Hyperfibrinogenemia decreases rates of femoropopliteal bypass thrombosis. | CONTRADICT | SUPPORT | judgement error |
| full | 578: In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis. | CONTRADICT | SUPPORT | judgement error |
| full | 1368: Vitamin D deficiency effects the term of delivery. | SUPPORT | NEI | judgement error |
| full | 268: Cold exposure increases BAT recruitment. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 619: Increased vessel density along with a reduction in fibrosis decreases the efficacy of chem | CONTRADICT | SUPPORT | judgement error |
| full | 728: Ly6C hi monocytes have a lower inflammatory capacity than Ly6C lo monocytes. | CONTRADICT | SUPPORT | judgement error |
| full | 303: DMRT1 is a sex-determining gene that is epigenetically regulated by the MHM region. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 759: Mathematical models predict that using Artemisinin-based combination therapy over nongamet | CONTRADICT | SUPPORT | judgement error |
| full | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | SUPPORT | evidence miss |
| full | 350: Discrimination between the initiator and elongation tRNAs depends on the translation initi | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 903: PD-1 triggering on monocytes reduces IL-10 production by monocytes. | CONTRADICT | SUPPORT | judgement error |
| full | 922: Patients in stable partnerships have a faster progression from HIV to AIDS. | CONTRADICT | SUPPORT | judgement error |
| full | 437: Functional consequences of genomic alterations due to Myelodysplastic syndrome (MDS) are p | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1029: Reduced responsiveness to interleukin-2 in regulatory T cells is associated with greater r | CONTRADICT | NEI | judgement error |
| full | 1041: Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nu | CONTRADICT | SUPPORT | retrieval miss |
| full | 507: Helminths interfere with immune system control of macrophages activated by IL-4 favor Myco | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1336: UCB T cells reduce TCR diversity after transplantation. | CONTRADICT | SUPPORT | judgement error |
| full | 551: ITAM phosphorylation prevents the transfer of the T cell receptor (TCR) signal from the ec | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 552: IgA plasma cells that are specific for transglutaminase 2 accumulate in the duodenal mucos | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 637: Input from  mental and physical health care professionals is effective at decreasing homel | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 727: Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterpart | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 768: Mercaptopurine is anabolized into the inactive methylmercaptopurine by thiopurine methyltr | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 870: Obesity decreases life quality. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 913: PPAR-RXRs are inhibited by PPAR ligands. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1099: Statins decrease blood cholesterol. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1175: The PPR MDA5 has two N-terminal CARD domains. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1197: The availability of safe places to study is not effective at decreasing homelessness. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1207: The composition of myosin-II isoform switches from the polarizable B isoform to the more h | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1213: The deregulated and prolonged activation of monocytes has deleterious effects in inflammat | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1278: The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1279: The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune event | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1282: Therapeutic use of the drug Dapsone to treat pyoderma gangrenous is based on anecdotal evi | NEI | SUPPORT | judgement error (false decisive verdict) |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
