# Evaluation report - run `roberta_scifact_recheck_v1`

- Suite: **title_recheck** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `Scientific NLI verifier` digest `483ab906b890ceacd52fa90f143f53ab8736a9f394616cea0ad2184d414c74f5`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T12:57:27+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_title_full | 100 | 1.000 | 0.730 | 0.729 | 0.771 | 0.732 | 0.684 | 0.676 | 0.824 | - | - | 1.000 | 0.100 | 0.000 | 2.447 | 3.666 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_title_full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 32 | 2 | 7 | 0 |
| CONTRADICT | 1 | 15 | 5 | 0 |
| NEI | 9 | 3 | 26 | 0 |

## Evidence metrics (denominators)

- nli_title_full / evidence_cited: precision 0.515 = 51/99 sentences; recall 0.421 = 51/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 105 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_title_full: 100/100 ok, 0 errors {}, 10 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_title_full | 0 | 0 | - | 100 | 0 | 100 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (27 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_title_full | 338: Dexamethasone decreases risk of postoperative bleeding. | CONTRADICT | NEI | judgement error |
| nli_title_full | 619: Increased vessel density along with a reduction in fibrosis decreases the efficacy of chem | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | NEI | judgement error |
| nli_title_full | 350: Discrimination between the initiator and elongation tRNAs depends on the translation initi | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 759: Mathematical models predict that using Artemisinin-based combination therapy over nongamet | CONTRADICT | SUPPORT | judgement error |
| nli_title_full | 1029: Reduced responsiveness to interleukin-2 in regulatory T cells is associated with greater r | CONTRADICT | NEI | judgement error |
| nli_title_full | 148: Autophagy declines in aged organisms. | SUPPORT | NEI | judgement error |
| nli_title_full | 507: Helminths interfere with immune system control of macrophages activated by IL-4 favor Myco | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1041: Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nu | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 1336: UCB T cells reduce TCR diversity after transplantation. | CONTRADICT | NEI | judgement error |
| nli_title_full | 179: Birth-weight is positively associated with breast cancer. | SUPPORT | NEI | judgement error |
| nli_title_full | 544: IFIT1 restricts viral replication by sequestrating mis-capped viral RNAs. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 208: CHEK2 is not associated with breast cancer. | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 637: Input from  mental and physical health care professionals is effective at decreasing homel | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 236: Cell autonomous sex determination in somatic cells occurs in Passeriformes. | SUPPORT | NEI | judgement error |
| nli_title_full | 727: Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterpart | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 314: Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophi | SUPPORT | NEI | retrieval miss |
| nli_title_full | 775: Mice defective for deoxyribonucleic acid (DNA) polymerase I (polI) reveal increased sensit | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 431: FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS). | SUPPORT | NEI | retrieval miss |
| nli_title_full | 870: Obesity decreases life quality. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 501: Headaches are not correlated with cognitive impairment. | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 913: PPAR-RXRs are inhibited by PPAR ligands. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1337: Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164. | SUPPORT | NEI | judgement error |
| nli_title_full | 1099: Statins decrease blood cholesterol. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1213: The deregulated and prolonged activation of monocytes has deleterious effects in inflammat | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 1278: The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1279: The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune event | NEI | CONTRADICT | judgement error (false decisive verdict) |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
