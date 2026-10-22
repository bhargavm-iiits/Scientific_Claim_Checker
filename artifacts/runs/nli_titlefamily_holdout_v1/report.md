# Evaluation report - run `nli_titlefamily_holdout_v1`

- Suite: **title_holdout** (split `accuracy_holdout`), claims requested per variant: **200**
- Gold label counts: {'SUPPORT': 83, 'CONTRADICT': 43, 'NEI': 74}
- Model: `Scientific NLI verifier` digest `70f1a6de62293c74712804c58a12ce407b21df4f40cd3107d51978eaa0e5f646`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T05:22:43+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_title_full | 200 | 1.000 | 0.835 | 0.822 | 0.887 | 0.753 | 0.826 | 0.723 | 0.849 | - | - | 1.000 | 0.015 | 0.000 | 1.753 | 2.574 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_title_full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 67 | 3 | 13 | 0 |
| CONTRADICT | 0 | 29 | 14 | 0 |
| NEI | 1 | 2 | 71 | 0 |

## Evidence metrics (denominators)

- nli_title_full / evidence_cited: precision 0.594 = 107/180 sentences; recall 0.437 = 107/245 sentences (200 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 45 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_title_full: 200/200 ok, 0 errors {}, 3 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_title_full | 0 | 0 | - | 200 | 0 | 200 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (33 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_title_full | 48: A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection. | CONTRADICT | NEI | judgement error |
| nli_title_full | 54: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in  | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 3: 1,000 genomes project enables mapping of genetic sequence variation consisting of rare var | SUPPORT | NEI | judgement error |
| nli_title_full | 354: Downregulation and mislocalization of Scribble prevents cell transformation and mammary tu | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 115: Anthrax spores can be disposed of easily after they are dispersed. | CONTRADICT | NEI | judgement error |
| nli_title_full | 183: Bone marrow cells contribute to adult macrophage compartments. | CONTRADICT | NEI | judgement error |
| nli_title_full | 70: Activation of PPM1D suppresses p53 function. | SUPPORT | NEI | retrieval miss |
| nli_title_full | 907: PGE 2 promotes intestinal tumor growth by altering the expression of tumor suppressing and | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 274: Combination nicotine replacement therapies with varenicline or bupropion lead to significa | CONTRADICT | NEI | judgement error |
| nli_title_full | 718: Low nucleosome occupancy correlates with low methylation levels across species. | CONTRADICT | NEI | judgement error |
| nli_title_full | 118: Antibiotic induced alterations in the gut microbiome reduce resistance against Clostridium | SUPPORT | NEI | judgement error |
| nli_title_full | 1100: Statins increase blood cholesterol. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 847: New drugs for tuberculosis often do not penetrate the necrotic portion of a tuberculosis l | CONTRADICT | NEI | judgement error |
| nli_title_full | 879: Occupancy of ribosomes by IncRNAs do not make functional peptides. | CONTRADICT | NEI | judgement error |
| nli_title_full | 239: Cellular aging closely links to an older appearance. | SUPPORT | NEI | retrieval miss |
| nli_title_full | 1021: Rapid up-regulation and higher basal expression of interferon-induced genes reduce surviva | CONTRADICT | NEI | judgement error |
| nli_title_full | 1049: Ribosomopathies have a low degree of cell and tissue specific pathology. | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 343: Diabetic patients with acute coronary syndrome experience increased short-term and long-te | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 1110: Suboptimal nutrition is not predictive of chronic disease | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 1221: The genomic aberrations found in matasteses are very similar to those found in the primary | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 475: Glycolysis is one of the primary glycometabolic pathways in cells. | SUPPORT | NEI | retrieval miss |
| nli_title_full | 1303: Tirasemtiv has no effect on fast-twitch muscle. | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 1320: Transplanted human glial progenitor cells are incapable of forming a neural network with h | CONTRADICT | NEI | judgement error |
| nli_title_full | 593: Incidence of heart failure decreased by 10% in women since 1979. | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 808: Most termination events in Okazaki fragments are sequence specific. | SUPPORT | NEI | judgement error |
| nli_title_full | 967: Pretreatment with the Arp2/3 inhibitor CK-666 affects lamelliopodia formation. | SUPPORT | NEI | judgement error |
| nli_title_full | 982: Proteins synthesized at the growth cone are ubiquitinated at a higher rate than proteins f | SUPPORT | NEI | judgement error |
| nli_title_full | 1019: Rapid phosphotransfer rates govern fidelity in two component systems | SUPPORT | NEI | judgement error |
| nli_title_full | 1024: Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes. | SUPPORT | NEI | judgement error |
| nli_title_full | 1104: Stroke patients with prior use of direct oral anticoagulants have a lower risk of in-hospi | SUPPORT | NEI | judgement error |
| nli_title_full | 1121: Synaptic activity enhances local release of brain derived neurotrophic factor from postsyn | SUPPORT | NEI | judgement error |
| nli_title_full | 1335: UCB T cells maintain high TCR diversity after transplantation. | SUPPORT | NEI | judgement error |
| nli_title_full | 1385: cSMAC formation enhances weak ligand signalling. | SUPPORT | CONTRADICT | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
