# Evaluation report - run `verifier_holdout_v2`

- Suite: **verifier_holdout** (split `accuracy_holdout`), claims requested per variant: **200**
- Gold label counts: {'SUPPORT': 83, 'CONTRADICT': 43, 'NEI': 74}
- Model: `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-21T19:18:24+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_full | 200 | 1.000 | 0.735 | 0.717 | 0.795 | 0.623 | 0.733 | 0.723 | 0.849 | - | - | 1.000 | 0.030 | 0.000 | 1.515 | 2.141 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 60 | 6 | 17 | 0 |
| CONTRADICT | 1 | 24 | 18 | 0 |
| NEI | 7 | 4 | 63 | 0 |

## Evidence metrics (denominators)

- nli_full / evidence_cited: precision 0.542 = 96/177 sentences; recall 0.392 = 96/245 sentences (200 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 71 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_full: 200/200 ok, 0 errors {}, 6 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_full | 0 | 0 | - | 200 | 0 | 200 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (53 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_full | 48: A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection. | CONTRADICT | NEI | judgement error |
| nli_full | 54: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in  | CONTRADICT | NEI | retrieval miss |
| nli_full | 3: 1,000 genomes project enables mapping of genetic sequence variation consisting of rare var | SUPPORT | NEI | judgement error |
| nli_full | 300: Cytosolic proteins bind to iron-responsive elements on mRNAs coding for DMT1. Cytosolic pr | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 129: Articles published in open access format are less likely to be cited than traditional jour | CONTRADICT | NEI | judgement error |
| nli_full | 183: Bone marrow cells contribute to adult macrophage compartments. | CONTRADICT | NEI | judgement error |
| nli_full | 70: Activation of PPM1D suppresses p53 function. | SUPPORT | NEI | retrieval miss |
| nli_full | 354: Downregulation and mislocalization of Scribble prevents cell transformation and mammary tu | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 212: CR is associated with higher methylation age. | CONTRADICT | NEI | judgement error |
| nli_full | 274: Combination nicotine replacement therapies with varenicline or bupropion lead to significa | CONTRADICT | NEI | judgement error |
| nli_full | 239: Cellular aging closely links to an older appearance. | SUPPORT | NEI | retrieval miss |
| nli_full | 410: Febrile seizures increase the threshold for development of epilepsy. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 692: Leuko-increased blood increases infectious complications in red blood cell transfusion. | CONTRADICT | NEI | judgement error |
| nli_full | 718: Low nucleosome occupancy correlates with low methylation levels across species. | CONTRADICT | NEI | judgement error |
| nli_full | 261: Chronic aerobic exercise alters endothelial function, improving vasodilating mechanisms me | SUPPORT | NEI | judgement error |
| nli_full | 411: Febrile seizures reduce the threshold for development of epilepsy. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 847: New drugs for tuberculosis often do not penetrate the necrotic portion of a tuberculosis l | CONTRADICT | NEI | judgement error |
| nli_full | 859: Normal expression of RUNX1 has tumor-promoting effects. | CONTRADICT | NEI | judgement error |
| nli_full | 279: Commelina yellow mottle virus' (ComYMV) genome consists of 7489 baise pairs. | SUPPORT | CONTRADICT | judgement error |
| nli_full | 554: Immune complex triggered cell death leads to extracellular release of neutrophil protein H | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 879: Occupancy of ribosomes by IncRNAs do not make functional peptides. | CONTRADICT | NEI | judgement error |
| nli_full | 993: Pyridostatin destabilizes the G - quadruplex in the telomeric region. | CONTRADICT | SUPPORT | judgement error |
| nli_full | 475: Glycolysis is one of the primary glycometabolic pathways in cells. | SUPPORT | NEI | judgement error |
| nli_full | 834: NOX2-independent pathways can generate peroxynitrite by reacting with nitrogen intermediat | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 1021: Rapid up-regulation and higher basal expression of interferon-induced genes reduce surviva | CONTRADICT | NEI | judgement error |
| nli_full | 1049: Ribosomopathies have a low degree of cell and tissue specific pathology. | CONTRADICT | NEI | retrieval miss |
| nli_full | 525: Histone demethylase recruitment and a transient decrease in histone methylation is necessa | SUPPORT | NEI | judgement error |
| nli_full | 907: PGE 2 promotes intestinal tumor growth by altering the expression of tumor suppressing and | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 1088: Silencing of Bcl2 is important for the maintenance and progression of tumors. | CONTRADICT | NEI | judgement error |
| nli_full | 1110: Suboptimal nutrition is not predictive of chronic disease | CONTRADICT | NEI | retrieval miss |
| nli_full | 533: Hyperfibrinogenemia increases rates of femoropopliteal bypass thrombosis. | SUPPORT | NEI | judgement error |
| nli_full | 1100: Statins increase blood cholesterol. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 1140: Taking 400mg of α-tocopheryl acetate helps to prevent prostate cancer. | CONTRADICT | NEI | judgement error |
| nli_full | 1221: The genomic aberrations found in matasteses are very similar to those found in the primary | CONTRADICT | NEI | retrieval miss |
| nli_full | 589: In young and middle-aged adults, current or remote uses of ADHD medications do not increas | SUPPORT | NEI | judgement error |
| nli_full | 1292: There is no association between HNF4A mutations and diabetes risks. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 1303: Tirasemtiv has no effect on fast-twitch muscle. | CONTRADICT | NEI | judgement error |
| nli_full | 593: Incidence of heart failure decreased by 10% in women since 1979. | SUPPORT | CONTRADICT | judgement error |
| nli_full | 1316: Transferred UCB T cells acquire a memory-like phenotype in recipients. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 756: Many proteins in human cells can be post-translationally modified at lysine residues via a | SUPPORT | NEI | retrieval miss |
| nli_full | 1332: Tumor necrosis factor alpha (TNF-α) and interleukin-1 (IL-1) are pro-inflammatory cytokine | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 808: Most termination events in Okazaki fragments are sequence specific. | SUPPORT | CONTRADICT | judgement error |
| nli_full | 967: Pretreatment with the Arp2/3 inhibitor CK-666 affects lamelliopodia formation. | SUPPORT | NEI | judgement error |
| nli_full | 982: Proteins synthesized at the growth cone are ubiquitinated at a higher rate than proteins f | SUPPORT | NEI | judgement error |
| nli_full | 1019: Rapid phosphotransfer rates govern fidelity in two component systems | SUPPORT | NEI | judgement error |
| nli_full | 1024: Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes. | SUPPORT | NEI | judgement error |
| nli_full | 1104: Stroke patients with prior use of direct oral anticoagulants have a lower risk of in-hospi | SUPPORT | CONTRADICT | judgement error |
| nli_full | 1121: Synaptic activity enhances local release of brain derived neurotrophic factor from postsyn | SUPPORT | NEI | judgement error |
| nli_full | 1163: The DdrB protein from Deinococcus radiodurans is an alternative SSB. | SUPPORT | NEI | judgement error |
| nli_full | 1266: The risk of breast cancer among parous women increases with placental weight of pregnancie | SUPPORT | NEI | judgement error |
| nli_full | 1290: There is an inverse relationship between hip fractures and statin use. | SUPPORT | CONTRADICT | judgement error |
| nli_full | 1335: UCB T cells maintain high TCR diversity after transplantation. | SUPPORT | NEI | judgement error |
| nli_full | 1385: cSMAC formation enhances weak ligand signalling. | SUPPORT | CONTRADICT | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
