# Evaluation report - run `trained_evidence_holdout_v2`

- Suite: **trained_holdout** (split `accuracy_holdout`), claims requested per variant: **200**
- Gold label counts: {'SUPPORT': 83, 'CONTRADICT': 43, 'NEI': 74}
- Model: `Scientific NLI verifier` digest `70f1a6de62293c74712804c58a12ce407b21df4f40cd3107d51978eaa0e5f646`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T09:06:28+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trained_evidence | 200 | 1.000 | 0.770 | 0.768 | 0.795 | 0.753 | 0.757 | 0.723 | 0.849 | 0.873 | 0.392 | 1.000 | 0.005 | 0.000 | 2.475 | 3.445 |

## Confusion matrices (rows = gold, columns = predicted)

**trained_evidence**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 58 | 2 | 23 | 0 |
| CONTRADICT | 1 | 29 | 13 | 0 |
| NEI | 4 | 3 | 67 | 0 |

## Evidence metrics (denominators)

- trained_evidence / evidence_selected: precision 0.873 = 96/110 sentences; recall 0.392 = 96/245 sentences (200 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 135 predicted sentences in unannotated documents (excluded from precision).
- trained_evidence / evidence_cited: precision 0.914 = 85/93 sentences; recall 0.347 = 85/245 sentences (200 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 36 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- trained_evidence: 200/200 ok, 0 errors {}, 1 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| trained_evidence | 0 | 0 | - | 200 | 0 | 200 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (46 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| trained_evidence | 48: A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection. | CONTRADICT | NEI | evidence miss |
| trained_evidence | 54: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in  | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 3: 1,000 genomes project enables mapping of genetic sequence variation consisting of rare var | SUPPORT | NEI | evidence miss |
| trained_evidence | 132: Aspirin inhibits the production of PGE2. | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 115: Anthrax spores can be disposed of easily after they are dispersed. | CONTRADICT | NEI | judgement error |
| trained_evidence | 274: Combination nicotine replacement therapies with varenicline or bupropion lead to significa | CONTRADICT | SUPPORT | judgement error |
| trained_evidence | 70: Activation of PPM1D suppresses p53 function. | SUPPORT | NEI | retrieval miss |
| trained_evidence | 300: Cytosolic proteins bind to iron-responsive elements on mRNAs coding for DMT1. Cytosolic pr | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 598: Incidence rates of cervical cancer have increased due to nationwide screening programs bas | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 692: Leuko-increased blood increases infectious complications in red blood cell transfusion. | CONTRADICT | NEI | judgement error |
| trained_evidence | 118: Antibiotic induced alterations in the gut microbiome reduce resistance against Clostridium | SUPPORT | NEI | judgement error |
| trained_evidence | 354: Downregulation and mislocalization of Scribble prevents cell transformation and mammary tu | NEI | CONTRADICT | judgement error (false decisive verdict) |
| trained_evidence | 811: Mutant mice lacking SVCT2 have greatly increased ascorbic acid levels in both brain and ad | CONTRADICT | NEI | evidence miss |
| trained_evidence | 879: Occupancy of ribosomes by IncRNAs do not make functional peptides. | CONTRADICT | NEI | evidence miss |
| trained_evidence | 143: Autologous transplantation of mesenchymal stem cells causes fewer opportunistic infections | SUPPORT | NEI | judgement error |
| trained_evidence | 384: Epidemiological disease burden from noncommunicable diseases is more prevalent in low econ | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 1021: Rapid up-regulation and higher basal expression of interferon-induced genes reduce surviva | CONTRADICT | NEI | evidence miss |
| trained_evidence | 1049: Ribosomopathies have a low degree of cell and tissue specific pathology. | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 239: Cellular aging closely links to an older appearance. | SUPPORT | NEI | retrieval miss |
| trained_evidence | 785: Microarray results from culture-amplified mixtures of serotypes correlate poorly with micr | NEI | CONTRADICT | judgement error (false decisive verdict) |
| trained_evidence | 1110: Suboptimal nutrition is not predictive of chronic disease | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 1137: TNFAIP3 is a tumor suppressor in glioblastoma. | CONTRADICT | NEI | judgement error |
| trained_evidence | 295: Crosstalk between dendritic cells (DCs) and innate lymphoid cells (ILCs) is important in t | SUPPORT | NEI | judgement error |
| trained_evidence | 907: PGE 2 promotes intestinal tumor growth by altering the expression of tumor suppressing and | NEI | SUPPORT | judgement error (false decisive verdict) |
| trained_evidence | 1221: The genomic aberrations found in matasteses are very similar to those found in the primary | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 1303: Tirasemtiv has no effect on fast-twitch muscle. | CONTRADICT | NEI | retrieval miss |
| trained_evidence | 343: Diabetic patients with acute coronary syndrome experience increased short-term and long-te | SUPPORT | CONTRADICT | judgement error |
| trained_evidence | 1100: Statins increase blood cholesterol. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| trained_evidence | 380: Enhanced early production of inflammatory chemokines improves viral control in the lung. | SUPPORT | NEI | judgement error |
| trained_evidence | 475: Glycolysis is one of the primary glycometabolic pathways in cells. | SUPPORT | NEI | retrieval miss |
| trained_evidence | 528: Human T-lymphotropic virus type-I-associated myelopathy / tropical spastic paraparesis (HA | SUPPORT | NEI | evidence miss |
| trained_evidence | 533: Hyperfibrinogenemia increases rates of femoropopliteal bypass thrombosis. | SUPPORT | NEI | evidence miss |
| trained_evidence | 593: Incidence of heart failure decreased by 10% in women since 1979. | SUPPORT | CONTRADICT | judgement error |
| trained_evidence | 756: Many proteins in human cells can be post-translationally modified at lysine residues via a | SUPPORT | NEI | evidence miss |
| trained_evidence | 808: Most termination events in Okazaki fragments are sequence specific. | SUPPORT | NEI | evidence miss |
| trained_evidence | 967: Pretreatment with the Arp2/3 inhibitor CK-666 affects lamelliopodia formation. | SUPPORT | NEI | judgement error |
| trained_evidence | 982: Proteins synthesized at the growth cone are ubiquitinated at a higher rate than proteins f | SUPPORT | NEI | evidence miss |
| trained_evidence | 985: Pseudogene PTENP1 regulates the expression of PTEN by functioning as an miRNA decoy. | SUPPORT | NEI | judgement error |
| trained_evidence | 1019: Rapid phosphotransfer rates govern fidelity in two component systems | SUPPORT | NEI | evidence miss |
| trained_evidence | 1024: Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes. | SUPPORT | NEI | evidence miss |
| trained_evidence | 1225: The locus rs647161 is associated with colorectal carcinoma. | SUPPORT | NEI | evidence miss |
| trained_evidence | 1266: The risk of breast cancer among parous women increases with placental weight of pregnancie | SUPPORT | NEI | evidence miss |
| trained_evidence | 1335: UCB T cells maintain high TCR diversity after transplantation. | SUPPORT | NEI | evidence miss |
| trained_evidence | 1352: Upregulation of mosGCTL-1 is induced upon infection with West Nile virus. | SUPPORT | NEI | evidence miss |
| trained_evidence | 1385: cSMAC formation enhances weak ligand signalling. | SUPPORT | NEI | judgement error |
| trained_evidence | 1389: mTORC2 regulates intracellular cysteine levels through xCT inhibition. | SUPPORT | NEI | evidence miss |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
