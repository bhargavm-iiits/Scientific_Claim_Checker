# Evaluation report - run `selection_evidence_holdout_v1`

- Suite: **selection_holdout** (split `accuracy_holdout`), claims requested per variant: **200**
- Gold label counts: {'SUPPORT': 83, 'CONTRADICT': 43, 'NEI': 74}
- Model: `Scientific NLI verifier` digest `37ec9d414d82188543815463770880d4005d1eb8763b47b30a1c2fe9505fea7e`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T09:19:19+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| selection_evidence | 200 | 1.000 | 0.760 | 0.758 | 0.779 | 0.747 | 0.750 | 0.723 | 0.849 | 0.847 | 0.384 | 1.000 | 0.015 | 0.000 | 2.481 | 3.640 |

## Confusion matrices (rows = gold, columns = predicted)

**selection_evidence**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 58 | 1 | 24 | 0 |
| CONTRADICT | 3 | 28 | 12 | 0 |
| NEI | 5 | 3 | 66 | 0 |

## Evidence metrics (denominators)

- selection_evidence / evidence_selected: precision 0.847 = 94/111 sentences; recall 0.384 = 94/245 sentences (200 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 175 predicted sentences in unannotated documents (excluded from precision).
- selection_evidence / evidence_cited: precision 0.874 = 83/95 sentences; recall 0.339 = 83/245 sentences (200 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 38 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- selection_evidence: 200/200 ok, 0 errors {}, 3 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| selection_evidence | 0 | 0 | - | 200 | 0 | 200 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (48 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| selection_evidence | 48: A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection. | CONTRADICT | NEI | evidence miss |
| selection_evidence | 54: AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in  | CONTRADICT | NEI | retrieval miss |
| selection_evidence | 3: 1,000 genomes project enables mapping of genetic sequence variation consisting of rare var | SUPPORT | NEI | evidence miss |
| selection_evidence | 132: Aspirin inhibits the production of PGE2. | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 115: Anthrax spores can be disposed of easily after they are dispersed. | CONTRADICT | NEI | evidence miss |
| selection_evidence | 274: Combination nicotine replacement therapies with varenicline or bupropion lead to significa | CONTRADICT | SUPPORT | judgement error |
| selection_evidence | 70: Activation of PPM1D suppresses p53 function. | SUPPORT | NEI | retrieval miss |
| selection_evidence | 300: Cytosolic proteins bind to iron-responsive elements on mRNAs coding for DMT1. Cytosolic pr | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 743: Macrolides protect against myocardial infarction. | CONTRADICT | SUPPORT | evidence miss |
| selection_evidence | 847: New drugs for tuberculosis often do not penetrate the necrotic portion of a tuberculosis l | CONTRADICT | NEI | evidence miss |
| selection_evidence | 239: Cellular aging closely links to an older appearance. | SUPPORT | NEI | retrieval miss |
| selection_evidence | 354: Downregulation and mislocalization of Scribble prevents cell transformation and mammary tu | NEI | CONTRADICT | judgement error (false decisive verdict) |
| selection_evidence | 879: Occupancy of ribosomes by IncRNAs do not make functional peptides. | CONTRADICT | SUPPORT | evidence miss |
| selection_evidence | 1021: Rapid up-regulation and higher basal expression of interferon-induced genes reduce surviva | CONTRADICT | NEI | evidence miss |
| selection_evidence | 279: Commelina yellow mottle virus' (ComYMV) genome consists of 7489 baise pairs. | SUPPORT | NEI | judgement error |
| selection_evidence | 384: Epidemiological disease burden from noncommunicable diseases is more prevalent in low econ | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 1049: Ribosomopathies have a low degree of cell and tissue specific pathology. | CONTRADICT | NEI | retrieval miss |
| selection_evidence | 1110: Suboptimal nutrition is not predictive of chronic disease | CONTRADICT | NEI | retrieval miss |
| selection_evidence | 295: Crosstalk between dendritic cells (DCs) and innate lymphoid cells (ILCs) is important in t | SUPPORT | NEI | judgement error |
| selection_evidence | 535: Hypertension is frequently observed in type 1 diabetes patients. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| selection_evidence | 1137: TNFAIP3 is a tumor suppressor in glioblastoma. | CONTRADICT | NEI | judgement error |
| selection_evidence | 1140: Taking 400mg of α-tocopheryl acetate helps to prevent prostate cancer. | CONTRADICT | NEI | judgement error |
| selection_evidence | 343: Diabetic patients with acute coronary syndrome experience increased short-term and long-te | SUPPORT | NEI | judgement error |
| selection_evidence | 785: Microarray results from culture-amplified mixtures of serotypes correlate poorly with micr | NEI | CONTRADICT | judgement error (false decisive verdict) |
| selection_evidence | 1221: The genomic aberrations found in matasteses are very similar to those found in the primary | CONTRADICT | NEI | retrieval miss |
| selection_evidence | 1303: Tirasemtiv has no effect on fast-twitch muscle. | CONTRADICT | NEI | retrieval miss |
| selection_evidence | 475: Glycolysis is one of the primary glycometabolic pathways in cells. | SUPPORT | NEI | retrieval miss |
| selection_evidence | 914: PPAR-RXRs can be activated by PPAR ligands. | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 1382: aPKCz causes tumour enhancement by affecting glutamine metabolism. | CONTRADICT | NEI | judgement error |
| selection_evidence | 521: High-sensitivity cardiac troponin T (HSCT-T) dosage may not be diagnostic if the onset of  | SUPPORT | NEI | evidence miss |
| selection_evidence | 1226: The loss of the TET protein functions may have dire biological consequences, such as myelo | NEI | SUPPORT | judgement error (false decisive verdict) |
| selection_evidence | 528: Human T-lymphotropic virus type-I-associated myelopathy / tropical spastic paraparesis (HA | SUPPORT | NEI | evidence miss |
| selection_evidence | 533: Hyperfibrinogenemia increases rates of femoropopliteal bypass thrombosis. | SUPPORT | NEI | evidence miss |
| selection_evidence | 593: Incidence of heart failure decreased by 10% in women since 1979. | SUPPORT | CONTRADICT | judgement error |
| selection_evidence | 597: Incidence rates of cervical cancer have decreased. | SUPPORT | NEI | judgement error |
| selection_evidence | 756: Many proteins in human cells can be post-translationally modified at lysine residues via a | SUPPORT | NEI | evidence miss |
| selection_evidence | 808: Most termination events in Okazaki fragments are sequence specific. | SUPPORT | NEI | evidence miss |
| selection_evidence | 837: NR5A2 is important in development of endometrial tissues. | SUPPORT | NEI | evidence miss |
| selection_evidence | 982: Proteins synthesized at the growth cone are ubiquitinated at a higher rate than proteins f | SUPPORT | NEI | evidence miss |
| selection_evidence | 985: Pseudogene PTENP1 regulates the expression of PTEN by functioning as an miRNA decoy. | SUPPORT | NEI | judgement error |
| selection_evidence | 1019: Rapid phosphotransfer rates govern fidelity in two component systems | SUPPORT | NEI | evidence miss |
| selection_evidence | 1024: Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes. | SUPPORT | NEI | judgement error |
| selection_evidence | 1132: TCR/CD3 microdomains are a required to induce the immunologic synapse to activate T cells. | SUPPORT | NEI | evidence miss |
| selection_evidence | 1225: The locus rs647161 is associated with colorectal carcinoma. | SUPPORT | NEI | evidence miss |
| selection_evidence | 1266: The risk of breast cancer among parous women increases with placental weight of pregnancie | SUPPORT | NEI | judgement error |
| selection_evidence | 1335: UCB T cells maintain high TCR diversity after transplantation. | SUPPORT | NEI | judgement error |
| selection_evidence | 1352: Upregulation of mosGCTL-1 is induced upon infection with West Nile virus. | SUPPORT | NEI | evidence miss |
| selection_evidence | 1385: cSMAC formation enhances weak ligand signalling. | SUPPORT | NEI | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
