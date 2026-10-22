# Evaluation report - run `roberta_scifact_holdout_v1`

- Suite: **title_holdout** (split `accuracy_holdout`), claims requested per variant: **200**
- Gold label counts: {'SUPPORT': 83, 'CONTRADICT': 43, 'NEI': 74}
- Model: `Scientific NLI verifier` digest `483ab906b890ceacd52fa90f143f53ab8736a9f394616cea0ad2184d414c74f5`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T12:49:24+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_title_full | 200 | 1.000 | 0.685 | 0.665 | 0.770 | 0.590 | 0.635 | 0.723 | 0.849 | - | - | 1.000 | 0.055 | 0.000 | 2.391 | 3.406 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_title_full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 67 | 4 | 12 | 0 |
| CONTRADICT | 5 | 23 | 15 | 0 |
| NEI | 19 | 8 | 47 | 0 |

## Evidence metrics (denominators)

- nli_title_full / evidence_cited: precision 0.461 = 89/193 sentences; recall 0.363 = 89/245 sentences (200 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 173 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_title_full: 200/200 ok, 0 errors {}, 11 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_title_full | 0 | 0 | - | 200 | 0 | 200 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (63 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_title_full | 42: A high microerythrocyte count raises vulnerability to severe anemia in homozygous alpha (+ | CONTRADICT | NEI | judgement error |
| nli_title_full | 48: A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection. | CONTRADICT | NEI | judgement error |
| nli_title_full | 70: Activation of PPM1D suppresses p53 function. | SUPPORT | NEI | retrieval miss |
| nli_title_full | 132: Aspirin inhibits the production of PGE2. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 57: APOE4 expression in iPSC-derived neurons increases AlphaBeta production and tau phosphoryl | CONTRADICT | NEI | judgement error |
| nli_title_full | 115: Anthrax spores can be disposed of easily after they are dispersed. | CONTRADICT | NEI | judgement error |
| nli_title_full | 113: Angiotensin converting enzyme inhibitors are associated with increased risk for functional | SUPPORT | NEI | judgement error |
| nli_title_full | 238: Cells undergoing methionine restriction may activate miRNAs. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 183: Bone marrow cells contribute to adult macrophage compartments. | CONTRADICT | SUPPORT | judgement error |
| nli_title_full | 219: CX3CR1 on the Th2 cells suppresses airway inflammation. | CONTRADICT | SUPPORT | judgement error |
| nli_title_full | 118: Antibiotic induced alterations in the gut microbiome reduce resistance against Clostridium | SUPPORT | NEI | judgement error |
| nli_title_full | 269: Cold exposure reduces BAT recruitment. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 233: Cell autonomous sex determination in somatic cells does not occur in Galliformes. | CONTRADICT | NEI | judgement error |
| nli_title_full | 274: Combination nicotine replacement therapies with varenicline or bupropion lead to significa | CONTRADICT | NEI | judgement error |
| nli_title_full | 239: Cellular aging closely links to an older appearance. | SUPPORT | NEI | retrieval miss |
| nli_title_full | 294: Crossover hot spots are not found within gene promoters in Saccharomyces cerevisiae. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 781: Mice that lack Interferon-γ or its receptor exhibit high resistance to experimental autoim | CONTRADICT | NEI | judgement error |
| nli_title_full | 847: New drugs for tuberculosis often do not penetrate the necrotic portion of a tuberculosis l | CONTRADICT | NEI | judgement error |
| nli_title_full | 298: Cytochrome c is released from the mitochondrial intermembrane space to cytosol during apop | SUPPORT | NEI | judgement error |
| nli_title_full | 312: De novo assembly of sequence data has more specific contigs than unassembled sequence data | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 879: Occupancy of ribosomes by IncRNAs do not make functional peptides. | CONTRADICT | NEI | judgement error |
| nli_title_full | 993: Pyridostatin destabilizes the G - quadruplex in the telomeric region. | CONTRADICT | SUPPORT | judgement error |
| nli_title_full | 343: Diabetic patients with acute coronary syndrome experience increased short-term and long-te | SUPPORT | NEI | judgement error |
| nli_title_full | 354: Downregulation and mislocalization of Scribble prevents cell transformation and mammary tu | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 1021: Rapid up-regulation and higher basal expression of interferon-induced genes reduce surviva | CONTRADICT | SUPPORT | judgement error |
| nli_title_full | 1049: Ribosomopathies have a low degree of cell and tissue specific pathology. | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 533: Hyperfibrinogenemia increases rates of femoropopliteal bypass thrombosis. | SUPPORT | NEI | judgement error |
| nli_title_full | 384: Epidemiological disease burden from noncommunicable diseases is more prevalent in low econ | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1110: Suboptimal nutrition is not predictive of chronic disease | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 1140: Taking 400mg of α-tocopheryl acetate helps to prevent prostate cancer. | CONTRADICT | NEI | judgement error |
| nli_title_full | 589: In young and middle-aged adults, current or remote uses of ADHD medications do not increas | SUPPORT | NEI | judgement error |
| nli_title_full | 410: Febrile seizures increase the threshold for development of epilepsy. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1221: The genomic aberrations found in matasteses are very similar to those found in the primary | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 1303: Tirasemtiv has no effect on fast-twitch muscle. | CONTRADICT | NEI | retrieval miss |
| nli_title_full | 593: Incidence of heart failure decreased by 10% in women since 1979. | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 411: Febrile seizures reduce the threshold for development of epilepsy. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 1359: Varenicline monotherapy is more effective after 12 weeks of treatment compared to combinat | CONTRADICT | NEI | judgement error |
| nli_title_full | 1382: aPKCz causes tumour enhancement by affecting glutamine metabolism. | CONTRADICT | SUPPORT | judgement error |
| nli_title_full | 597: Incidence rates of cervical cancer have decreased. | SUPPORT | NEI | judgement error |
| nli_title_full | 508: Hematopoietic Stem Cell purification reaches purity rate of up to 50%. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 832: NFAT4 activation requires IP3R-mediated Ca2+ mobilization. | SUPPORT | NEI | judgement error |
| nli_title_full | 514: High dietary calcium intakes are unnecessary for prevention of secondary hyperparathyroidi | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1012: Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume. | SUPPORT | NEI | judgement error |
| nli_title_full | 535: Hypertension is frequently observed in type 1 diabetes patients. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1104: Stroke patients with prior use of direct oral anticoagulants have a lower risk of in-hospi | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 560: Immune responses result in the development of inflammatory Th17 cells and anti-inflammator | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1290: There is an inverse relationship between hip fractures and statin use. | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 623: Individuals with low serum vitamin D concentrations have increased risk of multiple sclero | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1335: UCB T cells maintain high TCR diversity after transplantation. | SUPPORT | NEI | judgement error |
| nli_title_full | 628: Infection of human T-cell lymphotropic virus type 1 is most frequent in individuals of Afr | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1385: cSMAC formation enhances weak ligand signalling. | SUPPORT | CONTRADICT | judgement error |
| nli_title_full | 684: Lack of clpC does not affect sporulation efficiency in Bacillus subtilis cells. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 785: Microarray results from culture-amplified mixtures of serotypes correlate poorly with micr | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 800: Modifying the epigenome in the brain affects the normal human aging process by affecting c | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 907: PGE 2 promotes intestinal tumor growth by altering the expression of tumor suppressing and | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 914: PPAR-RXRs can be activated by PPAR ligands. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 956: Pleiotropic coupling of GLP-1R to intracellular effectors promotes distinct profiles of ce | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 975: Primary pro-inflammatory cytokines induce secondary pro- and anti-inflammatory mediators. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1226: The loss of the TET protein functions may have dire biological consequences, such as myelo | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1280: The ureABIEFGH gene cluster encodes urease maturation proteins : UreD/UreH, UreE, UreF, an | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1281: The ureABIEFGH gene cluster is induced by nickel (II) ion. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_full | 1292: There is no association between HNF4A mutations and diabetes risks. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_full | 1344: Up-regulation of the p53 pathway and related molecular events casues cancer resistance and | NEI | CONTRADICT | judgement error (false decisive verdict) |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
