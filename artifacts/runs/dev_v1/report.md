# Evaluation report - run `dev_v1`

- Suite: **dev** (split `dev_tune`), claims requested per variant: **120**
- Gold label counts: {'SUPPORT': 49, 'CONTRADICT': 26, 'NEI': 45}
- Model: `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-21T11:30:09+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 120 | 1.000 | 0.658 | 0.590 | 0.750 | 0.389 | 0.632 | 0.749 | 0.844 | 0.603 | 0.570 | 1.000 | 0.017 | 0.000 | 12.795 | 32.044 |

## Confusion matrices (rows = gold, columns = predicted)

**full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 48 | 0 | 1 | 0 |
| CONTRADICT | 13 | 7 | 6 | 0 |
| NEI | 18 | 3 | 24 | 0 |

## Evidence metrics (denominators)

- full / evidence_selected: precision 0.603 = 85/141 sentences; recall 0.570 = 85/149 sentences (120 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 261 predicted sentences in unannotated documents (excluded from precision).
- full / evidence_cited: precision 0.750 = 72/96 sentences; recall 0.483 = 72/149 sentences (120 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 123 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- full: 120/120 ok, 0 errors {}, 2 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| full | 526 | 5 | 1 % | 120 | 0 | 120 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (41 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| full | 2: 1 in 5 million in UK have abnormal PrP positivity. | CONTRADICT | SUPPORT | judgement error |
| full | 32: A country's Vaccine Alliance (GAVI) eligibility is not indictivate of accelerated adoption | CONTRADICT | SUPPORT | judgement error |
| full | 121: Antimicrobial agents are less effective due to the pressure of antimicrobial usage. | SUPPORT | NEI | evidence miss |
| full | 63: ATM and Rad3 related protein are critical for sensing DNA damage. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 119: Antidepressants increase the severity of migraines. | CONTRADICT | NEI | evidence miss |
| full | 156: BCL-2 promotes the apoptotic effects of c-Myc. | CONTRADICT | SUPPORT | judgement error |
| full | 64: ATM and Rad3 related protein have no role in sensing DNA damage. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 283: Consumption of whole fruits increases the risk of type 2 diabetes. | CONTRADICT | NEI | judgement error |
| full | 299: Cytochrome c is transferred from cytosol to the mitochondrial intermembrane space during a | CONTRADICT | SUPPORT | judgement error |
| full | 89: Adult tissue-resident macrophages stem from the embryonal yolk sac and fetal liver. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 317: Decreased diastolic blood pressure (DBP) is associated with abdominal aortic aneurysm. | CONTRADICT | NEI | evidence miss |
| full | 331: Deltex interacts with eIF3. There is no known interaction between Deltex and elF3 | CONTRADICT | NEI | evidence miss |
| full | 174: Beta-band coherence is diminished for visible stimuli over invisible stimuli. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 368: Early patent ductus ateriosus (PDA) screening increases in-hospital mortality. | CONTRADICT | SUPPORT | judgement error |
| full | 453: General exercise therapy is more effective than rotator cuff exercises in reducing pain an | CONTRADICT | SUPPORT | judgement error |
| full | 211: COPI coatmer is involved in viral replication. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 466: Genomic sequences involved in alternative splicing responsible for Hutchinson-Gilford prog | CONTRADICT | SUPPORT | judgement error |
| full | 590: In young and middle-aged adults, current or remote uses of ADHD medications increase the r | CONTRADICT | NEI | evidence miss |
| full | 224: Cancer-associated fibroblasts (CAFs) are constituents of tumor microenvironments | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 680: Lack of FGF21 in mice increases life expectancy. | CONTRADICT | SUPPORT | judgement error |
| full | 959: Polymeal nutrition increases cardiovascular mortality. | CONTRADICT | SUPPORT | judgement error |
| full | 407: FACT and other histone chaperone(s) compensate for Histone 2A (H2A)-histone 2B (H2B) dimer | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1065: Satellite cell dysfunction is a key factor in sarcopenia development. | CONTRADICT | SUPPORT | judgement error |
| full | 1087: Sildenafil worsens erectile function in men who experience sexual dysfunction as a result  | CONTRADICT | SUPPORT | judgement error |
| full | 449: Gastric lavage is an effective treatment for acute paraquat poisoning. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full | 1135: TNFAIP3 is a glioblastoma tumor suppressor. | CONTRADICT | SUPPORT | judgement error |
| full | 1184: The US health care system can save up to $5 billion if 7% of patients waiting for kidney t | CONTRADICT | SUPPORT | judgement error |
| full | 602: Increase of p62 in prostate tumor stroma results in defective autophagy. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full | 1327: Tuberculosis incidence occurs at higher rates in more sparsely populated areas in the UK. | CONTRADICT | NEI | evidence miss |
| full | 609: Increased flux of microbial products provokes immune responses. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 611: Increased lipolysis leads to higher P38 phosphorylation in adipose tissue. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 689: Less than 10% of patients exposed to radiation have activated markers of mesenchymal stem  | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 710: Long chain polyunsaturated fatty acids supplementation reduces wheezing and asthma. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 753: Malaria has a high vectorial capacity. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 761: MeCP2 influences the synaptic maturation of neurons. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 813: Mutations in G-Beta protein GNB1 are present in many cancers, resulting in loss of interac | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 908: PGE 2 suppresss intestinal tumor growth by altering the expression of tumor suppressing an | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full | 938: Persister cells provide relapse resistance in cancer patients. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1074: Sequence conservation in gene regulatory regions between species is a high-accuracy predic | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1111: Sudden death can occur in patients with complex ventricular arrhythmias. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 1394: miRNAs regulate a unique suite of mRNAs to maintain homeostasis in cells. | NEI | SUPPORT | judgement error (false decisive verdict) |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
