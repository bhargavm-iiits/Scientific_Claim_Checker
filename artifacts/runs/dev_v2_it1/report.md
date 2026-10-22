# Evaluation report - run `dev_v2_it1`

- Suite: **dev** (split `dev_tune`), claims requested per variant: **120**
- Gold label counts: {'SUPPORT': 49, 'CONTRADICT': 26, 'NEI': 45}
- Model: `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-21T11:56:14+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_v2_nothink | 120 | 1.000 | 0.675 | 0.649 | 0.756 | 0.511 | 0.680 | 0.749 | 0.844 | 0.603 | 0.570 | 1.000 | 0.033 | 0.000 | 6.305 | 11.638 |
| full_v2 | 120 | 1.000 | 0.642 | 0.631 | 0.667 | 0.571 | 0.655 | 0.749 | 0.844 | 0.603 | 0.570 | 1.000 | 0.017 | 0.000 | 18.831 | 40.258 |

## Confusion matrices (rows = gold, columns = predicted)

**full_v2_nothink**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 34 | 4 | 11 | 0 |
| CONTRADICT | 2 | 12 | 12 | 0 |
| NEI | 5 | 5 | 35 | 0 |

**full_v2**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 27 | 3 | 19 | 0 |
| CONTRADICT | 2 | 14 | 10 | 0 |
| NEI | 3 | 6 | 36 | 0 |

## Evidence metrics (denominators)

- full_v2_nothink / evidence_selected: precision 0.603 = 85/141 sentences; recall 0.570 = 85/149 sentences (120 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 261 predicted sentences in unannotated documents (excluded from precision).
- full_v2_nothink / evidence_cited: precision 0.697 = 62/89 sentences; recall 0.416 = 62/149 sentences (120 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 87 predicted sentences in unannotated documents (excluded from precision).
- full_v2 / evidence_selected: precision 0.603 = 85/141 sentences; recall 0.570 = 85/149 sentences (120 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 261 predicted sentences in unannotated documents (excluded from precision).
- full_v2 / evidence_cited: precision 0.759 = 63/83 sentences; recall 0.423 = 63/149 sentences (120 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 52 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- full_v2_nothink: 120/120 ok, 0 errors {}, 4 conflicts, 0 claims with a downgraded judgement.
- full_v2: 120/120 ok, 0 errors {}, 2 conflicts, 1 claims with a downgraded judgement (decisive answer without a valid citation -> NEI; claims 296).

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| full_v2_nothink | 174 | 360 | 67 % | 120 | 0 | 120 | 0 | 0 % |
| full_v2 | 352 | 366 | 51 % | 120 | 120 | 0 | 0 | 100 % |

- **Caveat (retrieval cache):** 120 claim(s) were served from the in-process retrieval cache (full_v2 120). For these claims `retrieve` is the near-zero lookup actually measured and `total` latency excludes the search (about one uncached retrieval per claim); compare latency only between variants with the same cache state.
- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (39 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| full_v2_nothink | 119: Antidepressants increase the severity of migraines. | CONTRADICT | NEI | evidence miss |
| full_v2_nothink | 299: Cytochrome c is transferred from cytosol to the mitochondrial intermembrane space during a | CONTRADICT | SUPPORT | judgement error |
| full_v2_nothink | 121: Antimicrobial agents are less effective due to the pressure of antimicrobial usage. | SUPPORT | NEI | evidence miss |
| full_v2_nothink | 89: Adult tissue-resident macrophages stem from the embryonal yolk sac and fetal liver. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full_v2_nothink | 317: Decreased diastolic blood pressure (DBP) is associated with abdominal aortic aneurysm. | CONTRADICT | NEI | evidence miss |
| full_v2_nothink | 331: Deltex interacts with eIF3. There is no known interaction between Deltex and elF3 | CONTRADICT | NEI | evidence miss |
| full_v2_nothink | 200: CD28 initiates tonic signaling in conventional T cells, which causes an exhaustion phenoty | SUPPORT | CONTRADICT | judgement error |
| full_v2_nothink | 174: Beta-band coherence is diminished for visible stimuli over invisible stimuli. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full_v2_nothink | 453: General exercise therapy is more effective than rotator cuff exercises in reducing pain an | CONTRADICT | NEI | judgement error |
| full_v2_nothink | 466: Genomic sequences involved in alternative splicing responsible for Hutchinson-Gilford prog | CONTRADICT | NEI | judgement error |
| full_v2_nothink | 334: Depletion of nitric oxide is responsible for vasospasm. | SUPPORT | CONTRADICT | judgement error |
| full_v2_nothink | 224: Cancer-associated fibroblasts (CAFs) are constituents of tumor microenvironments | NEI | SUPPORT | judgement error (false decisive verdict) |
| full_v2_nothink | 563: In British Men, haplogroup I decreases risk of cardiovascular disease. | CONTRADICT | NEI | judgement error |
| full_v2_nothink | 590: In young and middle-aged adults, current or remote uses of ADHD medications increase the r | CONTRADICT | NEI | evidence miss |
| full_v2_nothink | 394: Excess gestational weight gain is associated with obesity-related pregnancy outcomes. | SUPPORT | NEI | evidence miss |
| full_v2_nothink | 449: Gastric lavage is an effective treatment for acute paraquat poisoning. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full_v2_nothink | 615: Increased microtubule acetylation worsens interference of axonal transport caused by LRRK2 | CONTRADICT | NEI | judgement error |
| full_v2_nothink | 798: Mitochondria play a trivial role in calcium homeostasis. | CONTRADICT | NEI | evidence miss |
| full_v2_nothink | 607: Increased diastolic blood pressure (DBP) is associated with abdominal aortic aneurysm. | SUPPORT | NEI | evidence miss |
| full_v2_nothink | 582: In rhesus macaques, daily subcutaneous injections of emtricitabine are ineffective at prot | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full_v2_nothink | 1065: Satellite cell dysfunction is a key factor in sarcopenia development. | CONTRADICT | NEI | judgement error |
| full_v2_nothink | 1087: Sildenafil worsens erectile function in men who experience sexual dysfunction as a result  | CONTRADICT | NEI | judgement error |
| full_v2_nothink | 667: Klf2 is important for proper myeloid cell function. | SUPPORT | NEI | judgement error |
| full_v2_nothink | 602: Increase of p62 in prostate tumor stroma results in defective autophagy. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full_v2_nothink | 1135: TNFAIP3 is a glioblastoma tumor suppressor. | CONTRADICT | SUPPORT | judgement error |
| full_v2_nothink | 1327: Tuberculosis incidence occurs at higher rates in more sparsely populated areas in the UK. | CONTRADICT | NEI | evidence miss |
| full_v2_nothink | 670: Knockout proximal tubule-specific deletion of the BMP receptor Alk3 causes epithelial dama | SUPPORT | NEI | judgement error |
| full_v2_nothink | 609: Increased flux of microbial products provokes immune responses. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full_v2_nothink | 726: Ly6C hi monocytes have a higher inflammatory capacity than Ly6C lo monocytes. | SUPPORT | CONTRADICT | judgement error |
| full_v2_nothink | 813: Mutations in G-Beta protein GNB1 are present in many cancers, resulting in loss of interac | NEI | SUPPORT | judgement error (false decisive verdict) |
| full_v2_nothink | 942: Phase information is useful for predicting donor-recipient matches in organ transplantatio | SUPPORT | NEI | evidence miss |
| full_v2_nothink | 908: PGE 2 suppresss intestinal tumor growth by altering the expression of tumor suppressing an | NEI | CONTRADICT | judgement error (false decisive verdict) |
| full_v2_nothink | 973: Primary cervical cytology screening with HPV detection has higher longitudinal sensitivity | SUPPORT | CONTRADICT | judgement error |
| full_v2_nothink | 1111: Sudden death can occur in patients with complex ventricular arrhythmias. | NEI | SUPPORT | judgement error (false decisive verdict) |
| full_v2_nothink | 1114: Surfactin producing cells and exopolymer producing cells cooperate to generate "Van Gogh"  | SUPPORT | NEI | judgement error |
| full_v2_nothink | 1127: Systemic lupus erythematosus is a risk factor for cardiovascular disease. | SUPPORT | NEI | evidence miss |
| full_v2_nothink | 1159: The DESMOND program demonstrates no significant impact on weight loss. | SUPPORT | NEI | judgement error |
| full_v2_nothink | 1210: The density of cytokine receptor bearing cells affects the distance over which cytokines a | SUPPORT | NEI | evidence miss |
| full_v2_nothink | 1311: Trans-acting factors, such as lncRNAs, influence mRNA translation. | SUPPORT | NEI | retrieval miss |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
