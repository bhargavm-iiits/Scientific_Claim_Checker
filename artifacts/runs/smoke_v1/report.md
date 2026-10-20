# Evaluation report - run `smoke_v1`

- Suite: **smoke** (split `smoke`), claims requested per variant: **20**
- Gold label counts: {'SUPPORT': 8, 'CONTRADICT': 4, 'NEI': 8}
- Model: `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-20T14:44:04+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 20 | 1.000 | 0.600 | 0.432 | 0.762 | 0.000 | 0.533 | 0.600 | 0.775 | 0.588 | 0.526 | 1.000 | 0.050 | 0.000 | 4.220 | 8.177 |
| oracle | 20 | 1.000 | 0.700 | 0.517 | 0.800 | 0.000 | 0.750 | - | - | 0.588 | 0.526 | 1.000 | 0.000 | 0.000 | 0.383 | 1.947 |

## Confusion matrices (rows = gold, columns = predicted)

**full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 8 | 0 | 0 | 0 |
| CONTRADICT | 1 | 0 | 3 | 0 |
| NEI | 4 | 0 | 4 | 0 |

**oracle**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 8 | 0 | 0 | 0 |
| CONTRADICT | 2 | 0 | 2 | 0 |
| NEI | 2 | 0 | 6 | 0 |

## Evidence metrics (denominators)

- full / evidence_selected: precision 0.588 = 10/17 sentences; recall 0.526 = 10/19 sentences; 50 predicted sentences in unannotated documents (excluded from precision).
- full / evidence_cited: precision 0.615 = 8/13 sentences; recall 0.421 = 8/19 sentences; 18 predicted sentences in unannotated documents (excluded from precision).
- oracle / evidence_selected: precision 0.588 = 10/17 sentences; recall 0.526 = 10/19 sentences; 6 predicted sentences in unannotated documents (excluded from precision).
- oracle / evidence_cited: precision 0.615 = 8/13 sentences; recall 0.421 = 8/19 sentences; 5 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- full: 20/20 ok, 0 errors {}, 1 conflicts, 0 claims with a downgraded judgement.
- oracle: 20/20 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.

## Failure cases (14 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| full | 278: Commelina yellow mottle virus' (ComYMV) genome consists of 2140 baise pairs. | CONTRADICT | NEI | evidence miss |
| full | 523: Higher plasma levels of I-carnitine, when associated with trimethylamine N-oxide, are nega | CONTRADICT | NEI | evidence miss |
| full | 182: Bone Marrow (BM) transplantation is used to treat acute myeloid leukemia (AML) | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 925: Patients in stable partnerships progress from HIV to death at the same rate as patients no | CONTRADICT | SUPPORT | judgement error |
| full | 1249: The peak incidence of fractures occurs in toddlers. | CONTRADICT | NEI | evidence miss |
| full | 214: CRP is positively correlated with postoperative mortality following Coronary Artery Bypass | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 550: ITAM phosphorylation allows for the transfer of the T cell receptor (TCR) signal from the  | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 939: Persistor cells are one reason for incomplete responses to Tyrosine kinase inhibitor (TKI) | NEI | SUPPORT | judgement error (false decisive verdict) |
| oracle | 278: Commelina yellow mottle virus' (ComYMV) genome consists of 2140 baise pairs. | CONTRADICT | NEI | evidence miss |
| oracle | 523: Higher plasma levels of I-carnitine, when associated with trimethylamine N-oxide, are nega | CONTRADICT | SUPPORT | evidence miss |
| oracle | 82: Addition of BMP4, activin A, CHIR99021, and SU504 to reprogramming fibroblasts generates,  | NEI | SUPPORT | judgement error (false decisive verdict) |
| oracle | 925: Patients in stable partnerships progress from HIV to death at the same rate as patients no | CONTRADICT | SUPPORT | judgement error |
| oracle | 1249: The peak incidence of fractures occurs in toddlers. | CONTRADICT | NEI | evidence miss |
| oracle | 182: Bone Marrow (BM) transplantation is used to treat acute myeloid leukemia (AML) | NEI | SUPPORT | judgement error (false decisive verdict) |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
