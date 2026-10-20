# Evaluation report - run `smoke_integration`

- Suite: **smoke** (split `smoke`), claims requested per variant: **5** (limit 5)
- Gold label counts: {'SUPPORT': 1, 'CONTRADICT': 1, 'NEI': 3}
- Model: `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-20T14:37:26+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 5 | 1.000 | 0.400 | 0.300 | 0.500 | 0.000 | 0.400 | 0.300 | 0.500 | 1.000 | 0.500 | 1.000 | 0.000 | 0.000 | 11.014 | 20.613 |
| oracle | 5 | 1.000 | 0.400 | 0.300 | 0.500 | 0.000 | 0.400 | - | - | 1.000 | 0.500 | 1.000 | 0.000 | 0.000 | 2.046 | 4.420 |

## Confusion matrices (rows = gold, columns = predicted)

**full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 1 | 0 | 0 | 0 |
| CONTRADICT | 0 | 0 | 1 | 0 |
| NEI | 2 | 0 | 1 | 0 |

**oracle**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 1 | 0 | 0 | 0 |
| CONTRADICT | 0 | 0 | 1 | 0 |
| NEI | 2 | 0 | 1 | 0 |

## Evidence metrics (denominators)

- full / evidence_selected: precision 1.000 = 1/1 sentences; recall 0.500 = 1/2 sentences; 16 predicted sentences in unannotated documents (excluded from precision).
- full / evidence_cited: precision 1.000 = 1/1 sentences; recall 0.500 = 1/2 sentences; 9 predicted sentences in unannotated documents (excluded from precision).
- oracle / evidence_selected: precision 1.000 = 1/1 sentences; recall 0.500 = 1/2 sentences; 5 predicted sentences in unannotated documents (excluded from precision).
- oracle / evidence_cited: precision 1.000 = 1/1 sentences; recall 0.500 = 1/2 sentences; 5 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- full: 5/5 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.
- oracle: 5/5 ok, 0 errors {}, 0 conflicts, 0 claims with a downgraded judgement.

## Failure cases (6 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| full | 278: Commelina yellow mottle virus' (ComYMV) genome consists of 2140 baise pairs. | CONTRADICT | NEI | evidence miss |
| full | 182: Bone Marrow (BM) transplantation is used to treat acute myeloid leukemia (AML) | NEI | SUPPORT | judgement error (false decisive verdict) |
| full | 214: CRP is positively correlated with postoperative mortality following Coronary Artery Bypass | NEI | SUPPORT | judgement error (false decisive verdict) |
| oracle | 278: Commelina yellow mottle virus' (ComYMV) genome consists of 2140 baise pairs. | CONTRADICT | NEI | evidence miss |
| oracle | 82: Addition of BMP4, activin A, CHIR99021, and SU504 to reprogramming fibroblasts generates,  | NEI | SUPPORT | judgement error (false decisive verdict) |
| oracle | 182: Bone Marrow (BM) transplantation is used to treat acute myeloid leukemia (AML) | NEI | SUPPORT | judgement error (false decisive verdict) |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
