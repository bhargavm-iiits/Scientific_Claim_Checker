# Evaluation report - run `roberta_scifact_oracle_v1`

- Suite: **title_oracle** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `Scientific NLI verifier` digest `483ab906b890ceacd52fa90f143f53ab8736a9f394616cea0ad2184d414c74f5`
- Index fingerprint: `None`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-22T13:01:36+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_title_oracle | 100 | 1.000 | 0.830 | 0.825 | 0.850 | 0.800 | 0.825 | - | - | - | - | 1.000 | 0.030 | 0.000 | 0.153 | 0.361 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_title_oracle**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 34 | 2 | 5 | 0 |
| CONTRADICT | 1 | 16 | 4 | 0 |
| NEI | 4 | 1 | 33 | 0 |

## Evidence metrics (denominators)

- nli_title_oracle / evidence_cited: precision 0.500 = 57/114 sentences; recall 0.471 = 57/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 20 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_title_oracle: 100/100 ok, 0 errors {}, 3 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_title_oracle | 0 | 0 | - | 0 | - | - | - | n/a (no retrieval) |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (17 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_title_oracle | 338: Dexamethasone decreases risk of postoperative bleeding. | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 759: Mathematical models predict that using Artemisinin-based combination therapy over nongamet | CONTRADICT | SUPPORT | judgement error |
| nli_title_oracle | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1: 0-dimensional biomaterials show inductive properties. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_oracle | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 1041: Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nu | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 179: Birth-weight is positively associated with breast cancer. | SUPPORT | NEI | judgement error |
| nli_title_oracle | 268: Cold exposure increases BAT recruitment. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_title_oracle | 1336: UCB T cells reduce TCR diversity after transplantation. | CONTRADICT | NEI | judgement error |
| nli_title_oracle | 208: CHEK2 is not associated with breast cancer. | SUPPORT | CONTRADICT | judgement error |
| nli_title_oracle | 350: Discrimination between the initiator and elongation tRNAs depends on the translation initi | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_oracle | 236: Cell autonomous sex determination in somatic cells occurs in Passeriformes. | SUPPORT | NEI | judgement error |
| nli_title_oracle | 870: Obesity decreases life quality. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_oracle | 501: Headaches are not correlated with cognitive impairment. | SUPPORT | CONTRADICT | judgement error |
| nli_title_oracle | 1099: Statins decrease blood cholesterol. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_title_oracle | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | NEI | judgement error |
| nli_title_oracle | 1337: Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164. | SUPPORT | NEI | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
