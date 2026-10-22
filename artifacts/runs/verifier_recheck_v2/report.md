# Evaluation report - run `verifier_recheck_v2`

- Suite: **verifier_recheck** (split `final`), claims requested per variant: **100**
- Gold label counts: {'SUPPORT': 41, 'CONTRADICT': 21, 'NEI': 38}
- Model: `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Index fingerprint: `3fa9e53a7857605c`; prompt hashes: `{'evidence': '25376c48b37ac36e', 'judge': 'ae8f024d7631b2f1', 'judge_only': 'dc2ea4990866fadd', 'retrieve_judge': '2ea3577c957ee917'}`
- Created: 2026-10-21T19:23:29+00:00

Accuracy and macro-F1 are computed over **all requested claims**; operational errors count as wrong (coverage shows the share with a verdict). Macro-F1 averages the classes present in the gold labels. Evidence precision excludes sentences selected in documents without reference annotations (unannotated documents are not proven irrelevant); evidence recall keeps claims with an operational error in its denominator (zero predicted sentences). Confidence values are model-reported and uncalibrated.

## Results

| variant | n | coverage | accuracy | macro-F1 | F1 S | F1 C | F1 N | R@3 | R@10 | ev P (sel) | ev R (sel) | quotes valid | conflict | fail | mean s | p95 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nli_full | 100 | 1.000 | 0.780 | 0.783 | 0.800 | 0.800 | 0.750 | 0.676 | 0.824 | - | - | 1.000 | 0.020 | 0.000 | 1.419 | 1.966 |

## Confusion matrices (rows = gold, columns = predicted)

**nli_full**

| gold \ pred | SUPPORT | CONTRADICT | NEI | ERROR |
|---|---|---|---|---|
| SUPPORT | 32 | 0 | 9 | 0 |
| CONTRADICT | 2 | 16 | 3 | 0 |
| NEI | 5 | 3 | 30 | 0 |

## Evidence metrics (denominators)

- nli_full / evidence_cited: precision 0.652 = 60/92 sentences; recall 0.496 = 60/121 sentences (100 claims, including 0 with an operational error = 0 reference sentences in 0 annotated claim-document pairs); 60 predicted sentences in unannotated documents (excluded from precision).

## Operational summary

- nli_full: 100/100 ok, 0 errors {}, 2 conflicts, 0 claims with a downgraded judgement.

## Cache usage (latency context)

| variant | model calls | LLM cache hits | LLM hit share | retrievals | retrieval cache hits | misses | unknown | retrieval hit rate |
|---|---|---|---|---|---|---|---|---|
| nli_full | 0 | 0 | - | 100 | 0 | 100 | 0 | 0 % |

- LLM hit share = cache hits / (model calls + cache hits); cached model answers also make a variant faster than a cold run.

## Failure cases (22 selected; CONTRADICT emphasised)

Stage = earliest failing stage (heuristic): operational / retrieval miss / evidence miss / judgement error.

| variant | claim | gold | pred | stage |
|---|---|---|---|---|
| nli_full | 532: Hyperfibrinogenemia decreases rates of femoropopliteal bypass thrombosis. | CONTRADICT | NEI | judgement error |
| nli_full | 578: In mouse models, the loss of CSF1R facilitates MOZ-TIF2-induced leuekmogenesis. | CONTRADICT | NEI | judgement error |
| nli_full | 5: 1/2000 in UK have abnormal PrP positivity. | SUPPORT | NEI | judgement error |
| nli_full | 213: CRP is not predictive of postoperative mortality following Coronary Artery Bypass Graft (C | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 759: Mathematical models predict that using Artemisinin-based combination therapy over nongamet | CONTRADICT | SUPPORT | judgement error |
| nli_full | 852: Non-invasive ventilation use should be decreased if there is inadequate response to conven | CONTRADICT | NEI | judgement error |
| nli_full | 130: Articles published in open access format are more likely to be cited than traditional jour | SUPPORT | NEI | judgement error |
| nli_full | 350: Discrimination between the initiator and elongation tRNAs depends on the translation initi | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 1336: UCB T cells reduce TCR diversity after transplantation. | CONTRADICT | SUPPORT | judgement error |
| nli_full | 314: Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophi | SUPPORT | NEI | retrieval miss |
| nli_full | 507: Helminths interfere with immune system control of macrophages activated by IL-4 favor Myco | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 431: FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS). | SUPPORT | NEI | retrieval miss |
| nli_full | 727: Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterpart | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 636: Inositol lipid 3-phosphatase PTEN converts Ptdlns(3,4)P 2 into phosphatidylinositol 4-phos | SUPPORT | NEI | judgement error |
| nli_full | 783: Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA. | NEI | CONTRADICT | judgement error (false decisive verdict) |
| nli_full | 1020: Rapid up-regulation and higher basal expression of interferon-induced genes increase survi | SUPPORT | NEI | judgement error |
| nli_full | 870: Obesity decreases life quality. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 1187: The YAP1 and TEAD complex tanslocates into the nucleus where it interacts with transcripti | SUPPORT | NEI | judgement error |
| nli_full | 1099: Statins decrease blood cholesterol. | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 1274: The tip of the inner tube of the toxic type VI secretion system (T6SS) antibacterial effec | SUPPORT | NEI | judgement error |
| nli_full | 1279: The treatment of cancer patients with co-IR blockade precipitates adverse autoimmune event | NEI | SUPPORT | judgement error (false decisive verdict) |
| nli_full | 1337: Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164. | SUPPORT | NEI | judgement error |

Human-reviewed analysis of these failures belongs in `failure_analysis.md`.
