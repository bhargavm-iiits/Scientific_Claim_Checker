# Failure analysis — run `final_v1` (human-reviewed)

Scope: every wrong prediction of the **`full`** variant (hybrid + re-rank, k = 3, evidence agent on)
on the 100 frozen final validation claims (gold SUPPORT 41 / CONTRADICT 21 / NEI 38), with the
`oracle` and `judge_only` answers for contrast. `full` got **44 of 100 claims wrong**, all with
status `ok` (0 operational errors).

Sources read for every case: `predictions.jsonl` (claim-level verdicts, retrieved / examined IDs),
`pair_predictions.jsonl` (per-abstract selected and cited sentence IDs, verdict, the model's reason),
`failure_cases.csv` (heuristic stage), and the abstracts themselves in `data/processed/corpus.jsonl`
with the gold labels and evidence positions from `data/processed/claim_gold_validation.jsonl`. All
counts below were computed from these files with a read-only script; the quoted sentences are copied
from the corpus. Sentence IDs are **zero-based** (as stored); "S3" means `sentence_id` 3.

> **Integrity statement.** The prompts were frozen before `final_v1` started and were **not changed
> after these failures were seen**. The run manifest records the prompt hashes
> (`evidence 25376c48b37ac36e`, `judge ae8f024d7631b2f1`, `judge_only dc2ea4990866fadd`,
> `retrieve_judge 2ea3577c957ee917`); the files in `prompts/` still hash to exactly these values
> (last modified 2026-10-20 13:52 UTC, before the run was created at 14:45 UTC). Prompt development
> used training claims only. Every improvement suggested here is a proposal for a **new, separately
> named experiment (`final_v2`)** — its results must not be reported as untouched evaluation on this
> subset, because the subset has now been inspected.

## 1. How the earliest responsible stage is assigned

| Stage | Meaning (human reading) |
|---|---|
| operational | The run produced an error instead of a verdict. |
| retrieval miss | No evidence-bearing reference abstract was among the k = 3 examined. |
| evidence miss | The evidence-bearing abstract was examined, but the evidence agent did not pass the decisive sentence(s) to the judge (none selected, or only an incomplete part). |
| judgement error | The judge saw sufficient evidence (or, for gold NEI, saw no decisive evidence) and still gave the wrong verdict. |
| annotation-ambiguity | A careful human reading of the examined abstracts would also disagree with the gold label, or the gold label depends on SciFact's incomplete annotation (uncited abstracts are not annotated). |

The automatic heuristic in `failure_cases.csv` (EVALUATION.md §4) cannot see partial evidence or
annotation problems, so the human stage differs from it for 9 of the 44 claims (listed in §3.2).

## 2. Case studies (19 cases: 9 CONTRADICT, 4 SUPPORT, 6 NEI)

Notation: `rank r: doc_id — title [label]`, where the label is the gold pair label for reference
(cited) abstracts and "unannotated" otherwise. "Ref ranks" are positions in the pipeline's kept
top-10 list (hybrid + re-rank).

### 2.1 CONTRADICT claims

#### C1 — claim 516: "High levels of CRP reduces the risk of exacerbations in chronic obstructive pulmonary disease (COPD)."

- **Gold** CONTRADICT. **full** SUPPORT · **oracle** SUPPORT · **judge_only** CONTRADICT.
- **Retrieved top 3:** 1: 29564505 — *Inflammatory biomarkers and exacerbations in chronic obstructive
  pulmonary disease.* [CONTRADICT, reference]; 2: 20109325 — *Global strategy for … COPD: GOLD
  executive summary* [unannotated]; 3: 26107000 — *Extrapulmonary effects of COPD on physical
  activity* [unannotated]. Reference retrieved at rank 1.
- **Gold evidence:** S8 (odds ratios 1.2 / 1.7 / 3.7 for 1 / 2 / 3 high biomarkers) and S13:
  "Simultaneously elevated levels of CRP and fibrinogen and leukocyte count in individuals with COPD
  were associated with increased risk of having exacerbations, even in those with milder COPD and in
  those without previous exacerbations."
- **Selected** S8, S9, S12 (S8 is gold); **cited** S8, S9, S12.
- **Model reason:** "The evidence shows higher biomarker levels correlate with increased risk of
  exacerbations (odds ratios >1 and hazard ratios >1), indicating that high CRP levels (as a
  biomarker) increase risk, not reduce it. Thus, the claim that high CRP reduces risk is contradicted."
- **Reading:** retrieval and evidence are correct and the reason is correct — it literally says
  "contradicted" — but the emitted label is SUPPORT. The JSON schema puts `verdict` before `reason`,
  so the label is decoded before the model has "reasoned"; the reason then describes the evidence
  correctly. The oracle makes the identical error (the request is identical; `full` reused the
  oracle's cached answer). The model knows the answer without documents (judge_only CONTRADICT).
- **Earliest stage:** judgement error (verdict–reason inconsistency).
- **Improvement:** emit `reason` before `verdict` in the judge schema; add a supervisor consistency
  check that re-asks (once) when a decisive verdict's reason argues for the other label (an NLI-style
  check or a second short "does this reason support or contradict the claim?" call — a keyword test
  catches only 2 of the 14 such failures here, see §3.3).

#### C2 — claim 180: "Blocking the interaction between TDP-43 and respiratory complex I proteins ND3 and ND6 leads to increased TDP-43-induced neuronal loss."

- **Gold** CONTRADICT. **full** SUPPORT · **oracle** SUPPORT · **judge_only** NEI.
- **Retrieved top 3:** 1: 16966326 — *The Inhibition of TDP-43 Mitochondrial Localization Blocks Its
  Neuronal Toxicity* [CONTRADICT, reference]; 2: 9217800 — *The ALS-associated proteins FUS and TDP-43
  function together to affect Drosophila locomotion and life span.* [unannotated]; 3: 13380011 —
  *Spare respiratory capacity rather than oxidative stress regulates glutamate excitotoxicity …*
  [unannotated]. Reference at rank 1.
- **Gold evidence:** S5: "The suppression of TDP-43 mitochondrial localization abolishes WT and mutant
  TDP-43-induced mitochondrial dysfunction and neuronal loss, and improves phenotypes of transgenic
  mutant TDP-43 mice."
- **Selected** S4, S5; **cited** S5 (gold).
- **Model reason:** "Sentence [5] states that suppressing TDP-43 mitochondrial localization abolishes
  TDP-43-induced neuronal loss, which directly supports the claim that blocking TDP-43 interaction with
  respiratory complex I proteins leads to increased neuronal loss."
- **Reading:** the reason quotes "abolishes … neuronal loss" and concludes "supports … increased
  neuronal loss" — a direction-of-effect inversion inside one sentence.
- **Earliest stage:** judgement error (direction of effect).
- **Improvement:** as C1, plus an explicit step in the judge prompt: "state the direction the evidence
  reports (increase / decrease / no effect) and the direction the claim asserts, then compare" —
  i.e. structured fields `claim_direction`, `evidence_direction` before the verdict.

#### C3 — claim 922: "Patients in stable partnerships have a faster progression from HIV to AIDS."

- **Gold** CONTRADICT. **full** SUPPORT · **oracle** SUPPORT · **judge_only** CONTRADICT.
- **Retrieved top 3:** 1: 17077004 — *Stable partnership and progression to AIDS or death in HIV
  infected patients receiving highly active antiretroviral therapy: Swiss HIV cohort study.*
  [CONTRADICT, reference]; 2: 25134146 — *Impact of human immunodeficiency virus infection on the
  histological features of chronic hepatitis C …* [unannotated]; 3: 22227889 — *Caregiver burden in
  HIV-positive and HIV-negative partners of men with AIDS.* [unannotated]. Reference at rank 1.
- **Gold evidence:** S7: "… the adjusted hazard ratio for progression to AIDS or death was 0.79 (95%
  confidence interval 0.63 to 0.98) for participants with a stable partnership compared with those
  without." and S9: "A stable partnership is associated with a slower rate of progression to AIDS or
  death in HIV infected patients receiving HAART."
- **Selected** S7; **cited** S7. The explicit conclusion S9 was not selected.
- **Model reason:** "The study shows that patients with stable partnerships had a lower hazard ratio
  (0.79) for progression to AIDS or death compared to those without stable partnerships, indicating
  faster progression."
- **Reading:** a numeric misreading — HR 0.79 < 1 means slower progression. With S9 ("slower rate")
  in the context the error would have been much less likely.
- **Earliest stage:** judgement error (numeric / quantitative). Contributing: evidence agent omitted
  the plain-language conclusion sentence.
- **Improvement:** always pass the abstract's conclusion sentence(s) alongside the selected evidence;
  add a prompt rule "hazard/odds/risk ratio < 1 means lower risk / slower"; a small deterministic
  helper that annotates ratios with "(<1: reduced)" before judging.

#### C4 — claim 338: "Dexamethasone decreases risk of postoperative bleeding."

- **Gold** CONTRADICT. **full** SUPPORT · **oracle** SUPPORT · **judge_only** SUPPORT
  (`retrieve_judge_single_prompt` CONTRADICT, `evidence_off` NEI via conflict).
- **Retrieved top 3:** 1: 23349986 — *Dexamethasone and risk of nausea and vomiting and postoperative
  bleeding after tonsillectomy in children: a randomized trial.* [CONTRADICT, reference]; 2: 24211561 —
  *Dexamethasone decreases vomiting by children after tonsillectomy.* [unannotated]; 3: 15948830 —
  *Prophylaxis of postoperative vomiting in children undergoing tonsillectomy: a systematic review and
  meta-analysis.* [unannotated]. Reference at rank 1.
- **Gold evidence:** S10 "Two of 53 (4%; 95% CI, 0.5%-13%)", S11 "children who received placebo had
  bleeding compared with 6 of 53 (11%; …), 2 of 51 (4%; …), and 12 of 50 (24%; …) who received
  dexamethasone at 0.05, 0.15, and 0.5 mg/kg, respectively (P = .003).", S12 "Dexamethasone,
  0.5 mg/kg, was associated with the highest bleeding risk (adjusted relative risk, 6.80; …)", S15
  "… dexamethasone decreased the risk of PONV dose dependently but was associated with an increased
  risk of postoperative bleeding."
- **Selected** S7, S9, S11 (only S11 gold); **cited** S11.
- **Model reason:** "Dexamethasone reduced postoperative bleeding episodes from 13% in placebo to
  4%-12% in dexamethasone recipients, with a statistically significant difference (P = .003)."
- **Reading:** SciFact's sentence segmentation split one sentence into S10 (the placebo figure
  "Two of 53 (4%)") and S11. The evidence agent passed S11 without S10, so the judge never saw the
  placebo rate, invented "13% in placebo", and read the trend backwards. It also skipped S12 and the
  explicit conclusion S15 — the prompt says "usually select between 1 and 3 sentences" and there are
  four gold sentences. The heuristic calls this a judgement error because one gold sentence was
  selected; the decisive evidence was nevertheless missing.
- **Earliest stage:** evidence miss (partial evidence; sentence-fragment artefact), followed by a
  numeric judgement error.
- **Improvement:** when a selected sentence starts in lower case or its predecessor is a short
  fragment, also pass the neighbouring sentence; raise the selection budget to 5 and always include
  the conclusion sentence; tell the judge it may only quote numbers that appear in the supplied text.

#### C5 — claim 142: "Autologous transplantation of mesenchymal stem cells causes a higher rate of opportunistic infections than induction therapy with anti-interleukin-2 receptor antibodies."

- **Gold** CONTRADICT. **full** NEI · **oracle** NEI · **judge_only** NEI.
- **Retrieved top 3:** 1: 10582939 — *Induction therapy with autologous mesenchymal stem cells in
  living-related kidney transplants: a randomized controlled trial.* [CONTRADICT, reference]; 2:
  40164383 — *Comparison of allogeneic vs autologous bone marrow–derived mesenchymal stem cells …
  POSEIDON randomized trial.* [unannotated]; 3: 24101431 — *The antidiabetic effect of mesenchymal stem
  cells …* [unannotated]. Reference at rank 1.
- **Gold evidence:** S12: "Also, during the 1-year follow-up, combined analysis of MSC-treated groups
  revealed significantly decreased risk of opportunistic infections than the control group (hazard
  ratio, 0.42; 95% CI, 0.20-0.85, P=.02) CONCLUSION Among patients undergoing renal transplant, the use
  of autologous MSCs compared with anti-IL-2 receptor antibody induction therapy resulted in lower
  incidence of acute rejection, decreased risk of opportunistic infection, and better estimated renal
  function at 1 year."
- **Selected: nothing** from any of the three abstracts, so the judge was skipped and every abstract
  became NEI without a model call.
- **Reading:** the answer is stated verbatim in S12, which is a very long merged sentence (results +
  conclusion). The evidence agent returned an empty list for an abstract that is exactly on topic.
  The oracle fails for the same reason (identical cached evidence call).
- **Earliest stage:** evidence miss.
- **Improvement:** if the top-ranked abstract gets an empty selection, re-ask once with "the abstract
  is highly ranked for this claim; select the result/conclusion sentences that mention the claim's
  outcome" or fall back to passing the last two sentences (conclusions); split over-long merged
  sentences at "CONCLUSION(S)" markers for display (keeping the original IDs for quoting).

#### C6 — claim 1041: "Replacement of histone H2A with H2A.Z slows gene activation in yeasts by stabilizing +1 nucleosomes."

- **Gold** CONTRADICT. **full** SUPPORT · **oracle** SUPPORT · **judge_only** SUPPORT.
- **Retrieved top 3:** 1: 9451052 — *Stepwise Histone Replacement by SWR1 Requires Dual Activation with
  Histone H2A.Z and Canonical Nucleosome* [unannotated]; 2: 11844791 — *Conserved Histone Variant H2A.Z
  Protects Euchromatin from the Ectopic Spread of Silent Heterochromatin* [unannotated]; 3: 15778034 —
  *Variant Histone H2A.Z Is Globally Localized to the Promoters of Inactive Yeast Genes and Regulates
  Nucleosome Positioning* [unannotated]. **Ref ranks:** evidence-bearing 25254425 *Nucleosome stability
  mediated by histone variants H3.3 and H2A.Z.* — **not in the kept top 10** (it was rank 7 in hybrid
  RRF and fell out after re-ranking); the second cited abstract 16626264 (no evidence annotated) —
  rank 6.
- **Gold evidence:** 25254425 S3: "Immunoprecipitation studies of nucleosome core particles (NCPs)
  show that NCPs that contain both H3.3 and H2A.Z are even less stable than NCPs containing H3.3 and
  H2A."
- **full:** SUPPORT from 11844791 (cited S6, about Sir2/Sir3 spreading in htz1Δ cells) and 15778034
  (cited S3, S4, S6: "… configures chromatin structure to poise genes for transcriptional activation.").
  Reason for 15778034: "… regulates nucleosome positioning which poises genes for activation,
  supporting the claim that replacing H2A with H2A.Z slows gene activation by stabilizing +1
  nucleosomes."
- **Oracle** (given 25254425): selected S3, S4, S5, cited S3, reason "NCPs with H3.3 and H2A.Z are
  less stable … which supports the claim that H2A.Z stabilizes +1 nucleosomes" — again an inversion.
- **Reading:** a genuine retrieval miss in `full`, but even with the right abstract the judge inverts
  "less stable" into "stabilizes". Also, 16626264 (rank 6, outside k = 3) says "Htz1 promotes full
  gene activation" (S6) — arguably contradicting the claim too.
- **Earliest stage:** retrieval miss (and a latent judgement error, shown by the oracle).
- **Improvement:** biomedical re-ranker (MS MARCO cross-encoder demoted the evidence abstract from
  hybrid rank 7), or fuse re-ranker and RRF scores instead of replacing the order; the C1/C2 judge
  fixes for the latent error.

#### C7 — claim 1029: "Reduced responsiveness to interleukin-2 in regulatory T cells is associated with greater resistance to autoimmune diseases such as Type 1 Diabetes."

- **Gold** CONTRADICT. **full** NEI (`conflict=true`) · **oracle** CONTRADICT · **judge_only**
  CONTRADICT.
- **Retrieved top 3:** 1: 11899391 — *Type 1 diabetes-associated IL2RA variation lowers IL-2 signaling
  and contributes to diminished CD4+CD25+ regulatory T cell function.* [CONTRADICT, reference]; 2:
  13923140 — *Interleukin-2 gene variation impairs regulatory T cell function and causes autoimmunity*
  [CONTRADICT, reference]; 3: 2388819 — *In Vitro–expanded Antigen-specific Regulatory T Cells Suppress
  Autoimmune Diabetes* [unannotated]. Ref ranks 1, 2; third reference 13940200 not in top 10.
- **Gold evidence:** 11899391 S6 ("… an autoimmune disease-associated IL2RA haplotype correlates with
  diminished IL-2 responsiveness … and a reduction in their ability to suppress proliferation of
  autologous effector T cells."), 13923140 S2–S3.
- **Selected/cited** exactly the gold sentences in both reference abstracts → both judged
  **CONTRADICT** correctly. Abstract 3: selected S2–S4, cited S4 "Most significantly, small numbers of
  antigen-specific Tregs can reverse diabetes after disease onset …" → **SUPPORT**, reason "… supporting
  the claim that reduced responsiveness to interleukin-2 in regulatory T cells is associated with
  greater resistance to autoimmune diseases."
- **Reading:** two correct CONTRADICT decisions were cancelled by one spurious SUPPORT on a
  topically related, unannotated abstract (expanded Tregs say nothing about reduced IL-2
  responsiveness). The documented conflict → NEI convention turned it into a wrong claim verdict.
  The oracle, which never sees the distractor, is right.
- **Earliest stage:** judgement error (false SUPPORT on a distractor), amplified by aggregation.
- **Improvement:** weight document decisions by retrieval rank / evidence strength, or require a
  minority decisive verdict to cite a sentence that mentions the claim's key entities; report
  "2 contradict vs 1 support" in the explanation (already flagged as conflict).

#### C8 — claim 852: "Non-invasive ventilation use should be decreased if there is inadequate response to conventional treatment."

- **Gold** CONTRADICT. **full** SUPPORT · **oracle** SUPPORT · **judge_only** SUPPORT.
- **Retrieved top 3:** 1: 474325 — *Non-invasive ventilation in chronic obstructive pulmonary disease
  patients: helmet versus facial mask* [unannotated]; 2: 13843341 — *Cost effectiveness of ward based
  non-invasive ventilation for acute exacerbations of chronic obstructive pulmonary disease …*
  [CONTRADICT, reference]; 3: 8190282 — *Noninvasive ventilation for treatment of acute respiratory
  failure in patients undergoing solid organ transplantation: a randomized trial.* [unannotated].
  Reference at rank 2.
- **Gold evidence:** S9 (a typical hospital "will avoid six deaths and three to nine admissions to
  intensive care units per year …") and S10: "Non-invasive ventilation is a highly cost effective
  treatment that both reduced total costs and improved mortality in hospital."
- **Selected** S6, S7, S8 ("24/118 died in the group receiving standard treatment and 12/118 in the
  group receiving non-invasive ventilation (P=0.05)", cost reduction, "dominant (more effective and
  less costly) strategy"); **cited** S6–S8.
- **Model reason:** "The evidence shows non-invasive ventilation reduced deaths and costs compared to
  standard treatment, supporting that it should be decreased if there's inadequate response to
  conventional treatment."
- **Reading:** the heuristic says "evidence miss" (no gold sentence selected), but S6–S8 carry the same
  finding as the gold sentences; the judge understood them ("reduced deaths and costs") and drew the
  opposite conclusion. The claim is a recommendation, so the judge has to map "NIV is beneficial" to
  "NIV use should not be decreased" — a negated, normative claim.
- **Earliest stage:** judgement error (polarity on a normative claim). Heuristic disagreement.
- **Improvement:** the C1/C2 fixes; a prompt example for recommendation-style claims ("should be
  increased/decreased").

#### C9 — claim 759: "Mathematical models predict that using Artemisinin-based combination therapy over nongametocytocidal drugs have a dramatic impact in reducing malaria transmission."

- **Gold** CONTRADICT. **full** SUPPORT · **oracle** SUPPORT · **judge_only** SUPPORT.
- **Retrieved top 3:** 1: 1805641 — *Modelling the Impact of Artemisinin Combination Therapy and
  Long-Acting Treatments on Malaria Transmission Intensity* [CONTRADICT, reference]; 2: 3929361 —
  *Optimising Strategies for Plasmodium falciparum Malaria Elimination in Cambodia …* [unannotated];
  3: 10617916 — *Moderate effect of artemisinin-based combination therapy on transmission of
  Plasmodium falciparum.* [unannotated]. Reference at rank 1.
- **Gold evidence:** S13: "An efficacious antimalarial regimen with no specific gametocytocidal
  properties but a long prophylactic time was estimated to be more effective at reducing transmission
  than a short-acting ACT in the highest-transmission setting."
- **Selected and cited** S8, S10, S13 (S13 gold).
- **Model reason:** "The evidence shows significant reductions in malaria transmission when switching
  to ACT, with higher impact in high-transmission areas and a more effective regimen in
  high-transmission settings."
- **Reading:** the judge had the decisive comparative sentence but summarised it as support for ACT;
  it also ignored the third abstract's title ("Moderate effect …"). Comparative claims ("A over B has
  a dramatic impact") need the comparison, not just "ACT reduces transmission". This one is genuinely
  hard — S14 says ACTs could approach the effect of insecticide-treated nets in low-transmission
  settings.
- **Earliest stage:** judgement error (comparative / quantitative).
- **Improvement:** decompose comparative claims ("what is compared with what, and in which
  direction?") before judging; ask the judge to address each cited sentence separately.

### 2.2 SUPPORT claims

#### S1 — claim 1298: "Thigh-length graduated compression stockings (GCS) did not reduce deep vein thrombosis in patients admitted to hospital who are immobile because of acute stroke."

- **Gold** SUPPORT. **full** CONTRADICT · **oracle** CONTRADICT · **judge_only** NEI.
- **Retrieved top 3:** 1: 11718220 — *Effectiveness of thigh-length graduated compression stockings to
  reduce the risk of deep vein thrombosis after stroke (CLOTS trial 1) …* [SUPPORT, reference]; 2:
  79231308 — *Role of Computerized Physician Order Entry Systems in Facilitating Medication Errors*
  [unannotated]; 3: 29981186 — *Venous thrombosis in cancer patients: insights from the FRONTLINE
  survey.* [unannotated]. Reference at rank 1.
- **Gold evidence = selected:** S11 "The primary outcome occurred in 126 (10.0%) patients allocated to
  thigh-length GCS and in 133 (10.5%) allocated to avoid GCS, resulting in a non-significant absolute
  reduction in risk of 0.5% (95% CI -1.9% to 2.9%)." and S13 "These data do not lend support to the
  use of thigh-length GCS in patients admitted to hospital with acute stroke."; **cited** S13.
- **Model reason:** "The interpretation states that the data do not support the use of thigh-length
  GCS in stroke patients, indicating no significant benefit for reducing deep vein thrombosis."
- **Reading:** perfect retrieval and evidence; the reason restates the claim ("no significant benefit")
  yet the label is CONTRADICT — the negations in the claim ("did not reduce") and in S13 ("do not lend
  support") were treated as disagreement.
- **Earliest stage:** judgement error (negation).
- **Improvement:** rewrite negated claims into an explicit proposition before judging ("claim asserts:
  GCS has no effect on DVT"); C1 reason-first decoding.

#### S2 — claim 208: "CHEK2 is not associated with breast cancer."

- **Gold** SUPPORT. **full** NEI · **oracle** NEI · **judge_only** CONTRADICT.
- **Retrieved top 3:** 1: 13519661 — *Linkage Disequilibrium Mapping of CHEK2: Common Variation and
  Breast Cancer Risk* [SUPPORT, reference]; 2: 20280410 — *A common variant in BRCA2 is associated with
  both breast cancer risk and prenatal viability* [unannotated]; 3: 21439640 — *Macrophages induce COX-2
  expression in breast cancer cells …* [unannotated]. Reference at rank 1.
- **Gold evidence:** S7: "We genotyped these six tagSNPs in 1,577 postmenopausal breast cancer cases
  and 1,513 population controls, but found no convincing association between any common CHEK2
  haplotype and breast cancer risk."
- **Selected: nothing** in any abstract → all NEI without a judge call.
- **Reading:** an on-topic abstract with a clear null finding received an empty selection. The
  evidence agent appears not to treat "no association" as evidence. The gold label is itself narrow:
  S8 reports an odds ratio of 2.26 for the rare 1100delC mutation, and judge_only says CHEK2 *is*
  associated (true for 1100delC) — the claim is only supported for common variants.
- **Earliest stage:** evidence miss.
- **Improvement:** evidence prompt: "null results ('no association', 'did not differ') are evidence";
  the C5 re-ask on an empty selection for the top-ranked abstract.

#### S3 — claim 1020: "Rapid up-regulation and higher basal expression of interferon-induced genes increase survival of granule cell neurons that are infected by West Nile virus."

- **Gold** SUPPORT. **full** CONTRADICT · **oracle** NEI · **judge_only** NEI.
- **Retrieved top 3:** 1: 10559501 — *Beta interferon controls West Nile virus infection and pathogenesis
  in mice.* [unannotated]; 2: 9433958 — *Differential innate immune response programs in neuronal
  subtypes determine susceptibility to infection in the brain by positive stranded RNA viruses*
  [SUPPORT, reference]; 3: 6163801 — *Kinetics of early T cell receptor signaling …* [unannotated].
  Reference at rank 2.
- **Gold evidence:** 9433958 S3: "By transducing cortical neurons with genes that were expressed more
  highly in granule cell neurons, we identified three interferon-stimulated genes (ISGs; Ifi27, Irg1
  and Rsad2 (also known as Viperin)) that mediated the antiviral effects against different neurotropic
  viruses."
- **Selected: nothing** from the reference abstract (skipped → NEI). From 10559501: selected S3–S5,
  cited S5 ("… viral titers in ex vivo cultures of … cerebellar granule cell neurons, but not cortical
  neurons, from IFN-β(-)(/)(-) mice were greater than in wild-type cells.") → CONTRADICT, reason "The
  evidence shows that IFN-β knockout mice have higher viral replication in granule cell neurons, which
  contradicts the claim …".
- **Reading:** two errors. The evidence agent missed the reference abstract's key sentence; and the
  distractor's evidence (removing interferon → more virus in granule cells) actually agrees with the
  claim, but was judged CONTRADICT — a knockout-logic inversion.
- **Earliest stage:** evidence miss (plus a judgement error on the distractor).
- **Improvement:** C5 empty-selection re-ask; prompt guidance for loss-of-function logic ("knockout
  of X worsens Y ⇒ X protects against Y").

#### S4 — claim 1368: "Vitamin D deficiency effects the term of delivery."

- **Gold** SUPPORT. **full** NEI · **oracle** NEI · **judge_only** SUPPORT.
- **Retrieved top 3:** 1: 2425364 — *Association between maternal serum 25-hydroxyvitamin D level and
  pregnancy and neonatal outcomes: systematic review and meta-analysis of observational studies.*
  [SUPPORT, reference]; 2: 8856690 — *Molecular Mechanisms of Vitamin D Action* [unannotated]; 3:
  23267371 — *Vitamin D: The "sunshine" vitamin.* [unannotated]. Reference at rank 1.
- **Gold evidence:** S9–S12, e.g. S11 "Vitamin D insufficiency is associated with an increased risk of
  gestational diabetes, pre-eclampsia, and small for gestational age infants." and S12 "Pregnant women
  with low 25-OHD levels had an increased risk of bacterial vaginosis and lower birth weight infants,
  but not delivery by caesarean section."
- **Selected** S9, S10, S12 (all gold); **cited** none.
- **Model reason:** "The evidence sentences discuss associations with gestational diabetes,
  pre-eclampsia, small for gestational age infants, bacterial vaginosis, and low birth weight, but do
  not directly address the term of delivery (i.e., the length of pregnancy or time to delivery)."
- **Reading:** the abstract never mentions preterm birth or gestational length; "term of delivery" is
  not addressed literally. The judge's NEI is a defensible reading; the gold SUPPORT relies on an
  interpretation of the (ungrammatical) claim as "affects pregnancy outcomes".
- **Earliest stage:** annotation-ambiguity.
- **Improvement:** none for the system; report such cases separately (a second annotator would be
  needed to resolve them).

### 2.3 NEI claims (false decisive verdicts)

#### N1 — claim 268: "Cold exposure increases BAT recruitment."

- **Gold** NEI. **full** SUPPORT · **oracle** NEI · **judge_only** SUPPORT.
- **Retrieved top 3:** 1: 36838958 — *Sestrin2 inhibits uncoupling protein 1 expression through
  suppressing reactive oxygen species.* [unannotated]; 2: 970012 — *Cold Exposure Promotes
  Atherosclerotic Plaque Growth and Instability via UCP1-Dependent Lipolysis* [NEI, reference]; 3:
  1568684 — *The Bile Acid Chenodeoxycholic Acid Increases Human Brown Adipose Tissue Activity.*
  [unannotated]. Reference at rank 2 (nothing selected from it).
- **Selected/cited** in 1568684: S3 "Treatment of 12 healthy female subjects with CDCA for 2 days
  resulted in increased BAT activity." → SUPPORT.
- **Model reason:** "Sentence [3] states that CDCA treatment increased BAT activity in healthy females,
  which directly supports the claim that cold exposure increases BAT recruitment (as CDCA mimics cold
  exposure effects)."
- **Reading:** entity substitution — a bile acid is not cold exposure. The claim is textbook-true
  (judge_only SUPPORT), but SciFact's NEI label says the cited abstract does not settle it; the model
  fills the gap with world knowledge and a loosely related abstract.
- **Earliest stage:** judgement error (entity substitution; outside knowledge).
- **Improvement:** require that a decisive verdict's cited sentence mention the claim's intervention
  and outcome (or a stated synonym) — a cheap lexical/embedding entity-overlap check in the supervisor;
  prompt rule "if the evidence concerns a different intervention, answer NEI".

#### N2 — claim 1175: "The PPR MDA5 has two N-terminal CARD domains."

- **Gold** NEI. **full** SUPPORT · **oracle** NEI · **judge_only** NEI (only `full` and `full_k5`
  say SUPPORT).
- **Retrieved top 3:** 1: 22191759 — *Purification and structural characterization of bovine
  cathelicidins …* [unannotated]; 2: 15419873 — *Cytosolic viral sensor RIG-I is a
  5'-triphosphate-dependent translocase on double-stranded RNA.* [unannotated]; 3: 17402386 — *THE EMBO
  JOURNAL* [unannotated]. Reference 31272411 (*Immune signaling by RIG-I-like receptors.*) not in the
  top 10.
- **Selected/cited** in 15419873: S1 "Two N-terminal caspase activation and recruitment domains (CARDs)
  transmit the signal, and the regulatory domain prevents signaling in the absence of viral RNA." →
  SUPPORT, reason "The sentence states that there are two N-terminal CARDs, which directly supports the
  claim about PPR MDA5 …".
- **Reading:** the sentence describes RIG-I, not MDA5 (a related RIG-I-like receptor). Same-family
  entity substitution. (MDA5 does in fact have two CARDs, but this abstract does not say so.)
- **Earliest stage:** judgement error (entity substitution).
- **Improvement:** as N1; show the abstract title next to each sentence and instruct the judge to check
  that the sentence's subject is the claim's subject.

#### N3 — claim 1099: "Statins decrease blood cholesterol."

- **Gold** NEI. **full** SUPPORT · **oracle** NEI · **judge_only** SUPPORT (every retrieval variant
  SUPPORT).
- **Retrieved top 3:** 1: 22420524 — *Statin use and risk of gallstone disease followed by
  cholecystectomy.* [unannotated]; 2: 21557614 — *Pleiotropic effects of statins.* [unannotated]; 3:
  30981192 — *How to control residual cardiovascular risk despite statin treatment: focusing on
  HDL-cholesterol.* [unannotated]. Reference 7662206 (*A Century of Cholesterol and Coronaries: From
  Plaques to Genes to Statins*) at rank 4.
- **Cited:** 22420524 S1 "Statins decrease hepatic cholesterol biosynthesis and may therefore lower the
  risk of cholesterol gallstones by reducing the cholesterol concentration in the bile."; 21557614 S2
  "However, the overall benefits observed with statins appear to be greater than what might be expected
  from changes in lipid levels alone, suggesting effects beyond cholesterol lowering."
- **Reading:** the uncited abstracts contain reasonable (if indirect) support for a well-established
  fact. The gold NEI only reflects that the one cited abstract never mentions statins lowering
  cholesterol. A human reading these three abstracts would probably answer SUPPORT.
- **Earliest stage:** annotation-ambiguity (incomplete annotation of uncited abstracts).
- **Improvement:** none for the judge; in reporting, keep counting these as errors (the benchmark rule)
  but list them separately, as here.

#### N4 — claim 727: "Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterparts."

- **Gold** NEI. **full** SUPPORT · **oracle** NEI · **judge_only** CONTRADICT.
- **Retrieved top 3:** 1: 36444198 — *Subpopulations of mouse blood monocytes differ in maturation stage
  and inflammatory response.* [unannotated for this claim]; 2: 22406695 — *Origin and functions of
  tissue macrophages.* [unannotated]; 3: 22621251 — *Ly6Chi monocytes direct alternatively activated
  profibrotic macrophage regulation of lung fibrosis.* [unannotated]. Reference 7521113 at rank 9.
- **Cited:** 36444198 S9 "Under inflammatory conditions … there was a significant increase in immature
  Ly-6C(high) monocytes, resembling the inflammatory left shift of granulocytes." and 22621251 S9
  "Depletion of Ly6Chi circulating monocytes reduced pulmonary fibrosis …" — both judged SUPPORT.
- **Model reasons:** "… which suggests they have higher inflammatory capacity compared to Ly-6:lo
  counterparts." and "… indicating Ly6Chi monocytes promote inflammation and fibrosis, thus having
  higher inflammatory capacity than Ly6Clo."
- **Reading:** two problems. (a) The reasons say "higher", the verdict says SUPPORT for "lower" —
  verdict–reason inconsistency. (b) The near-identical claim **728** ("… lower inflammatory capacity
  than Ly6C lo monocytes.") is gold **CONTRADICT** with 36444198 S9–S10 as its gold evidence. So for
  727 a correct reading of the retrieved abstract gives CONTRADICT, which would still be scored wrong
  against the gold NEI.
- **Earliest stage:** annotation-ambiguity (gold depends on which abstract was cited), with a
  verdict–reason inconsistency on top.
- **Improvement:** C1 fixes for the inconsistency; flag near-duplicate claims with different gold
  labels in the data report.

#### N5 — claim 437: "Functional consequences of genomic alterations due to Myelodysplastic syndrome (MDS) are poorly understood due to the lack of an animal model."

- **Gold** NEI. **full** SUPPORT · **oracle** NEI · **judge_only** NEI.
- **Retrieved top 3:** 1: 39851630 — *Reserves, functional, immunoregulatory, and cytogenetic properties
  of bone marrow mesenchymal stem cells in patients with myelodysplastic syndromes.* [unannotated]; 2:
  5765455 — *Reactive oxygen species, DNA damage, and error-prone repair: a model for genomic
  instability with progression in myeloid leukemia?* [unannotated]; 3: 2359152 — *Landscape of genetic
  lesions in 944 patients with myelodysplastic syndromes* [unannotated]. Reference 18399038 not in the
  top 10.
- **Cited:** 5765455 S2 "Using our two-step mouse model for myeloid leukemic disease progression …"
  and S3; 39851630 S5 (impaired proliferative potential of patient MSCs). Both SUPPORT.
- **Model reason (5765455):** "… indicating that genomic alterations can be studied in an animal model,
  thus addressing the claim that functional consequences are poorly understood due to lack of animal
  models."
- **Reading:** the reason argues that an animal model exists (evidence *against* the claim, if
  anything) and labels it SUPPORT: "addresses the claim" is being treated as "supports the claim".
- **Earliest stage:** judgement error (polarity: relevance mistaken for support).
- **Improvement:** C1 fixes; a prompt contrast example "evidence that is on topic but argues the
  opposite is CONTRADICT; evidence that is on topic but does not settle it is NEI".

#### N6 — claim 1278: "The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events."

- **Gold** NEI. **full** SUPPORT · **oracle** NEI · **judge_only** NEI.
- **Retrieved top 3:** 1: 1454773 — *In vitro characterization of the anti-PD-1 antibody nivolumab,
  BMS-936558, and in vivo toxicology in non-human primates.* [unannotated]; 2: 25589047 —
  *Treatment-related mortality with bevacizumab in cancer patients: a meta-analysis.* [unannotated]; 3:
  11254040 — *Adverse Events Associated With the Treatment of Multidrug-Resistant Tuberculosis …*
  [unannotated]. Reference 11335781 (*Is autoimmunity the Achilles' heel of cancer immunotherapy?*) not
  in the top 10.
- **Cited:** 1454773 S7 "Nivolumab treatment did not induce adverse immune-related events when given to
  cynomolgus macaques at high concentrations, independent of circulating anti-nivolumab antibodies where
  observed." → SUPPORT.
- **Reading:** population substitution (macaques ≠ cancer patients) and scope (a universal "does not
  cause any" from one toxicology study). The reference abstract, had it been retrieved, says the
  opposite ("immunotoxicity and autoimmunity are emerging as the Achilles' heel of immunotherapy") but
  is a perspective without data, hence NEI.
- **Earliest stage:** judgement error (population / scope). Retrieval also missed the reference, but
  that does not matter for an NEI claim.
- **Improvement:** prompt rule "evidence in animals or cell lines does not settle a claim about
  patients"; treat universal quantifiers ("any", "all", "never") as requiring direct evidence.

## 3. Summary over all 44 `full` failures

### 3.1 Gold → predicted cells (unit: claims)

| Gold (n claims) | → SUPPORT | → CONTRADICT | → NEI | → ERROR | Wrong / gold claims |
|---|---|---|---|---|---|
| SUPPORT (41) | — | 2 | 2 | 0 | 4 / 41 (9.8 %) |
| CONTRADICT (21) | 15 | — | 2 | 0 | 17 / 21 (81.0 %) |
| NEI (38) | 23 | 0 | — | 0 | 23 / 38 (60.5 %) |
| **All (100)** | 38 | 2 | 4 | 0 | **44 / 100** |

38 of the 44 failures (86 %) are false SUPPORT verdicts: NEI → SUPPORT (23) and CONTRADICT →
SUPPORT (15).

### 3.2 Earliest responsible stage (unit: claims; denominator 44 failures)

| Stage | Heuristic (`failure_cases.csv`) | Human reading | Human: gold S / C / N |
|---|---|---|---|
| operational | 0 | 0 (0 %) | 0 / 0 / 0 |
| retrieval miss | 1 | 1 (2 %) | 0 / 1 / 0 |
| evidence miss | 5 | 5 (11 %) | 2 / 3 / 0 |
| judgement error | 38 (15 + 23 "false decisive verdict") | 31 (70 %) | 1 / 13 / 17 |
| annotation-ambiguity | — (not a heuristic category) | 7 (16 %) | 1 / 0 / 6 |
| **Total** | 44 | 44 | 4 / 17 / 23 |

Human stage vs heuristic, the 9 reclassified claims: **852** evidence miss → judgement error (the
selected S6–S8 carry the gold finding); **338** judgement error → evidence miss (decisive fragment
S10 and conclusion S15 not passed); **1368** judgement error → annotation-ambiguity; and six gold-NEI
claims judgement error → annotation-ambiguity: **507** (uncited 7433668 "Preexisting helminth infection
induces inhibition of innate pulmonary anti-tuberculosis defense by engaging the IL-4 receptor pathway"
genuinely supports the claim), **727** (twin of gold-CONTRADICT claim 728), **870** (uncited bariatric
abstract: "Significant correlations between weight loss and improvements in … physical aspects of
HRQOL"), **1099** (statins, N3), **1207** (cited abstract S1–S2: MIIB "downregulated in differentiated
cells", MIIA "constitutive" — close to the claimed B → A switch), **1213** (uncited arthritis/EAE
abstracts show harmful monocyte activation). These remain errors in every metric; the
reclassification only explains them.

### 3.3 Recurring patterns (primary pattern per failure; unit: claims, denominator 44)

| Pattern | Claims | n | Typical form |
|---|---|---|---|
| Polarity / direction inversion | 180, 249, 516, 532, 578, 619, 728, 852, 903, 1336 (gold C); 437, 1197 (gold N) | 12 | Evidence (often quoted correctly in the reason) says the opposite of the claim; label SUPPORT |
| Entity / outcome / population substitution | 1, 36, 268, 552, 637, 768, 1175, 1278, 1279 | 9 | Folate for B12 (36), CDCA for cold (268), RIG-I for MDA5 (1175), macaques for patients (1278), bevacizumab for co-IR blockade (1279), mental-health scores for homelessness (637) |
| Over-inference from mechanistically related text | 237, 303, 350, 551, 913, 1029, 1282 | 7 | A topically related mechanism sentence taken as proof |
| Annotation-ambiguity | 507, 727, 870, 1099, 1207, 1213, 1368 | 7 | Uncited abstracts support the claim, or the gold label is debatable |
| Evidence miss | 142, 208, 338, 452, 1020 | 5 | Empty or partial selection in an on-topic reference abstract |
| Numeric / comparative | 922, 759 | 2 | HR 0.79 read as "faster"; "A over B" comparison ignored (338 also, secondary) |
| Negation | 1298 | 1 | Negated claim + negated evidence read as disagreement (208, 1197, 1278 also contain negation) |
| Retrieval miss | 1041 | 1 | Evidence abstract demoted out of the top 10 by the re-ranker |

Cross-cutting observations (computed):

1. **NEI → SUPPORT over-calling when topically related papers are retrieved.** In 18 of the 23
   NEI → SUPPORT errors the SUPPORT came *only* from uncited, unannotated abstracts; in 5 it came from
   the cited (NEI) abstract. The oracle, which sees only the cited abstracts, answers NEI on 16 of these
   23. Over all 226 unannotated abstracts examined by `full`, the judge said SUPPORT 60 times and
   CONTRADICT 8 times. judge_only also says SUPPORT on 12 of the 23: many SciFact NEI claims are
   plausible or true statements whose cited abstract simply does not settle them, and the model fills
   the gap with outside knowledge.
2. **CONTRADICT → SUPPORT is a judgement problem, not a retrieval problem.** In 14 of the 15
   CONTRADICT → SUPPORT errors the gold-CONTRADICT abstract *was* examined and was itself judged
   SUPPORT (the 15th is the retrieval miss 1041). The oracle gives SUPPORT on all 15. At pair level,
   gold-CONTRADICT abstracts were judged SUPPORT in 14 of 20 examined pairs (`full`) and 15 of 24
   (`oracle`).
3. **The model's reason often contradicts its own verdict.** In 14 of the 44 failures (180, 249, 516,
   532, 578, 619, 728, 852, 903, 1336, 437, 1197, 1298, 727) the stated reason, read literally, argues
   for a different label than the one emitted — 10 of the 17 gold-CONTRADICT failures. A simple keyword
   test (reason mentions "contradict", "refute", "not support", "opposite") flags only 2 of the 115
   SUPPORT document verdicts in `full` (516, 619), so a semantic check is needed.
4. **Empty evidence selections on on-topic abstracts.** The evidence agent selected nothing in 142 of
   the 300 examined abstracts; that is usually right (distractors), but it included 4 evidence-bearing
   reference abstracts (claims 142, 208, 793, 1020; 793 was still answered correctly).
5. **Distractors also cause conflicts.** Of the 3 conflicts in `full`, two fall on gold-NEI claims
   (213, 783; scored correct by the conflict → NEI convention) and one (1029) turns two correct
   CONTRADICT decisions into a wrong NEI.

### 3.4 Improvements per category (for a future `final_v2`, not applied here)

| Category | n | Concrete improvement |
|---|---|---|
| Polarity / verdict–reason inconsistency | 12 (+2 secondary) | Put `reason` (and explicit `claim_direction` / `evidence_direction` fields) **before** `verdict` in the judge JSON schema; supervisor consistency check that re-asks once when the reason argues for another label (NLI-style or a short second model call). |
| Entity / population substitution | 9 | Supervisor check that a decisive cited sentence mentions the claim's key entities (lexical or embedding overlap, using the title as context); prompt rules for animal vs human and different intervention ⇒ NEI. |
| Over-inference / outside knowledge on NEI | 7 | Stronger "use only the supplied sentences" instruction with an NEI example; require that a SUPPORT cites a sentence that states the claimed relation, not just the entities; consider rank-weighted aggregation so a single weak SUPPORT from a low-ranked distractor cannot decide (or cause a conflict). |
| Evidence miss | 5 | Re-ask once when the top-ranked abstract gets an empty selection; always include the conclusion sentence; pass neighbouring sentences for fragments; raise the 1–3 sentence budget to ≤ 5; state that null results are evidence. |
| Numeric / comparative | 2 (+1) | Prompt rule for ratios (< 1 = reduced risk / slower); deterministic annotation of ratios and percentages; decompose "A versus B" claims. |
| Negation | 1 (+3) | Normalise negated claims into an explicit proposition before judging. |
| Retrieval miss | 1 | Biomedical cross-encoder, or score fusion of RRF and re-ranker instead of re-ordering by the re-ranker alone. |
| Annotation-ambiguity | 7 | No system change; keep them as errors, report them separately, and resolve with a second annotator if the subset is ever re-used. |
| Operational | 0 | None needed in this run (0 errors in 800 claim runs). |

The largest expected gains are in the judge (31 of 44 failures, 70 %), not in retrieval (1) or
evidence selection (5). Because the judge is a 4B quantised model shared by all agents, a LoRA-tuned
judge trained on the SciFact training split (with a held-out comparison against the same base model)
is the most direct next experiment — again as `final_v2`, not as a revision of `final_v1`.

## 4. All 44 `full` failures

H = heuristic stage, R = human-reviewed stage. Pattern as in §3.3.

| Claim | Gold | full | oracle | judge_only | H | R | Pattern |
|---|---|---|---|---|---|---|---|
| 142 | C | NEI | NEI | NEI | evidence miss | evidence miss | empty selection (C5) |
| 180 | C | S | S | NEI | judgement | judgement | polarity (C2) |
| 249 | C | S | S | NEI | judgement | judgement | polarity: "increased" cited as support for "reduces" |
| 338 | C | S | S | S | judgement | evidence miss | partial evidence + numeric (C4) |
| 452 | C | S | S | C | evidence miss | evidence miss | gold S0 ("Genetically identical cells … markedly different phenotypes") not selected |
| 516 | C | S | S | C | judgement | judgement | polarity, reason says "contradicted" (C1) |
| 532 | C | S | S | NEI | judgement | judgement | polarity: lower fibrinogen → higher patency read as support |
| 578 | C | S | S | NEI | judgement | judgement | polarity: reason says loss "would not facilitate" |
| 619 | C | S | S | C | judgement | judgement | polarity, reason says "contradicts the claim" |
| 728 | C | S | S | C | judgement | judgement | polarity (twin of 727) |
| 759 | C | S | S | S | judgement | judgement | comparative (C9) |
| 852 | C | S | S | S | evidence miss | judgement | polarity on normative claim (C8) |
| 903 | C | S | S | NEI | judgement | judgement | polarity: "induced IL-10" cited for "reduces IL-10" |
| 922 | C | S | S | C | judgement | judgement | numeric HR 0.79 (C3) |
| 1029 | C | NEI | C | C | judgement | judgement | distractor SUPPORT → conflict (C7) |
| 1041 | C | S | S | S | retrieval miss | retrieval miss | evidence doc outside top 10 (C6) |
| 1336 | C | S | S | NEI | judgement | judgement | polarity: cord blood "approximated healthy diversity" read as "reduces" |
| 208 | S | NEI | NEI | C | evidence miss | evidence miss | empty selection, null result (S2) |
| 1020 | S | C | NEI | NEI | evidence miss | evidence miss | empty selection + knockout logic (S3) |
| 1298 | S | C | C | NEI | judgement | judgement | negation (S1) |
| 1368 | S | NEI | NEI | S | judgement | annotation | "term of delivery" not addressed (S4) |
| 1 | N | S | NEI | NEI | judgement | judgement | substitution: EMT stem-cell lines for "0-dimensional biomaterials" |
| 36 | N | S | NEI | S | judgement | judgement | substitution: folate for vitamin B12 |
| 237 | N | S | S | S | judgement | judgement | over-inference: ctsR deletion → clpC sporulation |
| 268 | N | S | NEI | S | judgement | judgement | substitution: CDCA for cold (N1) |
| 303 | N | S | NEI | NEI | judgement | judgement | over-inference: MHM adjacent to DMRT1 ≠ regulates it |
| 350 | N | S | S | S | judgement | judgement | over-inference: IF3 conformational change ≠ tRNA discrimination |
| 437 | N | S | NEI | NEI | judgement | judgement | polarity: "addresses" taken as "supports" (N5) |
| 507 | N | S | S | NEI | judgement | annotation | uncited helminth/IL-4R abstract supports claim |
| 551 | N | S | NEI | NEI | judgement | judgement | over-inference / misread ("dissociation requires phosphorylation") |
| 552 | N | S | S | NEI | judgement | judgement | substitution: active disease for gluten-free diet |
| 637 | N | S | NEI | S | judgement | judgement | substitution: child mental-health scores for homelessness |
| 727 | N | S | NEI | C | judgement | annotation | twin claim of 728 (N4) |
| 768 | N | S | NEI | S | judgement | judgement | substitution: 6MP dose intensity for TPMT metabolism |
| 870 | N | S | NEI | S | judgement | annotation | uncited bariatric HRQOL abstract supports claim |
| 913 | N | S | NEI | S | judgement | judgement | over-inference (PPAR ligands activate, not inhibit) |
| 1099 | N | S | NEI | S | judgement | annotation | statins (N3) |
| 1175 | N | S | NEI | NEI | judgement | judgement | substitution: RIG-I for MDA5 (N2) |
| 1197 | N | S | NEI | NEI | judgement | judgement | polarity: reason implies support "could decrease homelessness" for a "not effective" claim |
| 1207 | N | S | S | NEI | judgement | annotation | cited abstract close to the claimed B → A switch |
| 1213 | N | S | S | S | judgement | annotation | uncited monocyte/arthritis abstracts support claim |
| 1278 | N | S | NEI | NEI | judgement | judgement | population/scope: macaques, "any" (N6) |
| 1279 | N | S | NEI | S | judgement | judgement | substitution: bevacizumab FAEs for co-IR blockade autoimmunity |
| 1282 | N | S | S | S | judgement | judgement | over-inference: case series "effective" → "anecdotal" |

Counts check: 17 CONTRADICT + 4 SUPPORT + 23 NEI = 44 rows.

## 5. Contrast with the oracle and the judge-only baseline

- **oracle** is right and `full` wrong on 17 claims, of which **16 are gold NEI** (1 CONTRADICT,
  claim 1029); `full` is right and oracle wrong on 6. The oracle's advantage is therefore almost
  entirely *not being shown distractor abstracts* on NEI claims (it reads 114 abstracts for 100 claims;
  `full` reads 300), not better evidence on SUPPORT/CONTRADICT claims. On CONTRADICT the oracle is no
  better (3 / 21 correct vs 4 / 21).
- **judge_only** is right and `full` wrong on 17 claims (NEI 10, CONTRADICT 6, SUPPORT 1); `full` is
  right and judge_only wrong on 23 (SUPPORT 14, NEI 8, CONTRADICT 1). Of the 15 CONTRADICT → SUPPORT
  failures, judge_only answers CONTRADICT on 5 (452, 516, 619, 728, 922 — note 452 is an evidence-miss
  case, the others are judgement errors), NEI on 6 and SUPPORT on 4. Without documents the model
  recognises several inverted claims as false; shown an abstract on exactly that topic it tends to say
  SUPPORT. judge_only answers are not grounded in any quotation, so they cannot be checked.

## 6. Development observation (not part of the final population)

The known case *"1 in 5 million in UK have abnormal PrP positivity."* is **training claim 2** (gold
CONTRADICT, evidence abstract 13734012, S4). It is **not** in the final 100 claims and not in the 20
`smoke_v1` claims; it was observed during development with an interactive check, and the model's
answers are still in the LLM cache: the evidence agent selected S4 (note: "Sentence 4 provides the
prevalence rate of 493 per million, which directly relates to the claim of 1 in 5 million (200 per
million).") and the judge answered SUPPORT, citing S4, with the reason "The study found 16 positive
appendix samples out of 32,441, indicating a prevalence of 493 per million, which is higher than 1 in
5 million (200,000 per million)." The reason states that the observed prevalence is higher than the
claimed one and still labels it SUPPORT; it also converts 1 in 5 million wrongly (it is 0.2 per
million, not 200 or 200,000). It is the same polarity / numeric pattern as C1–C3 above and is reported
only as a development observation.

## 7. Other observations

- **Stray ":" inside words in model reasons** (for example "IL-:2", "t:RNA", "gliob:astoma",
  "cynomolg:macaques"): 117 of the 974 non-skipped judge reasons across all variants contain such a
  mid-token colon. The same text is present in the raw responses stored in `artifacts/cache/llm/`, so
  it comes from the model output, not from post-processing. Reasons are displayed as model text;
  quotations are copied from the corpus and are unaffected (quote validity 1.000 in every variant).
- **No operational errors:** 0 errors in 800 claim runs (8 variants × 100 claims), and no downgraded
  judgement in any pipeline variant. **Correction (final review):** the earlier statement "0 claims
  with a downgraded judgement" was wrong for `retrieve_judge_single_prompt`: claims **350, 437, 793
  and 852** were downgraded to NEI because the model wrote its citations as `"Document <id>"`
  (echoing the prompt's `Document <id>:` header) and the citation parser did not strip the label, so
  valid citations were rejected (`downgraded_no_citation` events in `events.jsonl`; the metrics code
  only counted per-document downgrades, so the report showed 0). Raw model answers: 350 SUPPORT and
  437 SUPPORT (both gold NEI, so the downgrade made them correct), 793 CONTRADICT (gold CONTRADICT, a
  correct answer lost), 852 SUPPORT (gold CONTRADICT, wrong either way). The frozen result stays
  0.560 / 0.450; the raw answers would score 0.550 / 0.458 — within noise. The parser and the
  downgrade count are fixed for future runs (`Docs/EVALUATION.md` §8.5).
