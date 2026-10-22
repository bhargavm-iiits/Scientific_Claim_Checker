# LoRA scientific verifier: ten audit failures

Completed 23 October 2026. Population: all 163 frozen TRAIN-audit claims.
The model made 42 errors, with no operational failures. This is a reused
internal audit, not an independent test. Review does not change labels,
thresholds, predictions, checkpoint selection or the rejected candidate.

Cases are selected deterministically by cycling CONTRADICT, SUPPORT and NEI,
with lexical claim-ID order within each class, up to ten. This is an explanatory
sample, not a filtered performance population. Exact claims, predictions and
quotations are in `failure_cases.csv` and `predictions.jsonl`.

| Claim ID | Claim | Gold → prediction | Source observation and interpretation |
|---|---|---|---|
| 1081 | Side effects associated with antidepressants do not represent a mortality risk to postmenopausal women. | CONTRADICT → NEI | The reported annotated paper 5691302 contains increased mortality hazard/absolute-risk findings in sentences 10 and 13. The candidate emits no quotations. Retaining the paper did not produce the annotated contradiction; population scope and negative phrasing are possible difficulties, not established causes. |
| 1133 | TMEM27 is a marker for beta cells. | SUPPORT → NEI | Reported paper 24142891 identifies Tmem27 in pancreatic beta cells and describes beta-cell-specific cleavage (sentences 1, 4 and 6). No quotation is emitted. The claim compresses several observations into the term “marker”; the source is available, so this is a verifier failure under the saved annotation. |
| 1085 | Side effects associated with antidepressants lower risk of myocardial infarction. | NEI → CONTRADICT | The output cites paper 15984735 about migraine and cardiovascular disease, including greater risk among women. Those quotations do not identify antidepressant exposure. This illustrates an entity/exposure mismatch despite exact source copying. |
| 1156 | The DESMOND program caused substantial weight loss in most participants. | CONTRADICT → NEI | Reported paper 12584053 concludes no difference in biomedical/lifestyle outcomes at three years (sentence 13). No quotations are emitted. The verifier fails to connect the program outcome to the annotated contradiction. |
| 1297 | There was an estimated 30 million cases of pneumonia in young children worldwide in 2010. | SUPPORT → NEI | Reported paper 9167230 instead annotates a sentence estimating 11.9 million severe and 3.0 million very severe ALRI hospital admissions. The full stored abstract does not explicitly state 30 million pneumonia cases. This is a possible numerical/scope or annotation ambiguity requiring separate adjudication, not a basis for changing the gold label or excluding the error. |
| 1302 | Tirasemtiv has no effect on cardiac muscle. | NEI → CONTRADICT | The output cites paper 29526125 about tirofiban and troponin measurements in acute coronary syndromes. The cited source discusses a different named treatment. A semantically unrelated drug context is accepted as contradiction. |
| 1237 | The most prevalent adverse events to Semaglutide are cardiovascular. | CONTRADICT → NEI | Reported paper 3654468 says gastrointestinal events were most common (sentence 12). No quotations are emitted. The source provides an explicit adverse-event category comparison that the model does not turn into the annotated verdict. |
| 1305 | Tirasemtiv targets fast-twitch muscle. | SUPPORT → NEI | Annotated paper 12631697 was scored but not retained; the final reported list and quotations are empty. Its evidence names CK-2017357 and fast-skeletal-troponin, rather than spelling out the claim's treatment name. Alias resolution is a plausible difficulty; no external alias lookup or label correction was performed. |
| 1331 | Tumor development occurs in conjunction with upregulation of pro-inflammatory cytokines. | NEI → SUPPORT | The output cites paper 17223891 about Nlrp12-deficient mice with increased inflammation and tumorigenesis. This is related evidence, but the fixed claim's gold label is NEI. NEI in this benchmark reflects the annotation/evidence protocol, not a declaration that all related propositions are false. The mismatch remains scored as an error; it is not silently relabelled. |
| 1327 | Tuberculosis incidence occurs at higher rates in more sparsely populated areas in the UK. | CONTRADICT → NEI | Reported annotated papers 24241932 and 22194407 describe associations with overcrowding, including sentences 6–7 and 3/9 respectively. The model emits no quotations. Inferring the annotated contradiction requires handling population density/direction and the papers' incidence-versus-mortality distinctions. |

The aggregate error categories are 22 wrong verdicts despite retained annotated
papers, 16 decisive outputs on gold NEI, and four annotated papers scored but not
retained. These categories describe pipeline positions; they do not prove every
human hypothesis above. No errors are attributed to missing retrieval candidates
in this disjoint categorization, because the relevant papers were scored for all
non-NEI errors.

The model also over-selects citations: only 118 of 466 cited sentences in annotated
papers are gold evidence, even though all 843 total quotations match source text.
Removing the two-sentence cap corrects a recall-capacity limit but does not supply
a sufficiently selective evidence model. The wider retrieval pool and rescue
path can expose more unrelated material; this combined experiment has no causal
ablation establishing how much each component contributes.

The frozen audit rejects this candidate. The 13 fixed errors versus 10 new errors
give only three net correct claims, and paired group-bootstrap intervals include
zero. No fresh 807-claim refit, original-300 validation run, oracle substitution,
threshold sweep or claim-specific patch followed this review. A future model
needs a new declared protocol and evidence of improvement, with selection reuse
and any annotation adjudication disclosed.
