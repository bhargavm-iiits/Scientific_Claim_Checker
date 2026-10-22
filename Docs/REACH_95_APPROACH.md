# Proposed route toward >95% accuracy and Macro-F1

Status: a proposed experiment sequence, not a demonstrated >95% recipe. No new
model training or promotion is claimed here. The best measured original-300
result remains 242/300 (80.6667%) and Macro-F1 0.7959743388. Strict >95%
accuracy requires at least 286 correct, a net gain of 44; that alone does not
establish Macro-F1 >0.95 or all other metrics >0.95.

## What the current errors imply

A fresh read-only count of the two original best-model prediction files finds:

| Error category | Claims |
|---|---:|
| Wrong verdict despite examining an annotated paper | 35 |
| Annotated paper outside the top three | 10 |
| Annotated paper in top three but removed before judging | 3 |
| Decisive verdict on gold NEI | 10 |
| Total errors | 58 |

Prediction transitions are SUPPORT→NEI 23, CONTRADICT→NEI 22, SUPPORT→CONTRADICT
3, NEI→SUPPORT 5 and NEI→CONTRADICT 5. Thus 45 errors are missed decisive
verdicts; merely making the model more conservative is insufficient.
Correcting just the 13 retrieval/gating cases while preserving every other
decision would yield 255/300, or 85%, not the requested target. This is a
case-count diagnostic, not an upper bound on every possible retrieval change.

The LoRA audit additionally exposed false contradictions from mismatched
entities and over-selection of irrelevant sentences. Its data/source checks
and exact quotations passed, but its semantic quality gates failed.

## First experiment: learn claim aggregation on unseen predictions

Keep the strongest architecture's original retrieval settings and use the
existing grouped TRAIN recipe for the reader. The new component is a small,
regularized claim-level classifier that combines evidence signals. Inputs can
include source scores, support/contradiction/neutral probabilities, agreement
between sentence and document contexts, entity/relationship/scope alignment,
and numeric-comparison features. The classifier sees source text and predictions,
never gold reference IDs or gold sentence IDs at inference.

This differs from averaging two NLI models or tuning fixed probability cutoffs:
it learns when a candidate set establishes SUPPORT, CONTRADICT or NEI. A missing
lexical match is not automatically NEI; aliases and implicit relationships must
remain possible. A changed number is not automatically CONTRADICT; the entity,
outcome, units, population and time must be comparable first.

Fit calibration/aggregation using predictions for TRAIN claims the underlying
reader did not fit. Evaluate it on separate outer groups. This requires nested
paper/text-grouped splitting, with early stopping and every synthetic descendant
kept inside its parent group. Use one declared feature set and strong fixed
regularization, not a new large threshold grid. Existing stopping/audit claims
have already influenced earlier work; disclose their reuse.

Run the unchanged baseline and new aggregator on the same outer folds. Count
every claim, class and operational failure. Freeze model selection and inference
before a new original-300 confirmation. A gain on reused validation alone cannot
establish absence of overfitting; a separate newly annotated locked evaluation
is needed for an additional independent generalization check.

## Additional changes if the first experiment shows a useful gain

1. **Build better evidence-aligned training data.** Add source-backed examples
   from fitting groups that test negation, relation direction, quantities/units,
   entity aliases, populations and time qualifiers. Verify labels and exact
   evidence before training. A generated negation or an unannotated paper is
   not automatically a valid contradiction or NEI example. Filter validation
   claims, close paraphrases and claim-family overlap from augmentation. No
   per-validation-claim rules, copied answers or uncertain benchmark-fitted weights.
2. **Train on the evidence distribution encountered at inference.** Use genuine
   retrieval candidates, with verified hard distractors, alongside reference
   evidence. Preserve full-document context and sentence IDs. Teach the model
   to reject a different drug/outcome while recognizing a valid alias or a
   quantitatively opposite result. We already tried a joint verdict/rationale
   reader; repeating that architecture alone is not a new approach. Reviewed
   hard-case supervision and retrieval-aware claim-level learning are the
   proposed substantive additions.
3. **Improve scientific paper ranking separately.** Train or adapt the paper
   reranker on fitting TRAIN query/reference pairs and checked distractors.
   Evaluate all-claim reference Recall@3/@10 separately from evidence-bearing
   recall. Increasing the candidate pool alone already failed; assess ranking
   quality rather than assuming more candidates are better. If the embedding
   encoder changes, rebuild and fingerprint its index.
4. **Learn selective evidence sets.** Retain neighboring context for reasoning,
   but train rationale selection using actual sentence annotations. Permit
   multiple necessary sentences, without a fixed two-sentence ceiling. Measure
   precision and recall together; the LoRA candidate's 843 exact quotations
   did not make its 25.32% evidence precision acceptable.
5. **Add bounded verification checks.** Compare structured entity, relation,
   population, negation and numerical fields against cited spans. Use local
   models within the brief's compute limit. These checks need their own
   accuracy evaluation; schema-conforming output does not prove correct facts.
   Route uncertain evidence to a separately calibrated decision. NEI remains
   a scored class, and no difficult claims disappear from the denominator.

Each stage needs an ablation against the preceding accepted version. Combining
all changes immediately would make gains and regressions difficult to attribute.
The original best profile and all frozen prior experiments remain preserved.

## Evidence and constraints

The local P08 brief permits open models of at most 4B parameters on a free T4
or laptop CPU and asks for oracle/retrieved comparisons, retrieval recall and
sentence evidence precision/recall. General-science evaluation remains separate
from the original SciFact-300 target. Oracle information is diagnostic only.

Relevant research supports ingredients, not a promise of 95% in this project:

- [MultiVerS](https://aclanthology.org/2022.findings-naacl.6/) studies full-context
  verdict/rationale learning and scientific domain adaptation. Its published
  SciFact-trained weights are not automatically a clean validation checkpoint.
- [What Makes Medical Claims (Un)Verifiable?](https://aclanthology.org/2024.eacl-long.124/)
  examines entities, relations and query normalization, and finds unverifiability
  recognition difficult.
- [The Alignment Bottleneck in Decomposition-Based Claim Verification](https://www.ijcai.org/proceedings/2026/608)
  finds decomposition benefits depend on aligned evidence. It evaluates other
  datasets; its results are not SciFact scores or evidence of >95% here.

The recommended priority is a retrieval-aware, calibrated claim decision plus
better verified training evidence. There is currently no demonstrated recipe
in this repository that guarantees >95% across all requested metrics under the
brief's constraints.
