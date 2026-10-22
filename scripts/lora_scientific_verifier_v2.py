"""All-layer LoRA and source-preserving, verification-guided scientific reading."""
import re
import time
from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model

from claim_checker.agents.evidence import build_quotes
from claim_checker.agents.explainer import explain
from claim_checker.agents.judge_nli import judge_nli
from claim_checker.data.store import sha256_file
from claim_checker.data.verifier_context import format_premise
from claim_checker.models.verifier import NLIVerifier
from claim_checker.pipeline.aggregate import aggregate
from claim_checker.pipeline.supervisor import validate_claim_text
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import ClaimResult, DocumentReport, ErrorInfo


class CachingVerifier(NLIVerifier):
    """Cache identical premise/claim scores within one claim, without annotations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_cache()

    def reset_cache(self):
        self._scores = {}

    def score(self, premises, hypotheses, batch_size=4):
        if len(premises) != len(hypotheses):
            raise ValueError("Each premise needs one hypothesis")
        keys = list(zip(premises, hypotheses, strict=True))
        missing = list(dict.fromkeys(key for key in keys if key not in self._scores))
        if missing:
            values = super().score([p for p, _ in missing], [h for _, h in missing], batch_size=batch_size)
            self._scores.update(zip(missing, values, strict=True))
        return [dict(self._scores[key]) for key in keys]


def judge_document(verifier, claim, doc, context_ids, *, threshold):
    context_ids = sorted(set(context_ids))
    result = judge_nli(verifier, claim, doc, context_ids, threshold=threshold)
    if result.verdict == "NEI":
        return result
    # All original source sentences remain eligible; no fixed two-quote ceiling.
    scores = verifier.score([verifier.format_premise(doc.title, doc.sentences[sid]) for sid in context_ids],
                            [claim] * len(context_ids))
    chosen = [sid for sid, score in zip(context_ids, scores, strict=True) if score[result.verdict] >= .5]
    if not chosen:
        chosen = result.cited_sentence_ids
    return result.model_copy(update={"cited_sentence_ids": chosen,
                                     "reason": "The source reports: " + " ".join(doc.sentences[sid] for sid in chosen)})


def adapt_encoder(base, *, rank=16, alpha=32, dropout=.1):
    """Adapt attention in every layer while retaining frozen pretrained weights."""
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["query_proj", "value_proj"], modules_to_save=["pooler", "classifier"],
        bias="none",
    ))
    layers = {int(m.group(1)) for name, p in model.named_parameters() if p.requires_grad
              and (m := re.search(r"encoder\.layer\.(\d+)\..*\.lora_[AB]\.", name))}
    assert layers == set(range(base.config.num_hidden_layers)), "Every encoder layer must receive LoRA"
    unexpected = [name for name, p in model.named_parameters() if p.requires_grad
                  and not any(part in name for part in (".lora_A.", ".lora_B.", ".modules_to_save."))]
    assert not unexpected, unexpected
    return model


def weak_sentence_negatives(rows, corpus, gold):
    """One lexical hard non-rationale sentence per positive TRAIN claim/paper.

    These are weak NEI labels, not a claim that unannotated sentences cannot be
    evidence. They receive weight 0.1; annotation alternatives are unioned first.
    """
    result, seen = [], set()
    for row in rows:
        key = (row["claim_id"], row["doc_id"])
        if row["label"] == "NEI" or key in seen:
            continue
        seen.add(key)
        g, doc = gold[key[0]], corpus[key[1]]
        positives = set(g.evidence_by_doc.get(doc.doc_id, []))
        candidates = [sid for sid, sentence in enumerate(doc.sentences) if sid not in positives and sentence.strip()]
        if not candidates:
            continue
        words = set(re.findall(r"\w+", row["claim"].casefold()))
        sid = max(candidates, key=lambda i: (len(words & set(re.findall(r"\w+", doc.sentences[i].casefold()))), -i))
        result.append({"claim_id": row["claim_id"], "claim": row["claim"], "doc_id": doc.doc_id,
                       "premise": format_premise(doc.title, doc.sentences[sid], True), "label": "NEI",
                       "view": "weak_nonrationale", "context_sentence_ids": [sid], "class_weight": .1})
    return result


def row_weight(row):
    return .25 if row["view"] == "hard_negative" else .1 if row["view"] == "weak_nonrationale" else 1.


def pick_rescues(initial, candidates, cutoff=.9):
    """Gold-free fallback only when the initial reader supplied no decisive label."""
    if any(j.verdict != "NEI" or "window_conflict" in j.consistency_flags for j in initial):
        return []
    conflicts = [item for item in candidates if "window_conflict" in item[1].consistency_flags]
    if conflicts:
        return [min(conflicts, key=lambda item: item[0].rank)]
    eligible = [(hit, judgement) for hit, judgement in candidates
                if judgement.verdict != "NEI" and judgement.confidence is not None and judgement.confidence >= cutoff]
    return [max((item for item in eligible if item[1].verdict == label),
                key=lambda item: (item[1].confidence, -item[0].rank))
            for label in ("SUPPORT", "CONTRADICT") if any(item[1].verdict == label for item in eligible)]


def promote_hits(hits, doc_ids):
    """Keep the real candidate pool and move a verified rescue into the top three."""
    if len({h.doc_id for h in hits}) != len(hits):
        raise ValueError("Duplicate retrieval documents")
    if not 1 <= len(doc_ids) <= 2 or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("One or two distinct rescue sources are required")
    if not set(doc_ids) <= {h.doc_id for h in hits}:
        raise ValueError("Rescue source was not genuinely retrieved")
    selected = [next(h for h in hits if h.doc_id == did) for did in doc_ids]
    other = [h for h in hits if h.doc_id not in doc_ids]
    split = min(3 - len(selected), len(other))
    order = other[:split] + selected + other[split:]
    return [h.model_copy(update={"rank": i}) for i, h in enumerate(order, 1)]


class FeedbackChecker:
    """Experimental adapter; original product configuration is unchanged."""
    def __init__(self, cfg, model_dir, *, feedback=True):
        self.cfg, self.feedback = cfg, feedback
        self.resources = ClaimChecker(cfg)
        self.verifier = CachingVerifier(model_dir, device=cfg.verifier.device, include_title=True, window_overlap=2)
        self.identity = {"model_dir": str(model_dir), "weights_sha256": sha256_file(Path(model_dir) / "model.safetensors"),
                         "labels": self.verifier.labels, "feedback": feedback}

    @property
    def corpus(self):
        return self.resources.corpus

    def check(self, claim, claim_id=None, oracle_doc_ids=None):
        started = time.perf_counter()
        metadata = {"strategy": "pipeline", "judge_backend": "nli", "verifier": self.identity,
                    "options": {"oracle_doc_ids": oracle_doc_ids}, "rescue_attempted": False}
        try:
            if hasattr(self.verifier, "reset_cache"):
                self.verifier.reset_cache()
            text = validate_claim_text(claim, self.cfg.runtime.max_claim_chars, self.cfg.runtime.max_claim_tokens,
                                       self.resources.count_claim_tokens)
            if oracle_doc_ids is not None:
                if not oracle_doc_ids:
                    raise ValueError("Oracle needs source IDs")
                from claim_checker.schemas import RetrievalHit

                hits = [RetrievalHit(doc_id=did, rank=rank, method="oracle", score=0., title=self.corpus[did].title)
                        for rank, did in enumerate(dict.fromkeys(oracle_doc_ids), 1)]
                initial_hits = hits
            else:
                hits = self.resources._retrieve(text, "hybrid", True, 20).hits
                metadata["initial_retrieval_hits"] = [h.model_dump(mode="json") for h in hits]
                initial_hits = [h for h in hits[:3] if h.score >= -2]
            judged = {}
            for hit in initial_hits:
                doc = self.corpus[hit.doc_id]
                judged[hit.doc_id] = judge_document(self.verifier, text, doc, list(range(len(doc.sentences))),
                                              threshold={"SUPPORT": .5, "CONTRADICT": .9})
            chosen = list(initial_hits)
            if oracle_doc_ids is None and self.feedback and not any(
                j.verdict != "NEI" or "window_conflict" in j.consistency_flags for j in judged.values()
            ):
                metadata["rescue_attempted"] = True
                candidates = []
                for hit in hits:
                    if hit.doc_id in judged:
                        continue
                    doc = self.corpus[hit.doc_id]
                    judgement = judge_document(self.verifier, text, doc, list(range(len(doc.sentences))), threshold=.9)
                    judged[hit.doc_id] = judgement
                    candidates.append((hit, judgement))
                rescues = pick_rescues([judged[h.doc_id] for h in initial_hits], candidates)
                if rescues:
                    ids = [hit.doc_id for hit, _ in rescues]
                    metadata["rescued_doc_ids"] = ids
                    hits = promote_hits(hits, ids)
                    chosen_ids = [h.doc_id for h in chosen[:3 - len(rescues)]] + ids
                    chosen = [next(h for h in hits if h.doc_id == did) for did in chosen_ids]
            metadata["scored_candidate_doc_ids"] = list(judged)
            metadata["candidate_judgements"] = {did: j.model_dump(mode="json") for did, j in judged.items()}
            if oracle_doc_ids is None:
                metadata["retrieval_hits"] = [h.model_dump(mode="json") for h in hits]
            reports, quotes = [], []
            for hit in chosen:
                doc, judgement = self.corpus[hit.doc_id], judged[hit.doc_id]
                reports.append(DocumentReport(doc_id=hit.doc_id, title=doc.title, rank=hit.rank,
                                              retrieval_score=hit.score, retrieval_method=hit.method,
                                              judgement=judgement, sentences=doc.sentences, source=doc.source))
                quotes.extend(build_quotes(doc, judgement.cited_sentence_ids, judgement.verdict))
            judgements = [r.judgement for r in reports]
            verdict, conflict = aggregate(judgements)
            return ClaimResult(status="ok", claim=text, claim_id=claim_id, verdict=verdict, conflict=conflict,
                               documents=reports, evidence=quotes,
                               explanation=explain(verdict, conflict, judgements, {r.doc_id: r.title for r in reports},
                                                   context="supplied" if oracle_doc_ids is not None else "retrieved"),
                               timings={"total": time.perf_counter() - started}, metadata=metadata)
        except Exception as exc:
            return ClaimResult(status="error", claim=claim, claim_id=claim_id,
                               error=ErrorInfo(error_type=type(exc).__name__, message=str(exc)),
                               timings={"total": time.perf_counter() - started}, metadata=metadata)
