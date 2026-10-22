"""Local rationale selection with original source IDs and optional NLI fusion."""
import math

from claim_checker.models.evidence_classifier import select_sentence_ids
from claim_checker.schemas import EvidenceSelection


def select_trained_evidence(selector, verifier, claim, doc, cfg):
    probabilities = selector.score(claim, doc)
    if len(probabilities) != len(doc.sentences) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Rationale classifier returned invalid source sentence scores")
    nonneutral = [1.0] * len(probabilities)
    if cfg.alpha < 1:
        scores = verifier.score([verifier.format_premise(doc.title, s) for s in doc.sentences], [claim] * len(doc.sentences))
        if len(scores) != len(doc.sentences):
            raise ValueError("NLI fusion omitted a source sentence")
        nonneutral = [1 - score["NEI"] for score in scores]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in nonneutral):
            raise ValueError("NLI fusion returned invalid sentence scores")
    fused = [p ** cfg.alpha * n ** (1 - cfg.alpha) for p, n in zip(probabilities, nonneutral, strict=True)]
    ids = select_sentence_ids(fused, threshold=cfg.threshold, maximum=cfg.maximum)
    selection = EvidenceSelection(doc_id=doc.doc_id, sentence_ids=ids, note="A local rationale classifier selected original source sentences.")
    events = [{"status": "info", "message": f"Selected {len(ids)} source sentences with the trained rationale selector.",
               "data": {"doc_id": doc.doc_id, "sentence_ids": ids, "alpha": cfg.alpha, "threshold": cfg.threshold,
                        "maximum": cfg.maximum, "rationale_scores": probabilities, "selection_scores": fused}}]
    return selection, events
