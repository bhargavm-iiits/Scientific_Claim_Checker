"""Reject evaluation overlap with the actual fitted claim population."""
from claim_checker.config import sha256_text
from claim_checker.data.splits import normalize_claim_text
from claim_checker.errors import ConfigError


def assert_unfitted_population(training, population):
    fitted_ids = set(training.get("fitted_claim_ids", []))
    fitted_texts = set(training.get("fitted_claim_text_sha256", []))
    overlap = [claim.claim_id for claim in population if claim.claim_id in fitted_ids
               or sha256_text(normalize_claim_text(claim.claim)) in fitted_texts]
    if overlap:
        raise ConfigError(
            f"Evaluation contains {len(overlap)} claims used to fit this verifier; this is not held-out performance.",
            stage="evaluate", hint="Use a population whose IDs and normalized texts are absent from the fitted training manifest.")
