"""Frozen experiment populations: development, smoke and final.

* final       - stratified (by claim label) sample of eligible validation claims, seed 42.
                Annotation conflicts are excluded and recorded. Never silently overwritten.
* development - training claims minus any whose normalized text overlaps the final subset.
* smoke       - stratified sample of development claims (conflicts excluded), seed 42.
* dev_tune    - (added for the v2 experiment) stratified sample of development claims that are
                not in the smoke split (conflicts excluded), seed 42, default 120 claims. All v2
                prompt / thinking / rule iteration uses only this population; it is disjoint from
                the final split because the development split already is.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from claim_checker.config import AppConfig, canonical_json, sha256_text
from claim_checker.data.store import GOLD_FILE, read_json, sha256_file, write_json
from claim_checker.errors import DATA_SETUP_HINT, DataNotReadyError, FrozenSplitError
from claim_checker.schemas import ClaimGold, utc_now

SPLIT_FILES = {"development": "development.json", "smoke": "smoke.json", "final": "final.json",
               "dev_tune": "dev_tune.json", "accuracy_holdout": "accuracy_holdout.json"}
SPLIT_SOURCE = {"development": "train", "smoke": "train", "final": "validation", "dev_tune": "train",
                "accuracy_holdout": "validation"}
DEV_TUNE_SIZE = 120
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_claim_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _id_key(cid: str) -> tuple[int, int | str]:
    return (0, int(cid)) if cid.isdigit() else (1, cid)


def stratified_sample(
    items: list[ClaimGold], size: int, seed: int
) -> tuple[list[ClaimGold], dict[str, int]]:
    """Proportional stratified sample by claim label (largest-remainder allocation).

    Deterministic: candidates are sorted by id before seeded shuffling.
    """
    by_label: dict[str, list[ClaimGold]] = defaultdict(list)
    for g in sorted(items, key=lambda g: _id_key(g.claim_id)):
        by_label[str(g.label)].append(g)
    total = len(items)
    if size >= total:
        return sorted(items, key=lambda g: _id_key(g.claim_id)), {k: len(v) for k, v in by_label.items()}
    labels = sorted(by_label)
    exact = {lab: size * len(by_label[lab]) / total for lab in labels}
    alloc = {lab: int(exact[lab]) for lab in labels}
    remaining = size - sum(alloc.values())
    for lab in sorted(labels, key=lambda lab: (-(exact[lab] - alloc[lab]), lab))[:remaining]:
        alloc[lab] += 1
    rng = random.Random(seed)
    chosen: list[ClaimGold] = []
    for lab in labels:
        pool = list(by_label[lab])
        rng.shuffle(pool)
        chosen.extend(pool[: alloc[lab]])
    return sorted(chosen, key=lambda g: _id_key(g.claim_id)), alloc


def _payload(name: str, chosen: list[ClaimGold], **meta: Any) -> dict:
    ids = [g.claim_id for g in chosen]
    return {
        "name": name,
        "source_split": SPLIT_SOURCE[name],
        "claim_ids": ids,
        "size": len(ids),
        "class_counts": dict(sorted(Counter(str(g.label) for g in chosen).items())),
        "ids_sha256": sha256_text(canonical_json(ids)),
        **meta,
    }


def _identity(payload: dict) -> dict:
    """Content that must match for a frozen split to be considered unchanged."""
    return {k: v for k, v in payload.items() if k not in ("created_at",)}


def write_split(path: Path, payload: dict, *, frozen: bool, force: bool) -> str:
    if path.exists():
        existing = read_json(path)
        if _identity(existing) == _identity(payload):
            return "unchanged"
        if frozen and not force:
            raise FrozenSplitError(
                f"{path} already exists with different content; the final evaluation population is "
                "frozen. Re-run with --force only if you intend to start a new, separately reported "
                "experiment.",
                stage="splits",
            )
    write_json(path, {**payload, "created_at": utc_now()})
    return "written"


def build_splits(
    gold_train: dict[str, ClaimGold],
    gold_validation: dict[str, ClaimGold],
    *,
    seed: int = 42,
    smoke_size: int = 20,
    final_size: int = 100,
    validation_fingerprint: str = "",
    train_fingerprint: str = "",
) -> dict[str, dict]:
    val = list(gold_validation.values())
    conflicts = sorted((g.claim_id for g in val if g.conflict), key=_id_key)
    eligible = [g for g in val if not g.conflict and g.label is not None]
    final, final_alloc = stratified_sample(eligible, final_size, seed)
    final_payload = _payload(
        "final",
        final,
        seed=seed,
        requested_size=final_size,
        eligible_claims=len(eligible),
        all_fewer_than_requested=len(eligible) < final_size,
        stratified_by="claim-level reference label",
        excluded_annotation_conflicts=conflicts,
        allocation=final_alloc,
        source_fingerprint=validation_fingerprint,
    )

    final_texts = {normalize_claim_text(g.claim): g.claim_id for g in final}
    val_texts = {normalize_claim_text(g.claim) for g in val}
    overlap_final, overlap_val_only = [], []
    development = []
    for g in sorted(gold_train.values(), key=lambda g: _id_key(g.claim_id)):
        t = normalize_claim_text(g.claim)
        if t in final_texts:
            overlap_final.append({"train_claim_id": g.claim_id, "final_claim_id": final_texts[t]})
            continue
        if t in val_texts:
            overlap_val_only.append(g.claim_id)
        development.append(g)
    dev_payload = _payload(
        "development",
        development,
        removed_overlap_with_final=overlap_final,
        overlap_with_other_validation_claims_kept=overlap_val_only,
        source_fingerprint=train_fingerprint,
        note="All training claims except those whose normalized text appears in the final subset.",
    )
    smoke_pool = [g for g in development if not g.conflict and g.label is not None]
    smoke, smoke_alloc = stratified_sample(smoke_pool, smoke_size, seed)
    smoke_payload = _payload(
        "smoke",
        smoke,
        seed=seed,
        requested_size=smoke_size,
        stratified_by="claim-level reference label",
        allocation=smoke_alloc,
        source_fingerprint=train_fingerprint,
        # Same representation as the final split: the sorted ids of excluded conflicting claims.
        excluded_annotation_conflicts=sorted((g.claim_id for g in development if g.conflict), key=_id_key),
    )
    return {"final": final_payload, "development": dev_payload, "smoke": smoke_payload}


def freeze_splits(cfg: AppConfig, gold_train, gold_validation, *, force: bool = False) -> dict:
    processed = cfg.path("processed_dir")
    payloads = build_splits(
        gold_train,
        gold_validation,
        seed=cfg.splits.seed,
        smoke_size=cfg.splits.smoke_size,
        final_size=cfg.splits.final_size,
        validation_fingerprint=sha256_file(processed / GOLD_FILE.format(split="validation")),
        train_fingerprint=sha256_file(processed / GOLD_FILE.format(split="train")),
    )
    out_dir = cfg.path("splits_dir")
    status = {}
    # Final first: if it is frozen with different content nothing else is touched.
    for name in ("final", "development", "smoke"):
        status[name] = write_split(
            out_dir / SPLIT_FILES[name], payloads[name], frozen=(name == "final"), force=force
        )
    return {"status": status, "payloads": payloads}


def build_dev_tune_split(
    development: dict,
    smoke: dict,
    gold_train: dict[str, ClaimGold],
    *,
    size: int = DEV_TUNE_SIZE,
    seed: int = 42,
) -> dict:
    """Stratified (by claim label) tuning subset of the development split.

    Pool = development claims minus the smoke claims, minus annotation conflicts / unlabeled claims.
    Deterministic for a given (development split, smoke split, gold, size, seed). The development
    split is already disjoint from the final split (training claims whose normalized text appears in
    the final subset were removed when it was frozen)."""
    smoke_ids = set(smoke.get("claim_ids") or [])
    dev_ids = list(development.get("claim_ids") or [])
    missing = [c for c in dev_ids if c not in gold_train]
    if missing:
        raise DataNotReadyError(f"development split references unknown training claims {missing[:5]}",
                                stage="splits", hint=DATA_SETUP_HINT)
    pool = [gold_train[c] for c in dev_ids
            if c not in smoke_ids and not gold_train[c].conflict and gold_train[c].label is not None]
    chosen, alloc = stratified_sample(pool, size, seed)
    return _payload(
        "dev_tune",
        chosen,
        seed=seed,
        requested_size=size,
        stratified_by="claim-level reference label",
        allocation=alloc,
        pool="development claims minus smoke claims, annotation conflicts excluded",
        pool_size=len(pool),
        excluded_smoke_claims=sorted(smoke_ids & set(dev_ids), key=_id_key),
        development_ids_sha256=development.get("ids_sha256"),
        source_fingerprint=development.get("source_fingerprint", ""),
        purpose="Tuning population for the v2 judge (prompt, thinking, consistency rules). The final split "
                "is never used for tuning.",
    )


def freeze_dev_tune_split(cfg: AppConfig, *, size: int = DEV_TUNE_SIZE, seed: int | None = None,
                          force: bool = False) -> dict:
    """Write ``splits_dir/dev_tune.json`` (frozen: an existing file with different content is kept
    unless ``force``). Reads only the existing development / smoke splits and the training gold."""
    from claim_checker.data.store import load_gold

    development = load_split(cfg, "development")
    smoke = load_split(cfg, "smoke")
    gold_train = load_gold(cfg.path("processed_dir"), "train")
    payload = build_dev_tune_split(development, smoke, gold_train, size=size,
                                   seed=cfg.splits.seed if seed is None else seed)
    status = write_split(cfg.path("splits_dir") / SPLIT_FILES["dev_tune"], payload, frozen=True, force=force)
    return {"status": status, "payload": payload}


def load_split(cfg: AppConfig, name: str) -> dict:
    if name not in SPLIT_FILES:
        raise DataNotReadyError(f"Unknown split {name!r}; expected one of {sorted(SPLIT_FILES)}")
    path = cfg.path("splits_dir") / SPLIT_FILES[name]
    if not path.exists():
        raise DataNotReadyError(f"Split file missing: {path}", stage="splits", hint=DATA_SETUP_HINT)
    return read_json(path)


def split_fingerprint(cfg: AppConfig, name: str) -> str:
    return sha256_text(canonical_json(_identity(load_split(cfg, name))))
