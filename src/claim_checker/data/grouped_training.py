"""TRAIN-only folds that keep related claims and reference papers together."""
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold

from claim_checker.data.splits import normalize_claim_text


def claim_groups(claims):
    """Connected components of shared reference papers or normalized claim text."""
    by_id = {g.claim_id: g for g in claims}
    if len(by_id) != len(claims):
        raise ValueError("Duplicate claim IDs in training population")
    parents = {cid: cid for cid in by_id}

    def find(cid):
        while parents[cid] != cid:
            parents[cid] = parents[parents[cid]]
            cid = parents[cid]
        return cid

    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    owners = {}
    for cid, gold in sorted(by_id.items()):
        keys = [("text", normalize_claim_text(gold.claim))]
        keys += [("paper", did) for did in gold.reference_doc_ids]
        for key in keys:
            if key in owners:
                union(cid, owners[key])
            else:
                owners[key] = cid
    return {cid: find(cid) for cid in by_id}


def grouped_folds(train, validation, *, folds=5, seed=42):
    if folds < 3:
        raise ValueError("At least three folds are needed for train, stop and audit")
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    claims = sorted((g for g in train.values() if not g.conflict and g.label is not None
                     and normalize_claim_text(g.claim) not in forbidden), key=lambda g: g.claim_id)
    if any(g.split != "train" for g in claims):
        raise ValueError("Only original TRAIN claims may enter grouped training")
    groups = claim_groups(claims)
    if len(set(groups.values())) < folds:
        raise ValueError("Too few independent paper groups for the requested folds")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    parts = []
    for _, indices in splitter.split(claims, [g.label for g in claims], [groups[g.claim_id] for g in claims]):
        parts.append(sorted(claims[i].claim_id for i in indices))
    protocols = []
    for number in range(folds):
        audit, stop = parts[number], parts[(number + 1) % folds]
        fitted = sorted(cid for i, part in enumerate(parts) if i not in {number, (number + 1) % folds} for cid in part)
        buckets = {"train": fitted, "stop": stop, "audit": audit}
        group_sets = {key: {groups[c] for c in ids} for key, ids in buckets.items()}
        assert not (group_sets["train"] & group_sets["stop"] or group_sets["train"] & group_sets["audit"]
                    or group_sets["stop"] & group_sets["audit"])
        protocols.append({"fold": number, **buckets,
                          "label_counts": {key: dict(Counter(train[c].label for c in ids)) for key, ids in buckets.items()}})
    return {"seed": seed, "n_folds": folds, "eligible_claims": len(claims),
            "independent_groups": len(set(groups.values())), "group_by_claim": groups,
            "excluded_validation_texts": sorted(g.claim_id for g in train.values()
                                                if normalize_claim_text(g.claim) in forbidden),
            "folds": protocols}
