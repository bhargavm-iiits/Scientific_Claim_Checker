"""Freeze the 200 validation claims outside the previously inspected final-100."""
from claim_checker.config import canonical_json, load_config, sha256_text
from claim_checker.data.splits import load_split
from claim_checker.data.store import load_gold, read_json, sha256_file, write_json


def main():
    cfg = load_config()
    previous = load_split(cfg, "final")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    ids = sorted(set(validation) - set(previous["claim_ids"]), key=int)
    payload = {
        "name": "accuracy_holdout", "source_split": "validation", "size": len(ids),
        "claim_ids": ids, "ids_sha256": sha256_text(canonical_json(ids)),
        "source_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
        "excluded_prior_final_ids_sha256": previous["ids_sha256"],
        "purpose": "Confirmation after freezing the accuracy-improvement configuration on dev_tune. Do not tune on these results.",
    }
    path = cfg.path("splits_dir") / "accuracy_holdout.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("Existing holdout identity differs; refusing to overwrite")
    write_json(path, payload)
    print(f"Frozen {len(ids)} claims: {payload['ids_sha256']}")


if __name__ == "__main__":
    main()
