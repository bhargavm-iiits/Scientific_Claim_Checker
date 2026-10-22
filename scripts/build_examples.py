"""Build the frozen training demonstration pool, excluding all assessed populations."""
from pathlib import Path

from claim_checker.agents.examples import ExampleLibrary, build_examples
from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import load_corpus, load_gold, sha256_file, write_json


def main():
    cfg = load_config()
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    excluded = {g.claim for g in validation.values()}
    for split in ("dev_tune", "smoke"):
        excluded.update(train[c].claim for c in load_split(cfg, split)["claim_ids"])
    examples = build_examples(list(train.values()), load_corpus(cfg.path("processed_dir")), excluded)
    target = Path(cfg.root) / "artifacts/examples/train.jsonl"
    ExampleLibrary(examples).save(target)
    write_json(target.with_suffix(".manifest.json"), {
        "source": "SciFact train only", "excluded": "all validation, dev_tune and smoke claim texts",
        "examples": len(examples), "sha256": sha256_file(target),
        "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
    })
    print(f"Saved {len(examples)} training-only demonstrations to {target}")


if __name__ == "__main__":
    main()
