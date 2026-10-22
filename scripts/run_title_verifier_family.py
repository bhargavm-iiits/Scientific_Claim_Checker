"""Reproduce development-only recipe selection, refit and validation confirmation.

Requires the local independent NLI base weights and the GPU training dependencies.
Existing complete selection artifacts are preserved, not regenerated with new settings.
"""
import subprocess
import sys
from pathlib import Path

from claim_checker.data.store import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def run(arguments):
    print("Starting:", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, "-X", "utf8", *arguments], cwd=ROOT, check=True)


def main():
    plan = ROOT / "artifacts/improvement/nli_title_family_selection_plan.json"
    weak = ROOT / "artifacts/improvement/frozen_selection_nli_title_v2.json"
    clean = ROOT / "artifacts/improvement/frozen_selection_nli_title_clean_v3.json"
    if not weak.exists():
        run(["scripts/run_title_verifier_selection.py"])
    if not clean.exists():
        run(["scripts/run_title_verifier_selection.py", "--name", "nli_large_title_clean_train_v3",
             "--plan", str(plan), "--hard-negatives", "0", "--frozen-output", clean.name])
    candidates = [(path, read_json(path)) for path in (weak, clean)]
    winner_path, winner = max(candidates, key=lambda item: (item[1]["chosen"]["macro_f1"], item[1]["chosen"]["accuracy"]))
    assert all(record["validation_used_for_selection"] is False for _, record in candidates)
    family = {**winner, "family_plan_sha256": sha256_file(plan), "chosen_recipe": str(winner_path.relative_to(ROOT)),
              "family_candidates": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                                     "chosen": record["chosen"]} for path, record in candidates]}
    frozen_path = ROOT / "artifacts/improvement/frozen_selection_nli_title_family_v1.json"
    if frozen_path.exists():
        assert read_json(frozen_path) == family, "Refuse to change the frozen family selection"
    else:
        write_json(frozen_path, family)
    print("Frozen family winner:", family["training_name"], family["chosen"], flush=True)
    run(["scripts/refit_and_confirm_title_verifier.py", "--frozen", str(frozen_path),
         "--final-name", "nli_large_titlefamily_final_v1", "--run-prefix", "nli_titlefamily", "--run-version", "v1",
         "--plan-file", str(ROOT / "artifacts/improvement/nli_title_family_refit_plan.json")])


if __name__ == "__main__":
    main()
