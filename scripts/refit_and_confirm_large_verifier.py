"""Execute the predeclared larger-verifier refit, integrity audit and confirmation."""
import subprocess
import sys
from pathlib import Path

from claim_checker.data.store import read_json

ROOT = Path(__file__).resolve().parents[1]


def run(arguments):
    print("Starting:", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, "-X", "utf8", *arguments], cwd=ROOT, check=True)


def main():
    plan = read_json(ROOT / "artifacts/improvement/nli_large_refit_plan.json")
    chosen = read_json(ROOT / plan["selection"])["chosen"]
    assert chosen["epoch"] == plan["epochs"]
    assert chosen["settings"]["k"] == plan["inference"]["k"]
    arguments = ["scripts/train_verifier.py"]
    for key in ("name", "base_dir", "epochs", "schedule_epochs", "layers", "batch_size", "accumulate",
                "learning_rate", "max_length", "hard_negatives"):
        arguments.extend(["--" + key.replace("_", "-"), str(plan[key])])
    arguments.extend(["--device", "cuda", "--amp", "--gradient-checkpointing", "--global-class-balance", "--final-fit"])
    run(arguments)
    run(["scripts/verify_training_artifact.py", str(ROOT / "artifacts/models" / plan["name"])])
    run(["scripts/run_large_verifier_confirmation.py"])


if __name__ == "__main__":
    main()
