"""Run fixed fitting, provenance registration and actual SciFact confirmation."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    steps = [
        ["scripts.prepare_scifact95_protocol"],
        ["scripts.train_roberta_verifier", "--stage", "select"],
        ["scripts.train_roberta_verifier", "--stage", "refit"],
        ["scripts.register_roberta_training"],
        ["scripts.train_roberta_verifier", "--stage", "confirm"],
        ["scripts.summarize_roberta_confirmation"],
    ]
    for step in steps:
        print("Starting:", " ".join(step), flush=True)
        subprocess.run([sys.executable, "-X", "utf8", "-m", *step], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
