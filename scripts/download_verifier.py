"""Fetch an independent NLI verifier into the workspace, with a pinned revision."""
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    repo = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    revision = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
    target = Path(__file__).resolve().parents[1] / "artifacts/models/nli_base"
    snapshot_download(repo, revision=revision, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "README.md"])
    (target / "source_revision.txt").write_text(f"{repo}\n{revision}\n", encoding="utf-8")
    print(f"Downloaded {repo}@{revision} into {target}")


if __name__ == "__main__":
    main()
