"""Download the brief's Qwen judge base and record the immutable HF revision."""
import argparse
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download

from claim_checker.data.store import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen", "nli_large"), default="qwen")
    args = parser.parse_args()
    repo, directory = ("Qwen/Qwen2.5-1.5B-Instruct", "qwen25_15b_base") if args.model == "qwen" else (
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "nli_large_base")
    target = Path(__file__).resolve().parents[1] / "artifacts/models" / directory
    record = target / "source_revision.txt"
    revision = record.read_text(encoding="utf-8").splitlines()[1] if record.exists() else HfApi().model_info(repo).sha
    target.mkdir(parents=True, exist_ok=True)
    record.write_text(f"{repo}\n{revision}\n", encoding="utf-8")
    snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=[
        "*.json", "*.txt", "*.model", "*.jinja", "README.md",
    ])
    # A fresh URL avoids stale, cached signed redirects on some Windows networks.
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    metadata = next(f for f in info.siblings if f.rfilename == "model.safetensors")
    weights = target / "model.safetensors"
    if not weights.exists():
        partial = target / "model.safetensors.partial"
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors?download=true&fresh={time.time_ns()}"
            try:
                with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                                  stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    print(f"Weights response {response.status_code}; resuming from {offset:,} bytes", flush=True)
                    with partial.open("ab" if offset else "wb") as file:
                        progress = offset
                        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                            file.write(chunk)
                            offset += len(chunk)
                            if offset - progress >= 200 * 1024 * 1024:
                                print(f"Downloaded {offset / 2**20:.0f} / {metadata.size / 2**20:.0f} MiB", flush=True)
                                progress = offset
                break
            except requests.RequestException as exc:
                if attempt == 4:
                    raise
                print(f"Download retry: {exc}", flush=True)
        if partial.stat().st_size != metadata.size or sha256_file(partial) != metadata.lfs.sha256:
            raise ValueError("Downloaded model failed its published size/SHA256 check")
        partial.replace(weights)
    print(f"Downloaded {repo}@{revision} into {target}", flush=True)


if __name__ == "__main__":
    main()
