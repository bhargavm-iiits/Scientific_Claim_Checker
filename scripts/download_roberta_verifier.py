"""Download an independent, revision-pinned generic NLI base; no SciFact checkpoint."""
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi

from claim_checker.data.store import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
REPO = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
TARGET = ROOT / "artifacts/models/roberta_large_nli_base"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = TARGET / "download_manifest.json"
    if manifest.exists():
        frozen = read_json(manifest)
    else:
        info = HfApi().model_info(REPO, files_metadata=True)
        files = {f.rfilename: f for f in info.siblings}
        weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
        names = [n for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                             "vocab.json", "merges.txt", "README.md", weight) if n in files]
        assert all(n in names for n in ("config.json", "vocab.json", "merges.txt", weight))
        assert files[weight].lfs and files[weight].lfs.sha256
        frozen = {"repository": REPO, "revision": info.sha, "weights": weight,
                  "files": [{"name": n, "size": files[n].size,
                             "sha256": files[n].lfs.sha256 if files[n].lfs else None} for n in names],
                  "pretraining": "SNLI, MNLI, FEVER-NLI, ANLI R1-R3 per author model card; no SciFact checkpoint",
                  "model_card": f"https://huggingface.co/{REPO}/blob/{info.sha}/README.md"}
        write_json(manifest, frozen)
    assert frozen["repository"] == REPO
    for item in frozen["files"]:
        target = TARGET / item["name"]
        if target.exists():
            assert target.stat().st_size == item["size"]
            if item["sha256"]:
                assert sha256_file(target) == item["sha256"]
            continue
        partial = TARGET / (item["name"] + ".partial")
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            url = f"https://huggingface.co/{REPO}/resolve/{frozen['revision']}/{item['name']}?download=true&fresh={time.time_ns()}"
            try:
                with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as output:
                        progress = offset
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            output.write(chunk)
                            offset += len(chunk)
                            if offset - progress >= 200 * 1024 * 1024:
                                print(f"{item['name']}: {offset / 2**20:.0f}/{item['size'] / 2**20:.0f} MiB", flush=True)
                                progress = offset
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
        assert partial.stat().st_size == item["size"], f"Incomplete source file: {item['name']}"
        if item["sha256"]:
            assert sha256_file(partial) == item["sha256"], "Published weights checksum mismatch"
        partial.replace(target)
        print(f"Verified {item['name']}", flush=True)
    (TARGET / "source_revision.txt").write_text(f"{REPO}\n{frozen['revision']}\n", encoding="utf-8")
    print(f"Downloaded pinned generic RoBERTa NLI base into {TARGET}", flush=True)


if __name__ == "__main__":
    main()
