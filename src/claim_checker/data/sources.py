"""Dataset acquisition with an explicit, recorded compatibility route.

Route attempted first (as given in the project brief)::

    load_dataset("mteb/scifact", "corpus")["corpus"]
    load_dataset("allenai/scifact_entailment")

``allenai/scifact_entailment`` is a *script-based* dataset. Recent ``datasets`` releases refuse
dataset scripts. Remote code is never enabled (no ``trust_remote_code``). When the loader fails,
the exact exception is recorded, the official builder script is downloaded **as text only** to
read its archive URL, the official SciFact archive is cached under ``data/raw/scifact/`` and the
original JSONL files are read with ``load_dataset("json", ...)``. The builder's pair
transformation is reproduced in :mod:`claim_checker.data.prepare`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tarfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from claim_checker.config import AppConfig, canonical_json
from claim_checker.data.store import sha256_file, write_json
from claim_checker.errors import DataIntegrityError, DataNotReadyError
from claim_checker.schemas import utc_now

ARCHIVE_MEMBERS = {
    "data/corpus.jsonl": "corpus.jsonl",
    "data/claims_train.jsonl": "claims_train.jsonl",
    "data/claims_dev.jsonl": "claims_dev.jsonl",
}
SOURCE_MANIFEST = "source_manifest.json"


def pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def describe_exception(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"


@dataclass
class SourceBundle:
    mteb_rows: list[dict]
    original_corpus_rows: list[dict]
    claims_rows: dict[str, list[dict]]  # {"train": [...], "validation": [...]}
    raw_json_rows: dict[str, list[dict]]  # same files parsed with json for cross-checking
    hf_pairs: dict[str, list[dict]] | None  # rows from the HF loader if it worked
    manifest: dict[str, Any] = field(default_factory=dict)


def _dataset_revision(repo_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo_id).sha
    except Exception:  # offline or API change: recorded as unknown
        return None


def _content_sha(rows: list[dict], key: str) -> str:
    ordered = sorted(rows, key=lambda r: str(r[key]))
    return hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest()


def load_mteb_corpus(cfg: AppConfig) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg.data.mteb_dataset, cfg.data.mteb_config)
    split_name = "corpus" if "corpus" in ds else next(iter(ds))
    corpus = ds[split_name]
    columns = list(corpus.column_names)
    missing = {"_id", "title", "text"} - set(columns)
    if missing:
        raise DataIntegrityError(
            f"{cfg.data.mteb_dataset}/{cfg.data.mteb_config} lacks columns {sorted(missing)}; got {columns}",
            stage="data",
        )
    rows = [{"_id": r["_id"], "title": r["title"], "text": r["text"]} for r in corpus]
    info = {
        "repo_id": cfg.data.mteb_dataset,
        "config": cfg.data.mteb_config,
        "split": split_name,
        "call": f'load_dataset("{cfg.data.mteb_dataset}", "{cfg.data.mteb_config}")["{split_name}"]',
        "revision": _dataset_revision(cfg.data.mteb_dataset),
        "columns": columns,
        "features": {k: str(v) for k, v in corpus.features.items()},
        "rows": len(rows),
        "content_sha256": _content_sha(rows, "_id"),
    }
    return rows, info


def try_entailment_loader(cfg: AppConfig) -> tuple[dict[str, list[dict]] | None, dict]:
    """Attempt the brief's loader. Never enables remote code."""
    call = f'load_dataset("{cfg.data.claims_dataset}")'
    info: dict[str, Any] = {"repo_id": cfg.data.claims_dataset, "call": call, "trust_remote_code": False}
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.data.claims_dataset)
    except Exception as exc:  # recorded verbatim; the fallback route follows
        info.update(success=False, exception=describe_exception(exc))
        return None, info
    info.update(
        success=True,
        splits={k: len(v) for k, v in ds.items()},
        features={k: {f: str(t) for f, t in v.features.items()} for k, v in ds.items()},
    )
    return {k: [dict(r) for r in v] for k, v in ds.items()}, info


def read_builder_archive_url(cfg: AppConfig, dest_dir: Path) -> tuple[str, dict]:
    """Download the builder script as plain text (never executed) and read ``_URL``."""
    info: dict[str, Any] = {
        "repo_id": cfg.data.claims_dataset,
        "filename": cfg.data.builder_script,
        "revision": _dataset_revision(cfg.data.claims_dataset),
        "executed": False,
    }
    try:
        from huggingface_hub import hf_hub_download

        script_path = hf_hub_download(
            repo_id=cfg.data.claims_dataset,
            filename=cfg.data.builder_script,
            repo_type="dataset",
            local_dir=str(dest_dir / "builder"),
        )
        text = Path(script_path).read_text(encoding="utf-8")
        info["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        info["local_path"] = str(Path(script_path))
        match = re.search(r'^_URL\s*=\s*["\']([^"\']+)["\']', text, flags=re.MULTILINE)
        if match:
            info["archive_url_source"] = "builder_script"
            return match.group(1), info
        info["archive_url_source"] = "config_fallback (no _URL in script)"
    except Exception as exc:
        info["error"] = describe_exception(exc)
        info["archive_url_source"] = "config_fallback (script unavailable)"
    return cfg.data.archive_url_fallback, info


def download_archive(url: str, dest: Path, force: bool = False) -> dict:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    reused = dest.exists() and not force
    if not reused:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
        os.replace(tmp, dest)
    return {
        "url": url,
        "path": dest.as_posix(),
        "sha256": sha256_file(dest),
        "bytes": dest.stat().st_size,
        "reused_cached_file": reused,
        "retrieved_at": utc_now(),
    }


def extract_members(archive: Path, out_dir: Path) -> dict[str, dict]:
    """Extract only the expected regular files (no extractall, no path traversal)."""
    out: dict[str, dict] = {}
    with tarfile.open(archive, "r:gz") as tar:
        names = {m.name: m for m in tar.getmembers()}
        for member_name, local_name in ARCHIVE_MEMBERS.items():
            member = names.get(member_name) or names.get("./" + member_name)
            if member is None or not member.isfile():
                raise DataIntegrityError(
                    f"Archive {archive} lacks {member_name}; members: {sorted(names)[:20]}",
                    stage="data",
                )
            fh = tar.extractfile(member)
            assert fh is not None
            target = out_dir / local_name
            target.write_bytes(fh.read())
            with open(target, encoding="utf-8") as tf:
                lines = sum(1 for line in tf if line.strip())
            out[local_name] = {"member": member_name, "sha256": sha256_file(target), "lines": lines}
    return out


def _load_json_rows(path: Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")
    return [dict(r) for r in ds]


def _parse_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def acquire_sources(cfg: AppConfig, *, force_download: bool = False, log=print) -> SourceBundle:
    raw_dir = cfg.path("raw_dir")
    scifact_dir = raw_dir / "scifact"
    scifact_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "datasets": pkg_version("datasets"),
            "huggingface_hub": pkg_version("huggingface_hub"),
            "pyarrow": pkg_version("pyarrow"),
        },
    }

    log(f"Loading retrieval corpus: {cfg.data.mteb_dataset} / {cfg.data.mteb_config}")
    try:
        mteb_rows, mteb_info = load_mteb_corpus(cfg)
    except DataIntegrityError:
        raise
    except Exception as exc:
        raise DataNotReadyError(
            f"Could not load {cfg.data.mteb_dataset}: {describe_exception(exc)}",
            stage="data",
            hint="Check the internet connection for the first download, then rerun prepare-data.",
        ) from exc
    manifest["mteb_corpus"] = mteb_info
    log(f"  {len(mteb_rows)} documents")

    log(f"Attempting claims loader: load_dataset(\"{cfg.data.claims_dataset}\")")
    hf_pairs, loader_info = try_entailment_loader(cfg)
    manifest["entailment_loader"] = loader_info
    if not loader_info["success"]:
        log(f"  loader failed (recorded): {loader_info['exception'][:300]}")

    url, script_info = read_builder_archive_url(cfg, scifact_dir)
    manifest["builder_script"] = script_info
    log(f"Official archive URL ({script_info['archive_url_source']}): {url}")
    try:
        archive_info = download_archive(url, scifact_dir / "data.tar.gz", force=force_download)
    except Exception as exc:
        raise DataNotReadyError(
            f"Could not download the SciFact archive from {url}: {describe_exception(exc)}",
            stage="data",
            hint="Check the internet connection, or place data.tar.gz in data/raw/scifact/.",
        ) from exc
    manifest["archive"] = archive_info
    manifest["extracted"] = extract_members(scifact_dir / "data.tar.gz", scifact_dir)

    log("Reading original JSONL files with load_dataset(\"json\", ...)")
    files = {
        "corpus": scifact_dir / "corpus.jsonl",
        "train": scifact_dir / "claims_train.jsonl",
        "validation": scifact_dir / "claims_dev.jsonl",
    }
    json_rows = {name: _load_json_rows(p) for name, p in files.items()}
    raw_rows = {name: _parse_jsonl(p) for name, p in files.items()}

    manifest["route"] = (
        "hf_loader+official_archive" if loader_info["success"] else "fallback_official_archive_json"
    )
    manifest["route_description"] = (
        "MTEB corpus via load_dataset; claims and original sentence lists from the official SciFact "
        "archive referenced by the builder script, read with load_dataset('json'); the builder's "
        "claim-abstract transformation is reproduced in claim_checker.data.prepare."
    )
    manifest["json_loader_rows"] = {k: len(v) for k, v in json_rows.items()}

    write_json(raw_dir / SOURCE_MANIFEST, manifest)
    return SourceBundle(
        mteb_rows=mteb_rows,
        original_corpus_rows=json_rows["corpus"],
        claims_rows={"train": json_rows["train"], "validation": json_rows["validation"]},
        raw_json_rows=raw_rows,
        hf_pairs=hf_pairs,
        manifest=manifest,
    )
