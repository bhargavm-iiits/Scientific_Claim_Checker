"""UTF-8 JSON/JSONL storage helpers and loaders for prepared data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from claim_checker.errors import DATA_SETUP_HINT, DataNotReadyError

# File names inside the processed directory.
CORPUS_FILE = "corpus.jsonl"
PAIRS_FILE = "claim_pairs_{split}.jsonl"
GOLD_FILE = "claim_gold_{split}.jsonl"
QUALITY_FILE = "data_quality.json"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


atomic_write_text = _atomic_write_text  # public name (also used by the LLM response cache)


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    _atomic_write_text(Path(path), json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    lines = [json.dumps(to_jsonable(r), ensure_ascii=False, sort_keys=False) for r in rows]
    _atomic_write_text(Path(path), "".join(line + "\n" for line in lines))
    return len(lines)


def append_jsonl(path: str | Path, row: Any) -> None:
    """Append one record and flush it to disk (used for checkpointing after each claim)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line ({exc})") from exc


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


# ---------------------------------------------------------------------------
# Loaders for prepared records
# ---------------------------------------------------------------------------


def _require(path: Path) -> Path:
    if not path.exists():
        raise DataNotReadyError(f"Prepared data file is missing: {path}", stage="data", hint=DATA_SETUP_HINT)
    return path


def load_corpus(processed_dir: str | Path):
    """Return ``{doc_id: CorpusDocument}`` in file order."""
    from claim_checker.schemas import CorpusDocument

    path = _require(Path(processed_dir) / CORPUS_FILE)
    return {row["doc_id"]: CorpusDocument.model_validate(row) for row in iter_jsonl(path)}


def load_gold(processed_dir: str | Path, split: str):
    """Return ``{claim_id: ClaimGold}`` for ``split`` (train | validation)."""
    from claim_checker.schemas import ClaimGold

    path = _require(Path(processed_dir) / GOLD_FILE.format(split=split))
    return {row["claim_id"]: ClaimGold.model_validate(row) for row in iter_jsonl(path)}


def load_pairs(processed_dir: str | Path, split: str):
    from claim_checker.schemas import ClaimPair

    path = _require(Path(processed_dir) / PAIRS_FILE.format(split=split))
    return [ClaimPair.model_validate(row) for row in iter_jsonl(path)]
