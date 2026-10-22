"""Jinja2 prompt rendering. Each template holds a system part and a user part separated by
a line ``---USER---``. Prompt hashes are recorded in run manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

SEPARATOR = "---USER---"
TEMPLATES = ("evidence", "judge", "judge_only", "retrieve_judge")  # v1 (frozen; top-level prompts/)
# v2 templates live in a subfolder (``judge_v2.templates_dir``, default ``prompts/v2/``). They are
# hashed separately so the v1 hash record (and every v1 run identity) is unchanged.
TEMPLATES_V2 = ("judge",)


class PromptLibrary:
    def __init__(self, prompts_dir: str | Path, v2_dir: str = "v2"):
        self.dir = Path(prompts_dir)
        self.v2_dir = v2_dir.strip("/\\") or "v2"
        self.env = Environment(
            loader=FileSystemLoader(str(self.dir), encoding="utf-8"),
            undefined=StrictUndefined,
            keep_trailing_newline=False,
            autoescape=False,
        )

    def source(self, name: str) -> str:
        return (self.dir / f"{name}.jinja2").read_text(encoding="utf-8")

    def render(self, name: str, **ctx) -> list[dict[str, str]]:
        text = self.env.get_template(f"{name}.jinja2").render(**ctx)
        if SEPARATOR not in text:
            raise ValueError(f"Template {name} lacks the {SEPARATOR} separator")
        system, user = text.split(SEPARATOR, 1)
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]

    def hashes(self) -> dict[str, str]:
        out = {}
        for name in TEMPLATES:
            path = self.dir / f"{name}.jinja2"
            if path.exists():
                out[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return out

    def v2_name(self, name: str) -> str:
        """Template name of a v2 template for :meth:`render` (e.g. ``v2/judge``)."""
        return f"{self.v2_dir}/{name}"

    def hashes_v2(self) -> dict[str, str]:
        out = {}
        for name in TEMPLATES_V2:
            path = self.dir / self.v2_dir / f"{name}.jinja2"
            if path.exists():
                out[f"v2/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return out

    def hashes_v3(self) -> dict[str, str]:
        path = self.dir / "v3/judge.jinja2"
        return {"v3/judge": hashlib.sha256(path.read_bytes()).hexdigest()[:16]} if path.exists() else {}
