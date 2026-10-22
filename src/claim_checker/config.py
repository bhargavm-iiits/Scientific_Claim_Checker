"""Configuration loading: YAML files merged over typed defaults, with path resolution.

The CLI, Streamlit UI and notebook all call :func:`load_config` so that they share settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from claim_checker.errors import ConfigError

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def local_host_problem(url: str) -> str | None:
    """Return why ``url`` is not an allowed local Ollama endpoint, or ``None`` if it is.

    Allowed: an ``http``/``https`` URL whose parsed hostname is exactly ``localhost``,
    ``127.0.0.1`` or ``::1`` (written ``[::1]`` in a URL), with no user-info part.
    """
    from urllib.parse import urlsplit

    text = str(url or "").strip()
    try:
        parts = urlsplit(text)
        parts.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError as exc:
        return f"not a valid URL ({exc})"
    if parts.scheme.lower() not in ("http", "https"):
        return "the URL must start with http:// or https://"
    if parts.username is not None or parts.password is not None:
        return "credentials in the URL are not allowed"
    host = (parts.hostname or "").lower()
    if not host:
        return "the URL has no hostname"
    if host not in LOOPBACK_HOSTS:
        return f"hostname {host!r} is not one of {sorted(LOOPBACK_HOSTS)}"
    return None


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    index_dir: str = "artifacts/index"
    runs_dir: str = "artifacts/runs"
    retrieval_dir: str = "artifacts/retrieval"
    cache_dir: str = "artifacts/cache"
    prompts_dir: str = "prompts"


class DataConfig(_Section):
    mteb_dataset: str = "mteb/scifact"
    mteb_config: str = "corpus"
    claims_dataset: str = "allenai/scifact_entailment"
    builder_script: str = "scifact_entailment.py"
    archive_url_fallback: str = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
    expected_corpus_size: int = 5183


class KnowledgeConfig(_Section):
    kind: Literal["scifact", "general_science"] = "scifact"
    name: str = "SciFact biomedical abstracts"


class SplitsConfig(_Section):
    seed: int = 42
    smoke_size: int = 20
    final_size: int = 100


class ModelConfig(_Section):
    provider: Literal["ollama"] = "ollama"
    name: str = "qwen3:4b"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    think: bool = False
    seed: int = 42
    context_tokens: int = 8192
    max_output_tokens: int = 768
    request_timeout_seconds: float = 180
    max_schema_repair_attempts: int = Field(1, ge=0, le=3)
    max_concurrent_generations: int = 1
    keep_alive: str = "30m"
    cache_enabled: bool = True

    @field_validator("host")
    @classmethod
    def _local_only(cls, value: str) -> str:
        # The prototype must not silently switch to a hosted model: the URL is parsed and only an
        # exact loopback hostname is accepted (a substring test would accept e.g.
        # ``http://localhost.example.com`` or ``http://127.0.0.1.evil.net``).
        reason = local_host_problem(value)
        if reason:
            raise ValueError(
                f"model.host must be a local loopback endpoint such as http://127.0.0.1:11434 "
                f"(got {value!r}: {reason}); remote or cloud endpoints are not allowed"
            )
        return value.strip().rstrip("/")


class RetrievalConfig(_Section):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    max_window_tokens: int = 512
    window_overlap_sentences: int = Field(1, ge=0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_device: str = "cpu"
    reranker_max_tokens: int = 512
    reranker_batch_size: int = 16
    candidates_per_method: int = 30
    rrf_k: int = 60
    rerank_top_n: int = 20
    keep_top: int = 10
    pipeline_k: int = 3
    allowed_k: list[int] = Field(default_factory=lambda: [1, 3, 5])


class EvidenceConfig(_Section):
    max_sentences_per_call: int = 40
    max_chars_per_call: int = 12000


class EvidenceSelectorConfig(_Section):
    enabled: bool = False
    model_dir: str = "artifacts/models/evidence_deberta_final_v2/epoch4"
    device: Literal["cpu", "cuda"] = "cpu"
    threshold: float = Field(0.475, ge=0, le=1)
    maximum: int = Field(1, ge=1, le=40)
    alpha: float = Field(1, ge=0, le=1)


class JudgeConfig(_Section):
    skip_when_no_evidence: bool = True
    max_citation_repairs: int = Field(1, ge=0, le=3)


PromptSet = Literal["v1", "v2", "v3"]


class JudgeV3Config(_Section):
    examples_path: str = "artifacts/examples/train.jsonl"
    use_examples: bool = True
    max_context_chars: int = Field(8000, ge=1000)


class VerifierConfig(_Section):
    model_dir: str = "artifacts/models/nli_scifact_final_v2/epoch3"
    device: Literal["cpu", "cuda"] = "cpu"
    threshold: float = Field(0.5, ge=0, le=1)
    support_threshold: float | None = Field(None, ge=0, le=1)
    contradict_threshold: float | None = Field(None, ge=0, le=1)
    include_title: bool = False
    window_overlap: int = Field(1, ge=0, le=16)
    min_retrieval_score: float | None = Field(None, allow_inf_nan=False)


class LoraJudgeConfig(_Section):
    base_dir: str = "artifacts/models/qwen25_15b_base"
    adapter_dir: str = "artifacts/models/qwen25_scifact_lora_train_v1/epoch2"
    device: Literal["cpu", "cuda"] = "cpu"
    max_length: int = Field(1024, ge=256)


class JudgeV2Config(_Section):
    """Settings of the v2 judge (prompt set ``v2``). A separate section so that adding it leaves the
    v1 configuration fingerprint (model / retrieval / evidence / judge / runtime) unchanged."""

    templates_dir: str = "v2"  # subfolder of paths.prompts_dir holding the v2 templates
    # Thinking budget (tokens) of a judge call with qwen3 thinking on. Unbounded qwen3:4b thinking
    # took 1,000-3,000+ tokens per judge call on dev claims (30-60 s), so thinking is budget-forced:
    # when it reaches this many tokens it is closed and the JSON answer is decoded under the schema.
    think_budget_tokens: int = Field(512, ge=64)
    # Supervisor verdict / checklist / reason consistency check (re-ask once, then NEI).
    consistency_check: bool = True


class ProductConfig(_Section):
    """What ``ClaimChecker.check()`` uses when the caller does not choose a prompt set (UI, CLI
    ``check`` without ``--variant``, latency benchmark). Evaluation variants always state theirs."""

    prompt_set: PromptSet = "v1"
    judge_think: bool = False
    evidence_mode: Literal["evidence", "full_abstract"] = "evidence"
    judge_backend: Literal["ollama", "nli", "lora"] = "ollama"

    @model_validator(mode="after")
    def _lora_contract(self):
        if self.judge_backend == "lora" and (self.evidence_mode != "full_abstract" or self.prompt_set != "v1" or self.judge_think):
            raise ValueError("The LoRA backend uses full_abstract, prompt_set=v1 and judge_think=false; its own prompt is fixed in the training contract")
        return self


class RuntimeConfig(_Section):
    max_claim_chars: int = 2000
    max_claim_tokens: int = 256
    graph_recursion_limit: int = 25
    random_seed: int = 42


class AppConfig(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    evidence_selector: EvidenceSelectorConfig = Field(default_factory=EvidenceSelectorConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    judge_v2: JudgeV2Config = Field(default_factory=JudgeV2Config)
    judge_v3: JudgeV3Config = Field(default_factory=JudgeV3Config)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    lora: LoraJudgeConfig = Field(default_factory=LoraJudgeConfig)
    product: ProductConfig = Field(default_factory=ProductConfig)
    # Not part of the YAML: resolved at load time.
    root: str = Field(default="", exclude=True)

    def path(self, name: str) -> Path:
        """Absolute path for a key of the ``paths`` section."""
        rel = getattr(self.paths, name)
        p = Path(rel)
        return p if p.is_absolute() else Path(self.root) / p

    def fingerprint(self, *sections: str) -> str:
        """Stable hash of the selected configuration sections (all if none given)."""
        data = self.model_dump(mode="json")
        if sections:
            data = {s: data[s] for s in sections}
        return sha256_text(canonical_json(data))


Strategy = Literal["pipeline", "judge_only", "single_prompt"]


class VariantSpec(BaseModel):
    """One evaluation variant. All variants share the same frozen claim population."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    strategy: Strategy = "pipeline"
    k: int = 3
    retrieval_method: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True
    evidence_mode: Literal["evidence", "full_abstract"] = "evidence"
    oracle: bool = False
    # Added for v2. v1 variants leave both at their defaults, and :func:`variant_dump` then omits
    # them, so v1 variant definitions (and run manifests) are byte-identical to before.
    prompt_set: PromptSet = "v1"
    judge_think: bool = False
    judge_backend: Literal["ollama", "nli", "lora"] = "ollama"

    @model_validator(mode="after")
    def _v2_needs_pipeline(self):
        if self.judge_backend == "lora" and (self.evidence_mode != "full_abstract" or self.prompt_set != "v1" or self.judge_think):
            raise ValueError("The LoRA variant must use full_abstract, prompt_set=v1 and judge_think=false")
        if (self.prompt_set != "v1" or self.judge_think or self.judge_backend != "ollama") and self.strategy != "pipeline":
            raise ValueError(f"variant {self.name!r}: prompt_set/judge_think apply only to the pipeline strategy")
        return self


V2_ONLY_FIELDS = ("prompt_set", "judge_think")


def variant_dump(spec: VariantSpec) -> dict:
    """Variant definition as recorded in run manifests; the v2 fields appear only when used."""
    d = spec.model_dump()
    if spec.judge_backend == "ollama":
        d.pop("judge_backend")
    if spec.prompt_set == "v1" and not spec.judge_think:
        for k in V2_ONLY_FIELDS:
            d.pop(k, None)
    return d


class SuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    split: str
    variants: list[str]


class ExperimentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variants: dict[str, VariantSpec]
    suites: dict[str, SuiteSpec]

    def variant(self, name: str) -> VariantSpec:
        if name not in self.variants:
            raise ConfigError(
                f"Unknown variant {name!r}. Known: {', '.join(sorted(self.variants))}",
                stage="config",
            )
        return self.variants[name]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root.

    Order: ``CLAIM_CHECKER_ROOT`` environment variable; the source checkout containing this
    package (editable install); then the nearest parent of ``start``/cwd with ``configs/default.yaml``.
    """
    env = os.environ.get("CLAIM_CHECKER_ROOT")
    if env:
        return Path(env).resolve()
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "configs" / "default.yaml").exists():
        return pkg_root
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "configs" / "default.yaml").exists():
            return candidate
    return here


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}", stage="config") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping", stage="config")
    return data


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
    root: str | Path | None = None,
) -> AppConfig:
    """Load ``configs/default.yaml`` (or ``path``) and apply ``overrides``."""
    root_path = Path(root).resolve() if root else find_project_root()
    cfg_path = Path(path) if path else root_path / "configs" / "default.yaml"
    data = _read_yaml(cfg_path) if cfg_path.exists() or path else {}
    data = deep_merge(data, overrides or {})
    try:
        cfg = AppConfig.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError
        raise ConfigError(f"Invalid configuration in {cfg_path}: {exc}", stage="config") from exc
    cfg.root = str(root_path)
    return cfg


def load_experiments(path: str | Path | None = None, root: str | Path | None = None) -> ExperimentsConfig:
    root_path = Path(root).resolve() if root else find_project_root()
    exp_path = Path(path) if path else root_path / "configs" / "experiments.yaml"
    raw = _read_yaml(exp_path)
    defaults = raw.get("defaults", {}) or {}
    variants = {}
    for name, spec in (raw.get("variants") or {}).items():
        merged = {**defaults, **(spec or {}), "name": name}
        variants[name] = VariantSpec.model_validate(merged)
    suites = {n: SuiteSpec.model_validate(s) for n, s in (raw.get("suites") or {}).items()}
    for suite_name, suite in suites.items():
        for v in suite.variants:
            if v not in variants:
                raise ConfigError(f"Suite {suite_name!r} references unknown variant {v!r}")
    return ExperimentsConfig(variants=variants, suites=suites)
