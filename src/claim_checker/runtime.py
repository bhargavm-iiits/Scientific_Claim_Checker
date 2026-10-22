"""Shared entry point used by the CLI, the Streamlit UI and the notebook.

    from claim_checker.runtime import ClaimChecker, check_readiness
    checker = ClaimChecker()              # nothing heavy is loaded yet
    result = checker.check("Claim text", k=3)

Heavy resources (corpus, BM25, embeddings, cross-encoder, model client) are loaded lazily on
first use and reused for later calls. :func:`check_readiness` inspects files and the local
Ollama endpoint without loading any model.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claim_checker import __version__
from claim_checker.agents.explainer import display_reason, explain, explain_single_prompt
from claim_checker.agents.judge import judge_claim_only, retrieve_and_judge
from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import (
    AppConfig,
    ExperimentsConfig,
    VariantSpec,
    load_config,
    load_experiments,
)
from claim_checker.data.store import CORPUS_FILE, load_corpus
from claim_checker.errors import (
    DATA_SETUP_HINT,
    INDEX_SETUP_HINT,
    OLLAMA_SETUP_HINT,
    ClaimCheckerError,
    DataNotReadyError,
    InputValidationError,
)
from claim_checker.models.ollama_client import OllamaClient, model_name_matches
from claim_checker.pipeline.graph import (
    RETRIEVAL_CACHE_NOTE,
    PipelineResources,
    RetrievalLookup,
    build_graph,
    retrieval_cache_metadata,
)
from claim_checker.pipeline.state import PipelineOptions
from claim_checker.pipeline.supervisor import validate_claim_text
from claim_checker.schemas import (
    ClaimResult,
    CorpusDocument,
    DocumentReport,
    ErrorInfo,
    EvidenceQuote,
    StageEvent,
    utc_now,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ``progress(stage, events)`` is called after each graph node finishes: ``stage`` is the node name
# (``validate_input`` ... ``explain`` or ``handle_error``) and ``events`` the StageEvents it emitted.
ProgressCallback = Callable[[str, list[StageEvent]], None]


# ---------------------------------------------------------------------------
# Readiness (cheap; loads no model)
# ---------------------------------------------------------------------------


@dataclass
class ReadinessReport:
    data_ready: bool = False
    splits_ready: bool = False
    index_ready: bool = False
    model_ready: bool = False
    server_version: str | None = None
    model_name: str = ""
    model_backend: str = "ollama"
    model_digest: str | None = None
    index_fingerprint: str | None = None
    corpus_size: int | None = None
    problems: list[dict[str, str]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.data_ready and self.index_ready and self.model_ready

    def to_dict(self) -> dict:
        return {**self.__dict__, "ready": self.ready}


def check_readiness(cfg: AppConfig | None = None, *, check_model: bool = True, timeout: float = 3.0) -> ReadinessReport:
    """Report whether prepared data, a fresh index and the local model are available."""
    from claim_checker.retrieval.index import check_index

    cfg = cfg or load_config()
    rep = ReadinessReport(model_name=cfg.model.name)
    if (cfg.path("processed_dir") / CORPUS_FILE).exists():
        rep.data_ready = True
    else:
        hint = ("Run python scripts/prepare_general_science.py --download, then build-index with configs/general_science.yaml."
                if cfg.knowledge.kind == "general_science" else DATA_SETUP_HINT)
        rep.problems.append({"component": "data", "message": "Prepared corpus not found.", "hint": hint})
    rep.splits_ready = (cfg.path("splits_dir") / "final.json").exists()
    if rep.data_ready:
        try:
            manifest = check_index(cfg)
            rep.index_ready = True
            rep.index_fingerprint = manifest.get("fingerprint")
            rep.corpus_size = manifest.get("documents")
        except ClaimCheckerError as exc:
            rep.problems.append({"component": "index", "message": exc.message, "hint": exc.hint or INDEX_SETUP_HINT})
    else:
        rep.problems.append({"component": "index", "message": "Index needs prepared data.", "hint": INDEX_SETUP_HINT})
    local_backend = cfg.product.judge_backend if cfg.product.evidence_mode == "full_abstract" or cfg.evidence_selector.enabled else None
    if local_backend in {"nli", "lora"}:
        rep.model_backend = local_backend
    if check_model and local_backend in {"nli", "lora"}:
        from importlib.util import find_spec

        if local_backend == "lora":
            base = Path(cfg.root) / cfg.lora.base_dir
            adapter = Path(cfg.root) / cfg.lora.adapter_dir
            files = [base / "model.safetensors", base / "config.json", base / "tokenizer.json", base / "tokenizer_config.json",
                     adapter / "adapter_model.safetensors", adapter / "adapter_config.json", adapter.parent / "training_manifest.json"]
            dependencies = ["transformers", "peft"]
            rep.model_name = "Qwen2.5-1.5B-Instruct+LoRA"
        else:
            base = Path(cfg.root) / cfg.verifier.model_dir
            files = [base / "model.safetensors", base / "tokenizer.json", base / "config.json"]
            dependencies = ["transformers"]
            rep.model_name = "General science NLI verifier" if cfg.knowledge.kind == "general_science" else "Scientific NLI verifier"
        if cfg.evidence_selector.enabled:
            selector = Path(cfg.root) / cfg.evidence_selector.model_dir
            files.extend(selector / name for name in ("model.safetensors", "config.json", "tokenizer.json"))
            files.append(selector.parent / "training_manifest.json")
            rep.model_name += " + trained rationale selector"
        missing = [str(path) for path in files if not path.is_file()]
        absent = [name for name in dependencies if find_spec(name) is None]
        rep.model_ready = not missing and not absent
        if not rep.model_ready:
            rep.problems.append({"component": "model", "message": f"Missing local files: {missing}; dependencies: {absent}",
                                 "hint": "Follow Docs/ACCURACY_IMPROVEMENT_2026-10-22.md to prepare the local judge."})
    elif check_model:
        import httpx

        hint = OLLAMA_SETUP_HINT.format(model=cfg.model.name)
        try:
            rep.server_version = httpx.get(f"{cfg.model.host}/api/version", timeout=timeout).json().get("version")
            tags = httpx.get(f"{cfg.model.host}/api/tags", timeout=timeout).json().get("models", [])
            for m in tags:
                if model_name_matches(m.get("model"), cfg.model.name) or model_name_matches(m.get("name"), cfg.model.name):
                    rep.model_ready, rep.model_digest = True, m.get("digest")
                    break
            if not rep.model_ready:
                rep.problems.append({"component": "model", "message": f"Model {cfg.model.name} is not downloaded.",
                                     "hint": f"Run `ollama pull {cfg.model.name}`."})
        except Exception as exc:
            rep.problems.append({"component": "model", "message": f"Ollama is not reachable at {cfg.model.host} ({type(exc).__name__}).",
                                 "hint": hint})
    return rep


def claim_confidence(verdict: str | None, conflict: bool, judgements) -> float | None:
    """Claim-level model-reported (uncalibrated) confidence: the highest confidence among the
    documents whose verdict equals the claim verdict and that went to the model. None for a
    conflict, an error (no verdict) or an NEI reached without any model judgement."""
    if verdict is None or conflict:
        return None
    vals = [j.confidence for j in judgements
            if j is not None and j.verdict == verdict and j.confidence is not None and not j.skipped_model]
    return float(max(vals)) if vals else None


def thinking_summary(judgements) -> dict:
    """Per-claim summary of the v2 judge's thinking (lengths only; the thinking text is discarded).
    Covers the final judgement of each examined abstract that went to the model."""
    called = [j for j in judgements if j is not None and not j.skipped_model and j.thinking]
    return {
        "judge_model_judgements": len(called),
        "think": any(j.thinking.get("think") for j in called),
        "thinking_chars_total": sum(int(j.thinking.get("thinking_chars") or 0) for j in called),
        "thinking_chars_per_judgement": [int(j.thinking.get("thinking_chars") or 0) for j in called],
        "eval_tokens_total": sum(int(j.thinking.get("eval_tokens") or 0) for j in called),
        "think_fallbacks": sum(1 for j in called if j.thinking.get("fallback")),
        "budget_forced": sum(int(j.thinking.get("budget_forced") or 0) for j in called),
    }


# ---------------------------------------------------------------------------
# ClaimChecker
# ---------------------------------------------------------------------------


class ClaimChecker:
    """One shared, lazily-initialised claim-checking pipeline."""

    def __init__(
        self,
        cfg: AppConfig | None = None,
        *,
        experiments: ExperimentsConfig | None = None,
        corpus: dict[str, CorpusDocument] | None = None,
        retriever: Any | None = None,
        llm: OllamaClient | None = None,
        token_counter=None,
        check_resources: bool = True,
    ):
        self.cfg = cfg or load_config()
        self._experiments = experiments
        self._corpus = corpus
        self._retriever = retriever
        self._llm = llm
        self._encoder = None
        self._examples = None
        self._verifier = None
        self._verifier_identity = None
        self._evidence_selector = None
        self._evidence_selector_identity = None
        self._lora_judge = None
        self._lora_identity = None
        self._token_counter = token_counter
        self._check_resources = check_resources
        self._ready_index = retriever is not None
        self._ready_model = False
        self.prompts = PromptLibrary(self.cfg.path("prompts_dir"), v2_dir=self.cfg.judge_v2.templates_dir)
        self._lock = threading.RLock()
        self._retrieval_cache: OrderedDict = OrderedDict()
        self._graph = build_graph(PipelineResources(
            cfg=self.cfg,
            get_corpus=lambda: self.corpus,
            get_client=lambda: self.llm,
            prompts=self.prompts,
            retrieve=self._retrieve,
            ensure_ready=self._ensure_ready,
            count_tokens=self._count_tokens_or_none(),
            get_examples=lambda: self.examples,
            get_verifier=lambda: self.verifier,
            get_lora_judge=lambda: self.lora_judge,
            ensure_local_ready=self._ensure_local_ready,
            get_evidence_selector=lambda: self.evidence_selector,
        ))

    # -- lazy resources ----------------------------------------------------
    @property
    def evidence_selector(self):
        if self._evidence_selector is None:
            from claim_checker.data.store import read_json, sha256_file
            from claim_checker.models.evidence_classifier import EvidenceClassifier

            path = Path(self.cfg.root) / self.cfg.evidence_selector.model_dir
            if not (path / "model.safetensors").is_file():
                raise DataNotReadyError("Trained rationale selector weights are missing.", stage="select_evidence")
            training = read_json(path.parent / "training_manifest.json")
            renderer = Path(__file__).resolve().parent / "models/evidence_classifier.py"
            if training["renderer_sha256"] != sha256_file(renderer):
                raise DataNotReadyError("The trained selector input contract changed.", stage="select_evidence")
            self._evidence_selector = EvidenceClassifier(path, device=self.cfg.evidence_selector.device)
            self._evidence_selector_identity = {"model_dir": str(path), "weights_sha256": sha256_file(path / "model.safetensors"),
                                               "model_config_sha256": sha256_file(path / "config.json"),
                                               "training_manifest_sha256": sha256_file(path.parent / "training_manifest.json")}
        return self._evidence_selector

    @property
    def lora_judge(self):
        if self._lora_judge is None:
            from claim_checker.data.store import read_json, sha256_file
            from claim_checker.models.lora_judge import LoraJudge

            base = Path(self.cfg.root) / self.cfg.lora.base_dir
            adapter = Path(self.cfg.root) / self.cfg.lora.adapter_dir
            if not (base / "model.safetensors").is_file() or not (adapter / "adapter_model.safetensors").is_file():
                raise DataNotReadyError("The local Qwen base or LoRA adapter is missing.", stage="judge",
                                        hint="Train and evaluate the judge following Docs/ACCURACY_IMPROVEMENT_2026-10-22.md.")
            training = read_json(adapter.parent / "training_manifest.json")
            base_sha = sha256_file(base / "model.safetensors")
            if training["base_weights_sha256"] != base_sha or training["settings"]["max_length"] != self.cfg.lora.max_length:
                raise DataNotReadyError("The LoRA base/context settings differ from training.", stage="judge")
            contract = Path(__file__).resolve().parent / "models/lora_judge.py"
            if training["contract_sha256"] != sha256_file(contract):
                raise DataNotReadyError("The trained LoRA prompt/inference contract has changed.", stage="judge")
            try:
                self._lora_judge = LoraJudge(base, adapter, device=self.cfg.lora.device, max_length=self.cfg.lora.max_length)
            except ImportError as exc:
                raise DataNotReadyError("LoRA inference dependencies are missing.", stage="judge",
                                        hint='Install this project with its lora extras: pip install -e ".[lora]"') from exc
            self._lora_identity = {"base_sha256": base_sha,
                                   "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                                   "adapter_dir": str(adapter)}
        return self._lora_judge

    @property
    def verifier(self):
        if self._verifier is None:
            from claim_checker.data.store import sha256_file
            from claim_checker.models.verifier import NLIVerifier

            path = Path(self.cfg.verifier.model_dir)
            if not path.is_absolute():
                path = Path(self.cfg.root) / path
            if not (path / "model.safetensors").is_file():
                raise DataNotReadyError(f"Scientific verifier weights are missing in {path}", stage="judge",
                                        hint="Follow Docs/ACCURACY_IMPROVEMENT_2026-10-22.md to train the selected checkpoint, or use the Ollama judge backend.")
            self._verifier = NLIVerifier(path, device=self.cfg.verifier.device, include_title=self.cfg.verifier.include_title,
                                         window_overlap=self.cfg.verifier.window_overlap)
            self._verifier_identity = {"model_dir": str(path),
                                       "weights_sha256": sha256_file(path / "model.safetensors"),
                                       "labels": self._verifier.labels}
        return self._verifier

    @property
    def examples(self):
        if self._examples is None and self.cfg.judge_v3.use_examples:
            from claim_checker.agents.examples import ExampleLibrary

            path = Path(self.cfg.judge_v3.examples_path)
            self._examples = ExampleLibrary.load(path if path.is_absolute() else Path(self.cfg.root) / path)
        return self._examples

    @property
    def experiments(self) -> ExperimentsConfig:
        if self._experiments is None:
            self._experiments = load_experiments(root=self.cfg.root)
        return self._experiments

    @property
    def corpus(self) -> dict[str, CorpusDocument]:
        if self._corpus is None:
            self._corpus = load_corpus(self.cfg.path("processed_dir"))
        return self._corpus

    @property
    def encoder(self):
        if self._encoder is None:
            from claim_checker.retrieval.dense import SentenceEncoder

            r = self.cfg.retrieval
            self._encoder = SentenceEncoder(r.embedding_model, r.embedding_device, r.embedding_batch_size)
        return self._encoder

    @property
    def retriever(self):
        if self._retriever is None:
            from claim_checker.retrieval.index import Retriever

            self._retriever = Retriever.load(self.cfg, self.corpus, encoder=self.encoder)
        return self._retriever

    @property
    def llm(self) -> OllamaClient:
        if self._llm is None:
            self._llm = OllamaClient(self.cfg.model, cache_dir=self.cfg.path("cache_dir"))
        return self._llm

    def _count_tokens_or_none(self):
        if self._token_counter is not None:
            return self._token_counter
        if self._retriever is not None:  # injected retriever (tests): no tokenizer available
            return None
        return self.count_claim_tokens

    def count_claim_tokens(self, text: str) -> int | None:
        """Tokens of ``text`` under the embedding model's tokenizer (special tokens included).

        Loads only the tokenizer (not the embedding model, index, re-ranker or Ollama), so the token
        budget can be checked before any heavy resource is loaded. Returns ``None`` when no
        tokenizer is configured (an injected test retriever without a ``token_counter``).
        """
        if self._token_counter is not None:
            return int(self._token_counter(text))
        if self._retriever is not None and self._encoder is None:
            return None
        return self.encoder.count_tokens_light(text)

    def validate_claim(self, claim: str | None) -> str:
        """Normalise ``claim`` and enforce the character and token budgets without loading the
        index or contacting the model. Raises :class:`InputValidationError` (``invalid_input``)."""
        rt = self.cfg.runtime
        text = validate_claim_text(claim, rt.max_claim_chars, rt.max_claim_tokens, None)
        counter = self._count_tokens_or_none()
        return validate_claim_text(text, rt.max_claim_chars, rt.max_claim_tokens, counter) if counter else text

    def _validate_then_ready(self, claim: str | None, need_index: bool) -> str:
        """Input checks before readiness: characters, then the tokenizer-only token budget, then
        the (expensive) resource checks. If the tokenizer itself cannot be loaded yet (setup
        incomplete), readiness runs first so the user gets the setup hint, then the token check."""
        rt = self.cfg.runtime
        text = validate_claim_text(claim, rt.max_claim_chars, rt.max_claim_tokens, None)
        counter = self._count_tokens_or_none()
        counted = False
        if counter is not None:
            try:
                text = validate_claim_text(text, rt.max_claim_chars, rt.max_claim_tokens, counter)
                counted = True
            except InputValidationError:
                raise
            except Exception:
                counted = False
        self._ensure_ready(need_index)
        if counter is not None and not counted:
            text = validate_claim_text(text, rt.max_claim_chars, rt.max_claim_tokens, counter)
        return text

    def _ensure_local_ready(self, need_index: bool) -> None:
        if not self._check_resources:
            return
        if self.cfg.evidence_selector.enabled and self._evidence_selector is None:
            selector = Path(self.cfg.root) / self.cfg.evidence_selector.model_dir
            if any(not (selector / name).is_file() for name in ("model.safetensors", "config.json", "tokenizer.json")):
                raise DataNotReadyError("Trained rationale selector files are missing.", stage="validate_input")
        if self._corpus is None and not (self.cfg.path("processed_dir") / CORPUS_FILE).exists():
            raise DataNotReadyError("Prepared data not found.", stage="validate_input", hint=DATA_SETUP_HINT)
        if need_index and not self._ready_index:
            _ = self.retriever  # loads and checks fingerprints (stale index -> StaleIndexError)
            self._ready_index = True

    def _ensure_ready(self, need_index: bool) -> None:
        self._ensure_local_ready(need_index)
        if not self._check_resources:
            return
        if not self._ready_model:
            self.llm.ensure_ready()  # raises ModelUnavailableError with setup guidance
            self._ready_model = True

    def _retrieve(self, claim: str, method: str, rerank: bool, keep: int) -> RetrievalLookup:
        """Retrieve with a small in-process LRU cache keyed by ``(claim, method, rerank, keep)``.

        A cache hit returns *no* freshly measured sub-timings (``timings={}``) and flags
        ``cache_hit=True``; the earlier computation's sub-timings travel separately in
        ``original_timings`` so latency reports never mistake copies for measurements."""
        key = (claim, method, rerank, keep)
        if key in self._retrieval_cache:
            self._retrieval_cache.move_to_end(key)
            cached = self._retrieval_cache[key]
            return RetrievalLookup(hits=list(cached.hits), timings={}, cache_hit=True,
                                   original_timings=dict(cached.timings or {}))
        out = self.retriever.retrieve(claim, method=method, rerank=rerank, keep=keep)
        self._retrieval_cache[key] = out
        if len(self._retrieval_cache) > 512:
            self._retrieval_cache.popitem(last=False)
        return RetrievalLookup(hits=list(out.hits), timings=dict(out.timings or {}), cache_hit=False)

    @property
    def _product_model_checked(self) -> bool:
        if self.cfg.product.evidence_mode == "full_abstract" or self.cfg.evidence_selector.enabled:
            if self.cfg.product.judge_backend == "lora":
                return self._lora_judge is not None
            if self.cfg.product.judge_backend == "nli":
                return self._verifier is not None and (not self.cfg.evidence_selector.enabled or self._evidence_selector is not None)
        return bool(self._ready_model)

    @property
    def is_warm(self) -> bool:
        """True once the corpus, the retrieval index and the model check have been loaded/done in this
        process (e.g. after :meth:`warmup` or a first check); the cross-encoder loads on first re-rank."""
        return self._corpus is not None and self._retriever is not None and self._product_model_checked

    def resources_loaded(self) -> dict[str, bool]:
        """Which heavy resources this process has loaded (cheap; loads nothing)."""
        scorer = getattr(self._retriever, "reranker", None) if self._retriever is not None else None
        rer = getattr(scorer, "_model", None)
        return {"corpus": self._corpus is not None, "retriever": self._retriever is not None,
                "encoder": self._encoder is not None, "reranker": rer is not None,
                "model_checked": self._product_model_checked}

    def warmup(self) -> None:
        """Load corpus, indexes, encoders and check the model (useful before a demo)."""
        if (self.cfg.product.evidence_mode == "full_abstract" or self.cfg.evidence_selector.enabled) and self.cfg.product.judge_backend in {"nli", "lora"}:
            self._ensure_local_ready(True)
            _ = self.lora_judge if self.cfg.product.judge_backend == "lora" else self.verifier
            if self.cfg.evidence_selector.enabled:
                _ = self.evidence_selector
        else:
            self._ensure_ready(True)
        self.retriever.retrieve("warm up query", rerank=True, keep=3)

    def readiness(self) -> ReadinessReport:
        return check_readiness(self.cfg)

    # -- metadata -------------------------------------------------------------
    def _metadata(self, opts: dict, strategy: str) -> dict:
        digest = None
        if self._llm is not None:
            digest = self._llm._digest
        fp = getattr(self._retriever, "fingerprint", None)
        extra: dict = {}
        if opts.get("prompt_set") == "v2":
            extra = {"prompt_hashes_v2": self.prompts.hashes_v2(),
                     "judge_v2_settings": self.cfg.judge_v2.model_dump()}
        elif opts.get("prompt_set") == "v3":
            extra = {"prompt_hashes_v3": self.prompts.hashes_v3(),
                     "judge_v3_settings": self.cfg.judge_v3.model_dump()}
        if opts.get("judge_backend") == "nli":
            extra.update(judge_backend="nli", verifier=self._verifier_identity,
                         verifier_settings=self.cfg.verifier.model_dump())
            extra.update(model="Scientific NLI verifier", model_digest=(self._verifier_identity or {}).get("weights_sha256"),
                         model_settings={"device": self.cfg.verifier.device, "max_length": 512},
                         llm_stats_scope="Ollama transport only; excludes local NLI forward calls")
        elif opts.get("judge_backend") == "lora":
            extra.update(judge_backend="lora", lora=self._lora_identity, lora_settings=self.cfg.lora.model_dump())
            extra.update(model="Qwen2.5-1.5B-Instruct+LoRA", model_digest=(self._lora_identity or {}).get("adapter_sha256"))
            extra.update(prompt_set="lora", model_settings={"decoding": "greedy", "max_context_tokens": self.cfg.lora.max_length,
                                                            "max_new_tokens": 96, "device": self.cfg.lora.device},
                         llm_stats_scope="Ollama transport only; excludes local Qwen generation")
        return {
            "package_version": __version__,
            "created_at": utc_now(),
            "strategy": strategy,
            "options": opts,
            "model": self.cfg.model.name,
            "model_digest": digest,
            "model_settings": {"temperature": self.cfg.model.temperature, "think": self.cfg.model.think,
                               "seed": self.cfg.model.seed, "num_ctx": self.cfg.model.context_tokens},
            "index_fingerprint": fp,
            "prompt_hashes": self.prompts.hashes(),
            "prompt_set": opts.get("prompt_set", "v1"),
            "judge_think": bool(opts.get("judge_think", False)),
            **extra,
            **({"knowledge": self.cfg.knowledge.model_dump()} if self.cfg.knowledge.kind == "general_science" else {}),
            **({"evidence_backend": "trained_classifier", "evidence_selector": self._evidence_selector_identity,
                "evidence_selector_settings": self.cfg.evidence_selector.model_dump()}
               if self.cfg.evidence_selector.enabled and opts.get("evidence_mode") == "evidence" else {}),
        }

    # -- main API -------------------------------------------------------------
    def check(
        self,
        claim: str,
        k: int | None = None,
        *,
        retrieval_method: str = "hybrid",
        rerank: bool = True,
        evidence_mode: str | None = None,
        oracle_doc_ids: list[str] | None = None,
        claim_id: str | None = None,
        variant: str | None = None,
        progress: ProgressCallback | None = None,
        prompt_set: str | None = None,
        judge_think: bool | None = None,
        judge_backend: str | None = None,
    ) -> ClaimResult:
        """Run the supervised LangGraph pipeline on one claim. Never raises for operational problems:
        those come back as ``status="error"`` with ``verdict=None`` and a setup hint.

        ``prompt_set`` (``"v1"`` / ``"v2"``) and ``judge_think`` select the judge configuration; when
        omitted they come from the ``product`` section of the configuration (the product default).
        Evaluation variants always pass them explicitly, so ``--variant full`` stays v1.

        ``progress`` (optional) is called as ``progress(stage, events)`` after each graph node
        finishes, for live stage status in a UI. The graph is then streamed
        (``stream_mode=["updates", "values"]``) instead of invoked; the returned result is the same.
        Exceptions raised by the callback are swallowed (counted in
        ``metadata["progress_callback_errors"]``) so a display problem cannot fail a check.
        """
        k = k or self.cfg.retrieval.pipeline_k
        evidence_mode = evidence_mode or self.cfg.product.evidence_mode
        judge_backend = judge_backend or self.cfg.product.judge_backend
        if prompt_set is None:
            prompt_set = self.cfg.product.prompt_set
            if judge_think is None:
                judge_think = self.cfg.product.judge_think
        try:
            opts = PipelineOptions(k=k, retrieval_method=retrieval_method, rerank=rerank,
                                   evidence_mode=evidence_mode, oracle_doc_ids=oracle_doc_ids, variant=variant,
                                   prompt_set=prompt_set, judge_think=bool(judge_think), judge_backend=judge_backend)
        except Exception as exc:
            return self._error_result(claim, claim_id, ErrorInfo(error_type="invalid_input", message=str(exc),
                                                                 stage="validate_input"), {}, "pipeline")
        with self._lock:
            stats0 = dict(self._llm.stats) if self._llm else {}
            t0 = time.perf_counter()
            callback_errors = [0]
            try:
                state = self._run_graph(
                    {"claim": claim, "claim_id": claim_id, "options": opts, "status": "running",
                     "citation_repairs": 0, "events": [], "timings": {}},
                    {"recursion_limit": self.cfg.runtime.graph_recursion_limit},
                    progress, callback_errors,
                )
            except Exception as exc:  # e.g. GraphRecursionError: bounded graph execution exceeded
                err = ErrorInfo(error_type="graph_error", message=f"{type(exc).__name__}: {exc}", stage="pipeline",
                                hint="The workflow did not reach a terminal state; this is an internal error.")
                return self._error_result(claim, claim_id, err, {"total": time.perf_counter() - t0}, "pipeline",
                                          opts.model_dump())
            total = time.perf_counter() - t0
        result = self._result_from_state(claim, claim_id, state, total, opts, stats0)
        if callback_errors[0]:
            result.metadata["progress_callback_errors"] = callback_errors[0]
        return result

    def _run_graph(self, inputs: dict, config: dict, progress: ProgressCallback | None,
                   callback_errors: list[int]):
        if progress is None:
            return self._graph.invoke(inputs, config=config)
        final = None
        for mode, chunk in self._graph.stream(inputs, config=config, stream_mode=["updates", "values"]):
            if mode == "values":
                final = chunk  # the last "values" chunk is exactly what invoke() returns
                continue
            for stage, update in (chunk or {}).items():
                events = list((update or {}).get("events") or []) if isinstance(update, dict) else []
                try:
                    progress(stage, events)
                except Exception:  # a UI/display failure must never fail the check itself
                    callback_errors[0] += 1
        return final

    def check_variant(
        self, claim: str, variant: str | VariantSpec, *, claim_id: str | None = None,
        oracle_doc_ids: list[str] | None = None, progress: ProgressCallback | None = None,
    ) -> ClaimResult:
        """Run a named variant. ``progress`` is forwarded to :meth:`check` for pipeline variants
        (the judge-only and single-prompt baselines are not graphs and do not report stages)."""
        spec = variant if isinstance(variant, VariantSpec) else self.experiments.variant(variant)
        if spec.strategy == "judge_only":
            return self.check_judge_only(claim, claim_id=claim_id, variant=spec.name)
        if spec.strategy == "single_prompt":
            return self.check_single_prompt(claim, k=spec.k, rerank=spec.rerank,
                                            retrieval_method=spec.retrieval_method, claim_id=claim_id,
                                            variant=spec.name)
        return self.check(claim, k=spec.k, retrieval_method=spec.retrieval_method, rerank=spec.rerank,
                          evidence_mode=spec.evidence_mode,
                          oracle_doc_ids=(oracle_doc_ids or []) if spec.oracle else None,
                          claim_id=claim_id, variant=spec.name, progress=progress,
                          prompt_set=spec.prompt_set, judge_think=spec.judge_think,
                          judge_backend=spec.judge_backend)

    # -- baselines --------------------------------------------------------------
    def _baseline(self, claim: str, claim_id: str | None, strategy: str, opts: dict, body) -> ClaimResult:
        timings: dict[str, float] = {}
        events: list[StageEvent] = []
        with self._lock:
            t0 = time.perf_counter()
            stats0 = dict(self._llm.stats) if self._llm else {}
            try:
                ts = time.perf_counter()
                text = self._validate_then_ready(claim, strategy == "single_prompt")
                timings["validate_input"] = time.perf_counter() - ts
                result = body(text, timings, events)
            except ClaimCheckerError as exc:
                err = ErrorInfo(**exc.to_dict())
                timings["total"] = time.perf_counter() - t0
                return self._error_result(claim, claim_id, err, timings, strategy, opts, events)
            except Exception as exc:
                err = ErrorInfo(error_type="unexpected_error", message=f"{type(exc).__name__}: {exc}", stage=strategy)
                timings["total"] = time.perf_counter() - t0
                return self._error_result(claim, claim_id, err, timings, strategy, opts, events)
            timings["total"] = time.perf_counter() - t0
        result.timings = {k: round(v, 4) for k, v in timings.items()}
        result.metadata = {**self._metadata(opts, strategy), **self._llm_delta(stats0), **result.metadata}
        result.claim_id = claim_id
        result.events = events + result.events
        return result

    def check_judge_only(self, claim: str, *, claim_id: str | None = None, variant: str | None = "judge_only") -> ClaimResult:
        opts = {"variant": variant, "k": 0}

        def body(text, timings, events):
            t = time.perf_counter()
            out, cached = judge_claim_only(self.llm, self.prompts, text)
            timings["judge"] = time.perf_counter() - t
            events.append(StageEvent(stage="judge", message="claim-only judgement", data={"cached": cached}))
            expl = (f"Baseline without retrieval: the model answered {out.verdict} from general knowledge only; "
                    f"no source evidence was consulted. Reason: {out.reason.strip()} "
                    f"Model-reported confidence (uncalibrated): {out.confidence:.2f}.")
            return ClaimResult(status="ok", claim=text, verdict=out.verdict, explanation=expl,
                               metadata={"confidence": float(out.confidence)}, documents=[], evidence=[],
                               events=[], timings={})
        return self._baseline(claim, claim_id, "judge_only", opts, body)

    def check_single_prompt(self, claim: str, *, k: int = 3, rerank: bool = True, retrieval_method: str = "hybrid",
                            claim_id: str | None = None, variant: str | None = "retrieve_judge_single_prompt") -> ClaimResult:
        opts = {"variant": variant, "k": k, "rerank": rerank, "retrieval_method": retrieval_method}

        def body(text, timings, events):
            t = time.perf_counter()
            out = self._retrieve(text, retrieval_method, rerank, max(self.cfg.retrieval.keep_top, k))
            timings["retrieve"] = time.perf_counter() - t  # actual wall time (near zero on a cache hit)
            cache_meta = retrieval_cache_metadata(out)
            if cache_meta.get("retrieval_cache_hit"):
                events.append(StageEvent(stage="retrieve", status="info", message=RETRIEVAL_CACHE_NOTE,
                                         data={"flag": "retrieval_cache_hit",
                                               "original_compute_s": cache_meta["retrieve_original_compute_s"]}))
            hits = list(out.hits)
            docs = [self.corpus[h.doc_id] for h in hits[:k]]
            t = time.perf_counter()
            res, ev = retrieve_and_judge(self.llm, self.prompts, text, docs)
            timings["judge"] = time.perf_counter() - t
            events.extend(StageEvent(stage="judge", status=e.get("status", "info"), message=e["message"],
                                     data=e.get("data", {})) for e in ev)
            by_doc: dict[str, list[int]] = {}
            for d, s in res["citations"]:
                by_doc.setdefault(d, []).append(s)
            quotes = [EvidenceQuote(doc_id=d, title=self.corpus[d].title, sentence_id=s,
                                    quote=self.corpus[d].sentences[s], doc_verdict=None)
                      for d, s in res["citations"]]
            documents = [DocumentReport(doc_id=h.doc_id, title=h.title, rank=h.rank, retrieval_score=h.score,
                                        retrieval_method=h.method) for h in hits[:k]]
            # One claim-level verdict over all abstracts: explained per cited document, never as a
            # pseudo-document judgement.
            expl = explain_single_prompt(
                res["verdict"], reason=res["reason"], citations=by_doc,
                titles={d.doc_id: d.title for d in docs}, n_documents=len(docs),
                confidence=res["confidence"], downgraded=res["downgraded"], model_verdict=res["model_verdict"],
            )
            return ClaimResult(status="ok", claim=text, verdict=res["verdict"], conflict=False, evidence=quotes,
                               documents=documents, explanation=expl,
                               events=[], timings={},
                               metadata={"retrieval_hits": [h.model_dump() for h in hits], **cache_meta,
                                         "confidence": None if res["downgraded"] else float(res["confidence"])}
                               | {"single_prompt": {
                                   k2: v for k2, v in res.items() if k2 not in ("citations",)}})
        return self._baseline(claim, claim_id, "single_prompt", opts, body)

    # -- result assembly ----------------------------------------------------------
    def _llm_delta(self, stats0: dict) -> dict:
        if self._llm is None:
            return {"llm_calls": 0, "llm_cache_hits": 0, "llm_repairs": 0}
        s = self._llm.stats
        return {
            "llm_calls": s["calls"] - stats0.get("calls", 0),
            "llm_cache_hits": s["cache_hits"] - stats0.get("cache_hits", 0),
            "llm_repairs": s["repairs"] - stats0.get("repairs", 0),
            "llm_think_fallbacks": s.get("think_fallbacks", 0) - stats0.get("think_fallbacks", 0),
            "model_digest": self._llm._digest,
        }

    def _error_result(self, claim, claim_id, err: ErrorInfo, timings, strategy, opts=None, events=None) -> ClaimResult:
        events = list(events or []) + [StageEvent(stage=err.stage or strategy, status="error", message=err.message,
                                                  data={"error_type": err.error_type})]
        return ClaimResult(
            status="error", claim=claim if isinstance(claim, str) else "", claim_id=claim_id, verdict=None,
            explanation=explain(None, False, [], {}, error=err), timings={k: round(v, 4) for k, v in timings.items()},
            metadata=self._metadata(opts or {}, strategy), events=events, error=err,
        )

    def _result_from_state(self, claim, claim_id, state, total, opts: PipelineOptions, stats0) -> ClaimResult:
        timings = {k: round(v, 4) for k, v in (state.get("timings") or {}).items()}
        timings["total"] = round(total, 4)
        meta = {**self._metadata(opts.model_dump(), "pipeline"), **self._llm_delta(stats0)}
        if state.get("retrieval_cache") is not None:  # absent for oracle mode / failures before retrieval
            meta.update(state["retrieval_cache"])
        events = list(state.get("events") or [])
        if state.get("status") != "ok":
            err = state.get("error") or ErrorInfo(error_type="unexpected_error", message="pipeline ended without result")
            res = self._error_result(claim, claim_id, err, {}, "pipeline", opts.model_dump(), [])
            res.timings, res.events = timings, events
            res.metadata = meta
            return res
        corpus = self.corpus
        hits = {h.doc_id: h for h in state.get("hits") or []}
        selections = state.get("selections") or {}
        judgements = state.get("judgements") or {}
        documents, quotes = [], []
        for rank, d in enumerate(state["doc_ids"], start=1):
            h = hits.get(d)
            j = judgements.get(d)
            if j is not None and j.prompt_set == "v2":
                j = j.model_copy(update={"reason_display": display_reason(j)})
            sel = selections.get(d)
            documents.append(DocumentReport(
                doc_id=d, title=corpus[d].title, rank=h.rank if h else rank,
                retrieval_score=h.score if h else None, retrieval_method=h.method if h else "",
                selected_sentence_ids=sel.sentence_ids if sel else None, judgement=j,
                sentences=list(corpus[d].sentences), source=dict(corpus[d].source)))
            if j is not None:
                for i in j.cited_sentence_ids:
                    quotes.append(EvidenceQuote(doc_id=d, title=corpus[d].title, sentence_id=i,
                                                quote=corpus[d].sentences[i], doc_verdict=j.verdict))
        meta["retrieval_hits"] = [h.model_dump() for h in state.get("hits") or []]
        meta["selection_notes"] = {d: s.note for d, s in selections.items()}
        meta["confidence"] = claim_confidence(state["verdict"], bool(state.get("conflict")),
                                              [judgements.get(d) for d in state["doc_ids"]])
        if opts.prompt_set == "v2":
            meta["judge_thinking"] = thinking_summary(judgements.values())
        return ClaimResult(
            status="ok", claim=state["claim"], claim_id=claim_id, verdict=state["verdict"],
            conflict=bool(state.get("conflict")), evidence=quotes, documents=documents,
            explanation=state.get("explanation", ""), timings=timings, metadata=meta, events=events,
        )
