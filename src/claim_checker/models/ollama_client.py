"""Local Ollama client: availability checks, JSON-schema structured output, Pydantic validation,
bounded repair, structured errors and a deterministic on-disk response cache.

Only a local endpoint is used (enforced by :class:`claim_checker.config.ModelConfig`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from claim_checker.config import ModelConfig, canonical_json
from claim_checker.data.store import atomic_write_text
from claim_checker.errors import (
    OLLAMA_SETUP_HINT,
    ClaimCheckerError,
    ContextOverflowError,
    ModelOutputError,
    ModelTimeoutError,
    ModelUnavailableError,
)

Message = dict[str, str]
CHARS_PER_TOKEN = 3.0  # conservative estimate for English scientific text
Validator = Callable[[BaseModel], list[str]]
log = logging.getLogger(__name__)


class Transport(Protocol):
    def chat(self, payload: dict) -> str: ...
    def list_models(self) -> list[dict]: ...
    def version(self) -> str: ...


def inline_refs(schema: dict) -> dict:
    """Resolve local ``$ref``/``$defs`` so the schema is self-contained for the grammar engine."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def schema_for(model: type[BaseModel]) -> dict:
    return inline_refs(model.model_json_schema())


_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


# Budget forcing for qwen3 thinking: the truncated thinking is closed with this sentence and the
# answer is decoded (under the JSON schema) in a raw continuation using qwen3's chat template.
BUDGET_CLOSE = "I have used my thinking budget, so I will answer now."


def qwen3_raw_prompt(messages: list[Message], thinking: str) -> str:
    """qwen3 chat template (ChatML) for ``messages`` followed by an assistant turn whose thinking block
    holds ``thinking`` (closed by :data:`BUDGET_CLOSE`), ready for the answer."""
    turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return f"{turns}<|im_start|>assistant\n<think>\n{thinking.strip()}\n\n{BUDGET_CLOSE}\n</think>\n\n"


def clean_content(raw: str) -> str:
    text = _THINK.sub("", raw or "").strip()
    return _FENCE.sub("", text).strip()


class OllamaTransport:
    """Thin adapter over the ``ollama`` Python client."""

    def __init__(self, host: str, timeout: float):
        import ollama

        self._ollama = ollama
        self.host = host
        self.client = ollama.Client(host=host, timeout=timeout)

    def _map_error(self, exc: Exception, model: str = "") -> ClaimCheckerError:
        import httpx

        hint = OLLAMA_SETUP_HINT.format(model=model or "qwen3:4b")
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError)):
            return ModelUnavailableError(f"Cannot reach Ollama at {self.host}: {exc}", stage="model", hint=hint)
        if isinstance(exc, httpx.TimeoutException):
            return ModelTimeoutError(f"Ollama request timed out: {exc}", stage="model", hint=
                                     "The local model did not answer within the timeout; check GPU/CPU load.")
        if isinstance(exc, self._ollama.ResponseError):
            if getattr(exc, "status_code", None) == 404:
                return ModelUnavailableError(f"Model {model!r} is not available in Ollama: {exc}",
                                             stage="model", hint=hint)
            return ModelOutputError(f"Ollama returned an error: {exc}", stage="model")
        return ModelUnavailableError(f"Ollama request failed: {type(exc).__name__}: {exc}", stage="model", hint=hint)

    def chat(self, payload: dict) -> str:
        return self.chat_full(payload)["content"]

    def chat_full(self, payload: dict) -> dict:
        """Answer content plus generation metadata. With ``think=True`` Ollama returns the reasoning
        separately in ``message.thinking``; only ``content`` is ever parsed as the answer, and only
        the length of the thinking text is passed on (never the text itself)."""
        try:
            resp = self.client.chat(**payload)
        except Exception as exc:
            raise self._map_error(exc, payload.get("model", "")) from exc

        def get(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        message = get(resp, "message")
        thinking = get(message, "thinking") or ""
        return {
            "content": get(message, "content") or "",
            # Kept in memory only for a budget-forced continuation; never cached or returned to callers.
            "thinking": thinking,
            "thinking_chars": len(thinking),
            "done_reason": get(resp, "done_reason"),
            "eval_count": get(resp, "eval_count"),
        }

    def generate_raw(self, payload: dict) -> dict:
        """``/api/generate`` with ``raw=True`` (the caller supplies the full chat-template prompt)."""
        try:
            resp = self.client.generate(**payload)
        except Exception as exc:
            raise self._map_error(exc, payload.get("model", "")) from exc

        def get(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        return {"content": get(resp, "response") or "", "thinking_chars": 0,
                "done_reason": get(resp, "done_reason"), "eval_count": get(resp, "eval_count")}

    def list_models(self) -> list[dict]:
        try:
            resp = self.client.list()
        except Exception as exc:
            raise self._map_error(exc) from exc
        models = resp["models"] if isinstance(resp, dict) else resp.models
        out = []
        for m in models:
            d = m if isinstance(m, dict) else m.model_dump()
            out.append({"name": d.get("model") or d.get("name"), "digest": d.get("digest"),
                        "size": d.get("size"), "details": d.get("details")})
        return out

    def version(self) -> str:
        import httpx

        try:
            r = httpx.get(f"{self.host}/api/version", timeout=5)
            r.raise_for_status()
            return r.json().get("version", "unknown")
        except Exception as exc:
            raise self._map_error(exc) from exc


@dataclass
class LLMResponse:
    parsed: BaseModel
    raw: str
    cached: bool
    attempts: int
    latency: float
    problems: list[str] = field(default_factory=list)
    cache_key: str = ""
    # True when the repair attempt was schema-invalid and the earlier schema-valid answer
    # (with its semantic ``problems``) was kept instead - see :meth:`OllamaClient.structured`.
    repair_fallback: bool = False
    # Set when an existing disk-cache entry for this request was unreadable or incompatible and the
    # answer was regenerated (and the entry rewritten). Agents turn it into a warning stage event.
    cache_problem: str | None = None
    # Thinking (qwen3 ``think=True``) bookkeeping: characters of thinking text over all attempts (the
    # text itself is discarded), generated tokens (thinking + answer) when the transport reports
    # them, the think flag actually requested, and whether a thinking attempt ran out of output
    # budget before answering and was retried once without thinking.
    thinking_chars: int = 0
    eval_tokens: int | None = None
    think: bool = False
    think_fallback: bool = False
    # Number of attempts whose thinking hit ``think_budget`` and was closed by budget forcing.
    budget_forced: int = 0


def cache_warning_event(resp: LLMResponse, where: str) -> dict | None:
    """Warning stage-event dict for a regenerated corrupt cache entry, or None."""
    if not resp.cache_problem:
        return None
    return {"status": "warning", "message": f"LLM cache entry for {where} was unreadable and was regenerated "
            f"({resp.cache_problem})", "data": {"flag": "llm_cache_corrupt", "cache_key": resp.cache_key}}


def _with_tag(name: str) -> str:
    """``qwen3`` -> ``qwen3:latest`` (Ollama's implicit tag); tagged names are unchanged."""
    name = (name or "").strip()
    last = name.rsplit("/", 1)[-1]  # a registry prefix may contain a ``host:port`` colon
    return name if ":" in last or not name else f"{name}:latest"


def model_name_matches(name: str | None, wanted: str) -> bool:
    """True if an installed Ollama model ``name`` is the configured model ``wanted``.

    Shared by :func:`claim_checker.runtime.check_readiness` and :class:`OllamaClient` so the
    readiness check and the client agree. An untagged name means ``:latest`` on either side;
    any other tag must match exactly (``qwen3:4b`` never matches ``qwen3:8b``).
    """
    if not name or not wanted:
        return False
    return _with_tag(name) == _with_tag(wanted)


_model_matches = model_name_matches  # backwards-compatible private alias


class OllamaClient:
    """Structured-output client shared by all agents."""

    def __init__(self, cfg: ModelConfig, cache_dir: str | Path | None = None, transport: Transport | None = None):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) / "llm" if cache_dir and cfg.cache_enabled else None
        self._transport = transport
        self._digest: str | None = None
        self._lock = threading.Semaphore(max(1, cfg.max_concurrent_generations))
        self.stats = {"calls": 0, "cache_hits": 0, "repairs": 0, "cache_corrupt": 0, "think_fallbacks": 0}

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            self._transport = OllamaTransport(self.cfg.host, self.cfg.request_timeout_seconds)
        return self._transport

    # -- availability -------------------------------------------------------
    def server_version(self) -> str:
        return self.transport.version()

    def model_info(self) -> dict:
        models = self.transport.list_models()
        for m in models:
            if _model_matches(m["name"] or "", self.cfg.name):
                return m
        raise ModelUnavailableError(
            f"Model {self.cfg.name!r} is not downloaded in Ollama (available: "
            f"{[m['name'] for m in models] or 'none'})",
            stage="model",
            hint=OLLAMA_SETUP_HINT.format(model=self.cfg.name),
        )

    def model_digest(self) -> str:
        if self._digest is None:
            self._digest = self.model_info().get("digest") or "unknown"
        return self._digest

    def ensure_ready(self) -> str:
        return self.model_digest()

    # -- structured generation ------------------------------------------------
    def options(self, max_output_tokens: int | None = None) -> dict:
        """Generation options. ``max_output_tokens`` overrides ``num_predict`` for one request (the
        v2 judge with thinking); ``None`` gives exactly the configured (v1) options."""
        return {
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "num_ctx": self.cfg.context_tokens,
            "num_predict": max_output_tokens or self.cfg.max_output_tokens,
        }

    def _think(self, think: bool | None) -> bool:
        return self.cfg.think if think is None else bool(think)

    def _payload(self, messages: list[Message], schema: dict, think: bool | None = None,
                 max_output_tokens: int | None = None) -> dict:
        return {
            "model": self.cfg.name,
            "messages": messages,
            "format": schema,
            "options": self.options(max_output_tokens),
            "think": self._think(think),
            "keep_alive": self.cfg.keep_alive,
            "stream": False,
        }

    def check_context(self, messages: list[Message], schema: dict, purpose: str = "",
                      max_output_tokens: int | None = None) -> int:
        """Conservative prompt-size estimate (chars / CHARS_PER_TOKEN). Raise instead of letting
        Ollama silently truncate the start of an oversized prompt."""
        chars = sum(len(m["content"]) for m in messages) + len(json.dumps(schema))
        est = int(chars / CHARS_PER_TOKEN) + 1
        budget = self.cfg.context_tokens - (max_output_tokens or self.cfg.max_output_tokens)
        if est > budget:
            raise ContextOverflowError(
                f"{purpose or 'prompt'} needs about {est} tokens but only {budget} fit the "
                f"{self.cfg.context_tokens}-token context (after reserving output tokens).",
                stage=purpose or "model",
                hint="Use the evidence agent (bounded windows), a smaller k, or raise model.context_tokens.",
            )
        return est

    def cache_key(self, messages: list[Message], schema: dict, think: bool | None = None,
                  max_output_tokens: int | None = None, think_budget: int | None = None) -> str:
        """SHA-256 of everything that determines the answer. With no per-request overrides the
        identity is exactly the v1 one (so v1 cache entries keep hitting); the effective think flag
        and output budget are always part of it, so think/no-think answers never collide."""
        ident = {
            "model": self.cfg.name,
            "digest": self.model_digest(),
            "messages": messages,
            "schema": schema,
            "options": self.options(max_output_tokens),
            "think": self._think(think),
            "repairs": self.cfg.max_schema_repair_attempts,
        }
        if think_budget is not None and self._think(think):
            ident["think_budget"] = int(think_budget)
        return hashlib.sha256(canonical_json(ident).encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def structured(
        self,
        messages: list[Message],
        schema_model: type[BaseModel],
        validator: Validator | None = None,
        purpose: str = "",
        *,
        think: bool | None = None,
        max_output_tokens: int | None = None,
        think_budget: int | None = None,
    ) -> LLMResponse:
        """Request JSON matching ``schema_model``; validate; one bounded repair; structured error.

        ``validator`` returns semantic problems (e.g. sentence ids outside the supplied context).
        Schema errors or semantic problems trigger the repair attempt (at most
        ``max_schema_repair_attempts``, default 1). If problems persist the parsed result is
        returned with ``problems`` set so the caller can sanitize it.

        Repair policy - a repair can only improve the outcome, never make it worse:

        * the most recent *schema-valid* answer is the result; a schema-valid repair replaces an
          earlier answer (it was produced with explicit feedback about the earlier problems);
        * a *schema-invalid* repair never discards an earlier schema-valid answer: that answer is
          returned with its semantic ``problems`` and ``repair_fallback=True``, and the caller's
          existing semantic handling (invalid-id filtering, supervisor re-ask / NEI downgrade)
          deals with it; agents record a warning stage event for the fallback;
        * only when no attempt was schema-valid is :class:`ModelOutputError` raised.

        ``think`` / ``max_output_tokens`` override the configured think flag and ``num_predict`` for
        this request (``None`` = configuration, i.e. v1 behaviour). With thinking on, an attempt
        that spends the whole output budget on thinking and returns no answer is retried once with
        thinking off (``think_fallback``); this retry is extra to the schema-repair budget.

        ``think_budget`` (thinking on only) caps thinking at that many tokens (budget forcing): the
        first request runs with ``num_predict = think_budget``; if it stops for length, the thinking so
        far is closed and the answer is decoded in a raw qwen3-template continuation under the same
        schema (``budget_forced``). Transports or models without that capability fall back to one
        request without thinking.
        """
        schema = schema_for(schema_model)
        self.check_context(messages, schema, purpose, max_output_tokens)
        use_think = self._think(think)
        key = self.cache_key(messages, schema, think, max_output_tokens, think_budget)
        path = self._cache_path(key)
        cache_problem: str | None = None
        if path is not None and path.exists():
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                parsed = schema_model.model_validate_json(entry["raw"])
                self.stats["cache_hits"] += 1
                return LLMResponse(parsed, entry["raw"], True, entry.get("attempts", 1), 0.0,
                                   entry.get("problems", []), key, bool(entry.get("repair_fallback", False)),
                                   thinking_chars=int(entry.get("thinking_chars", 0) or 0),
                                   eval_tokens=entry.get("eval_tokens"), think=use_think,
                                   think_fallback=bool(entry.get("think_fallback", False)),
                                   budget_forced=int(entry.get("budget_forced", 0) or 0))
            except Exception as exc:  # corrupt or incompatible entry: regenerate, but say so
                cache_problem = f"{type(exc).__name__}: {str(exc)[:200]}"
                self.stats["cache_corrupt"] += 1
                log.warning("Corrupt or incompatible LLM cache entry %s (%s); regenerating it.", path, cache_problem)

        convo = list(messages)
        t0 = time.perf_counter()
        last_error = ""
        # Most recent schema-valid answer: (parsed, cleaned text, semantic problems).
        best: tuple[BaseModel, str, list[str]] | None = None
        last_valid = False
        attempts = 0
        thinking_chars = 0
        eval_tokens: int | None = None
        think_fallback = False
        budget_forced = 0
        for attempt in range(1 + self.cfg.max_schema_repair_attempts):
            attempts = attempt + 1
            if use_think and think_budget is not None:
                out = self._chat(convo, schema, True, think_budget)
                if out.get("done_reason") == "length":
                    forced = self._budget_forced(convo, schema, out.get("thinking") or "", max_output_tokens)
                    thinking_chars += out.get("thinking_chars") or 0
                    eval_tokens = (eval_tokens or 0) + (out.get("eval_count") or 0)
                    if forced is not None:
                        budget_forced += 1
                        out = forced
                    else:  # no raw-template continuation available: answer without thinking
                        think_fallback, use_think = True, False
                        self.stats["think_fallbacks"] += 1
                        out = self._chat(convo, schema, False, max_output_tokens)
            else:
                out = self._chat(convo, schema, use_think, max_output_tokens)
            if use_think and not clean_content(out["content"]) and out.get("done_reason") == "length":
                # Thinking used the whole budget and no answer was produced: retry once without it.
                think_fallback, use_think = True, False
                self.stats["think_fallbacks"] += 1
                thinking_chars += out.get("thinking_chars") or 0
                eval_tokens = (eval_tokens or 0) + (out.get("eval_count") or 0)
                log.warning("%s: thinking exhausted num_predict without an answer; retrying without thinking",
                            purpose or "model")
                out = self._chat(convo, schema, False, max_output_tokens)
            thinking_chars += out.get("thinking_chars") or 0
            if out.get("eval_count") is not None:
                eval_tokens = (eval_tokens or 0) + out["eval_count"]
            raw = out["content"]
            text = clean_content(raw)
            try:
                parsed = schema_model.model_validate_json(text)
            except ValidationError as exc:
                last_error = _short_validation_error(exc)
                problems = [f"output does not match the required JSON schema: {last_error}"]
                last_valid = False
            else:
                problems = list(validator(parsed)) if validator else []
                best, last_valid = (parsed, text, problems), True
                if not problems:
                    break
            if attempt < self.cfg.max_schema_repair_attempts:
                self.stats["repairs"] += 1
                convo = convo + [
                    {"role": "assistant", "content": text[:4000]},
                    {"role": "user", "content": "Your previous answer was invalid: " + "; ".join(problems)
                     + ". Answer again with only a JSON object that follows the schema and the rules."},
                ]
        latency = time.perf_counter() - t0
        if best is None:
            raise ModelOutputError(
                f"{purpose or 'model'} output invalid after {attempts} attempt(s): {last_error}",
                stage=purpose or "model",
                hint="The local model did not return valid structured output; retry or inspect the prompt.",
            )
        parsed, raw, problems = best
        repair_fallback = not last_valid  # the final (repair) attempt was schema-invalid
        if path is not None:
            entry = {"raw": raw, "attempts": attempts, "problems": problems, "purpose": purpose}
            if repair_fallback:
                entry["repair_fallback"] = True
            if self._think(think):  # thinking bookkeeping only for thinking requests (v1 entries unchanged)
                entry.update(thinking_chars=thinking_chars, eval_tokens=eval_tokens, think_fallback=think_fallback,
                             budget_forced=budget_forced)
            # Atomic (temp file + os.replace): an interrupted write can never leave a truncated entry.
            atomic_write_text(path, json.dumps(entry, ensure_ascii=False))
        return LLMResponse(parsed, raw, False, attempts, latency, problems, key, repair_fallback, cache_problem,
                           thinking_chars=thinking_chars, eval_tokens=eval_tokens, think=self._think(think),
                           think_fallback=think_fallback, budget_forced=budget_forced)

    def _budget_forced(self, convo: list[Message], schema: dict, thinking: str,
                       max_output_tokens: int | None) -> dict | None:
        """Close truncated thinking and decode the answer (raw qwen3 template, same schema and options).
        ``None`` when the transport has no raw generate or the model is not a qwen3 model."""
        raw = getattr(self.transport, "generate_raw", None)
        if not callable(raw) or not self.cfg.name.split("/")[-1].lower().startswith("qwen3"):
            return None
        payload = {
            "model": self.cfg.name,
            "prompt": qwen3_raw_prompt(convo, thinking),
            "raw": True,
            "format": schema,
            "options": self.options(max_output_tokens),
            "keep_alive": self.cfg.keep_alive,
            "stream": False,
        }
        with self._lock:
            self.stats["calls"] += 1
            self.stats["budget_forced"] = self.stats.get("budget_forced", 0) + 1
            return raw(payload)

    def _chat(self, convo: list[Message], schema: dict, think: bool, max_output_tokens: int | None) -> dict:
        """One model request. Transports without ``chat_full`` (test fakes) return content only."""
        payload = self._payload(convo, schema, think, max_output_tokens)
        with self._lock:
            self.stats["calls"] += 1
            full = getattr(self.transport, "chat_full", None)
            if callable(full):
                return full(payload)
            return {"content": self.transport.chat(payload), "thinking_chars": 0, "done_reason": None,
                    "eval_count": None}


def _short_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)
