import json

import httpx
import pytest

from claim_checker.config import ModelConfig, load_config
from claim_checker.errors import (
    ConfigError,
    ModelOutputError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from claim_checker.models.ollama_client import (
    OllamaClient,
    OllamaTransport,
    clean_content,
    schema_for,
)
from claim_checker.schemas import JudgeOutput, RetrieveJudgeOutput
from tests.conftest import ROOT, FakeTransport

GOOD = json.dumps({"verdict": "SUPPORT", "cited_sentence_ids": [1], "reason": "r", "confidence": 0.9})
MSG = [{"role": "user", "content": "x"}]


def client_with(responses, cache_dir=None, **cfg_kw):
    it = iter(responses)
    t = FakeTransport(lambda p: next(it))
    return OllamaClient(ModelConfig(**cfg_kw), cache_dir=cache_dir, transport=t), t


def test_valid_structured_response_is_parsed_with_expected_options():
    c, t = client_with([GOOD])
    r = c.structured(MSG, JudgeOutput)
    assert r.parsed.verdict == "SUPPORT" and r.attempts == 1
    p = t.calls[0]
    assert p["think"] is False and p["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 768}
    assert p["format"]["properties"]["verdict"]["enum"] == ["SUPPORT", "CONTRADICT", "NEI"]


def test_unsupported_label_triggers_one_repair_then_succeeds():
    bad = json.dumps({"verdict": "TRUE", "cited_sentence_ids": [], "reason": "r", "confidence": 0.5})
    c, t = client_with([bad, GOOD])
    r = c.structured(MSG, JudgeOutput)
    assert r.attempts == 2 and len(t.calls) == 2
    assert "invalid" in t.calls[1]["messages"][-1]["content"]


def test_invalid_twice_is_a_terminal_structured_error():
    c, t = client_with(["not json", "{\"verdict\": 1}"])
    with pytest.raises(ModelOutputError):
        c.structured(MSG, JudgeOutput, purpose="judge")
    assert len(t.calls) == 2  # bounded: exactly one repair


def test_semantic_problems_are_repaired_then_returned():
    c, t = client_with([GOOD, GOOD])
    r = c.structured(MSG, JudgeOutput, validator=lambda o: ["id 1 not allowed"])
    assert len(t.calls) == 2 and r.problems == ["id 1 not allowed"]


def test_timeout_and_unavailable_propagate_as_operational_errors():
    for exc, cls in ((ModelTimeoutError("t"), ModelTimeoutError), (ModelUnavailableError("u"), ModelUnavailableError)):
        c = OllamaClient(ModelConfig(), transport=FakeTransport(fail=exc))
        with pytest.raises(cls):
            c.structured(MSG, JudgeOutput)


def test_transport_maps_httpx_errors():
    t = OllamaTransport("http://127.0.0.1:9", timeout=1)
    assert isinstance(t._map_error(httpx.ConnectError("refused"), "qwen3:4b"), ModelUnavailableError)
    assert isinstance(t._map_error(httpx.ReadTimeout("slow"), "qwen3:4b"), ModelTimeoutError)
    with pytest.raises(ModelUnavailableError):
        t.list_models()  # nothing listens on port 9


def test_missing_model_gives_setup_guidance():
    c = OllamaClient(ModelConfig(), transport=FakeTransport(models=[{"name": "llama3:8b", "digest": "x"}]))
    with pytest.raises(ModelUnavailableError) as ei:
        c.ensure_ready()
    assert "ollama pull qwen3:4b" in ei.value.hint


def test_disk_cache_reuses_identical_calls(tmp_path):
    c, t = client_with([GOOD], cache_dir=tmp_path)
    a = c.structured(MSG, JudgeOutput)
    b = c.structured(MSG, JudgeOutput)
    assert not a.cached and b.cached and len(t.calls) == 1 and c.stats["cache_hits"] == 1
    assert list((tmp_path / "llm").rglob("*.json"))


def test_cloud_hosts_are_rejected():
    with pytest.raises(ConfigError):
        load_config(root=ROOT, overrides={"model": {"host": "https://api.example.com"}})


def test_oversized_prompt_is_flagged_not_truncated():
    from claim_checker.errors import ContextOverflowError

    c, t = client_with([GOOD], context_tokens=1000, max_output_tokens=200)
    with pytest.raises(ContextOverflowError):
        c.structured([{"role": "user", "content": "word " * 2000}], JudgeOutput, purpose="judge")
    assert t.calls == []


def test_schema_is_self_contained_and_content_cleaned():
    s = schema_for(RetrieveJudgeOutput)
    assert "$defs" not in json.dumps(s) and "$ref" not in json.dumps(s)
    assert clean_content("<think>\n</think>\n```json\n{\"a\": 1}\n```") == '{"a": 1}'


@pytest.mark.parametrize("host", ["http://127.0.0.1:11434", "http://localhost:11434/", "http://[::1]:11434",
                                  "https://LOCALHOST:443"])
def test_loopback_hosts_are_accepted(host):
    assert ModelConfig(host=host).host == host.rstrip("/")


@pytest.mark.parametrize("host", [
    "http://localhost.example.com:11434",  # substring of "localhost" - was accepted before
    "http://127.0.0.1.nip.io:11434",
    "http://api.example.com/?h=127.0.0.1",
    "http://user@localhost:11434",
    "http://0.0.0.0:11434",
    "127.0.0.1:11434",  # no scheme
    "ftp://127.0.0.1",
    "http://localhost:notaport",
])
def test_non_loopback_hosts_are_rejected_with_clear_error(host):
    with pytest.raises(ConfigError) as ei:
        load_config(root=ROOT, overrides={"model": {"host": host}})
    assert "local loopback endpoint" in ei.value.message


@pytest.mark.parametrize("installed,wanted,ok", [
    ("qwen3:4b", "qwen3:4b", True),
    ("qwen3:latest", "qwen3", True),
    ("qwen3", "qwen3:latest", True),
    ("qwen3:8b", "qwen3:4b", False),
    ("qwen3:4b", "qwen3", False),
    (None, "qwen3:4b", False),
])
def test_readiness_and_client_share_model_name_matching(monkeypatch, tmp_path, installed, wanted, ok):
    from claim_checker import runtime
    from claim_checker.models.ollama_client import model_name_matches

    assert model_name_matches(installed, wanted) is ok
    cfg = load_config(root=ROOT, overrides={"model": {"name": wanted}, "paths": {"processed_dir": str(tmp_path)}})

    class Resp:
        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    tags = {"models": [{"model": installed, "name": installed, "digest": "sha256:d"}]}
    monkeypatch.setattr(httpx, "get", lambda url, **kw: Resp({"version": "x"} if url.endswith("/api/version") else tags))
    rep = runtime.check_readiness(cfg, check_model=True)
    client = OllamaClient(cfg.model, transport=FakeTransport(models=[{"name": installed, "digest": "sha256:d"}]))
    if ok:
        assert rep.model_ready and client.ensure_ready() == "sha256:d"
    else:
        assert not rep.model_ready
        with pytest.raises(ModelUnavailableError):
            client.ensure_ready()


def test_schema_invalid_repair_keeps_first_schema_valid_answer(tmp_path):
    # Attempt 1 parses but cites an id outside the context; the repair is not even JSON. The repair
    # must not discard attempt 1: it is returned with its problems for the caller to sanitize.
    first = json.dumps({"verdict": "SUPPORT", "cited_sentence_ids": [1, 9], "reason": "r", "confidence": 0.9})
    c, t = client_with([first, "not json at all"], cache_dir=tmp_path)
    r = c.structured(MSG, JudgeOutput, validator=lambda o: ["id 9 not allowed"] if 9 in o.cited_sentence_ids else [])
    assert len(t.calls) == 2  # bounded: still exactly one repair
    assert r.repair_fallback and r.attempts == 2 and r.problems == ["id 9 not allowed"]
    assert r.parsed.verdict == "SUPPORT" and r.parsed.cited_sentence_ids == [1, 9]
    cached = c.structured(MSG, JudgeOutput, validator=lambda o: [])
    assert cached.cached and cached.repair_fallback and cached.parsed == r.parsed


def test_schema_valid_repair_still_replaces_first_answer():
    first = json.dumps({"verdict": "SUPPORT", "cited_sentence_ids": [9], "reason": "r", "confidence": 0.9})
    c, _ = client_with([first, GOOD])
    r = c.structured(MSG, JudgeOutput, validator=lambda o: ["bad"] if 9 in o.cited_sentence_ids else [])
    assert not r.repair_fallback and r.parsed.cited_sentence_ids == [1] and r.problems == []


# ---------------------------------------------------------------------------
# Review fixes: atomic cache writes, visible corrupt-entry regeneration
# ---------------------------------------------------------------------------


def test_cache_entry_is_written_atomically(tmp_path, monkeypatch):
    import claim_checker.models.ollama_client as oc

    seen = []
    real = oc.atomic_write_text
    monkeypatch.setattr(oc, "atomic_write_text", lambda path, text: (seen.append(path), real(path, text)))
    c, _ = client_with([GOOD], cache_dir=tmp_path)
    c.structured(MSG, JudgeOutput)
    files = list((tmp_path / "llm").rglob("*.json"))
    assert len(seen) == 1 and files == [seen[0]]
    assert not list((tmp_path / "llm").rglob("*.tmp"))  # no temp file left behind
    assert json.loads(files[0].read_text(encoding="utf-8"))["raw"] == GOOD


def test_interrupted_cache_write_leaves_no_partial_entry(tmp_path, monkeypatch):
    import os

    c, _ = client_with([GOOD], cache_dir=tmp_path)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        c.structured(MSG, JudgeOutput)
    assert not list((tmp_path / "llm").rglob("*.json")) and not list((tmp_path / "llm").rglob("*.tmp"))


def test_corrupt_cache_entry_is_regenerated_with_a_warning(tmp_path, caplog):
    from claim_checker.models.ollama_client import cache_warning_event

    c, t = client_with([GOOD, GOOD], cache_dir=tmp_path)
    first = c.structured(MSG, JudgeOutput)
    path = next((tmp_path / "llm").rglob("*.json"))
    path.write_text('{"raw": "{truncated', encoding="utf-8")  # e.g. an old non-atomic, interrupted write
    with caplog.at_level("WARNING", logger="claim_checker.models.ollama_client"):
        again = c.structured(MSG, JudgeOutput)
    assert not again.cached and len(t.calls) == 2 and again.parsed == first.parsed
    assert again.cache_problem and c.stats["cache_corrupt"] == 1
    assert any("Corrupt or incompatible LLM cache entry" in r.getMessage() for r in caplog.records)
    ev = cache_warning_event(again, "judge/101")
    assert ev["status"] == "warning" and ev["data"]["flag"] == "llm_cache_corrupt"
    assert cache_warning_event(first, "judge/101") is None
    assert json.loads(path.read_text(encoding="utf-8"))["raw"] == GOOD  # entry repaired
    assert c.structured(MSG, JudgeOutput).cached


def test_corrupt_cache_entry_becomes_a_stage_event(cfg, corpus, tmp_path):
    from claim_checker.agents.judge import judge_document
    from claim_checker.agents.prompting import PromptLibrary

    c, _ = client_with([GOOD, GOOD], cache_dir=tmp_path)
    prompts = PromptLibrary(cfg.path("prompts_dir"))
    judge_document(c, prompts, "claim", corpus["101"], "full_abstract", None, cfg.judge)
    next((tmp_path / "llm").rglob("*.json")).write_text("not json", encoding="utf-8")
    _, events = judge_document(c, prompts, "claim", corpus["101"], "full_abstract", None, cfg.judge)
    assert any((e.get("data") or {}).get("flag") == "llm_cache_corrupt" for e in events)


def test_connect_error_maps_to_unavailable_after_dead_branch_removed():
    t = OllamaTransport("http://127.0.0.1:1", timeout=1)
    err = t._map_error(httpx.ConnectError("refused"), "qwen3:4b")
    assert isinstance(err, ModelUnavailableError) and "Cannot reach Ollama" in err.message
    assert isinstance(t._map_error(ConnectionError("x")), ModelUnavailableError)
    assert isinstance(t._map_error(httpx.ReadTimeout("slow")), ModelTimeoutError)
