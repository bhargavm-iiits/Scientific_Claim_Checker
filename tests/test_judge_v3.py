import json

import pytest
from conftest import FakeTransport, make_doc

from claim_checker.agents.judge_v3 import judge_v3
from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import JudgeV3Config
from claim_checker.models.ollama_client import OllamaClient


def answer(verdict, ids):
    return json.dumps({"reason": "The reported finding establishes the claim.", "verdict": verdict,
                       "cited_sentence_ids": ids, "confidence": 0.9})


def test_v3_windows_keep_original_citation_positions(cfg):
    doc = make_doc("101", "Results", ["a " * 300, "b " * 300, "c " * 300])
    calls = []

    def respond(payload):
        text = payload["messages"][1]["content"]
        calls.append(text)
        return answer("SUPPORT", [2] if "[2]" in text else [0] if "[0]" in text else [1])

    client = OllamaClient(cfg.model, transport=FakeTransport(respond))
    result, _ = judge_v3(client, PromptLibrary(cfg.path("prompts_dir")), "target claim", doc,
                         [0, 1, 2], JudgeV3Config(use_examples=False, max_context_chars=1000))
    assert len(calls) == 3
    assert result.cited_sentence_ids == [0, 1, 2]
    assert result.context_sentence_ids == [0, 1, 2]
    assert result.verdict == "SUPPORT"


def test_v3_rejects_outside_context_ids(cfg):
    doc = make_doc("101", "Results", ["First sentence."])
    client = OllamaClient(cfg.model, transport=FakeTransport())
    with pytest.raises(ValueError, match="Invalid context"):
        judge_v3(client, PromptLibrary(cfg.path("prompts_dir")), "claim", doc, [9], JudgeV3Config())


def test_v3_empty_evidence_uses_no_model(cfg):
    transport = FakeTransport()
    client = OllamaClient(cfg.model, transport=transport)
    result, _ = judge_v3(client, PromptLibrary(cfg.path("prompts_dir")), "claim",
                         make_doc("101", "Results", ["First sentence."]), [], JudgeV3Config())
    assert result.verdict == "NEI" and result.skipped_model
    assert not transport.calls
