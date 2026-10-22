"""v2 judge (checklist + reason before verdict, qwen3 thinking with a budget, consistency check),
its configuration wiring, the dev_tune split and the guarantees that v1 is unchanged. Offline:
scripted fake transports only."""

from __future__ import annotations

import json

import pytest

from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import ModelConfig, VariantSpec, load_config, variant_dump
from claim_checker.data.splits import build_dev_tune_split, freeze_dev_tune_split, load_split
from claim_checker.data.store import write_json, write_jsonl
from claim_checker.evaluation.analysis import load_predictions
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.models.ollama_client import OllamaClient, qwen3_raw_prompt, schema_for
from claim_checker.pipeline.consistency import consistency_flags, nei_disagrees_with_checklist
from claim_checker.schemas import ClaimGold, DocumentJudgement, JudgeOutput, JudgeOutputV2
from tests.conftest import ROOT, FakeTransport, is_v2_judge, purpose_of, scripted, v2_answer

V1_HASHES = {"evidence": "25376c48b37ac36e", "judge": "ae8f024d7631b2f1", "judge_only": "dc2ea4990866fadd",
             "retrieve_judge": "2ea3577c957ee917"}
MSG = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


# ---------------------------------------------------------------------------
# v1 is untouched
# ---------------------------------------------------------------------------


def test_v1_prompt_hashes_are_unchanged():
    lib = PromptLibrary(ROOT / "prompts")
    assert lib.hashes() == V1_HASHES  # the v1 record holds only the frozen top-level templates
    assert set(lib.hashes_v2()) == {"v2/judge"}


def test_v1_request_and_cache_key_identity_unchanged():
    """Without per-request overrides the payload and the cache identity are exactly the v1 ones."""
    import hashlib

    from claim_checker.config import canonical_json

    t = FakeTransport(lambda p: json.dumps({"verdict": "NEI", "cited_sentence_ids": [], "reason": "r",
                                            "confidence": 0.5}))
    c = OllamaClient(ModelConfig(), transport=t)
    schema = schema_for(JudgeOutput)
    legacy = {"model": "qwen3:4b", "digest": "sha256:fake", "messages": MSG, "schema": schema,
              "options": {"temperature": 0.0, "seed": 42, "num_ctx": 8192, "num_predict": 768},
              "think": False, "repairs": 1}
    assert c.cache_key(MSG, schema) == hashlib.sha256(canonical_json(legacy).encode()).hexdigest()
    c.structured(MSG, JudgeOutput)
    assert t.calls[0]["think"] is False and t.calls[0]["options"]["num_predict"] == 768


def test_cache_key_separates_think_flag_and_budget():
    c = OllamaClient(ModelConfig(), transport=FakeTransport())
    schema = schema_for(JudgeOutputV2)
    k_plain = c.cache_key(MSG, schema)
    k_off = c.cache_key(MSG, schema, think=False)
    k_on = c.cache_key(MSG, schema, think=True)
    k_budget = c.cache_key(MSG, schema, think=True, think_budget=512)
    k_budget2 = c.cache_key(MSG, schema, think=True, think_budget=256)
    assert k_plain == k_off  # config default is think=False
    assert len({k_off, k_on, k_budget, k_budget2}) == 4
    assert c.cache_key(MSG, schema, think=False, think_budget=512) == k_off  # budget irrelevant without thinking


def test_v1_variant_definitions_are_recorded_without_v2_fields(experiments):
    full = variant_dump(experiments.variant("full"))
    assert "prompt_set" not in full and "judge_think" not in full
    v2 = variant_dump(experiments.variant("full_v2"))
    assert v2["prompt_set"] == "v2" and v2["judge_think"] is True


# ---------------------------------------------------------------------------
# Schema order and thinking
# ---------------------------------------------------------------------------


def test_v2_schema_generates_checklist_and_reason_before_verdict():
    props = list(schema_for(JudgeOutputV2)["properties"])
    assert props == ["claim_assertion", "evidence_finding", "same_subject", "evidence_direction", "reason",
                     "verdict", "cited_sentence_ids", "confidence"]
    assert props.index("reason") < props.index("verdict")


def test_v2_payload_keeps_property_order_and_requests_thinking(make_checker):
    checker = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])}))
    res = checker.check("Vitamin D increases bone density", k=1, prompt_set="v2", judge_think=True)
    assert res.status == "ok" and res.verdict == "SUPPORT"
    judge = [c for c in checker.transport.calls if purpose_of(c) == "judge"]
    assert len(judge) == 1 and is_v2_judge(judge[0])
    assert list(judge[0]["format"]["properties"])[:6] == ["claim_assertion", "evidence_finding", "same_subject",
                                                           "evidence_direction", "reason", "verdict"]
    assert judge[0]["think"] is True and judge[0]["options"]["num_predict"] == checker.cfg.judge_v2.think_budget_tokens
    assert "judge prompt v2" in judge[0]["messages"][0]["content"]
    ev = [c for c in checker.transport.calls if purpose_of(c) == "evidence"]
    assert ev and all(c["think"] is False for c in ev)  # thinking is for the judge step only
    j = res.documents[0].judgement
    assert j.prompt_set == "v2" and j.checklist["evidence_direction"] == "same_as_claim"
    assert j.thinking["think"] is True
    assert res.metadata["prompt_set"] == "v2" and "v2/judge" in res.metadata["prompt_hashes_v2"]
    assert res.metadata["judge_thinking"]["judge_model_judgements"] == 1


class ThinkingTransport(FakeTransport):
    """Fake transport that reports thinking and supports raw generation (budget forcing)."""

    def __init__(self, chat_results, raw_content=None):
        super().__init__()
        self.chat_results = list(chat_results)
        self.raw_content = raw_content
        self.raw_calls: list[dict] = []

    def chat_full(self, payload):
        self.calls.append(payload)
        return self.chat_results.pop(0)

    def generate_raw(self, payload):
        self.raw_calls.append(payload)
        return {"content": self.raw_content, "thinking_chars": 0, "done_reason": "stop", "eval_count": 40}


GOOD_V2 = json.dumps(v2_answer("CONTRADICT", [1]))


def test_thinking_text_is_never_parsed_or_cached(tmp_path):
    t = ThinkingTransport([{"content": GOOD_V2, "thinking": "SUPPORT? no. secret reasoning", "thinking_chars": 29,
                            "done_reason": "stop", "eval_count": 300}])
    c = OllamaClient(ModelConfig(), cache_dir=tmp_path, transport=t)
    r = c.structured(MSG, JudgeOutputV2, think=True, think_budget=512)
    assert r.parsed.verdict == "CONTRADICT" and r.thinking_chars == 29 and r.budget_forced == 0
    entry = json.loads(next((tmp_path / "llm").rglob("*.json")).read_text(encoding="utf-8"))
    assert "secret" not in json.dumps(entry) and entry["thinking_chars"] == 29
    again = c.structured(MSG, JudgeOutputV2, think=True, think_budget=512)
    assert again.cached and again.thinking_chars == 29 and len(t.calls) == 1


def test_thinking_over_budget_is_closed_and_answer_decoded_under_schema():
    t = ThinkingTransport([{"content": "", "thinking": "step 1 ... step 40", "thinking_chars": 18,
                            "done_reason": "length", "eval_count": 512}], raw_content=GOOD_V2)
    c = OllamaClient(ModelConfig(), transport=t)
    r = c.structured(MSG, JudgeOutputV2, think=True, think_budget=512)
    assert r.parsed.verdict == "CONTRADICT" and r.budget_forced == 1 and not r.think_fallback
    raw = t.raw_calls[0]
    assert raw["raw"] is True and raw["format"] == schema_for(JudgeOutputV2)
    assert raw["prompt"] == qwen3_raw_prompt(MSG, "step 1 ... step 40")
    assert raw["prompt"].endswith("</think>\n\n") and "<|im_start|>assistant\n<think>\nstep 1" in raw["prompt"]
    assert r.eval_tokens == 552


def test_thinking_over_budget_without_raw_support_falls_back_to_no_thinking():
    class NoRaw(FakeTransport):
        def __init__(self):
            super().__init__()
            self.results = [{"content": "", "thinking": "x" * 50, "thinking_chars": 50, "done_reason": "length",
                             "eval_count": 512},
                            {"content": GOOD_V2, "thinking": "", "thinking_chars": 0, "done_reason": "stop",
                             "eval_count": 100}]

        def chat_full(self, payload):
            self.calls.append(payload)
            return self.results.pop(0)

    t = NoRaw()
    c = OllamaClient(ModelConfig(), transport=t)
    r = c.structured(MSG, JudgeOutputV2, think=True, think_budget=512)
    assert r.parsed.verdict == "CONTRADICT" and r.think_fallback and c.stats["think_fallbacks"] == 1
    assert [p["think"] for p in t.calls] == [True, False]


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------


def _j(verdict, subject="yes", direction="same_as_claim", reason="ok"):
    return DocumentJudgement(doc_id="1", verdict=verdict, cited_sentence_ids=[0], reason=reason,
                             checklist={"claim_assertion": "a", "evidence_finding": "b", "same_subject": subject,
                                        "evidence_direction": direction})


@pytest.mark.parametrize("j,flags", [
    (_j("SUPPORT"), []),
    (_j("SUPPORT", direction="opposite_to_claim"), ["checklist_opposite"]),
    (_j("SUPPORT", direction="no_effect_found"), ["checklist_no_effect"]),
    (_j("SUPPORT", subject="partly"), ["checklist_partial_subject"]),
    (_j("SUPPORT", reason="The evidence contradicts the claim."), ["reason_negates_support"]),
    (_j("SUPPORT", reason="This does not support the claim."), ["reason_negates_support"]),
    (_j("CONTRADICT", direction="opposite_to_claim"), []),
    (_j("CONTRADICT", direction="no_effect_found", subject="partly"), []),
    (_j("CONTRADICT", direction="no_effect_found"), []),
    (_j("CONTRADICT", direction="same_as_claim"), ["checklist_same_direction"]),
    (_j("CONTRADICT", direction="opposite_to_claim", reason="It directly supports the claim."),
     ["reason_affirms_support"]),
    (_j("CONTRADICT", direction="opposite_to_claim", reason="It does not support the claim."), []),
    (_j("NEI", direction="opposite_to_claim"), []),
])
def test_consistency_flags(j, flags):
    assert consistency_flags(j) == flags


def test_v1_judgements_are_never_checked():
    j = DocumentJudgement(doc_id="1", verdict="SUPPORT", cited_sentence_ids=[0], reason="this contradicts the claim")
    assert consistency_flags(j) == [] and not nei_disagrees_with_checklist(j)


def _v2_responder(answers: list[dict]):
    """Evidence selects sentence 2 of doc 101; the judge returns ``answers`` in order."""
    base = scripted({"101": [2]}, {})
    queue = list(answers)

    def respond(p):
        if purpose_of(p) == "judge":
            return json.dumps(queue.pop(0))
        return base(p)

    return respond


def test_inconsistent_verdict_is_reasked_once_and_can_be_fixed(make_checker):
    checker = make_checker(_v2_responder([v2_answer("SUPPORT", [2], direction="opposite_to_claim"),
                                          v2_answer("CONTRADICT", [2])]))
    res = checker.check("Vitamin D decreases bone density", k=1, prompt_set="v2", judge_think=False)
    judge = [c for c in checker.transport.calls if purpose_of(c) == "judge"]
    assert len(judge) == 2 and "inconsistent" not in judge[0]["messages"][1]["content"]
    assert "opposite direction" in judge[1]["messages"][1]["content"]  # supervisor feedback names the problem
    assert res.verdict == "CONTRADICT"
    j = res.documents[0].judgement
    assert j.repaired and not j.downgraded and j.consistency_flags == ["before_repair:checklist_opposite"]
    assert any(e.data.get("flag") == "consistency_repair" for e in res.events)


def test_still_inconsistent_after_reask_is_downgraded_to_nei(make_checker):
    bad = v2_answer("SUPPORT", [2], direction="opposite_to_claim")
    checker = make_checker(_v2_responder([bad, bad]))
    res = checker.check("Vitamin D decreases bone density", k=1, prompt_set="v2")
    assert len([c for c in checker.transport.calls if purpose_of(c) == "judge"]) == 2  # bounded: one re-ask
    assert res.status == "ok" and res.verdict == "NEI"
    j = res.documents[0].judgement
    assert j.downgraded and j.consistency_flags == ["checklist_opposite"] and "inconsistent" in j.reason
    assert any(e.data.get("flag") == "downgraded_inconsistent" for e in res.events)


def test_citation_and_consistency_repairs_share_one_round(make_checker, corpus):
    base = scripted({"101": [2], "202": [1]}, {})
    queue = {"101": [v2_answer("SUPPORT", []), v2_answer("SUPPORT", [])],
             "202": [v2_answer("CONTRADICT", [1], direction="same_as_claim"),
                     v2_answer("CONTRADICT", [1], direction="same_as_claim")]}

    def respond(p):
        if purpose_of(p) == "judge":
            d = "101" if "Document 101" in p["messages"][1]["content"] or "document 101" in p["messages"][1]["content"] \
                else "202"
            return json.dumps(queue[d].pop(0))
        return base(p)

    checker = make_checker(respond)
    res = checker.check("Vitamin D increases bone density", k=2, prompt_set="v2")
    assert len([c for c in checker.transport.calls if purpose_of(c) == "judge"]) == 4  # 2 docs x (1 + 1 re-ask)
    assert res.verdict == "NEI" and all(d.judgement.downgraded for d in res.documents)
    flags = {e.data.get("flag") for e in res.events}
    assert {"citation_repair", "consistency_repair", "downgraded_no_citation", "downgraded_inconsistent"} <= flags


def test_v1_pipeline_ignores_consistency_rules(make_checker):
    """A v1 SUPPORT whose reason says 'contradicts' is left alone (v1 behaviour unchanged)."""
    def respond(p):
        if purpose_of(p) == "judge":
            return json.dumps({"verdict": "SUPPORT", "cited_sentence_ids": [2], "reason": "this contradicts the claim",
                               "confidence": 0.9})
        return scripted({"101": [2]}, {})(p)

    checker = make_checker(respond)
    res = checker.check("Vitamin D increases bone density", k=1)
    assert res.verdict == "SUPPORT" and len([c for c in checker.transport.calls if purpose_of(c) == "judge"]) == 1
    assert res.metadata["prompt_set"] == "v1" and "prompt_hashes_v2" not in res.metadata


# ---------------------------------------------------------------------------
# Configuration wiring
# ---------------------------------------------------------------------------


def test_experiments_define_v2_variants_and_suites(experiments):
    assert experiments.variant("full_v2").prompt_set == "v2" and experiments.variant("full_v2").judge_think
    assert not experiments.variant("full_v2_nothink").judge_think
    assert experiments.variant("oracle_v2").oracle and experiments.variant("oracle_v2").prompt_set == "v2"
    assert experiments.suites["final_v2"].variants == ["full_v2_nothink", "oracle_v2", "full_v2"]
    assert experiments.variant("full").prompt_set == "v1" and not experiments.variant("full").judge_think
    assert experiments.suites["final_v2"].split == "final" and "full_v2" in experiments.suites["final_v2"].variants
    assert experiments.suites["dev"].split == "dev_tune"
    # the frozen v1 suite is unchanged
    assert experiments.suites["final"].variants == ["judge_only", "retrieve_judge_single_prompt", "oracle", "full",
                                                   "evidence_off", "rerank_off", "full_k1", "full_k5"]


def test_v2_options_are_rejected_for_non_pipeline_strategies():
    with pytest.raises(ValueError):
        VariantSpec(name="x", strategy="judge_only", prompt_set="v2")


def test_check_variant_full_stays_v1_even_when_product_default_is_v2(cfg, corpus, experiments):
    from claim_checker.runtime import ClaimChecker
    from tests.conftest import FakeRetriever

    v2cfg = cfg.model_copy(deep=True)
    v2cfg.product.prompt_set, v2cfg.product.judge_think = "v2", True
    t = FakeTransport(scripted({"101": [2]}, {"101": ("SUPPORT", [2])}))
    checker = ClaimChecker(v2cfg, experiments=experiments, corpus=corpus, retriever=FakeRetriever(corpus),
                           llm=OllamaClient(v2cfg.model, cache_dir=None, transport=t),
                           token_counter=lambda s: len(s.split()))
    product = checker.check("Vitamin D increases bone density", k=1)  # product default -> v2 + thinking
    judge = [c for c in t.calls if purpose_of(c) == "judge"]
    assert product.metadata["prompt_set"] == "v2" and is_v2_judge(judge[-1]) and judge[-1]["think"] is True
    v1 = checker.check_variant("Vitamin D increases bone density", "full")
    judge = [c for c in t.calls if purpose_of(c) == "judge"]
    assert v1.metadata["prompt_set"] == "v1" and not is_v2_judge(judge[-1]) and judge[-1]["think"] is False
    explicit = checker.check("Vitamin D increases bone density", k=1, prompt_set="v1")
    assert explicit.metadata["prompt_set"] == "v1"


def test_default_config_loads_with_v2_sections():
    cfg = load_config()
    assert cfg.product.prompt_set in ("v1", "v2")
    assert cfg.judge_v2.templates_dir == "v2" and cfg.judge_v2.think_budget_tokens >= 64
    # adding the v2 sections does not change the v1 configuration fingerprint inputs
    assert "judge_v2" not in cfg.model_dump()["judge"]


# ---------------------------------------------------------------------------
# dev_tune split
# ---------------------------------------------------------------------------


def _gold(i, label, conflict=False):
    return ClaimGold(claim_id=str(i), claim=f"claim {i}", split="train", cited_doc_ids=["1"], reference_doc_ids=["1"],
                     evidence_by_doc={}, label_by_doc={}, label=None if conflict else label, conflict=conflict)


def test_dev_tune_split_is_stratified_excludes_smoke_and_is_deterministic():
    labels = ["SUPPORT"] * 40 + ["CONTRADICT"] * 20 + ["NEI"] * 40
    gold = {str(i): _gold(i, lab) for i, lab in enumerate(labels)}
    gold["999"] = _gold(999, None, conflict=True)
    development = {"claim_ids": list(gold), "ids_sha256": "x", "source_fingerprint": "fp"}
    smoke = {"claim_ids": ["0", "40", "60"]}
    a = build_dev_tune_split(development, smoke, gold, size=20, seed=42)
    b = build_dev_tune_split(development, smoke, gold, size=20, seed=42)
    assert a == b and a["size"] == 20 and a["source_split"] == "train"
    assert a["class_counts"] == {"CONTRADICT": 4, "NEI": 8, "SUPPORT": 8}
    assert not {"0", "40", "60", "999"} & set(a["claim_ids"])
    assert a["pool_size"] == 97 and a["excluded_smoke_claims"] == ["0", "40", "60"]
    assert build_dev_tune_split(development, smoke, gold, size=20, seed=7)["claim_ids"] != a["claim_ids"]


def test_freeze_dev_tune_split_writes_once(cfg):
    labels = ["SUPPORT"] * 10 + ["NEI"] * 10
    golds = [_gold(i, lab) for i, lab in enumerate(labels)]
    write_jsonl(cfg.path("processed_dir") / "claim_gold_train.jsonl", golds)
    write_json(cfg.path("splits_dir") / "development.json", {"name": "development",
                                                             "claim_ids": [g.claim_id for g in golds]})
    write_json(cfg.path("splits_dir") / "smoke.json", {"name": "smoke", "claim_ids": ["0"]})
    first = freeze_dev_tune_split(cfg, size=6)
    assert first["status"] == "written" and load_split(cfg, "dev_tune")["size"] == 6
    assert freeze_dev_tune_split(cfg, size=6)["status"] == "unchanged"
    from claim_checker.errors import FrozenSplitError

    with pytest.raises(FrozenSplitError):
        freeze_dev_tune_split(cfg, size=8)


def test_repository_dev_tune_split_is_disjoint_from_final_and_smoke():
    cfg = load_config()
    try:
        dev = load_split(cfg, "dev_tune")
    except Exception:
        pytest.skip("dev_tune split not created in this checkout")
    smoke, development = load_split(cfg, "smoke"), load_split(cfg, "development")
    assert set(dev["claim_ids"]) <= set(development["claim_ids"])
    assert not set(dev["claim_ids"]) & set(smoke["claim_ids"])
    assert dev["source_split"] == "train"  # final claims come from the validation split


# ---------------------------------------------------------------------------
# Runner identity
# ---------------------------------------------------------------------------


@pytest.fixture
def v2_eval_setup(cfg, corpus):
    processed = cfg.path("processed_dir")
    write_jsonl(processed / "corpus.jsonl", corpus.values())
    golds = [
        ClaimGold(claim_id="1", claim="Vitamin D increases bone density", split="validation", cited_doc_ids=["101"],
                  reference_doc_ids=["101"], evidence_by_doc={"101": [2]}, label_by_doc={"101": "SUPPORT"},
                  label="SUPPORT"),
        ClaimGold(claim_id="2", claim="Vitamin D changes density in older women", split="validation",
                  cited_doc_ids=["202"], reference_doc_ids=["202"], evidence_by_doc={"202": [1]},
                  label_by_doc={"202": "CONTRADICT"}, label="CONTRADICT"),
    ]
    write_jsonl(processed / "claim_gold_validation.jsonl", golds)
    write_json(cfg.path("splits_dir") / "final.json", {"name": "final", "claim_ids": ["1", "2"]})
    return golds


def test_run_identity_adds_v2_fields_only_for_v2_runs(cfg, experiments, make_checker, v2_eval_setup):
    checker = make_checker(scripted({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("CONTRADICT", [1])}))
    v1_dir = run_evaluation(cfg, experiments, checker, suite="final", variants=["oracle"], run_id="v1run",
                            log=lambda m: None)
    v1_ident = json.loads((v1_dir / "manifest.json").read_text(encoding="utf-8"))["identity"]
    assert "prompt_hashes_v2" not in v1_ident and v1_ident["prompt_hashes"] == V1_HASHES
    v2_dir = run_evaluation(cfg, experiments, checker, suite="final_v2", variants=["oracle_v2"], run_id="v2run",
                            log=lambda m: None)
    man = json.loads((v2_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["identity"]["prompt_hashes_v2"]) == {"v2/judge"} and man["identity"]["config_fingerprint_v2"]
    assert man["variants"]["oracle_v2"]["prompt_set"] == "v2"
    preds = load_predictions(v2_dir)["oracle_v2"]
    assert [p["verdict"] for p in preds] == ["SUPPORT", "CONTRADICT"]


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def test_mcnemar_exact_by_hand():
    from claim_checker.evaluation.compare import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 1) == 1.0
    assert abs(mcnemar_exact(0, 5) - 2 / 32) < 1e-12  # 2 * 0.5**5
    assert abs(mcnemar_exact(2, 8) - 2 * (1 + 10 + 45) / 1024) < 1e-12


def _write_run(runs_dir, run_id, variant, rows):
    d = runs_dir / run_id
    d.mkdir(parents=True)
    write_json(d / "manifest.json", {"run_id": run_id, "variants": {variant: {}}})
    write_jsonl(d / "predictions.jsonl", [{"variant": variant, "claim_id": c, "gold_label": g, "verdict": p,
                                           "status": "ok", "timings": {"total": 1.0}} for c, g, p in rows])
    write_jsonl(d / "pair_predictions.jsonl", [{"variant": variant, "claim_id": c, "doc_id": "d", "predicted": p,
                                                "reason": "this contradicts the claim" if c == "1" else "fine",
                                                "skipped_model": False} for c, g, p in rows])


def test_compare_runs_pairs_claims_and_counts_false_support(tmp_path):
    from claim_checker.evaluation.compare import compare_runs, render_comparison_md

    gold = [("1", "CONTRADICT"), ("2", "NEI"), ("3", "SUPPORT"), ("4", "SUPPORT")]
    _write_run(tmp_path, "ra", "full", [(c, g, "SUPPORT") for c, g in gold])
    _write_run(tmp_path, "rb", "full_v2", [("1", "CONTRADICT", "CONTRADICT"), ("2", "NEI", "NEI"),
                                            ("3", "SUPPORT", "SUPPORT"), ("4", "SUPPORT", "NEI")])
    out = compare_runs(tmp_path, ("ra", "full"), ("rb", "full_v2"), n_boot=200)
    assert out["n_claims"] == 4 and out["a"]["accuracy"] == 0.5 and out["b"]["accuracy"] == 0.75
    assert out["a"]["errors"]["false_support"] == 2 and out["b"]["errors"]["false_support"] == 0
    assert out["paired"]["only_a_correct"] == 1 and out["paired"]["only_b_correct"] == 2
    assert out["a"]["judge_consistency"]["support_reason_negates_support"] == 1
    lo, hi = out["bootstrap"]["accuracy_diff_b_minus_a"]
    assert lo <= 0.25 <= hi
    assert "McNemar" in render_comparison_md(out)


def test_explanation_names_the_consistency_downgrade(make_checker):
    bad = v2_answer("SUPPORT", [2], direction="opposite_to_claim")
    res = make_checker(_v2_responder([bad, bad])).check("Vitamin D decreases bone density", k=1, prompt_set="v2")
    assert "contradicted the judge's own checklist" in res.explanation
    assert "without a valid cited sentence" not in res.explanation
