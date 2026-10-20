import json

import pytest

from claim_checker.data.store import write_json, write_jsonl
from claim_checker.errors import RunMismatchError
from claim_checker.evaluation.analysis import analyse_run, failure_stage, load_predictions
from claim_checker.evaluation.metrics import (
    evidence_metrics,
    latency_stats,
    operational_metrics,
    recall_at_k,
    retrieval_metrics,
    verdict_metrics,
)
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.schemas import ClaimGold
from tests.conftest import scripted


def test_verdict_metrics_count_errors_as_wrong():
    gold = ["SUPPORT", "SUPPORT", "CONTRADICT", "NEI"]
    pred = ["SUPPORT", None, "NEI", "NEI"]
    m = verdict_metrics(gold, pred)
    assert m["n_claims"] == 4 and m["covered"] == 3 and m["coverage"] == 0.75
    assert m["accuracy"] == 0.5 and m["accuracy_on_covered"] == pytest.approx(2 / 3)
    # SUPPORT: P=1 R=0.5 F1=2/3; CONTRADICT: F1=0; NEI: P=0.5 R=1 F1=2/3
    assert m["per_class"]["SUPPORT"]["f1"] == pytest.approx(2 / 3)
    assert m["per_class"]["NEI"]["f1"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)
    assert m["confusion"]["matrix"]["SUPPORT"]["ERROR"] == 1


def test_retrieval_and_evidence_metrics_by_hand():
    assert recall_at_k(["a", "b", "c"], ["c", "z"], 3) == 0.5
    assert recall_at_k(["a"], [], 3) is None
    recs = [
        {"retrieved_doc_ids": ["a", "b", "c", "d"], "reference_doc_ids": ["c"], "evidence_doc_ids": ["c"],
         "examined_doc_ids": ["a", "b", "c"]},
        {"retrieved_doc_ids": ["x", "y"], "reference_doc_ids": ["q"], "evidence_doc_ids": [], "examined_doc_ids": ["x"]},
    ]
    m = retrieval_metrics(recs, ks=(1, 3))
    assert m["recall@1"] == 0.0 and m["recall@3"] == 0.5 and m["recall@3_n_claims"] == 2
    assert m["evidence_doc_recall@3"] == 1.0 and m["evidence_doc_recall@3_n_claims"] == 1
    ev = evidence_metrics([
        {"selected": {"c": [1, 2], "a": [0]}, "evidence_by_doc": {"c": [2, 3]}},
        {"selected": {}, "evidence_by_doc": {"q": [0]}},
        {"selected": None, "evidence_by_doc": {"z": [5]}},  # not evaluated (e.g. judge-only)
    ])
    assert ev["true_positive_sentences"] == 1 and ev["predicted_sentences_in_annotated_docs"] == 2
    assert ev["precision"] == 0.5 and ev["reference_sentences"] == 3 and ev["recall"] == pytest.approx(1 / 3)
    assert ev["predicted_sentences_in_unannotated_docs"] == 1 and ev["claims"] == 2


def test_latency_and_operational_metrics():
    lat = latency_stats([1.0, 2.0, 3.0, None])
    assert lat["n"] == 3 and lat["mean"] == 2.0 and lat["median"] == 2.0
    op = operational_metrics([{"status": "ok", "conflict": True}, {"status": "error", "error": {"error_type": "model_timeout"}}])
    assert op["operational_failure_rate"] == 0.5 and op["conflict_rate_of_ok"] == 1.0
    assert op["errors_by_type"] == {"model_timeout": 1}


def test_relevance_gate_rejecting_all_documents_keeps_citation_recall_denominator(make_checker):
    from claim_checker.evaluation.runner import prediction_record

    checker = make_checker()
    checker.cfg.verifier.min_retrieval_score = 100
    result = checker.check("A scientific claim.", k=3, evidence_mode="full_abstract", judge_backend="nli")
    assert result.status == "ok" and result.verdict == "NEI" and not result.documents
    gold = ClaimGold(claim_id="1", claim=result.claim, split="train", cited_doc_ids=["101"],
                     reference_doc_ids=["101"], evidence_by_doc={"101": [2]},
                     label_by_doc={"101": "SUPPORT"}, label="SUPPORT")
    record = prediction_record("nli_gated", gold, result)
    assert record["cited"] == {} and record["selected"] is None
    for saved in (record, {**record, "cited": None}):
        metrics = evidence_metrics([saved], "cited")
        assert metrics["claims_in_recall_denominator"] == 1
        assert metrics["reference_sentences"] == 1 and metrics["recall"] == 0
        assert evidence_metrics([saved], "selected")["claims_in_recall_denominator"] == 0
    assert not checker.transport.calls


def test_failure_stage_heuristic():
    base = {"status": "ok", "gold_label": "CONTRADICT", "evidence_doc_ids": ["d1"], "evidence_by_doc": {"d1": [3]},
            "examined_doc_ids": ["d2"], "selected": {"d2": [0]}}
    assert failure_stage({**base, "status": "error"}) == "operational"
    assert failure_stage(base) == "retrieval miss"
    assert failure_stage({**base, "examined_doc_ids": ["d1"], "selected": {"d1": [0]}}) == "evidence miss"
    assert failure_stage({**base, "examined_doc_ids": ["d1"], "selected": {"d1": [3]}}) == "judgement error"


@pytest.fixture
def eval_setup(cfg, corpus):
    processed = cfg.path("processed_dir")
    write_jsonl(processed / "corpus.jsonl", corpus.values())
    golds = [
        ClaimGold(claim_id="1", claim="Vitamin D increases bone density", split="validation", cited_doc_ids=["101"],
                  reference_doc_ids=["101"], evidence_by_doc={"101": [2]}, label_by_doc={"101": "SUPPORT"}, label="SUPPORT"),
        ClaimGold(claim_id="2", claim="Vitamin D changes density in older women", split="validation",
                  cited_doc_ids=["202"], reference_doc_ids=["202"], evidence_by_doc={"202": [1]},
                  label_by_doc={"202": "CONTRADICT"}, label="CONTRADICT"),
        ClaimGold(claim_id="3", claim="Stars form in clouds of cheese", split="validation", cited_doc_ids=["404"],
                  reference_doc_ids=["404"], evidence_by_doc={}, label_by_doc={}, label="NEI"),
    ]
    write_jsonl(processed / "claim_gold_validation.jsonl", golds)
    write_json(cfg.path("splits_dir") / "final.json", {"name": "final", "claim_ids": ["1", "2", "3"]})
    return golds


def test_runner_checkpoints_resumes_and_analyses(cfg, experiments, make_checker, eval_setup, corpus):
    checker = make_checker(scripted({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("SUPPORT", [1])}))
    run_dir = run_evaluation(cfg, experiments, checker, suite="final", variants=["oracle", "judge_only"],
                             run_id="t1", limit=2, log=lambda m: None)
    preds = load_predictions(run_dir)
    assert {v: len(r) for v, r in preds.items()} == {"oracle": 2, "judge_only": 2}
    assert all(r["claim_id"] in ("1", "2") for rs in preds.values() for r in rs)
    calls = len(checker.transport.calls)
    run_evaluation(cfg, experiments, checker, suite="final", variants=["oracle", "judge_only"],
                   run_id="t1", limit=2, log=lambda m: None)
    assert len(checker.transport.calls) == calls  # resume skipped completed claims
    out = analyse_run(run_dir, corpus)
    oracle = out["metrics"]["oracle"]
    assert oracle["verdict"]["n_claims"] == 2 and oracle["verdict"]["accuracy"] == 0.5
    assert oracle["evidence_selected"]["precision"] == 1.0 and oracle["quotes"]["validity"] == 1.0
    for f in ("metrics.json", "results.csv", "confusion_matrices.json", "failure_cases.csv", "report.md"):
        assert (run_dir / f).exists()
    assert any(f["gold_label"] == "CONTRADICT" for f in out["failures"])
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["identity"]["population_size"] == 2


def test_resume_refused_when_prompts_change(cfg, experiments, make_checker, eval_setup, monkeypatch):
    checker = make_checker(scripted({}, {}))
    run_evaluation(cfg, experiments, checker, suite="final", variants=["judge_only"], run_id="t2", log=lambda m: None)
    monkeypatch.setattr(checker.prompts, "hashes", lambda: {"judge": "changed"})
    with pytest.raises(RunMismatchError):
        run_evaluation(cfg, experiments, checker, suite="final", variants=["judge_only"], run_id="t2", log=lambda m: None)


def test_operational_failures_stay_in_population(cfg, experiments, make_checker, eval_setup):
    def respond(p):
        raise_once = respond.n == 0
        respond.n += 1
        return "garbage" if raise_once or respond.n == 2 else json.dumps(
            {"verdict": "NEI", "reason": "r", "confidence": 0.1})
    respond.n = 0
    checker = make_checker(respond)
    run_dir = run_evaluation(cfg, experiments, checker, suite="final", variants=["judge_only"], run_id="t3",
                             log=lambda m: None)
    m = analyse_run(run_dir)["metrics"]["judge_only"]
    assert m["verdict"]["n_claims"] == 3 and m["verdict"]["covered"] == 2
    assert m["operational"]["errors"] == 1 and m["verdict"]["accuracy"] == pytest.approx(1 / 3)


def test_claim_with_several_abstracts_counts_once(cfg, experiments, make_checker, corpus):
    processed = cfg.path("processed_dir")
    write_jsonl(processed / "corpus.jsonl", corpus.values())
    g = ClaimGold(claim_id="9", claim="Vitamin D increases bone density", split="validation",
                  cited_doc_ids=["101", "202"], reference_doc_ids=["101", "202"], evidence_by_doc={"101": [2]},
                  label_by_doc={"101": "SUPPORT"}, label="SUPPORT")
    write_jsonl(processed / "claim_gold_validation.jsonl", [g])
    write_json(cfg.path("splits_dir") / "final.json", {"name": "final", "claim_ids": ["9"]})
    checker = make_checker(scripted({"101": [2]}, {"101": ("SUPPORT", [2])}))
    run_dir = run_evaluation(cfg, experiments, checker, suite="final", variants=["oracle"], run_id="t4",
                             log=lambda m: None)
    pairs = [json.loads(line) for line in (run_dir / "pair_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pairs) == 2 and {p["doc_id"] for p in pairs} == {"101", "202"}
    m = analyse_run(run_dir, corpus)["metrics"]["oracle"]
    assert m["verdict"]["n_claims"] == 1 and m["verdict"]["accuracy"] == 1.0


def test_evidence_recall_keeps_operational_errors_in_denominator():
    ok = {"status": "ok", "selected": {"c": [1, 2]}, "evidence_by_doc": {"c": [2, 3]}}
    errored = {"status": "error", "selected": None, "evidence_by_doc": {"q": [0, 4], "r": [1], "s": []}}
    not_applicable = {"status": "ok", "selected": None, "evidence_by_doc": {"z": [5]}}
    ev = evidence_metrics([ok, errored, not_applicable])
    # recall: 1 TP over 2 (ok claim) + 3 (errored claim, zero predicted) reference sentences
    assert ev["recall"] == pytest.approx(1 / 5) and ev["recall_denominator"] == ev["reference_sentences"] == 5
    assert ev["errored_claims_included"] == 1 and ev["errored_annotated_pairs_included"] == 2
    assert ev["errored_reference_sentences_included"] == 3 and ev["claims_in_recall_denominator"] == 2
    # precision is unaffected by errors
    assert ev["precision"] == 0.5 and ev["precision_denominator"] == 2 and ev["claims"] == 1
    without_error = evidence_metrics([ok, not_applicable])
    assert without_error["precision"] == ev["precision"] and without_error["recall"] == 0.5
    assert without_error["errored_claims_included"] == 0


def test_errored_claims_reach_the_report_evidence_denominators(cfg, experiments, make_checker, eval_setup, corpus):
    calls = {"n": 0}
    base = scripted({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("CONTRADICT", [1])})

    def respond(p):  # claim 2 (doc 202) fails operationally: two invalid outputs from the evidence agent
        if "Document 202" in p["messages"][1]["content"] and "Evidence agent" in p["messages"][0]["content"]:
            calls["n"] += 1
            return "garbage"
        return base(p)

    run_dir = run_evaluation(cfg, experiments, make_checker(respond), suite="final", variants=["oracle"],
                             run_id="t5", log=lambda m: None)
    out = analyse_run(run_dir, corpus)
    ev = out["metrics"]["oracle"]["evidence_selected"]
    assert out["metrics"]["oracle"]["operational"]["errors"] == 1 and calls["n"] == 2
    assert ev["errored_claims_included"] == 1 and ev["reference_sentences"] == 2 and ev["recall"] == 0.5
    assert "including 1 with an operational error" in out["report"]


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_resume_drops_diagnostics_of_interrupted_claim(cfg, experiments, make_checker, eval_setup, monkeypatch):
    from claim_checker.evaluation.runner import EVENTS, PAIRS, PREDICTIONS, RETRIEVAL
    from claim_checker.retrieval import index

    monkeypatch.setattr(index, "check_index", lambda cfg: {"fingerprint": "fakefp"})  # fake retriever, no index

    checker = make_checker(scripted({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("CONTRADICT", [1])}))
    run_dir = run_evaluation(cfg, experiments, checker, suite="final", variants=["full"], run_id="t6", limit=None,
                             log=lambda m: None)
    full_bytes = {n: (run_dir / n).read_bytes() for n in (PREDICTIONS, PAIRS, RETRIEVAL, EVENTS)}
    # Simulate an interruption of claim 3: its diagnostics were appended, its prediction was not.
    preds = _rows(run_dir / PREDICTIONS)
    (run_dir / PREDICTIONS).write_text("".join(json.dumps(r) + "\n" for r in preds if r["claim_id"] != "3"),
                                       encoding="utf-8")
    logs: list[str] = []
    run_evaluation(cfg, experiments, checker, suite="final", variants=["full"], run_id="t6", log=logs.append)
    assert any("Removed" in m for m in logs)
    for name, per_claim in ((RETRIEVAL, 1), (EVENTS, 1), (PAIRS, 3)):  # k=3 pair rows per claim
        rows = _rows(run_dir / name)
        counts = {cid: sum(1 for r in rows if r["claim_id"] == cid) for cid in ("1", "2", "3")}
        assert counts == {"1": per_claim, "2": per_claim, "3": per_claim}, (name, counts)
    assert len(_rows(run_dir / PREDICTIONS)) == 3
    # Resuming a complete run must not touch any of its files.
    before = {n: (run_dir / n).read_bytes() for n in full_bytes}
    run_evaluation(cfg, experiments, checker, suite="final", variants=["full"], run_id="t6", log=lambda m: None)
    assert {n: (run_dir / n).read_bytes() for n in full_bytes} == before


def test_reconcile_removes_stale_duplicate_attempts():
    from claim_checker.evaluation.runner import _clean_diagnostic_rows

    # Older resume behaviour: claim b's first attempt was orphaned, then re-run immediately after.
    rows = [{"variant": "v", "claim_id": "a", "doc_id": "1"}, {"variant": "v", "claim_id": "a", "doc_id": "2"},
            {"variant": "v", "claim_id": "b", "doc_id": "1", "try": 1}, {"variant": "v", "claim_id": "b", "doc_id": "2", "try": 1},
            {"variant": "v", "claim_id": "b", "doc_id": "1", "try": 2}, {"variant": "v", "claim_id": "b", "doc_id": "2", "try": 2},
            {"variant": "v", "claim_id": "c", "doc_id": "1"}]
    kept = _clean_diagnostic_rows(rows, {("v", "a"), ("v", "b")})
    assert [(r["claim_id"], r.get("try")) for r in kept] == [("a", None), ("a", None), ("b", 2), ("b", 2)]
    one_per_claim = [{"variant": "v", "claim_id": "a", "n": 1}, {"variant": "v", "claim_id": "a", "n": 2}]
    assert _clean_diagnostic_rows(one_per_claim, {("v", "a")}) == [one_per_claim[1]]


def test_cache_metrics_by_hand():
    from claim_checker.evaluation.metrics import cache_metrics, subtimings_exceed_stage

    fresh = {"retrieve.bm25": 0.0125, "retrieve.dense": 0.0261, "retrieve.fuse": 0.0001,
             "retrieve.rerank": 1.0787, "retrieve": 1.1176}
    copied = {**fresh, "retrieve": 0.0032}  # legacy cache hit: sub-steps copied, stage itself ~0
    assert not subtimings_exceed_stage(fresh) and subtimings_exceed_stage(copied)
    assert not subtimings_exceed_stage({"retrieve": 0.5}) and not subtimings_exceed_stage({})
    assert not subtimings_exceed_stage({"retrieve.bm25": 0.3})  # no stage time: cannot tell
    recs = [
        {"retrieved_doc_ids": ["a"], "retrieval_cache_hit": False, "llm_calls": 3, "llm_cache_hits": 0,
         "timings": fresh},
        {"retrieved_doc_ids": ["a"], "retrieval_cache_hit": True, "llm_calls": 0, "llm_cache_hits": 3,
         "timings": {"retrieve": 0.001}},
        {"retrieved_doc_ids": ["a"], "retrieval_cache_hit": True, "llm_calls": 1, "llm_cache_hits": 1,
         "timings": {"retrieve": 0.001}},
        {"retrieved_doc_ids": ["a"], "llm_calls": None, "timings": copied},  # legacy, no flag
        {"retrieved_doc_ids": ["a"], "timings": fresh},  # legacy, no flag, looks fresh
        {"retrieved_doc_ids": None, "llm_calls": 2, "llm_cache_hits": 0, "timings": {}},  # oracle/judge-only
    ]
    c = cache_metrics(recs)
    assert c["llm"] == {"unit": "model requests", "model_calls": 6, "cache_hits": 4, "cache_hit_rate": 0.4}
    r = c["retrieval"]
    assert (r["claims_with_retrieval"], r["cache_hits"], r["cache_misses"], r["unknown"]) == (5, 2, 1, 2)
    assert r["cache_hit_rate"] == pytest.approx(2 / 3) and r["unknown_with_copied_subtimings"] == 1
    empty = cache_metrics([{"retrieved_doc_ids": None, "timings": {}}])
    assert empty["retrieval"]["cache_hit_rate"] is None and empty["llm"]["cache_hit_rate"] is None


def test_report_shows_retrieval_cache_hits_and_tolerates_legacy_records(cfg, experiments, make_checker,
                                                                        eval_setup, corpus, monkeypatch):
    from claim_checker.retrieval import index

    monkeypatch.setattr(index, "check_index", lambda cfg: {"fingerprint": "fakefp"})  # fake retriever, no index
    checker = make_checker(scripted({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("CONTRADICT", [1])}))
    run_dir = run_evaluation(cfg, experiments, checker, suite="final",
                             variants=["retrieve_judge_single_prompt", "full", "oracle"], run_id="tc",
                             log=lambda m: None)
    preds = _rows(run_dir / "predictions.jsonl")
    by = {(r["variant"], r["claim_id"]): r for r in preds}
    assert by[("retrieve_judge_single_prompt", "1")]["retrieval_cache_hit"] is False
    assert by[("full", "1")]["retrieval_cache_hit"] is True
    assert by[("full", "1")]["retrieve_original_compute_s"] == {"bm25": 0.001}
    assert by[("oracle", "1")]["retrieval_cache_hit"] is None
    out = analyse_run(run_dir, corpus)
    m = out["metrics"]
    assert m["full"]["cache"]["retrieval"]["cache_hits"] == 3
    assert m["full"]["cache"]["retrieval"]["cache_hit_rate"] == 1.0
    assert m["retrieve_judge_single_prompt"]["cache"]["retrieval"]["cache_misses"] == 3
    assert m["oracle"]["cache"]["retrieval"]["claims_with_retrieval"] == 0
    assert "retrieve.bm25" not in m["full"]["latency_stages"]  # copies are not reported as measured
    assert "## Cache usage" in out["report"] and "Caveat (retrieval cache)" in out["report"]
    assert "(full 3)" in out["report"]
    header = (run_dir / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "cache" not in header  # results.csv columns unchanged

    # Records saved before the flag existed: status unknown and the report must not crash; copied
    # sub-timings (sum > the stage's own wall time) are still recognised.
    legacy = []
    for r in preds:
        r = {k: v for k, v in r.items() if k not in ("retrieval_cache_hit", "retrieve_original_compute_s")}
        if r["variant"] == "full":
            r["timings"] = {**r["timings"], "retrieve.rerank": 1.0, "retrieve": 0.002}
        legacy.append(r)
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in legacy), encoding="utf-8")
    out = analyse_run(run_dir, corpus)
    rf = out["metrics"]["full"]["cache"]["retrieval"]
    assert (rf["cache_hits"], rf["cache_misses"], rf["unknown"], rf["unknown_with_copied_subtimings"]) == (0, 0, 3, 3)
    assert rf["cache_hit_rate"] is None
    assert "unknown for 6 claim(s)" in out["report"] and "Caveat (retrieval cache)" in out["report"]

    # Metrics computed by an older version (no "cache" entry) still render.
    from claim_checker.evaluation.analysis import render_cache_section

    assert render_cache_section({"x": {}})[-1].startswith("- LLM hit share")


# ---------------------------------------------------------------------------
# Review fixes: single-prompt downgrade count, run-id validation
# ---------------------------------------------------------------------------


def test_single_prompt_downgrades_are_counted_including_legacy_records(cfg, experiments, make_checker,
                                                                       eval_setup, corpus, monkeypatch):
    from claim_checker.retrieval import index

    monkeypatch.setattr(index, "check_index", lambda cfg: {"fingerprint": "fakefp"})

    def respond(p):  # every single-prompt answer cites a document that was not supplied
        return json.dumps({"verdict": "SUPPORT", "citations": [{"doc_id": "999", "sentence_id": 0}],
                           "reason": "r", "confidence": 0.5})

    run_dir = run_evaluation(cfg, experiments, make_checker(respond), suite="final",
                             variants=["retrieve_judge_single_prompt"], run_id="tdg", log=lambda m: None)
    preds = _rows(run_dir / "predictions.jsonl")
    assert all(r["downgraded"] and r["model_verdict"] == "SUPPORT" and r["verdict"] == "NEI" for r in preds)
    out = analyse_run(run_dir, corpus)
    m = out["metrics"]["retrieve_judge_single_prompt"]
    assert m["downgraded_claims"] == 3 and m["downgraded_claim_ids"] == ["1", "2", "3"]
    assert "3 claims with a downgraded judgement (decisive answer without a valid citation" in out["report"]

    # Records saved before the flag was propagated (frozen final_v1): downgraded=False, no
    # model_verdict. The count is recovered from the saved downgraded_no_citation events.
    legacy = [{k: v for k, v in r.items() if k != "model_verdict"} | {"downgraded": False} for r in preds]
    (run_dir / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in legacy), encoding="utf-8")
    m = analyse_run(run_dir, corpus)["metrics"]["retrieve_judge_single_prompt"]
    assert m["downgraded_claims"] == 3
    # ...and without an event log the legacy count is simply what the records say.
    (run_dir / "events.jsonl").unlink()
    assert analyse_run(run_dir, corpus)["metrics"]["retrieve_judge_single_prompt"]["downgraded_claims"] == 0


@pytest.mark.parametrize("bad", ["..", ".", "../outside", "a/b", r"a\b", r"C:\temp\x", "/abs", "", "x y",
                                 "run..id", "CON", "nul.txt", "ok;rm"])
def test_invalid_run_ids_are_rejected(cfg, bad):
    from claim_checker.errors import InputValidationError
    from claim_checker.evaluation.runner import run_directory, validate_run_id

    with pytest.raises(InputValidationError):
        validate_run_id(bad)
    with pytest.raises(InputValidationError):
        run_directory(cfg, bad)


@pytest.mark.parametrize("good", ["final_v1", "smoke-20261020-101500", "final.v2", "A1"])
def test_valid_run_ids_stay_inside_runs_dir(cfg, good):
    from claim_checker.evaluation.runner import run_directory

    assert run_directory(cfg, good) == cfg.path("runs_dir") / good


def test_run_evaluation_and_report_cli_refuse_path_traversal(cfg, experiments, make_checker, eval_setup, capsys):
    from claim_checker import cli
    from claim_checker.errors import InputValidationError

    with pytest.raises(InputValidationError):
        run_evaluation(cfg, experiments, make_checker(scripted({}, {})), suite="final", variants=["judge_only"],
                       run_id="../escape", log=lambda m: None)
    assert not (cfg.path("runs_dir").parent / "escape").exists()
    with pytest.raises(InputValidationError):
        cli.cmd_report(cfg, type("Args", (), {"run_id": r"..\..\x"})())


def test_report_cli_rejects_bad_run_id_with_exit_code_2(monkeypatch, cfg, capsys):
    from claim_checker import cli

    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    assert cli.main(["report", "--run-id", "../final_v1"]) == 2
    err = capsys.readouterr().err
    assert "invalid_input" in err and "Invalid run id" in err


# ---------------------------------------------------------------------------
# Latency benchmark (cold, reproducible)
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_setup(cfg, corpus):
    processed = cfg.path("processed_dir")
    write_jsonl(processed / "corpus.jsonl", corpus.values())
    golds = [ClaimGold(claim_id=str(i), claim=f"Vitamin D claim number {i}", split="train", cited_doc_ids=["101"],
                       reference_doc_ids=["101"], evidence_by_doc={"101": [2]}, label_by_doc={"101": "SUPPORT"},
                       label="SUPPORT") for i in range(1, 5)]
    write_jsonl(processed / "claim_gold_train.jsonl", golds)
    write_json(cfg.path("splits_dir") / "smoke.json", {"name": "smoke", "claim_ids": ["1", "2", "3", "4"]})
    return golds


def test_latency_benchmark_separates_warmup_and_checks_caches(cfg, make_checker, smoke_setup, tmp_path):
    from claim_checker.evaluation.latency import (
        render_latency_md,
        run_latency_benchmark,
        write_latency_outputs,
    )

    checker = make_checker(scripted({"101": [2], "202": [1]}, {"101": ("SUPPORT", [2]), "202": ("CONTRADICT", [1])}))
    out = run_latency_benchmark(cfg, checker, split="smoke", limit=3, k=3, no_cache=True, repeat=True,
                                command="python -m claim_checker latency --no-cache", log=lambda m: None)
    assert out["repeat"]["cache_check"]["retrieval_cache_hits"] == 3  # repeat pass reuses retrieval
    assert out["n_claims"] == 3 and out["claim_ids"] == ["1", "2", "3"] and out["errors"] == 0
    assert out["warmup_seconds"]["first_claim_id"] == "4"  # warm-up claim is not in the measured set
    assert out["warmup_seconds"]["total"] >= out["warmup_seconds"]["resources"]
    assert out["cache_check"] == {"llm_cache_hits_total": 0, "retrieval_cache_hits": 0,
                                  "llm_calls_total": out["cache_check"]["llm_calls_total"]}
    assert out["cache_check"]["llm_calls_total"] > 0
    assert all(c["retrieval_cache_hit"] is False for c in out["claims"])
    for stage in ("retrieve", "select_evidence", "judge", "total"):
        st = out["stages_seconds"][stage]
        assert st["n"] == 3 and st["mean"] is not None and st["p95"] >= st["median"] >= 0
    md = render_latency_md(out, "artifacts/latency/x.json")
    assert "| **total** | 3 |" in md and "LLM disk cache **off**" in md and "Repeat pass" in md
    cfg.root = str(tmp_path)  # write into the test's temp folder, not the project
    j, m = write_latency_outputs(cfg, out, "cold_test")
    assert j.exists() and m.exists() and j.parent == tmp_path / "artifacts" / "latency"


def test_latency_benchmark_refuses_cold_run_with_llm_cache_enabled(cfg, smoke_setup, corpus, experiments, tmp_path):
    from claim_checker.errors import ConfigError
    from claim_checker.evaluation.latency import run_latency_benchmark
    from claim_checker.models.ollama_client import OllamaClient
    from claim_checker.runtime import ClaimChecker
    from tests.conftest import FakeRetriever, FakeTransport

    llm = OllamaClient(cfg.model, cache_dir=tmp_path / "cache", transport=FakeTransport(scripted({}, {})))
    checker = ClaimChecker(cfg, experiments=experiments, corpus=corpus, retriever=FakeRetriever(corpus), llm=llm,
                           token_counter=lambda s: len(s.split()))
    with pytest.raises(ConfigError):
        run_latency_benchmark(cfg, checker, split="smoke", limit=2, no_cache=True, log=lambda m: None)
