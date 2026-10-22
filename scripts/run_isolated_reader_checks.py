"""Repeat fixed TRAIN screens in isolation after the recorded RAM-pressure incident.

Original errors and interrupted CPU artifacts are preserved. No prompt, sample,
weight, pooling rule or threshold changes are made. Use separate processes/envs
for native Ollama and GPU NLI, never concurrent model jobs.
"""
import argparse
import gc
import platform
from importlib.metadata import version
from pathlib import Path

from claim_checker.data.store import load_gold, read_json, read_jsonl, sha256_file, write_json
from claim_checker.models.ollama_client import OllamaClient
from scripts import run_polarity_consistency as polarity
from scripts import run_reasoned_reader as native

ROOT = Path(__file__).resolve().parents[1]


def freeze(stage):
    plan = {"stage": stage, "script_sha256": sha256_file(Path(__file__)),
            "native_plan_sha256": sha256_file(native.PLAN),
            "native_screen_plan_sha256": sha256_file(ROOT / "artifacts/improvement/reasoned_reader_screen_protocol_v1.json"),
            "polarity_plan_sha256": sha256_file(polarity.PLAN),
            "reason": "Original native run experienced RAM allocation failure during concurrent CPU-model experiment; preserve all original errors and repeat fixed screens without competing model jobs.",
            "changes_to_claims_prompts_weights_thresholds": False,
            "cache": "Same deterministic native response cache; cache hits are recorded, not new cold latency measurements",
            "runtime": {"python": platform.python_version(),
                        **{package: version(package) for package in ("torch", "transformers", "sentence-transformers", "ollama")}}}
    path = ROOT / f"artifacts/improvement/isolated_{stage}_protocol_v1.json"
    if path.exists():
        assert read_json(path) == plan
    else:
        write_json(path, plan)


def native_check():
    cfg = native.configuration()
    client = OllamaClient(cfg.model, ROOT / "artifacts/cache/reasoned_reader_v1")
    plan = native.prepare(cfg, client)
    ids = read_json(ROOT / "artifacts/improvement/reasoned_reader_screen_protocol_v1.json")["claim_ids"]
    gold = load_gold(cfg.path("processed_dir"), "train")
    checker = native.ReasonedChecker(cfg, client)
    report = native.evaluate(checker, [gold[c] for c in ids], "reasoned_reader_isolated20_audit_v1",
                             {**plan, "audit_population": "Same20fixedTRAINclaims, isolated repeat; notvalidation300"})
    rows = read_jsonl(ROOT / "artifacts/improvement/reasoned_reader_isolated20_audit_v1/predictions.jsonl")
    passed = report["verdict"]["accuracy"] >= .80 and report["verdict"]["macro_f1"] >= .80 and report["operational"]["errors"] == 0
    write_json(ROOT / "artifacts/improvement/reasoned_reader_isolated_decision_v1.json",
               {"passed": passed, "metrics": report, "cached_claims": sum(bool((r.get("generation") or {}).get("cached")) for r in rows),
                "all_original_errors_preserved": True, "validation300_used": False})
    print(f"Isolated fixed20TRAINscreen passed={passed}", flush=True)
    if passed:
        report = native.evaluate(checker, [gold[c] for c in plan["audit_ids"]], "reasoned_reader_isolated163_audit_v1", plan)
        passed = report["verdict"]["accuracy"] >= .80 and report["verdict"]["macro_f1"] >= .80 and report["operational"]["errors"] == 0
        if passed:
            validation = load_gold(cfg.path("processed_dir"), "validation")
            assert len(validation) == 300
            final = native.evaluate(checker, [validation[c] for c in sorted(validation)], "reasoned_reader_isolated_validation300_v1", plan)
            write_json(ROOT / "artifacts/improvement/final_isolated_reasoned_reader_v1.json",
                       {**final, "target_met": final["verdict"]["accuracy"] > .95 and final["verdict"]["macro_f1"] > .95})
    # Unload only this experiment's model before another model job starts.
    client.transport.client.generate(model=cfg.model.name, keep_alive=0)
    del checker, client
    gc.collect()


def polarity_check():
    plan = polarity.prepare()
    report = polarity.evaluate(plan, plan["screen_ids"], "screen_isolated_gpu", device="cuda")
    if not report["gate_passed"]:
        print("Isolated polarity screen rejected; no163audit or validation300.", flush=True)
        return
    report = polarity.evaluate(plan, plan["audit_ids"], "audit", device="cuda")
    if report["gate_passed"]:
        validation = load_gold(polarity.load_config().path("processed_dir"), "validation")
        assert len(validation) == 300
        polarity.evaluate(plan, sorted(validation), "validation300", device="cuda", final=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("native", "polarity"), required=True)
    args = parser.parse_args()
    freeze(args.stage)
    if args.stage == "native":
        native_check()
    else:
        polarity_check()


if __name__ == "__main__":
    main()
