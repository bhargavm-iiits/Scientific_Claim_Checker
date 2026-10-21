"""Fixed TRAIN-only viability screen after a slow synthetic runtime preflight."""
from pathlib import Path

from claim_checker.data.splits import stratified_sample
from claim_checker.data.store import load_gold, read_json, sha256_file, write_json
from claim_checker.models.ollama_client import OllamaClient
from scripts.run_reasoned_reader import (
    PLAN,
    ROOT,
    ReasonedChecker,
    configuration,
    evaluate,
    prepare,
)


def main():
    cfg = configuration()
    client = OllamaClient(cfg.model, ROOT / "artifacts/cache/reasoned_reader_v1")
    plan = prepare(cfg, client)
    groups = read_json(ROOT / "artifacts/improvement/grouped_training_protocol_v1.json")
    train = load_gold(cfg.path("processed_dir"), "train")
    claims, _ = stratified_sample([train[c] for c in groups["folds"][0]["stop"]], 20, 2026)
    assert len(claims) == 20 and not {g.claim_id for g in claims} & set(plan["audit_ids"])
    path = ROOT / "artifacts/improvement/reasoned_reader_screen_protocol_v1.json"
    frozen = {"parent_protocol_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
              "claim_ids": [g.claim_id for g in claims], "population": "20fixedTRAINstoppingclaims, notvalidation300",
              "selection": "Stratified sample preserving class proportions, seed2026; no inspected outcomes",
              "advance": "Both accuracy and three-class MacroF1>=0.80, zeroerrors; only then163audit",
              "rationale": "Synthetic preflight exhausted native thinking budget; runtime viability screen before long163audit",
              "preflight_sha256": sha256_file(ROOT / "artifacts/improvement/reasoned_reader_preflight.json")}
    if path.exists():
        assert read_json(path) == frozen
    else:
        write_json(path, frozen)
    # Evaluate's identity/plan argument is explicitly labeled as TRAIN development.
    checker = ReasonedChecker(cfg, client)
    report = evaluate(checker, claims, "reasoned_reader_screen20_audit_v1",
                      {**plan, "audit_population": frozen["population"]})
    passed = report["verdict"]["accuracy"] >= .80 and report["verdict"]["macro_f1"] >= .80 and report["operational"]["errors"] == 0
    write_json(ROOT / "artifacts/improvement/reasoned_reader_screen_decision_v1.json",
               {"screen_protocol_sha256": sha256_file(path), "passed": passed, "metrics": report,
                "validation300_used": False, "warning": "Small reusedTRAINscreen is not benchmark accuracy"})
    print(f"FixedTRAINscreen passed={passed}; no validation300score established.", flush=True)


if __name__ == "__main__":
    main()
