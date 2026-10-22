"""One fixed, training-free polarity-consistency experiment, gated on TRAIN only."""
import argparse
import gc
import math
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import normalize_claim_text, stratified_sample
from claim_checker.data.store import (
    append_jsonl,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from claim_checker.evaluation.metrics import operational_metrics, quote_validity, verdict_metrics
from claim_checker.evaluation.runner import prediction_record
from claim_checker.evaluation.training_provenance import assert_unfitted_population
from claim_checker.models.verifier import NLIVerifier
from claim_checker.runtime import ClaimChecker

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/improvement/polarity_consistency_protocol_v1.json"
SWAP = {"SUPPORT": "CONTRADICT", "CONTRADICT": "SUPPORT", "NEI": "NEI"}


def negate(claim):
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Logical negation requires a nonempty claim")
    return "It is not the case that " + claim.strip()


def consistent_scores(original, negated):
    """Equal-weight geometric pooling after swapping negated support/contradiction."""
    if set(original) != set(SWAP) or set(negated) != set(SWAP):
        raise ValueError("All three NLI probabilities are required")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*original.values(), *negated.values()]):
        raise ValueError("NLI probabilities must be finite and between zero and one")
    pooled = {label: math.sqrt(max(original[label], 1e-9) * max(negated[SWAP[label]], 1e-9)) for label in SWAP}
    total = sum(pooled.values())
    return {label: value / total for label, value in pooled.items()}


class ConsistentVerifier(NLIVerifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = False
        self.memo = {}

    def windows(self, claim, sentences, *, overlap=None, title=""):
        negated = negate(claim)
        longer = max((claim, negated), key=lambda t: len(self.tokenizer.encode(t, add_special_tokens=False)))
        return super().windows(longer, sentences, overlap=overlap, title=title)

    def ordinary(self, premises, hypotheses, batch_size):
        keys = list(zip(premises, hypotheses, strict=True))
        missing = list(dict.fromkeys(key for key in keys if key not in self.memo))
        if missing:
            values = super().score([p for p, _ in missing], [h for _, h in missing], batch_size=batch_size)
            self.memo.update(zip(missing, values, strict=True))
        return [self.memo[key] for key in keys]

    def score(self, premises, hypotheses, batch_size=4):
        original = self.ordinary(premises, hypotheses, batch_size)
        if not self.enabled:
            return original
        negated = self.ordinary(premises, [negate(h) for h in hypotheses], batch_size)
        return [consistent_scores(p, q) for p, q in zip(original, negated, strict=True)]


def prepare():
    cfg = load_config()
    source = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    grouped = read_json(source)
    train, validation = (load_gold(cfg.path("processed_dir"), split) for split in ("train", "validation"))
    screen, _ = stratified_sample([train[c] for c in grouped["folds"][0]["stop"]], 20, 2026)
    audit = grouped["folds"][0]["audit"]
    reader = "artifacts/models/regularized_grouped_v1_fold0/epoch4"
    final = "artifacts/models/nli_large_titlefamily_final_v1/epoch1"
    fit_rows = read_jsonl((ROOT / reader).parent / "training_pairs.jsonl")
    fitting = {r["claim_id"] for r in fit_rows}
    used = fitting | set(grouped["folds"][0]["stop"])
    assert not used & set(audit)
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    assert all(normalize_claim_text(train[c].claim) not in forbidden for c in used | set(audit))
    plan = {
        "objective": "Originalvalidation300 accuracy>0.95 AND three-class MacroF1>0.95",
        "algorithm": "Fixed equal geometric mean of p(label|P,H) and p(swappedlabel|P,notH), normalized",
        "negation": "It is not the case that <original claim>", "probability_floor": 1e-9,
        "new_training": False, "new_threshold_searches": 0,
        "reader": reader, "reader_sha256": sha256_file(ROOT / reader / "model.safetensors"),
        "final_reader": final, "final_reader_sha256": sha256_file(ROOT / final / "model.safetensors"),
        "model_metadata_sha256": {name: {filename: sha256_file(ROOT / path / filename)
                                         for filename in ("config.json", "tokenizer.json")}
                                  for name, path in (("audit_reader", reader), ("final_reader", final))},
        "reader_training_sha256": sha256_file((ROOT / reader).parent / "training_pairs.jsonl"),
        "screen_ids": [g.claim_id for g in screen], "audit_ids": audit,
        "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
        "validation_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
        "source_protocol_sha256": sha256_file(source), "script_sha256": sha256_file(Path(__file__)),
        "inference": {"k": 3, "gate": -2, "support_threshold": .5, "contradict_threshold": .9,
                      "include_title": True, "window_overlap": 2},
        "control": "Same selected weights and identical windows budgeted for the longer original/negated claim; no pooling",
        "screen_gate": "Candidate accuracy and MacroF1 not below paired control; zeroerrors",
        "audit_gate": "Candidate must improve both metrics >=0.02 over paired control, zeroerrors",
        "reader_transfer": "Only if audit passes, apply unchanged pooling to existing stronger807TRAINreader on300; audit is not an estimate of final combination",
        "limitations": "ReusedTRAINscreen/audit and reusedvalidation; not a blind test. Logical-negation equivalence is a model diagnostic, not a proof of scientific truth.",
    }
    if PLAN.exists():
        assert read_json(PLAN) == plan, "Frozen recipe changed; use a new version"
    else:
        write_json(PLAN, plan)
    return plan


def evaluate(plan, ids, stage, *, device="cuda", final=False):
    from claim_checker.retrieval.index import check_index

    model = plan["final_reader"] if final else plan["reader"]
    sha = plan["final_reader_sha256"] if final else plan["reader_sha256"]
    assert sha256_file(ROOT / model / "model.safetensors") == sha
    cfg = load_config(overrides={"product": {"judge_backend": "nli", "evidence_mode": "full_abstract", "prompt_set": "v1"},
                                "verifier": {"model_dir": model, "device": device, "include_title": True, "window_overlap": 2,
                                             "min_retrieval_score": -2, "support_threshold": .5, "contradict_threshold": .9},
                                "retrieval": {"pipeline_k": 3}})
    gold = load_gold(cfg.path("processed_dir"), "validation" if final else "train")
    claims = [gold[c] for c in ids]
    if not final:
        trained = read_jsonl((ROOT / model).parent / "training_pairs.jsonl")
        assert_unfitted_population({"fitted_claim_ids": list({r["claim_id"] for r in trained})}, claims)
    else:
        assert_unfitted_population(read_json((ROOT / model).parent / "training_manifest.json"), claims)
    folder = ROOT / f"artifacts/improvement/polarity_{stage}_v1"
    folder.mkdir(exist_ok=True)
    identity = {"plan_sha256": sha256_file(PLAN), "ids": ids, "weights_sha256": sha, "device": device,
                "configuration": cfg.model_dump(mode="json"),
                "index_fingerprint": check_index(cfg)["fingerprint"],
                "core_code_sha256": {str(p.relative_to(ROOT / "src/claim_checker")): sha256_file(p)
                                     for p in sorted((ROOT / "src/claim_checker").rglob("*.py"))}}
    identity_path = folder / "identity.json"
    if identity_path.exists():
        assert read_json(identity_path) == identity
    else:
        write_json(identity_path, identity)
    path = folder / "paired_predictions.jsonl"
    saved = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in saved}
    assert len(done) == len(saved) and done <= set(ids)
    checker = ClaimChecker(cfg)
    verifier = ConsistentVerifier(ROOT / model, device=device, include_title=True, window_overlap=2)
    checker._verifier = verifier
    checker._verifier_identity = {"model_dir": model, "weights_sha256": sha, "labels": verifier.labels}
    for index, claim in enumerate(claims, 1):
        if claim.claim_id in done:
            continue
        verifier.memo.clear()
        verifier.enabled = False
        control = checker.check(claim.claim, claim_id=claim.claim_id)
        verifier.enabled = True
        candidate = checker.check(claim.claim, claim_id=claim.claim_id)
        append_jsonl(path, {"claim_id": claim.claim_id, "control": prediction_record("common_window_control", claim, control),
                            "candidate": prediction_record("polarity_consistency_v1", claim, candidate)})
        if index % 10 == 0:
            print(f"Polarity {stage}: {index}/{len(claims)} complete", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == len(claims) and {r["claim_id"] for r in rows} == set(ids)
    report = {"claims": len(claims), "population": "Originalreusedvalidation300" if final else "ReusedTRAINdevelopment, notblind",
              "reference": "Controlled comparator uses same weights and longer-hypothesis window budget; original saved best300 remains separate"}
    for role in ("control", "candidate"):
        records = [r[role] for r in rows]
        report[role] = {"verdict": verdict_metrics([r["gold_label"] for r in records],
                                                  [r["verdict"] if r["status"] == "ok" else None for r in records]),
                        "operations": operational_metrics(records), "quotes": quote_validity(records, checker.corpus)}
    control, candidate = (report[role]["verdict"] for role in ("control", "candidate"))
    delta = .02 if stage == "audit" else 0
    report["gate_passed"] = (candidate["accuracy"] >= control["accuracy"] + delta
                             and candidate["macro_f1"] >= control["macro_f1"] + delta
                             and report["candidate"]["operations"]["errors"] == 0
                             and report["control"]["operations"]["errors"] == 0)
    report["target_met"] = final and candidate["accuracy"] > .95 and candidate["macro_f1"] > .95
    write_json(folder / "metrics.json", report)
    print(f"Polarity {stage}: control={control['accuracy']:.4%}/{control['macro_f1']:.6f}; candidate={candidate['accuracy']:.4%}/{candidate['macro_f1']:.6f}; gate={report['gate_passed']}", flush=True)
    del verifier, checker
    gc.collect()
    import torch

    if device == "cuda":
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "audit", "confirm", "all"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    plan = prepare()
    if args.stage in ("screen", "all"):
        if not evaluate(plan, plan["screen_ids"], "screen", device=args.device)["gate_passed"]:
            print("Polarity screen failed; no audit or validation run.", flush=True)
            return
    if args.stage in ("audit", "all"):
        assert read_json(ROOT / "artifacts/improvement/polarity_screen_v1/metrics.json")["gate_passed"]
        if not evaluate(plan, plan["audit_ids"], "audit", device=args.device)["gate_passed"]:
            print("Polarity audit failed; no validation run.", flush=True)
            return
    if args.stage in ("confirm", "all"):
        assert read_json(ROOT / "artifacts/improvement/polarity_audit_v1/metrics.json")["gate_passed"]
        gold = load_gold(load_config().path("processed_dir"), "validation")
        assert len(gold) == 300
        evaluate(plan, sorted(gold), "validation300", device=args.device, final=True)


if __name__ == "__main__":
    main()
