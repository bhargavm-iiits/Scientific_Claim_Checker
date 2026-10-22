"""Development-only soft-voting/retrieval-gate diagnostic for two frozen verifiers."""
import argparse
import itertools
from pathlib import Path

from analyze_verifier import decide

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import verdict_metrics

ROOT = Path(__file__).resolve().parents[1]


def contexts(row):
    return {(c["doc_id"], c["rank"], tuple(c["sentence_ids"])): c for c in row["contexts"] if c["view"] == "abstract"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs=2, type=Path, default=[ROOT / "artifacts/improvement/nli_scifact_train_v2_epoch3_dev120",
                                                               ROOT / "artifacts/improvement/nli_large_scifact_train_v1_epoch2_dev120"])
    parser.add_argument("--ks", nargs="+", type=int, choices=(1, 3, 5), default=[3])
    parser.add_argument("--name", default="nli_two_model_ensemble_diagnostic")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a file basename")
    folders = args.folders
    models = [{r["claim_id"]: r for r in read_jsonl(folder / "predictions.jsonl")} for folder in folders]
    ids = load_split(load_config(), "dev_tune")["claim_ids"]
    if any(set(model) != set(ids) for model in models):
        raise ValueError("Both complete development populations are required")
    if any(models[0][cid]["gold"] != models[1][cid]["gold"] for cid in ids):
        raise ValueError("Development labels must agree across both score artifacts")
    retrieval = {r["claim_id"]: {h["doc_id"]: h["score"] for h in r["hits"]}
                 for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    aligned = {}
    for cid in ids:
        first, second = contexts(models[0][cid]), contexts(models[1][cid])
        if first.keys() != second.keys():
            raise ValueError(f"Window alignment mismatch for {cid}; rescore canonical windows, do not drop cases")
        aligned[cid] = [(first[key], second[key]) for key in first]
    results = []
    for k, weight, gate, support, contradict in itertools.product(args.ks, (0, 0.25, 0.5, 0.75, 1),
            (None, -2, 0, 2), (0.5, 0.7, 0.9, 0.98), (0.5, 0.7, 0.9, 0.98)):
        predictions = []
        for cid in ids:
            mixed = []
            for first, second in aligned[cid]:
                if gate is not None and retrieval[cid][first["doc_id"]] < gate:
                    continue
                scores = {label: (1 - weight) * first["scores"][label] + weight * second["scores"][label]
                          for label in first["scores"]}
                mixed.append({**first, "scores": scores})
            predictions.append(decide({"contexts": mixed}, view="abstract", k=k,
                                      threshold={"SUPPORT": support, "CONTRADICT": contradict}))
        metrics = verdict_metrics([models[0][cid]["gold"] for cid in ids], predictions)
        results.append({"second_weight": weight, "k": k, "retrieval_gate": gate, "support_threshold": support,
                        "contradict_threshold": contradict, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    results.sort(key=lambda r: (-r["macro_f1"], -r["accuracy"], r["retrieval_gate"] is not None,
                                r["second_weight"] not in (0, 1)))
    report = {"population": "dev_tune120", "purpose": "Development-only ensemble diagnostic; no validation inputs",
              "source_folders": [str(folder) for folder in folders],
              "source_prediction_sha256": [sha256_file(folder / "predictions.jsonl") for folder in folders],
              "script_sha256": sha256_file(Path(__file__)),
              "all_windows_aligned": True, "candidates": results}
    write_json(ROOT / f"artifacts/improvement/{args.name}.json", report)
    print(results[:8])


if __name__ == "__main__":
    main()
