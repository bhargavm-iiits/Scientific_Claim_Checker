"""Record actual fitted populations for evaluation leakage checks and provenance."""
from pathlib import Path

from claim_checker.config import sha256_text
from claim_checker.data.splits import normalize_claim_text
from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = ROOT / "artifacts/improvement/scifact95_roberta_protocol_v1.json"
    for name in ("roberta_scifact_select_v1", "roberta_scifact_final_v1"):
        folder = ROOT / "artifacts/models" / name
        identity = read_json(folder / "identity.json")
        assert identity["protocol_sha256"] == sha256_file(protocol)
        assert identity["script_sha256"] == sha256_file(ROOT / "scripts/train_roberta_verifier.py")
        rows = read_jsonl(folder / "training_pairs.jsonl")
        audit = read_json(folder / "data_audit.json")
        assert len(rows) == audit["fit_pairs"]
        ids = sorted({r["claim_id"] for r in rows})
        assert len(ids) == audit["fit_claims"]
        manifest = {"source_split": "train", "settings": {"final_fit": identity["final_fit"]},
                    "fitted_claim_ids": ids,
                    "fitted_claim_text_sha256": sorted({sha256_text(normalize_claim_text(r["claim"])) for r in rows}),
                    "protocol_sha256": identity["protocol_sha256"], "base_weights_sha256": identity["base_weights_sha256"],
                    "training_pairs_sha256": sha256_file(folder / "training_pairs.jsonl"),
                    "data_audit_sha256": sha256_file(folder / "data_audit.json"), "training_identity_sha256": sha256_file(folder / "identity.json"),
                    "note": "Selected-model TRAIN overlap is population-specific; final refit includes all eligible807TRAINclaims. Neither may be scored as a held-out result on fitted texts."}
        path = folder / "training_manifest.json"
        if path.exists():
            assert read_json(path) == manifest, "Preserve frozen training provenance"
        else:
            write_json(path, manifest)
        print(f"Registered {name}: {len(ids)} actual fitted claims", flush=True)


if __name__ == "__main__":
    main()
