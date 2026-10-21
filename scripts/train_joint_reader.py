"""Fresh joint label/rationale reader: TRAIN selection, audit, conditional refit/confirmation."""
import argparse
import gc
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.agents.evidence import build_quotes
from claim_checker.agents.explainer import explain
from claim_checker.config import load_config, sha256_text
from claim_checker.data.joint_context import supervised_context
from claim_checker.data.splits import load_split, normalize_claim_text
from claim_checker.data.store import (
    append_jsonl,
    load_corpus,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from claim_checker.evaluation.metrics import operational_metrics, quote_validity, verdict_metrics
from claim_checker.evaluation.runner import prediction_record
from claim_checker.evaluation.training_provenance import assert_unfitted_population
from claim_checker.models.joint_verifier import JointReader, JointVerifier, rationale_loss
from claim_checker.pipeline.aggregate import aggregate
from claim_checker.pipeline.supervisor import validate_claim_text
from claim_checker.retrieval.index import check_index
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import ClaimResult, DocumentReport, ErrorInfo
from scripts.train_verifier import evidence_safe_rows, training_rows

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/improvement/joint_nli_protocol_v1.json"
SELECT = ROOT / "artifacts/models/joint_nli_select_v1"
FINAL = ROOT / "artifacts/models/joint_nli_final_v1"
BASE = ROOT / "artifacts/models/nli_large_base"


def prepare():
    cfg = load_config()
    parent = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    grouped = read_json(parent)
    split = grouped["folds"][0]
    gold = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    assert all(normalize_claim_text(gold[c].claim) not in forbidden for key in ("train", "stop", "audit") for c in split[key])
    prior = ROOT / "artifacts/models/regularized_grouped_v1_fold0"
    plan = {
        "objective": "Sameoriginal300claims: accuracy>0.95 AND three-class MacroF1>0.95",
        "architecture": "Shared generic DeBERTa encoder, joint token-rationale head, zero-initialized evidence projection into NLI classification",
        "inspiration": "https://aclanthology.org/2022.findings-naacl.6/; not a reproduction and no published SciFact-trained weights",
        "fit_ids": split["train"], "stop_ids": split["stop"], "audit_ids": split["audit"],
        "eligible_ids": sorted(grouped["group_by_claim"]), "source_protocol_sha256": sha256_file(parent),
        "base": (BASE / "source_revision.txt").read_text(), "base_weights_sha256": sha256_file(BASE / "model.safetensors"),
        "prior_fit_rows_sha256": sha256_file(prior / "training_pairs.jsonl"),
        "prior_stop_rows_sha256": sha256_file(prior / "stop_pairs.jsonl"),
        "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
        "validation_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                          [Path(__file__), ROOT / "src/claim_checker/models/joint_verifier.py",
                           ROOT / "src/claim_checker/data/joint_context.py", ROOT / "scripts/train_verifier.py"]},
        "recipe": {"seed": 42, "maximum_epochs": 4, "patience": 1, "min_delta": .002,
                   "encoder_layers": 4, "encoder_lr": 5e-6, "new_heads_lr": 1e-4,
                   "dropout": .2, "weight_decay": .05, "physical_batch": 2, "effective_batch": 16,
                   "max_length": 512, "auxiliary_weight": .5, "positive_token_weight": 3., "weak_negative_token_weight": .25,
                   "weak_nonreference_class_weight": .25, "class_balance": "sqrt inverse frequency on gradient-fit rows only",
                   "optimizer": "AdamW8bit", "precision": "FP32weights,BF16CUDAautocast", "gradient_checkpointing": True},
        "selection": "Minimum mean unweighted reference-pair CE +0.5 weighted source-token BCE; no audit or validation selection",
        "inference": {"k": 3, "retrieval_gate": -2, "class_threshold": .5, "quote_threshold": .5,
                      "max_quotes_per_window": 2, "window_overlap": 2, "include_title": True},
        "audit_gate": "Both accuracy and MacroF1 >=same-reader-oldbaseline+0.02; zeroerrors",
        "audit_baseline": {"accuracy": 118 / 163, "macro_f1": .725584736251403},
        "refit": "Only after audit passes, fresh generic base on807TRAIN for selectedepochs with original4epochLRhorizon",
        "limitations": "Old163TRAINaudit is reused, not blind. Stop loss uses reference-supplied contexts. Windowed512token model, not full-document Longformer. Nonreference/uncited evidence negatives are weak labels.",
        "new_training_recipes": 1, "new_threshold_searches": 0,
    }
    if PLAN.exists():
        assert read_json(PLAN) == plan, "Frozen experiment changed; use a new version"
    else:
        write_json(PLAN, plan)
    return plan


def data(plan, final=False):
    cfg = load_config()
    corpus, gold = load_corpus(cfg.path("processed_dir")), load_gold(cfg.path("processed_dir"), "train")
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    assert tokenizer.is_fast and tokenizer.padding_side == "right"
    prior = ROOT / "artifacts/models/regularized_grouped_v1_fold0"
    if final:
        rows, _ = training_rows(cfg, 1, exclude_development=False)
        rows, omitted = evidence_safe_rows(cfg, rows, tokenizer, 512, True)
        assert not omitted
        buckets = {"fit": rows, "stop": []}
    else:
        buckets = {name: read_jsonl(prior / filename) for name, filename in
                   (("fit", "training_pairs.jsonl"), ("stop", "stop_pairs.jsonl"))}
        assert {r["claim_id"] for r in buckets["fit"]} == set(plan["fit_ids"])
        assert {r["claim_id"] for r in buckets["stop"]} == set(plan["stop_ids"])
        papers = {name: {r["doc_id"] for r in rows} for name, rows in buckets.items()}
        audit_papers = {d for c in plan["audit_ids"] for d in gold[c].reference_doc_ids}
        assert not papers["fit"] & papers["stop"] and not (papers["fit"] | papers["stop"]) & audit_papers
    encoded = {}
    for role, rows in buckets.items():
        encoded[role] = []
        for row in rows:
            g = gold[row["claim_id"]]
            assert g.split == "train" and row["claim"] == g.claim
            if row["view"] == "hard_negative":
                assert row["doc_id"] not in g.reference_doc_ids and row["label"] == "NEI"
            else:
                assert row["label"] == g.label_by_doc.get(row["doc_id"], "NEI")
            encoded[role].append(supervised_context(tokenizer, row, corpus[row["doc_id"]], g))
    folder = ROOT / ("artifacts/improvement/joint_final_data_v1" if final else "artifacts/improvement/joint_selection_data_v1")
    folder.mkdir(exist_ok=True)
    for role, rows in encoded.items():
        write_jsonl(folder / f"{role}_tokens.jsonl", rows)
    write_json(folder / "integrity.json", {"plan_sha256": sha256_file(PLAN),
               "rows": {k: len(v) for k, v in encoded.items()}, "claims": {k: len({r['claim_id'] for r in v}) for k, v in encoded.items()},
               "all_positive_premises_fit_without_truncation": True, "source_reconstruction_mismatches": 0,
               "fit_stop_source_paper_overlap": None if final else 0, "audit_source_paper_overlap": None if final else 0,
               "truncated_NEI_contexts": "Permitted: target is nondecisive; no positive evidence is silently removed"})
    return tokenizer, encoded


def collator(tokenizer, device):
    def collate(rows):
        length = max(len(r["inputs"]["input_ids"]) for r in rows)
        tokens = [{k: v for k, v in r["inputs"].items() if k not in ("source_mask", "rationale_targets")} for r in rows]
        inputs = tokenizer.pad(tokens, return_tensors="pt")
        inputs["source_mask"] = torch.tensor([r["inputs"]["source_mask"] + [0] * (length - len(r["inputs"]["input_ids"])) for r in rows])
        targets = torch.tensor([r["inputs"]["rationale_targets"] + [-100] * (length - len(r["inputs"]["input_ids"])) for r in rows], device=device)
        return {k: v.to(device) for k, v in inputs.items()}, targets
    return collate


def new_model(recipe):
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    config.hidden_dropout_prob = config.pooler_dropout = config.cls_dropout = recipe["dropout"]
    base = AutoModelForSequenceClassification.from_pretrained(BASE, config=config, local_files_only=True, dtype=torch.float32)
    for p in base.parameters():
        p.requires_grad = False
    for block in [*list(base.deberta.encoder.layer)[-recipe["encoder_layers"]:], base.pooler, base.classifier]:
        for p in block.parameters():
            p.requires_grad = True
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return JointReader(base, recipe["dropout"]).cuda()


def train(plan, final=False, pilot=False):
    import bitsandbytes as bnb

    recipe = plan["recipe"]
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    tokenizer, buckets = data(plan, final)
    rows, stop = buckets["fit"], buckets["stop"]
    folder = FINAL if final else SELECT
    identity = {"plan_sha256": sha256_file(PLAN), "final": final}
    if not pilot:
        if folder.exists():
            assert read_json(folder / "identity.json") == identity
            assert (folder / "selection.json").exists(), "Interrupted training; retain artifacts and use a new version"
            return
        folder.mkdir(parents=True)
        write_json(folder / "identity.json", identity)
        gold = load_gold(load_config().path("processed_dir"), "train")
        used = sorted({r["claim_id"] for r in rows + stop})
        write_json(folder / "training_manifest.json", {
            "fitted_claim_ids": used,
            "fitted_claim_text_sha256": sorted({sha256_text(normalize_claim_text(gold[c].claim)) for c in used}),
            "gradient_fit_claim_ids": sorted({r["claim_id"] for r in rows}),
            "stopping_claim_ids": sorted({r["claim_id"] for r in stop}),
            "plan_sha256": sha256_file(PLAN), "source_split": "train", "final_fit": final,
            "note": "Strict overlap guard includes gradient and stopping populations; validation texts excluded"})
    model = new_model(recipe)
    mapping = {"SUPPORT": model.backbone.config.label2id["entailment"], "CONTRADICT": model.backbone.config.label2id["contradiction"],
               "NEI": model.backbone.config.label2id["neutral"]}
    encoder_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [*model.rationale_head.parameters(), *model.evidence_projection.parameters()]
    optimizer = bnb.optim.AdamW8bit([{"params": encoder_params, "lr": recipe["encoder_lr"]},
                                   {"params": head_params, "lr": recipe["new_heads_lr"]}], weight_decay=recipe["weight_decay"])
    collate = collator(tokenizer, "cuda")
    if pilot:
        longest = sorted(rows, key=lambda r: len(r["inputs"]["input_ids"]), reverse=True)[:2]
        inputs, targets = collate(longest)
        model.eval()
        with torch.inference_mode():
            actual, _ = model(**inputs)
            expected = model.backbone(**{k: v for k, v in inputs.items() if k != "source_mask"}).logits
        assert torch.allclose(actual, expected, atol=1e-5), "Zero projection must preserve generic NLI predictions"
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, rationale = model(**inputs)
            labels = torch.tensor([mapping[r["label"]] for r in longest], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits.float(), labels) + .5 * rationale_loss(rationale, targets).mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        write_json(ROOT / "artifacts/improvement/joint_hardware_pilot_v1.json", {
            "longest_tokens": [len(r["inputs"]["input_ids"]) for r in longest], "loss_finite": True,
            "zero_projection_base_parity": True, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "model_saved": False, "purpose": "Hardware/backprop compatibility only, not benchmark accuracy"})
        print("Joint hardware/backprop pilot passed; no checkpoint saved.", flush=True)
        return
    counts = Counter(r["label"] for r in rows)
    class_weights = torch.zeros(3, device="cuda")
    for label, index in mapping.items():
        class_weights[index] = math.sqrt(len(rows) / (3 * counts[label]))
    accumulation = recipe["effective_batch"] // recipe["physical_batch"]
    loader = torch.utils.data.DataLoader(rows, batch_size=recipe["physical_batch"], shuffle=True, collate_fn=lambda x: x)
    horizon = math.ceil(len(loader) / accumulation) * recipe["maximum_epochs"]
    warmup = max(1, horizon // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0., (horizon - step) / (horizon - warmup))))
    epochs = read_json(SELECT / "selection.json")["epoch"] if final else recipe["maximum_epochs"]
    best, best_epoch, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, current in enumerate(loader):
            inputs, targets = collate(current)
            labels = torch.tensor([mapping[r["label"]] for r in current], device="cuda")
            weak = torch.tensor([r["class_weight"] for r in current], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, rationale = model(**inputs)
                ce = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none") * class_weights[labels] * weak
                per_row = ce + recipe["auxiliary_weight"] * rationale_loss(rationale, targets)
                loss = per_row.sum() / min(recipe["effective_batch"], len(rows) - step // accumulation * recipe["effective_batch"])
            assert torch.isfinite(loss)
            loss.backward()
            losses.extend(per_row.detach().tolist())
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_([*encoder_params, *head_params], 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        stop_losses = []
        for start in range(0, len(stop), 2):
            current = stop[start:start + 2]
            inputs, targets = collate(current)
            labels = torch.tensor([mapping[r["label"]] for r in current], device="cuda")
            with torch.inference_mode():
                logits, rationale = model(**inputs)
                values = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none") + .5 * rationale_loss(rationale, targets)
            stop_losses.extend(values.tolist())
        stop_loss = float(np.mean(stop_losses)) if stop else None
        assert stop_loss is None or math.isfinite(stop_loss), "Stopping loss must be finite"
        improved = final or stop_loss < best - recipe["min_delta"]
        if improved:
            best, best_epoch, stale = stop_loss, epoch, 0
            # Final refit saves only its last epoch; selection retains improved checkpoints.
            if not final or epoch == epochs:
                model.save(folder / f"epoch{epoch}", tokenizer)
        else:
            stale += 1
        report = {"epoch": epoch, "weighted_train_loss": float(np.mean(losses)), "unweighted_stop_joint_loss": stop_loss,
                  "improved": improved, "elapsed_seconds": time.perf_counter() - started, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
        append_jsonl(folder / "learning_curve.jsonl", report)
        print(report, flush=True)
        if not final and stale >= recipe["patience"]:
            break
    write_json(folder / "selection.json", {"epoch": best_epoch, "stop_loss": best, "validation_used": False,
               "weights_sha256": sha256_file(folder / f"epoch{best_epoch}/model.safetensors"),
               "heads_sha256": sha256_file(folder / f"epoch{best_epoch}/joint_heads.safetensors")})
    del model, optimizer, encoder_params, head_params, scheduler
    gc.collect()
    torch.cuda.empty_cache()


class JointChecker:
    def __init__(self, folder):
        self.cfg = load_config(overrides={"product": {"judge_backend": "nli", "evidence_mode": "full_abstract"},
                                          "verifier": {"model_dir": str(folder), "device": "cuda"}})
        self.resources = ClaimChecker(self.cfg)
        self.verifier = JointVerifier(folder, "cuda")
        self.identity = {"model_dir": str(folder), "weights_sha256": sha256_file(folder / "model.safetensors"),
                         "heads_sha256": sha256_file(folder / "joint_heads.safetensors"), "labels": self.verifier.labels}

    @property
    def corpus(self):
        return self.resources.corpus

    def check(self, claim, claim_id=None, oracle_doc_ids=None):
        start = time.perf_counter()
        metadata = {"strategy": "pipeline", "judge_backend": "nli", "verifier": self.identity,
                    "options": {"oracle_doc_ids": oracle_doc_ids}}
        try:
            text = validate_claim_text(claim, self.cfg.runtime.max_claim_chars, self.cfg.runtime.max_claim_tokens, self.resources.count_claim_tokens)
            if oracle_doc_ids is None:
                hits = self.resources._retrieve(text, "hybrid", True, 10).hits
                metadata["retrieval_hits"] = [h.model_dump(mode="json") for h in hits]
                chosen = [(h.doc_id, h.rank, h.score) for h in hits[:3] if h.score >= -2]
            else:
                if not oracle_doc_ids:
                    raise ValueError("Oracle requires explicit source IDs")
                chosen = [(did, rank, None) for rank, did in enumerate(dict.fromkeys(oracle_doc_ids), 1)]
            reports, judgements, quotes = [], [], []
            for did, rank, score in chosen:
                doc = self.corpus[did]
                judgement = self.verifier.judge(text, doc, .5)
                judgements.append(judgement)
                reports.append(DocumentReport(doc_id=did, title=doc.title, rank=rank, retrieval_score=score,
                                              retrieval_method="oracle" if oracle_doc_ids is not None else "hybrid",
                                              judgement=judgement, sentences=doc.sentences, source=doc.source))
                quotes.extend(build_quotes(doc, judgement.cited_sentence_ids, judgement.verdict))
            verdict, conflict = aggregate(judgements)
            return ClaimResult(status="ok", claim=text, claim_id=claim_id, verdict=verdict, conflict=conflict, documents=reports,
                               evidence=quotes, explanation=explain(verdict, conflict, judgements, {d.doc_id: d.title for d in reports},
                                                                  context="supplied" if oracle_doc_ids is not None else "retrieved"),
                               timings={"total": time.perf_counter() - start}, metadata=metadata)
        except Exception as exc:
            return ClaimResult(status="error", claim=claim, claim_id=claim_id, error=ErrorInfo(error_type=type(exc).__name__, message=str(exc)),
                               timings={"total": time.perf_counter() - start}, metadata=metadata)


def evaluate(plan, final=False, oracle=False):
    selection = read_json((FINAL if final else SELECT) / "selection.json")
    model = (FINAL if final else SELECT) / f"epoch{selection['epoch']}"
    assert sha256_file(model / "model.safetensors") == selection["weights_sha256"]
    assert sha256_file(model / "joint_heads.safetensors") == selection["heads_sha256"]
    checker = JointChecker(model)
    gold = load_gold(checker.cfg.path("processed_dir"), "validation" if final else "train")
    ids = (load_split(checker.cfg, "final")["claim_ids"] if oracle else sorted(gold)) if final else plan["audit_ids"]
    assert_unfitted_population(read_json(model.parent / "training_manifest.json"), [gold[c] for c in ids])
    stage = "oracle100" if oracle else "validation300" if final else "audit163"
    folder = ROOT / f"artifacts/improvement/joint_{stage}_v1"
    folder.mkdir(exist_ok=True)
    identity = {"plan_sha256": sha256_file(PLAN), "selected_weights": selection, "ids": ids,
                "index": None if oracle else check_index(checker.cfg)["fingerprint"],
                "core_code_sha256": {str(p.relative_to(ROOT / "src/claim_checker")): sha256_file(p)
                                     for p in sorted((ROOT / "src/claim_checker").rglob("*.py"))}}
    if (folder / "identity.json").exists():
        assert read_json(folder / "identity.json") == identity
    else:
        write_json(folder / "identity.json", identity)
    path = folder / "predictions.jsonl"
    saved = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in saved}
    assert len(done) == len(saved) and done <= set(ids)
    for index, cid in enumerate(ids, 1):
        if cid not in done:
            g = gold[cid]
            result = checker.check(g.claim, cid, g.reference_doc_ids if oracle else None)
            append_jsonl(path, prediction_record("joint_nli_v1", g, result))
        if index % 20 == 0:
            print(f"Joint {stage}: {index}/{len(ids)}", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == len(ids) and {r["claim_id"] for r in rows} == set(ids)
    metric = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else None for r in rows])
    operations = operational_metrics(rows)
    report = {"verdict": metric, "operations": operations, "quotes": quote_validity(rows, checker.corpus),
              "population": "Reused163TRAINaudit, notblind" if not final else "Separateoracle100diagnostic" if oracle else "Originalreusedvalidation300, notblind",
              "target_met": final and not oracle and metric["accuracy"] > .95 and metric["macro_f1"] > .95}
    if not final:
        report["audit_gate_passed"] = (metric["accuracy"] >= plan["audit_baseline"]["accuracy"] + .02
                                       and metric["macro_f1"] >= plan["audit_baseline"]["macro_f1"] + .02
                                       and operations["errors"] == 0)
    write_json(folder / "metrics.json", report)
    print(f"Joint {stage}: {metric['accuracy']:.4%}/{metric['macro_f1']:.6f}, errors={operations['errors']}", flush=True)
    del checker
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "pilot", "fit", "audit", "refit", "confirm", "all"), required=True)
    args = parser.parse_args()
    plan = prepare()
    if args.stage == "pilot":
        train(plan, pilot=True)
        return
    if args.stage in ("fit", "all"):
        train(plan)
    if args.stage in ("audit", "all"):
        if not evaluate(plan)["audit_gate_passed"]:
            print("Joint reader rejected on TRAIN audit; no refit or validation300.", flush=True)
            return
    if args.stage in ("refit", "confirm", "all"):
        assert read_json(ROOT / "artifacts/improvement/joint_audit163_v1/metrics.json")["audit_gate_passed"]
    if args.stage in ("refit", "all"):
        train(plan, final=True)
    if args.stage in ("confirm", "all"):
        evaluate(plan, final=True)
        evaluate(plan, final=True, oracle=True)


if __name__ == "__main__":
    main()
