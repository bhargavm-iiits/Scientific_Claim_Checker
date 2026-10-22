"""Fit a small rationale classifier with audited TRAIN-only sentence supervision."""
import argparse
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.config import load_config
from claim_checker.data.splits import load_split, normalize_claim_text
from claim_checker.data.store import load_corpus, load_gold, sha256_file, write_json, write_jsonl
from claim_checker.models.evidence_classifier import candidate_context

ROOT = Path(__file__).resolve().parents[1]
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def build_rows(cfg, tokenizer, claim_ids, *, max_length=512, use_neighbors=True, input_order="query_first"):
    corpus, gold = load_corpus(cfg.path("processed_dir")), load_gold(cfg.path("processed_dir"), "train")
    rows, omissions = [], []
    for cid in claim_ids:
        item = gold[cid]
        assert not item.conflict
        for did in item.reference_doc_ids:
            doc = corpus[did]
            positives = set(item.evidence_by_doc.get(did, []))
            for sid in range(len(doc.sentences)):
                try:
                    premise, context_ids = candidate_context(tokenizer, item.claim, doc.title, doc.sentences, sid,
                                                             max_length=max_length, use_neighbors=use_neighbors, input_order=input_order)
                except ValueError:
                    omissions.append({"claim_id": cid, "doc_id": did, "sentence_id": sid, "target": int(sid in positives)})
                    continue
                rows.append({"claim_id": cid, "doc_id": did, "sentence_id": sid, "claim": item.claim,
                             "premise": premise, "context_sentence_ids": context_ids, "target": int(sid in positives)})
    return rows, omissions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="evidence_scifact_train_v1")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--schedule-epochs", type=int, default=6)
    parser.add_argument("--final-fit", action="store_true")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--input-order", choices=("query_first", "premise_first"), default="query_first")
    parser.add_argument("--no-neighbors", action="store_true")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.epochs <= args.schedule_epochs or min(args.batch_size, args.accumulate) < 1:
        parser.error("epochs must be positive and within the learning-rate horizon")
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = load_config()
    train, validation = load_gold(cfg.path("processed_dir"), "train"), load_gold(cfg.path("processed_dir"), "validation")
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    dev_ids = load_split(cfg, "dev_tune")["claim_ids"]
    if not args.final_fit:
        for split in ("dev_tune", "smoke"):
            forbidden.update(normalize_claim_text(train[cid].claim) for cid in load_split(cfg, split)["claim_ids"])
    eligible = [cid for cid, g in train.items() if not g.conflict and normalize_claim_text(g.claim) not in forbidden]
    base = args.base_dir or Path(snapshot_download("cross-encoder/ms-marco-MiniLM-L6-v2", revision=REVISION, local_files_only=True))
    if not base.is_absolute():
        base = ROOT / base
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    render_settings = {"use_neighbors": not args.no_neighbors, "input_order": args.input_order}
    rows, omissions = build_rows(cfg, tokenizer, eligible, **render_settings)
    folder = ROOT / "artifacts/models" / args.name
    if folder.exists():
        raise ValueError("Use a new training name; existing artifacts are preserved")
    folder.mkdir(parents=True)
    write_jsonl(folder / "training_sentences.jsonl", rows)
    assert all(normalize_claim_text(r["claim"]) not in forbidden for r in rows)
    corpus = load_corpus(cfg.path("processed_dir"))
    for row in rows:
        item, doc = train[row["claim_id"]], corpus[row["doc_id"]]
        assert row["target"] == int(row["sentence_id"] in item.evidence_by_doc.get(row["doc_id"], []))
        assert row["premise"] == candidate_context(tokenizer, item.claim, doc.title, doc.sentences, row["sentence_id"], **render_settings)[0]
    counts = Counter(row["target"] for row in rows)
    write_json(folder / "training_manifest.json", {
        "settings": {**vars(args), "base_dir": str(args.base_dir) if args.base_dir else None}, "seed": 42, "base": str(base),
        "revision": (base / "source_revision.txt").read_text() if args.base_dir else REVISION,
        "base_sha256": sha256_file(base / "model.safetensors"), "source_train_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
        "training_data_sha256": sha256_file(folder / "training_sentences.jsonl"), "eligible_claims": len(eligible),
        "rows": len(rows), "targets": dict(counts), "max_length": 512, "batch_size": args.batch_size,
        "accumulate": args.accumulate, "learning_rate": args.learning_rate, "render_settings": render_settings,
        "omissions": omissions, "script_sha256": sha256_file(Path(__file__)),
        "renderer_sha256": sha256_file(ROOT / "src/claim_checker/models/evidence_classifier.py"), "torch": torch.__version__,
    })
    write_json(folder / "training_integrity.json", {"claims": len({r["claim_id"] for r in rows}), "rows": len(rows),
               "excluded_text_overlaps": 0, "source_or_annotation_mismatches": 0, "final_fit": args.final_fit,
               "omissions": omissions, "training_data_sha256": sha256_file(folder / "training_sentences.jsonl")})
    def encode(row):
        return tokenizer(row["claim"], row["premise"]) if args.input_order == "query_first" else tokenizer(row["premise"], row["claim"])

    encoded = [encode(row) for row in rows]

    def collate(indices):
        batch = tokenizer.pad([encoded[i] for i in indices], return_tensors="pt")
        labels = torch.tensor([rows[i]["target"] for i in indices], dtype=torch.float32)
        return {k: v.cuda() for k, v in batch.items()}, labels.cuda()

    loader = DataLoader(list(range(len(rows))), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_rows, dev_encoded = [], []
    if not args.final_fit:
        dev_rows, dev_omissions = build_rows(cfg, tokenizer, dev_ids, **render_settings)
        assert not dev_omissions, "Development examples must not be silently omitted"
        dev_encoded = [encode(row) for row in dev_rows]
    model_config = AutoConfig.from_pretrained(base, local_files_only=True, num_labels=1,
                                              id2label={0: "rationale"}, label2id={"rationale": 0})
    model_config.rationale_use_neighbors = not args.no_neighbors
    model_config.rationale_input_order = args.input_order
    model = AutoModelForSequenceClassification.from_pretrained(base, config=model_config, local_files_only=True,
                                                              ignore_mismatched_sizes=True, dtype=torch.float32).cuda()
    if args.freeze_embeddings:
        for parameter in model.base_model.embeddings.parameters():
            parameter.requires_grad = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    horizon = math.ceil(len(loader) / args.accumulate) * args.schedule_epochs
    warmup = max(1, horizon // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0.0, (horizon - step) / max(1, horizon - warmup))))
    positive_weight = torch.tensor(counts[0] / counts[1], device="cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")
    started, checkpoints = time.perf_counter(), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, (batch, labels) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits.flatten()
                per_example = loss_fn(logits, labels)
                loss = per_example.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evidence-selection loss")
            effective = args.batch_size * args.accumulate
            group_start = ((step - 1) * args.batch_size) // effective * effective
            group_size = min(effective, len(rows) - group_start)
            (per_example.sum() / group_size).backward()
            if step % args.accumulate == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        target = folder / f"epoch{epoch}"
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        report = {"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "seconds": time.perf_counter() - started,
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
        if not args.final_fit:
            model.eval()
            probabilities = []
            for start in range(0, len(dev_encoded), 16):
                batch = tokenizer.pad(dev_encoded[start:start + 16], return_tensors="pt")
                with torch.inference_mode():
                    probabilities.extend(model(**{k: v.cuda() for k, v in batch.items()}).logits.float().sigmoid().flatten().tolist())
            precision, recall, f1, _ = precision_recall_fscore_support([row["target"] for row in dev_rows], [p >= 0.5 for p in probabilities], average="binary", zero_division=0)
            report.update(reference_dev_precision=precision, reference_dev_recall=recall, reference_dev_f1=f1)
        write_json(target / "training_report.json", report)
        checkpoints.append(report)
        print(report, flush=True)
    write_json(folder / "status.json", {"complete": True, "epochs": args.epochs})
    if not args.final_fit:
        checkpoints.sort(key=lambda c: (-c["reference_dev_f1"], c["epoch"]))
        write_json(folder / "selection.json", {"chosen": checkpoints[0], "checkpoints": checkpoints,
                   "purpose": "Reference-supplied rationale component selection; not claim accuracy", "validation_used": False})


if __name__ == "__main__":
    main()
