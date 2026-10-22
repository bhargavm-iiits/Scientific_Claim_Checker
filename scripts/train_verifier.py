"""Fine-tune the last NLI layers on SciFact TRAIN annotations only.

Selection runs exclude validation, development-tuning and smoke claim texts.
The separately marked final refit includes internal development after selection,
but still excludes every public validation text. Evaluation annotations never
enter the optimizer.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.config import load_config
from claim_checker.data.splits import load_split, normalize_claim_text
from claim_checker.data.store import (
    append_jsonl,
    load_corpus,
    load_gold,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from claim_checker.data.verifier_context import format_premise, positive_window
from claim_checker.retrieval.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]


def globally_weighted_losses(logits, labels, class_weights):
    """Per-example weights that do not cancel for a physical batch of one."""
    return torch.nn.functional.cross_entropy(logits, labels, reduction="none") * class_weights[labels]


def training_rows(cfg, hard_negatives=0, *, exclude_development=True):
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    excluded = {normalize_claim_text(g.claim) for g in validation.values()}
    if exclude_development:
        for name in ("dev_tune", "smoke"):
            excluded.update(normalize_claim_text(train[c].claim) for c in load_split(cfg, name)["claim_ids"])
    corpus = load_corpus(cfg.path("processed_dir"))
    rows = []
    excluded_ids = []
    for gold in train.values():
        if gold.conflict or normalize_claim_text(gold.claim) in excluded:
            excluded_ids.append(gold.claim_id)
            continue
        assert gold.split == "train"
        for doc_id in gold.reference_doc_ids:
            doc = corpus[doc_id]
            label = gold.label_by_doc.get(doc_id, "NEI")
            ids = gold.evidence_by_doc.get(doc_id, [])
            if label != "NEI" and not ids:
                continue
            rows.append({"claim_id": gold.claim_id, "claim": gold.claim, "doc_id": doc_id,
                         "premise": " ".join(doc.sentences), "label": label, "view": "abstract"})
            # Short rationale pairs retain negation/direction when a long abstract is truncated.
            if ids:
                rows.append({"claim_id": gold.claim_id, "claim": gold.claim, "doc_id": doc_id,
                             "premise": " ".join(doc.sentences[n] for n in ids),
                             "label": label, "view": "rationale"})
    if hard_negatives:
        index = BM25Index(list(corpus), [d.text for d in corpus.values()])
        for gold in train.values():
            if gold.claim_id in excluded_ids:
                continue
            added = 0
            for doc_id, _ in index.search(gold.claim, top_n=20):
                if doc_id in gold.reference_doc_ids:
                    continue
                rows.append({"claim_id": gold.claim_id, "claim": gold.claim, "doc_id": doc_id,
                             "premise": " ".join(corpus[doc_id].sentences), "label": "NEI",
                             "view": "hard_negative"})
                added += 1
                if added == hard_negatives:
                    break
    return rows, excluded_ids


def evidence_safe_rows(cfg, rows, tokenizer, max_length, include_title):
    corpus = load_corpus(cfg.path("processed_dir"))
    raw = {str(r["id"]): r for r in read_jsonl(cfg.path("raw_dir") / "scifact/claims_train.jsonl")}
    result, omitted, cache = [], [], {}
    for row in rows:
        doc = corpus[row["doc_id"]]
        if row["label"] == "NEI":
            result.append({**row, "premise": format_premise(doc.title, row["premise"], include_title),
                           "title": doc.title, "include_title": include_title})
            continue
        alternatives = [r["sentences"] for r in raw[row["claim_id"]]["evidence"][row["doc_id"]]
                        if r["label"] == row["label"] and r["sentences"]]
        key = (row["claim_id"], row["doc_id"])
        if key not in cache:
            cache[key] = positive_window(tokenizer, row["claim"], doc.title, doc.sentences, alternatives,
                                          max_length=max_length, include_title=include_title)
        window = cache[key]
        if window is None:
            omitted.append({"claim_id": row["claim_id"], "doc_id": row["doc_id"], "view": row["view"]})
            continue
        if row["view"] == "rationale":
            ids = window["rationale_sentence_ids"]
            window = {**window, "context_sentence_ids": ids,
                      "premise": format_premise(doc.title, " ".join(doc.sentences[i] for i in ids), include_title)}
        result.append({**row, **window, "title": doc.title, "include_title": include_title})
    return result, omitted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="nli_scifact_train_v1")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--hard-negatives", type=int, default=0)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--final-fit", action="store_true", help="Refit after hyperparameters are frozen; include internal development but exclude ALL public validation texts")
    p.add_argument("--base-dir", default="artifacts/models/nli_base")
    p.add_argument("--amp", action="store_true", help="BF16 CUDA autocast; weights and optimizer remain FP32")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--global-class-balance", action="store_true", help="Use physical-batch-independent inverse-frequency weighting")
    p.add_argument("--pilot-batches", type=int, default=0, help="Hardware pilot only; stop without saving a model")
    p.add_argument("--schedule-epochs", type=int, default=None, help="Keep an earlier checkpoint's original LR horizon when refitting fewer epochs")
    p.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    p.add_argument("--evidence-safe-windows", action="store_true")
    p.add_argument("--include-title", action="store_true")
    args = p.parse_args()
    if args.layers < 1 or min(args.epochs, args.batch_size, args.accumulate) < 1 or args.pilot_batches < 0:
        p.error("layers, epochs, batch-size and accumulate must be positive; pilot-batches cannot be negative")
    if args.amp and args.device != "cuda":
        p.error("BF16 autocast in this training recipe requires CUDA")
    if args.schedule_epochs is not None and args.schedule_epochs < args.epochs:
        p.error("schedule-epochs cannot be shorter than the actual training run")
    torch.set_num_threads(args.threads)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = load_config()
    base = ROOT / args.base_dir
    rows, excluded = training_rows(cfg, args.hard_negatives, exclude_development=not args.final_fit)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    omitted = []
    if args.evidence_safe_windows:
        rows, omitted = evidence_safe_rows(cfg, rows, tokenizer, args.max_length, args.include_title)
    elif args.include_title:
        p.error("include-title training requires evidence-safe-windows")
    folder = ROOT / "artifacts/models" / args.name
    if folder.exists():
        raise ValueError("Training run already exists; use a new --name")
    folder.mkdir(parents=True)
    write_jsonl(folder / "training_pairs.jsonl", rows)
    write_json(folder / "training_manifest.json", {
        "seed": 42, "source_split": "train", "excluded_claim_ids": excluded,
        "exclusions": "All validation normalized claim texts; annotation conflicts" if args.final_fit else "All validation, dev_tune and smoke normalized claim texts; annotation conflicts",
        "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
        "training_pairs_sha256": sha256_file(folder / "training_pairs.jsonl"),
        "labels": dict(Counter(r["label"] for r in rows)), "settings": vars(args),
        "base": (base / "source_revision.txt").read_text(),
        "base_weights_sha256": sha256_file(base / "model.safetensors"),
        "script_sha256": sha256_file(Path(__file__)),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "omitted_positive_views": omitted,
        "optimizer_version": version("bitsandbytes") if args.optimizer == "adamw8bit" else torch.__version__,
    })
    model = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True, dtype=torch.float32)
    model.to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if args.layers > len(model.deberta.encoder.layer):
        raise ValueError("Requested more trainable layers than this model has")
    for block in list(model.deberta.encoder.layer)[-args.layers:] + [model.pooler, model.classifier]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mapping = {"SUPPORT": model.config.label2id["entailment"],
               "CONTRADICT": model.config.label2id["contradiction"],
               "NEI": model.config.label2id["neutral"]}
    counts = Counter(r["label"] for r in rows)
    class_weights = torch.empty(3, device=args.device)
    for label, index in mapping.items():
        class_weights[index] = len(rows) / (3 * counts[label])
    encoded = []
    for row in rows:
        tokens = tokenizer(row["premise"], row["claim"], truncation="only_first", max_length=args.max_length)
        tokens["labels"] = mapping[row["label"]]
        encoded.append(tokens)

    def collate(batch):
        labels = torch.tensor([r["labels"] for r in batch])
        inputs = tokenizer.pad([{k: v for k, v in r.items() if k != "labels"} for r in batch], return_tensors="pt")
        return {**inputs, "labels": labels}

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.learning_rate, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    total_steps = math.ceil(len(loader) / args.accumulate) * (args.schedule_epochs or args.epochs)
    warmup = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.0, (total_steps - step) / max(1, total_steps - warmup))),
    )
    print(f"Training {len(rows)} pairs; {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters", flush=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext():
                logits = model(**batch).logits
                if args.global_class_balance:
                    per_example = globally_weighted_losses(logits, labels, class_weights)
                    effective_batch = args.batch_size * args.accumulate
                    offset = (step - 1) * args.batch_size
                    group_start = offset // effective_batch * effective_batch
                    group_size = min(effective_batch, len(encoded) - group_start)
                    loss = per_example.mean()
                    backward_loss = per_example.sum() / group_size
                else:
                    # Preserve the historical recipe for reproducibility.
                    loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
                    backward_loss = loss / args.accumulate
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite verifier training loss")
            backward_loss.backward()
            if step % args.accumulate == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            if args.pilot_batches and step >= args.pilot_batches:
                stats = {"pilot_batches": step, "mean_loss": float(np.mean(losses)),
                         "seconds": time.perf_counter() - started,
                         "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None,
                         "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if args.device == "cuda" else None,
                         "model_checkpoint_saved": False}
                write_json(folder / "pilot.json", stats)
                print(stats, flush=True)
                return
            if step % 20 == 0:
                event = {"epoch": epoch, "step": step, "steps_per_epoch": len(loader),
                         "loss_last20": float(np.mean(losses[-20:])), "seconds": time.perf_counter() - started,
                         "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None}
                append_jsonl(folder / "progress.jsonl", event)
                print(event, flush=True)
        target = folder / f"epoch{epoch}"
        model.save_pretrained(target)
        tokenizer.save_pretrained(target)
        write_json(target / "training_loss.json", {"epoch": epoch, "mean_loss": float(np.mean(losses)), "seconds": time.perf_counter() - started})
        print(f"Saved {target}", flush=True)
    write_json(folder / "status.json", {"complete": True, "epochs": args.epochs,
                                        "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
