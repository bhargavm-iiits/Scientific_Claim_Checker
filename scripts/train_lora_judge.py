"""Reproducible, train-only Qwen LoRA fine-tuning with cited verdict supervision."""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from claim_checker.models.lora_judge import context_windows, prompt_ids, target_text
from claim_checker.retrieval.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]


def continuation_loss(model, inputs, targets):
    """Teacher forcing with loss only on answer tokens, including the first one."""
    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False,
                   logits_to_keep=targets.numel() + 1).logits[0, :-1].float()
    return F.cross_entropy(logits, targets)


def batch_continuation_losses(model, rows, pad_id):
    """Right-pad causal inputs and project only supervised hidden states.

    Real tokens precede every padding token, so a causal attention mask already
    prevents them from attending to padding. Omitting an additional padding mask
    lets SDPA use its fast causal kernel. The frozen vocabulary projection is
    evaluated only at positions that predict actual answer tokens.
    """
    device = next(model.parameters()).device
    length = max(len(r["prefix_ids"]) + len(r["target_ids"]) for r in rows)
    inputs = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        sequence = row["prefix_ids"] + row["target_ids"]
        inputs[i, :len(sequence)] = torch.tensor(sequence, device=device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=inputs, use_cache=False).last_hidden_state
    states, targets, lengths = [], [], []
    for i, row in enumerate(rows):
        start = len(row["prefix_ids"]) - 1
        size = len(row["target_ids"])
        states.append(hidden[i, start:start + size])
        targets.extend(row["target_ids"])
        lengths.append(size)
    logits = base.lm_head(torch.cat(states, dim=0)).float()
    losses = F.cross_entropy(logits, torch.tensor(targets, device=device), reduction="none")
    return torch.stack([chunk.mean() for chunk in losses.split(lengths)])


def training_rows(cfg, tokenizer, max_length, *, final_fit=False, hard_negatives=1):
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    excluded = {normalize_claim_text(g.claim) for g in validation.values()}
    if not final_fit:
        for name in ("dev_tune", "smoke"):
            excluded.update(normalize_claim_text(train[c].claim) for c in load_split(cfg, name)["claim_ids"])
    corpus = load_corpus(cfg.path("processed_dir"))
    raw = {str(r["id"]): r for r in read_jsonl(cfg.path("raw_dir") / "scifact/claims_train.jsonl")}
    index = BM25Index(list(corpus), [d.text for d in corpus.values()])
    rows, omitted, eligible_ids = [], [], []
    for gold in train.values():
        if gold.conflict or normalize_claim_text(gold.claim) in excluded:
            continue
        eligible_ids.append(gold.claim_id)
        docs = [(d, gold.label_by_doc.get(d, "NEI"), "reference") for d in gold.reference_doc_ids]
        candidates = [d for d, _ in index.search(gold.claim, top_n=20) if d not in gold.reference_doc_ids]
        docs.extend((d, "NEI", "weak_negative") for d in candidates[:hard_negatives])
        for doc_id, label, provenance in docs:
            doc = corpus[doc_id]
            cited = []
            if label != "NEI":
                alternatives = [r["sentences"] for r in raw[gold.claim_id]["evidence"].get(doc_id, [])
                                if r["label"] == label and r["sentences"]]
                if not alternatives:
                    raise ValueError(f"No matching raw rationale for {gold.claim_id}/{doc_id}")
                cited = sorted(set(min(alternatives, key=lambda ids: (len(ids), ids))))
            try:
                windows = context_windows(tokenizer, gold.claim, doc.title, doc.sentences, max_length)
            except ValueError as exc:
                omitted.append({"claim_id": gold.claim_id, "doc_id": doc_id, "reason": str(exc)})
                continue
            # Keep all neutral windows, and positive windows containing a complete rationale.
            views = [(ids, "abstract_window") for ids in windows if not cited or set(cited).issubset(ids)]
            if cited and cited not in [ids for ids, _ in views]:
                # A second positive view teaches concise evidence without inventing explanations.
                views.append((cited, "rationale"))
            for ids, view in views:
                prefix = prompt_ids(tokenizer, gold.claim, doc.title, doc.sentences, ids)
                target = tokenizer.encode(target_text(label, cited) + tokenizer.eos_token, add_special_tokens=False)
                if len(prefix) + len(target) > max_length:
                    omitted.append({"claim_id": gold.claim_id, "doc_id": doc_id, "reason": "complete target exceeds budget"})
                    continue
                rows.append({"claim_id": gold.claim_id, "claim": gold.claim, "doc_id": doc_id,
                             "label": label, "provenance": provenance, "view": view,
                             "sentence_ids": ids, "cited_sentence_ids": cited,
                             "prefix_ids": prefix, "target_ids": target})
    return rows, eligible_ids, omitted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--hard-negatives", type=int, default=1)
    p.add_argument("--final-fit", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="Pilot only: maximum examples (not optimizer steps); 0 trains the full population")
    args = p.parse_args()
    if min(args.epochs, args.accumulate, args.rank, args.batch_size) < 1 or args.max_length < 256:
        p.error("Invalid positive training settings")
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("This training recipe requires the local CUDA torch environment")
    cfg = load_config()
    base = ROOT / "artifacts/models/qwen25_15b_base"
    folder = ROOT / "artifacts/models" / args.name
    if folder.exists():
        raise ValueError("Training folder exists; select a new name")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rows, eligible, omitted = training_rows(cfg, tokenizer, args.max_length,
                                           final_fit=args.final_fit, hard_negatives=args.hard_negatives)
    folder.mkdir(parents=True)
    write_jsonl(folder / "training_rows.jsonl", rows)
    counts = Counter(r["label"] for r in rows)
    weights = {label: math.sqrt(len(rows) / (3 * n)) for label, n in counts.items()}
    manifest = {"seed": 42, "settings": vars(args), "source_split": "train",
                "eligible_claim_ids": eligible, "labels": dict(counts), "loss_weights": weights,
                "omitted": omitted, "base": (base / "source_revision.txt").read_text(),
                "base_weights_sha256": sha256_file(base / "model.safetensors"),
                "libraries": {name: version(name) for name in ("transformers", "peft", "accelerate")},
                "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
                "validation_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
                "training_sha256": sha256_file(folder / "training_rows.jsonl"),
                "contract_sha256": sha256_file(ROOT / "src/claim_checker/models/lora_judge.py"),
                "script_sha256": sha256_file(Path(__file__)),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "exclusions": "All public validation texts; internal dev_tune/smoke unless final_fit; conflicting annotations"}
    write_json(folder / "training_manifest.json", manifest)
    print(f"Prepared {len(rows)} rows, {len(eligible)} eligible claims, labels={dict(counts)}, omitted={len(omitted)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=2 * args.rank,
        lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to("cuda").train()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=args.learning_rate, weight_decay=0.01)
    total_batches = min(len(rows) * args.epochs, args.max_steps or len(rows) * args.epochs)
    effective_batch = args.batch_size * args.accumulate
    updates = sum(math.ceil(min(len(rows), total_batches - start) / effective_batch)
                  for start in range(0, total_batches, len(rows)))
    warmup = max(1, int(updates * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda n: min((n + 1) / warmup, max(0.0, (updates - n) / max(1, updates - warmup))),
    )
    optimizer.zero_grad(set_to_none=True)
    started, batches, losses = time.perf_counter(), 0, []
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(rows)))
        random.shuffle(order)
        # Preserve optimizer groups while reducing padding within each group.
        for start in range(0, len(order), effective_batch):
            order[start:start + effective_batch] = sorted(
                order[start:start + effective_batch],
                key=lambda i: len(rows[i]["prefix_ids"]) + len(rows[i]["target_ids"]),
            )
        epoch_losses = []
        epoch_batches = min(len(rows), total_batches - batches)
        for position in range(0, epoch_batches, args.batch_size):
            batch_rows = [rows[index] for index in order[position:position + min(args.batch_size, epoch_batches - position)]]
            # Compute vocabulary logits only for the target continuation. The first
            # retained logit predicts the first target from the last prompt token.
            individual = batch_continuation_losses(model, batch_rows, tokenizer.pad_token_id)
            weighted = individual * torch.tensor([weights[r["label"]] for r in batch_rows], device="cuda")
            if not torch.isfinite(weighted).all():
                raise FloatingPointError("Non-finite training loss")
            group_start = position // effective_batch * effective_batch
            group_size = min(effective_batch, epoch_batches - group_start)
            (weighted.sum() / group_size).backward()
            values = weighted.detach().tolist()
            batches += len(batch_rows)
            epoch_losses.extend(values)
            losses.extend(values)
            end = position + len(batch_rows)
            if end % effective_batch == 0 or end == epoch_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batches % 40 < len(batch_rows) or batches == total_batches:
                event = {"epoch": epoch, "batches": batches, "loss_last10": sum(losses[-10:]) / len(losses[-10:]),
                         "seconds": time.perf_counter() - started,
                         "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30}
                append_jsonl(folder / "progress.jsonl", event)
                print(event, flush=True)
            if batches >= total_batches:
                break
        out = folder / f"epoch{epoch}"
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        write_json(out / "training_summary.json", {"epoch": epoch, "batches": batches,
                   "mean_epoch_loss": sum(epoch_losses) / len(epoch_losses),
                   "seconds": time.perf_counter() - started, "complete_epoch": len(epoch_losses) == len(rows)})
        if batches >= total_batches:
            break
    write_json(folder / "status.json", {"complete": True, "pilot": bool(args.max_steps), "batches": batches,
                                        "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
