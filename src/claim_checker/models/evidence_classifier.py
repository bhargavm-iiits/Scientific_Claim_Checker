"""Trainable sentence-rationale selector; inference never receives annotations."""
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def candidate_context(tokenizer, claim, title, sentences, index, *, max_length=512, use_neighbors=True, input_order="query_first"):
    if not 0 <= index < len(sentences):
        raise ValueError("Candidate sentence index is outside the source")
    ids = [index]

    def render(selected):
        parts = [f"Title: {title}", f"Sentence: {sentences[index]}"]
        parts.extend(f"{'Previous' if i < index else 'Next'}: {sentences[i]}" for i in sorted(selected) if i != index)
        return "\n".join(parts)

    def fits(selected):
        pair = (claim, render(selected)) if input_order == "query_first" else (render(selected), claim)
        return len(tokenizer(*pair, add_special_tokens=True)["input_ids"]) <= max_length

    if not fits(ids):
        raise ValueError("Complete candidate sentence exceeds the selector token budget")
    for neighbour in (index - 1, index + 1) if use_neighbors else ():
        if 0 <= neighbour < len(sentences) and fits([*ids, neighbour]):
            ids.append(neighbour)
    return render(ids), sorted(ids)


class EvidenceClassifier:
    def __init__(self, path: str | Path, *, device="cpu", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, dtype=torch.float32).to(device).eval()
        self.device, self.max_length = device, max_length
        self.use_neighbors = getattr(self.model.config, "rationale_use_neighbors", True)
        self.input_order = getattr(self.model.config, "rationale_input_order", "query_first")

    def score(self, claim, doc, batch_size=16):
        premises = [candidate_context(self.tokenizer, claim, doc.title, doc.sentences, i, max_length=self.max_length,
                                      use_neighbors=self.use_neighbors, input_order=self.input_order)[0]
                    for i in range(len(doc.sentences))]
        result = []
        for start in range(0, len(premises), batch_size):
            selected = premises[start:start + batch_size]
            pairs = ([claim] * len(selected), selected) if self.input_order == "query_first" else (selected, [claim] * len(selected))
            batch = self.tokenizer(*pairs, padding=True, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**{k: v.to(self.device) for k, v in batch.items()}).logits
            result.extend(logits.float().sigmoid().flatten().tolist())
        return result


def select_sentence_ids(probabilities, *, threshold, maximum):
    order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    return sorted(i for i in order[:maximum] if probabilities[i] >= threshold)
