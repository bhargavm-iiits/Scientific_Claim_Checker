"""Training/inference contract for the optional, locally fine-tuned Qwen judge.

Prompts contain only a claim, a title and original numbered source sentences.
Annotations are used by the training script, never by this module's inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SYSTEM = """Verify a scientific claim using only the supplied abstract.
SUPPORT: the abstract provides evidence for the claim.
CONTRADICT: the abstract provides evidence against the same claim.
NEI: the abstract does not establish either direction, including a different subject or outcome.
Compare the studied entities, intervention, outcome, direction and numerical quantities.
A general claim can be supported by a specific study; a missing detail alone is not contradiction.
Return only JSON with verdict (SUPPORT, CONTRADICT or NEI) and cited_sentence_ids.
For SUPPORT or CONTRADICT cite the original zero-based sentence numbers that justify the decision.
For NEI return an empty cited_sentence_ids list. Do not use outside knowledge."""


class LoraAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["SUPPORT", "CONTRADICT", "NEI"]
    cited_sentence_ids: list[int] = Field(max_length=32)

    @field_validator("cited_sentence_ids", mode="before")
    @classmethod
    def strict_ids(cls, values):
        if not isinstance(values, list) or any(type(i) is not int for i in values):
            raise ValueError("Sentence IDs must be JSON integers")
        return values


def messages(claim, title, sentences, ids):
    text = "\n".join(f"[{i}] {sentences[i]}" for i in ids)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Claim: {claim}\nTitle: {title}\nAbstract:\n{text}"}]


def prompt_ids(tokenizer, claim, title, sentences, ids):
    return tokenizer.apply_chat_template(messages(claim, title, sentences, ids),
                                         tokenize=True, add_generation_prompt=True, return_dict=False)


def context_windows(tokenizer, claim, title, sentences, max_length=1024, reserve=96):
    """Keep original indices and cover every sentence without silent truncation.

    Fail explicitly for an individual sentence that cannot fit. This avoids
    returning citations to text that the model never actually received.
    """
    windows, start = [], 0
    while start < len(sentences):
        end = start + 1
        first = prompt_ids(tokenizer, claim, title, sentences, [start])
        if len(first) + reserve > max_length:
            raise ValueError(f"Sentence {start} exceeds the Qwen context budget")
        while end < len(sentences):
            ids = list(range(start, end + 1))
            if len(prompt_ids(tokenizer, claim, title, sentences, ids)) + reserve > max_length:
                break
            end += 1
        windows.append(list(range(start, end)))
        if end == len(sentences):
            break
        start = max(start + 1, end - 1)
    return windows


def parse_answer(text, allowed):
    answer = LoraAnswer.model_validate_json(text.strip())
    if any(i not in allowed for i in answer.cited_sentence_ids):
        raise ValueError("Generated citation is outside the supplied context")
    if answer.verdict == "NEI" and answer.cited_sentence_ids:
        raise ValueError("NEI must not claim a decisive citation")
    if answer.verdict != "NEI" and not answer.cited_sentence_ids:
        raise ValueError("A decisive judgement must cite source evidence")
    return answer


class LoraJudge:
    def __init__(self, base, adapter=None, *, device="cuda", max_length=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device, self.max_length = torch, device, max_length
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(Path(adapter)), is_trainable=False)
        self.model.to(device).eval()

    def judge(self, claim, doc):
        outputs = []
        for ids in context_windows(self.tokenizer, claim, doc.title, doc.sentences, self.max_length):
            tokens = prompt_ids(self.tokenizer, claim, doc.title, doc.sentences, ids)
            batch = self.torch.tensor([tokens], device=self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    input_ids=batch, attention_mask=self.torch.ones_like(batch),
                    max_new_tokens=96, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            text = self.tokenizer.decode(generated[0, len(tokens):], skip_special_tokens=True)
            answer = parse_answer(text, ids)
            outputs.append({"sentence_ids": ids, "answer": answer.model_dump(), "raw": text})
        return outputs


def target_text(label, cited):
    return json.dumps({"verdict": label, "cited_sentence_ids": cited}, separators=(",", ":"))
