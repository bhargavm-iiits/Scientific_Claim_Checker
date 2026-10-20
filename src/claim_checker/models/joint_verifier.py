"""Joint scientific label/rationale model, initialized from a generic NLI encoder."""
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.agents.judge_nli import choose_label
from claim_checker.data.joint_context import encode_context
from claim_checker.data.store import read_json, write_json
from claim_checker.models.verifier import LABEL_MAP, NLIVerifier
from claim_checker.pipeline.aggregate import aggregate_verdicts
from claim_checker.schemas import DocumentJudgement


class JointReader(nn.Module):
    def __init__(self, backbone, dropout=.2):
        super().__init__()
        self.backbone = backbone
        size = backbone.config.hidden_size
        self.rationale_head = nn.Linear(size, 1)
        self.evidence_projection = nn.Linear(size, size, bias=False)
        self.evidence_dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.rationale_head.weight)
        nn.init.zeros_(self.rationale_head.bias)
        nn.init.zeros_(self.evidence_projection.weight)

    def forward(self, source_mask, **inputs):
        if torch.any(source_mask.sum(dim=1) == 0):
            raise ValueError("Joint reader needs at least one source token per example")
        hidden = self.backbone.deberta(**inputs).last_hidden_state
        rationale_logits = self.rationale_head(hidden).squeeze(-1)
        attention = rationale_logits.sigmoid() * source_mask
        evidence = (hidden * attention.unsqueeze(-1)).sum(dim=1) / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        features = self.backbone.pooler(hidden) + self.evidence_projection(self.evidence_dropout(evidence))
        logits = self.backbone.classifier(self.backbone.dropout(features))
        return logits, rationale_logits

    def save(self, folder, tokenizer):
        folder = Path(folder)
        self.backbone.save_pretrained(folder)
        tokenizer.save_pretrained(folder)
        weights = {f"rationale_head.{k}": v.detach().cpu().contiguous() for k, v in self.rationale_head.state_dict().items()}
        weights.update({f"evidence_projection.{k}": v.detach().cpu().contiguous() for k, v in self.evidence_projection.state_dict().items()})
        save_file(weights, folder / "joint_heads.safetensors")
        write_json(folder / "joint_config.json", {"architecture": "token_rationale_attention_v1", "dropout": self.evidence_dropout.p})

    @classmethod
    def load(cls, folder, device="cpu"):
        info = read_json(Path(folder) / "joint_config.json")
        assert info["architecture"] == "token_rationale_attention_v1"
        backbone = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True, dtype=torch.float32)
        model = cls(backbone, info["dropout"])
        heads = load_file(Path(folder) / "joint_heads.safetensors")
        for name in ("rationale_head", "evidence_projection"):
            getattr(model, name).load_state_dict({k.split(".", 1)[1]: v for k, v in heads.items() if k.startswith(name + ".")})
        return model.to(device).eval()


def rationale_loss(logits, targets):
    valid = targets != -100
    weights = torch.where(targets == 1, 3., .25)
    values = nn.functional.binary_cross_entropy_with_logits(logits.float(), targets.clamp_min(0).float(), reduction="none")
    return (values * weights * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class JointVerifier:
    def __init__(self, folder, device="cpu"):
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.model = JointReader.load(folder, device)
        self.device, self.max_length, self.include_title, self.window_overlap = device, 512, True, 2
        self.labels = [LABEL_MAP[self.model.backbone.config.id2label[i].lower()] for i in range(3)]

    def format_premise(self, title, text):
        from claim_checker.data.verifier_context import format_premise

        return format_premise(title, text, True)

    def judge(self, claim, doc, threshold=.5):
        windows = NLIVerifier.windows(self, claim, doc.sentences, title=doc.title)
        scores, rationales = [], []
        for ids in windows:
            row = encode_context(self.tokenizer, claim, doc, ids)
            inputs = {k: torch.tensor([v], device=self.device) for k, v in row["inputs"].items()}
            with torch.inference_mode():
                logits, token_logits = self.model(**inputs)
            scores.append(dict(zip(self.labels, logits.float().softmax(-1)[0].tolist(), strict=True)))
            probs = token_logits[0].float().sigmoid().tolist()
            rationales.append({sid: sum(p for p, source in zip(probs, row["token_source_ids"], strict=True) if source == sid)
                              / sum(source == sid for source in row["token_source_ids"])
                              for sid in set(row["token_source_ids"]) - {-1}})
        labels = [choose_label(s, threshold) for s in scores]
        verdict, conflict = aggregate_verdicts(labels)
        cited, confidence = [], None
        for label in (("SUPPORT", "CONTRADICT") if conflict else (() if verdict == "NEI" else (verdict,))):
            best = max((i for i, value in enumerate(labels) if value == label), key=lambda i: scores[i][label])
            order = sorted(rationales[best], key=lambda sid: (-rationales[best][sid], sid))
            chosen = [sid for sid in order[:2] if rationales[best][sid] >= .5] or order[:1]
            cited.extend(chosen)
            confidence = scores[best][label] if not conflict else None
        cited = sorted(set(cited))
        reason = ("Different source windows yielded opposing decisions." if conflict else
                  "The abstract reports: " + " ".join(doc.sentences[i] for i in cited) if cited else
                  "The retrieved abstract did not establish either direction of the claim.")
        return DocumentJudgement(doc_id=doc.doc_id, verdict=verdict, cited_sentence_ids=cited,
                                 context_sentence_ids=list(range(len(doc.sentences))), mode="full_abstract",
                                 reason=reason, confidence=confidence, prompt_set="joint_nli_v1",
                                 consistency_flags=["window_conflict"] if conflict else [])
