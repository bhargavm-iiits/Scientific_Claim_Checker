"""The optional judge must preserve provenance and reject fabricated citations."""
from pathlib import Path

import pytest

from claim_checker.models.lora_judge import context_windows, messages, parse_answer, prompt_ids


class WordTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(m["content"] for m in messages).split()


def test_context_windows_cover_source_without_renumbering():
    sentences = [f"Sentence {i} " + "word " * 10 for i in range(20)]
    windows = context_windows(WordTokenizer(), "claim", "title", sentences, max_length=220, reserve=20)
    assert len(windows) > 1
    assert set().union(*map(set, windows)) == set(range(20))
    assert all(a[-1] == b[0] for a, b in zip(windows, windows[1:], strict=False))
    rendered = messages("claim", "title", sentences, windows[1])[-1]["content"]
    assert f"[{windows[1][0]}] Sentence {windows[1][0]}" in rendered


def test_overlong_sentence_fails_instead_of_truncating_source():
    with pytest.raises(ValueError, match="exceeds"):
        context_windows(WordTokenizer(), "claim", "title", ["word " * 300], max_length=220)


@pytest.mark.parametrize("text", [
    '{"verdict":"SUPPORT","cited_sentence_ids":[9]}',
    '{"verdict":"CONTRADICT","cited_sentence_ids":[]}',
    '{"verdict":"NEI","cited_sentence_ids":[0]}',
    '{"verdict":"SUPPORT","cited_sentence_ids":[true]}',
    '{"verdict":"SUPPORT","cited_sentence_ids":["0"]}',
    '```json\n{"verdict":"NEI","cited_sentence_ids":[]}\n```',
])
def test_invalid_or_unverifiable_answers_fail(text):
    with pytest.raises(ValueError):
        parse_answer(text, [0, 1])


def test_valid_answer_keeps_original_citations():
    answer = parse_answer('{"verdict":"SUPPORT","cited_sentence_ids":[7]}', [5, 6, 7])
    assert answer.verdict == "SUPPORT"
    assert answer.cited_sentence_ids == [7]


def test_installed_tokenizer_returns_tokens_and_respects_context_budget():
    transformers = pytest.importorskip("transformers")
    base = Path(__file__).resolve().parents[1] / "artifacts/models/qwen25_15b_base"
    if not (base / "tokenizer.json").exists():
        pytest.skip("Optional local Qwen tokenizer is not downloaded")
    tokenizer = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
    sentences = [f"A study measured protein {i} in 100 participants. " * 4 for i in range(20)]
    windows = context_windows(tokenizer, "A claim about protein.", "Study", sentences)
    assert len(windows) > 1
    for ids in windows:
        prefix = prompt_ids(tokenizer, "A claim about protein.", "Study", sentences, ids)
        assert isinstance(prefix, list) and all(type(i) is int for i in prefix)
        assert 100 < len(prefix) <= 1024 - 96


def test_sparse_answer_loss_matches_full_masked_causal_loss():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from scripts.train_lora_judge import batch_continuation_losses, continuation_loss

    torch.manual_seed(7)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, attention_dropout=0.0)).eval()
    inputs = torch.tensor([[4, 8, 15, 16, 23, 42, 3, 2, 1]])
    labels = inputs.clone()
    labels[:, :5] = -100
    full = model(input_ids=inputs, labels=labels).loss
    sparse = continuation_loss(model, inputs, inputs[0, 5:])
    torch.testing.assert_close(sparse, full, rtol=1e-6, atol=1e-6)
    rows = [{"prefix_ids": [4, 8, 15, 16, 23], "target_ids": [42, 3, 2, 1]},
            {"prefix_ids": [15, 16, 23], "target_ids": [2, 1]}]
    batched = batch_continuation_losses(model, rows, pad_id=0)
    for i, row in enumerate(rows):
        sequence = torch.tensor([row["prefix_ids"] + row["target_ids"]])
        expected = continuation_loss(model, sequence, torch.tensor(row["target_ids"]))
        torch.testing.assert_close(batched[i], expected, rtol=1e-6, atol=1e-6)
    wrapped = peft.get_peft_model(model, peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM, r=2, lora_alpha=4,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
    )).train()
    batch_continuation_losses(wrapped, rows, pad_id=0).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in wrapped.named_parameters() if "lora_B" in name)
