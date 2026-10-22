"""Class weighting must work independently of physical microbatch size."""
import pytest


def test_global_class_weights_survive_single_example_batches():
    torch = pytest.importorskip("torch")
    from scripts.train_verifier import globally_weighted_losses

    logits = torch.tensor([[1.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
    labels = torch.tensor([0, 1, 2])
    weights = torch.tensor([0.5, 2.0, 3.0])
    whole = globally_weighted_losses(logits, labels, weights).mean()
    pieces = torch.stack([globally_weighted_losses(logits[i:i + 1], labels[i:i + 1], weights).mean()
                          for i in range(3)]).mean()
    torch.testing.assert_close(whole, pieces)
    unweighted = torch.nn.functional.cross_entropy(logits[:1], labels[:1])
    torch.testing.assert_close(globally_weighted_losses(logits[:1], labels[:1], weights)[0], unweighted * weights[0])
