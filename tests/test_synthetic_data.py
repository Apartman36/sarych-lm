import torch

from sarych.data_synthetic import SyntheticTokenDataset


def test_synthetic_batch_shapes_range_and_shift():
    dataset = SyntheticTokenDataset(
        total_tokens=1024,
        vocab_size=128,
        block_size=16,
        pattern_mode="mixed",
        seed=123,
    )

    x, y = dataset.get_batch(batch_size=4, device="cpu")

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert int(x.min()) >= 0
    assert int(y.max()) < 128
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_synthetic_data_is_deterministic_with_seed():
    first = SyntheticTokenDataset(1024, 64, 8, pattern_mode="mixed", seed=99)
    second = SyntheticTokenDataset(1024, 64, 8, pattern_mode="mixed", seed=99)

    x1, y1 = first.get_batch(batch_size=6, device="cpu")
    x2, y2 = second.get_batch(batch_size=6, device="cpu")

    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_structured_mode_is_learnable_not_uniform_only():
    dataset = SyntheticTokenDataset(512, 32, 8, pattern_mode="arithmetic", seed=7)
    tokens = dataset.tokens
    diffs = (tokens[1:32] - tokens[:31]) % 32

    assert torch.unique(diffs).numel() <= 4
