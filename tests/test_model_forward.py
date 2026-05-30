import torch

from sarych.model import SarychConfig, SarychLM


def test_model_forward_loss_and_backward_cpu():
    config = SarychConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        d_ff=96,
        dropout=0.0,
    )
    model = SarychLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, config.block_size))
    targets = torch.randint(0, config.vocab_size, (3, config.block_size))

    logits, loss = model(input_ids, targets)

    assert logits.shape == (3, config.block_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_forward_without_targets_returns_logits_only():
    config = SarychConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, d_ff=48)
    model = SarychLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits = model(input_ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, config.block_size, config.vocab_size)
