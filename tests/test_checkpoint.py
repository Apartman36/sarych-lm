import torch

from sarych.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from sarych.model import SarychConfig, SarychLM


def test_checkpoint_save_and_restore_model_optimizer_and_step(tmp_path):
    config = SarychConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, d_ff=48)
    model = SarychLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    checkpoint_path = save_checkpoint(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler_state={"last_lr": 1e-3},
        step=17,
        best_val_loss=3.5,
        config={"test": True},
        parameter_count=model.count_parameters(),
        is_best=True,
    )

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    metadata = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, map_location="cpu")

    assert metadata["step"] == 17
    assert metadata["best_val_loss"] == 3.5
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name])
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert (tmp_path / "checkpoint_best.pt").exists()


def test_restore_rng_state_accepts_serialized_cpu_byte_values():
    torch.manual_seed(123)
    state = {
        "python_random": __import__("random").getstate(),
        "numpy_random": __import__("numpy").random.get_state(),
        "torch_cpu": torch.get_rng_state().tolist(),
    }

    restore_rng_state(state)

    after_restore = torch.rand(4)
    torch.manual_seed(123)
    expected = torch.rand(4)
    assert torch.equal(after_restore, expected)


def test_checkpoint_restore_rng_state_after_optimizer_step(tmp_path):
    config = SarychConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, d_ff=48)
    model = SarychLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.manual_seed(99)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(x, x)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    checkpoint_path = save_checkpoint(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler_state={"last_lr": 1e-3},
        step=1,
        best_val_loss=None,
        config={"test": True},
        parameter_count=model.count_parameters(),
    )

    metadata = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, map_location="cpu")
    assert metadata["step"] == 1

    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, resumed_loss = model(x, x)
    resumed_loss.backward()
