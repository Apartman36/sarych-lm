import torch

from sarych.data_synthetic import FixedBatchDataset
from sarych.model import SarychConfig, SarychLM
from sarych.utils import choose_device, set_seed


def test_tiny_model_overfits_fixed_structured_batch():
    set_seed(2024)
    device = choose_device("auto")
    config = SarychConfig(
        vocab_size=32,
        block_size=12,
        n_layer=1,
        n_head=2,
        n_embd=32,
        d_ff=96,
        dropout=0.0,
    )
    model = SarychLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    dataset = FixedBatchDataset(batch_size=8, block_size=config.block_size, vocab_size=config.vocab_size, device=device)
    x, y = dataset.get_batch()

    with torch.no_grad():
        _, initial_loss = model(x, y)

    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, final_loss = model(x, y)

    assert float(final_loss) < float(initial_loss) * 0.65
