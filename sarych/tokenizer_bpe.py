from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


class SarychBPETokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> "SarychBPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))


def _iter_training_text(input_paths: Sequence[str | Path], limit_lines: int | None) -> Iterable[str]:
    remaining = limit_lines
    for input_path in input_paths:
        with Path(input_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if remaining is not None and remaining <= 0:
                    return
                yield line
                if remaining is not None:
                    remaining -= 1


def train_byte_bpe_tokenizer(
    *,
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    vocab_size: int = 4096,
    special_tokens: Sequence[str] | None = None,
    limit_lines: int | None = None,
) -> SarychBPETokenizer:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    paths = [Path(path) for path in input_paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Tokenizer input file(s) not found: {', '.join(missing)}")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=list(special_tokens or DEFAULT_SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(_iter_training_text(paths, limit_lines), trainer=trainer)

    wrapped = SarychBPETokenizer(tokenizer)
    wrapped.save(output_path)
    return wrapped
