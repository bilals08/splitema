from pathlib import Path
from types import SimpleNamespace
import torch
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from ..config import device


class CharTokenizer:
    def __init__(self, text):
        chars = ["[UNK]"] + sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    def encode(self, text):
        return SimpleNamespace(ids=[self.stoi.get(c, 0) for c in text])

    def decode(self, ids):
        return "".join(self.itos.get(i, "[UNK]") for i in ids)

    def get_vocab_size(self):
        return self.vocab_size


def build_bpe_tokenizer(text, vocab_size=5000):
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator([text], BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"]))
    return tok


_TEXT_FIELDS = {
    "wikitext2":   ("text",     "WikiText-2 (~2M tokens, Wikipedia)"),
    "ptb":         ("sentence", "Penn Treebank (~1M tokens, WSJ)"),
}


def load_text(choice, data_root="data/language_data"):
    path = Path(data_root) / choice
    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- run download_datasets.py first")
    ds = load_from_disk(str(path))
    if choice == "shakespeare":
        full = ds["train"]["text"][0]
        n80, n90 = int(len(full) * 0.8), int(len(full) * 0.9)
        return full[:n80], full[n80:n90], full[n90:], "Tiny Shakespeare (~1M chars)"
    if choice not in _TEXT_FIELDS:
        raise ValueError(f"unknown dataset {choice!r}; pick from {list(_TEXT_FIELDS)}")
    field, desc = _TEXT_FIELDS[choice]
    join = lambda split: "\n".join(ds[split][field])
    return join("train"), join("validation"), join("test"), desc


def make_get_batch(data, block_size, batch_size):
    def get_batch(seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x  = torch.stack([data[i:i+block_size]     for i in ix])
        y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    return get_batch
