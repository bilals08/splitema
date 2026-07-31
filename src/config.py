from dataclasses import dataclass
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 100
MAX_PEAKS = 200
MIN_PEPTIDE_LEN = 7
MAX_PEPTIDE_LEN = 50
PEAK_MIN_MZ = 10.0

# Peak tokens are encoded as [whole m/z, fractional m/z, intensity].
MZ_MAX      = 2500
VOCAB_WHOLE = MZ_MAX + 1
VOCAB_FRAC  = 1001
VOCAB_INT   = 101


@dataclass(frozen=True)
class DatasetHyperParams:
    block_size:   int   = 128
    d_model:      int   = 256
    n_head:       int   = 8
    n_layer:      int   = 6
    batch_size:   int   = 64
    lr:           float = 5e-4
    weight_decay: float = 0.1
    max_epochs:   int   = 120


DATASET_HYPERPARAMS: dict[str, DatasetHyperParams] = {
    "wikitext2":   DatasetHyperParams(),
    "ptb":         DatasetHyperParams(),
    "shakespeare": DatasetHyperParams(n_layer=4, batch_size=128, lr=1e-3, weight_decay=0.01),
}


@dataclass(frozen=True)
class ProteomicsConfig:
    d_model:      int   = 512
    n_head:       int   = 8
    n_layer:      int   = 2
    batch_size:   int   = 256
    lr:           float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int   = 100
    train_steps:  int   = 5200
    eval_batches: int   = 32
    temperature:  float = 0.1
