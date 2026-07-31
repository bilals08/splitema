from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from theoretical_spectrum_processing.sequence import SequenceProcessor
from theoretical_spectrum_processing.spectrum_theoretical import SpectrumTheoreticalProcessor
from ..config import (
    MAX_PEAKS, MZ_MAX, PEAK_MIN_MZ, MIN_PEPTIDE_LEN, MAX_PEPTIDE_LEN,
)
from ..logger import log
from .peptide import bracket_to_lowerletter, peptide_length


class ProteomicsDataset(Dataset):
    def __init__(self, h5_path: str | Path):
        self.path  = str(h5_path)
        self._file = None
        with h5py.File(self.path, "r") as f:
            self._indices = [
                i for i, seq in enumerate(f["sequence"])
                if MIN_PEPTIDE_LEN <= peptide_length(_decode_sequence(seq)) <= MAX_PEPTIDE_LEN
            ]
            self._length = len(self._indices)

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        f   = self._open()
        raw_i = self._indices[i]
        seq = _decode_sequence(f["sequence"][raw_i])
        return {
            "peaks":          torch.from_numpy(np.asarray(f["peaks"][raw_i], dtype=np.float32)),
            "sequence":       seq,
            "charge":         float(f["charge"][raw_i]),
            "precursor_mass": float(f["precursor_mass"][raw_i]),
            "protein":        "",
        }


def _decode_sequence(seq) -> str:
    if isinstance(seq, (bytes, np.bytes_)):
        return seq.decode()
    if isinstance(seq, np.ndarray):
        return seq.item().decode() if seq.ndim == 0 and isinstance(seq.item(), (bytes, np.bytes_)) else (
            seq.item() if seq.ndim == 0 else "".join(chr(int(c)) for c in seq)
        )
    return str(seq)


def build_proteomics_datasets(data_root: str | Path):
    root = Path(data_root)

    massive_trains = sorted(root.glob("Massive-KB-v1_train*.h5"))
    if not massive_trains:
        raise FileNotFoundError(f"Missing Massive-KB-v1_train*.h5 files in {root}")

    log.debug("Detected Massive-KB dataset layout in %s", root)
    train_parts = []
    for p in massive_trains:
        ds = ProteomicsDataset(p)
        log.debug(f"  train  {p.name:<40}  {len(ds):>8,} samples")
        train_parts.append(ds)

    val_path = root / "Massive-KB-v1_val.h5"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation file {val_path}")
    val_ds = ProteomicsDataset(val_path)
    log.debug(f"  val    {val_path.name:<40}  {len(val_ds):>8,} samples")

    test_path = root / "Massive-KB-v1_test_sorted.h5"
    if not test_path.exists():
        test_path = root / "Massive-KB-v1_test.h5"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing test file (Massive-KB-v1_test.h5 or _test_sorted.h5) in {root}"
        )
    test_ds = ProteomicsDataset(test_path)
    log.debug(f"  test   {test_path.name:<40}  {len(test_ds):>8,} samples")

    train_ds = ConcatDataset(train_parts)
    log.debug(f"  split  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    return train_ds, val_ds, test_ds


def tokenize_spectrum(peaks: torch.Tensor, precursor_neutral_mass: float = 0.0) -> tuple[torch.Tensor, int]:
    peaks = peaks.float()
    valid = peaks[(peaks[:, 0] > 0) | (peaks[:, 1] > 0)]
    valid = valid[(valid[:, 0] >= PEAK_MIN_MZ) & (valid[:, 0] <= MZ_MAX)]
    if valid.numel() == 0:
        return torch.zeros(MAX_PEAKS, 3, dtype=torch.long), 0

    if len(valid) > MAX_PEAKS:
        valid = valid[torch.argsort(valid[:, 1], descending=True)[:MAX_PEAKS]]
    valid = valid[torch.argsort(valid[:, 0], stable=True)]
    n = len(valid)

    mz    = valid[:, 0]
    inten = valid[:, 1].clamp(min=0)

    whole   = mz.long().clamp(1, MZ_MAX)
    frac    = ((mz - mz.floor()) * 1000).round().long().clamp(0, 999) + 1
    sqrt_i  = torch.sqrt(inten)
    max_sq  = sqrt_i.max()
    int_tok = ((sqrt_i / (max_sq + 1e-8)) * 99).round().long() + 1
    int_tok = int_tok.clamp(1, 100)

    tokens = torch.zeros(MAX_PEAKS, 3, dtype=torch.long)
    tokens[:n, 0] = whole
    tokens[:n, 1] = frac
    tokens[:n, 2] = int_tok
    return tokens, n


def _peaks_to_tensor(spectrum) -> torch.Tensor:
    if not spectrum:
        return torch.zeros(0, 2)
    return torch.tensor([[mz, i] for mz, i in spectrum], dtype=torch.float32)


def _gen_spectrum(sequence, charge, meta):
    result = SpectrumTheoreticalProcessor.generate_theoretical_spectrum(
        sequence, meta=meta, maxcharge=charge)
    return result[0] if isinstance(result, tuple) else []


def sequence_to_spectra(sequence: str, charge: int) -> tuple[torch.Tensor, int, torch.Tensor, int]:
    charge  = max(1, int(charge))
    meta    = {"charge": charge}
    seq_ll  = bracket_to_lowerletter(sequence)
    theo    = _gen_spectrum(seq_ll, charge, meta)
    decoy   = _gen_spectrum(SequenceProcessor.get_decoy(seq_ll), charge, meta)
    theo_tokens, theo_len = tokenize_spectrum(_peaks_to_tensor(theo))
    decoy_tokens, decoy_len = tokenize_spectrum(_peaks_to_tensor(decoy))
    return theo_tokens, theo_len, decoy_tokens, decoy_len


_collate_logged = False


def collate_contrastive(batch):
    global _collate_logged
    exp_tokens, theo_tokens, dec_tokens = [], [], []
    exp_lens, theo_lens, dec_lens = [], [], []
    metas, seqs = [], []

    for item in batch:
        seq    = item["sequence"]
        if not (MIN_PEPTIDE_LEN <= peptide_length(seq) <= MAX_PEPTIDE_LEN):
            continue
        charge = max(1, int(item["charge"]))
        mass   = float(item["precursor_mass"])

        exp_tok, exp_len = tokenize_spectrum(item["peaks"])
        if exp_len == 0:
            continue
        theo_tok, theo_len, dec_tok, dec_len = sequence_to_spectra(seq, charge)

        if not _collate_logged:
            log.debug(f"collate sample  seq={repr(seq[:40])}  charge={charge}  "
                      f"tokens={tuple(exp_tok.shape)}  n_valid={exp_len}")
            _collate_logged = True

        exp_tokens.append(exp_tok)
        theo_tokens.append(theo_tok)
        dec_tokens.append(dec_tok)
        exp_lens.append(exp_len)
        theo_lens.append(theo_len)
        dec_lens.append(dec_len)
        metas.append(torch.tensor([charge, mass], dtype=torch.float32))
        seqs.append(seq)

    if not exp_tokens:
        raise ValueError("Batch contains no valid spectra")

    return (
        torch.stack(exp_tokens), torch.stack(metas), torch.tensor(exp_lens, dtype=torch.long),
        torch.stack(theo_tokens), torch.tensor(theo_lens, dtype=torch.long),
        torch.stack(dec_tokens), torch.tensor(dec_lens, dtype=torch.long),
        seqs,
    )


def make_contrastive_loader(dataset, batch_size, shuffle, num_workers=0,
                             persistent_workers=False, drop_last=False, seed=None,
                             prefetch_factor=None):
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        def worker_init_fn(worker_id):
            worker_seed = int(seed) + worker_id
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": collate_contrastive,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
        "generator": generator,
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)
