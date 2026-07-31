#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
import run_language
import src
from src.train.utils import safe_name


# Load a YAML experiment configuration.
def _load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


# Rebuild the tokenizer expected by a language-model checkpoint.
def _build_tokenizer(ds_name: str, train_text: str):
    if ds_name == "shakespeare":
        return src.CharTokenizer(train_text)
    return src.build_bpe_tokenizer(train_text)


# Compute mean cross-entropy for a saved language model.
def _eval_split(model, batch_fn, vocab_size: int, seed_base: int, n_batches: int):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(n_batches):
            x, y = batch_fn(seed=seed_base + i)
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item())
    return sum(losses) / len(losses)


# Validate that checkpoint metadata matches the current evaluation setup.
def _check_checkpoint_config(checkpoint: dict, expected: dict, checkpoint_path: Path):
    actual = checkpoint["config"]
    mismatches = [
        key for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual[key]!r} current={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"{checkpoint_path} does not match current config ({details})")


# Evaluate each best language checkpoint and optionally write a CSV report.
def evaluate_best_checkpoints(config_path: str, output_csv: str | None, n_batches: int):
    c = _load_config(config_path)
    datasets = c["datasets"]
    data_root = c["data_root"]
    checkpoint_dir = Path(c["checkpoint_dir"])
    max_epochs = c["max_epochs"]
    seed = int(c["seed"])

    rows = []
    print(f"device: {src.device}")
    print(f"seed: {seed}")
    print(f"eval batches per split: {n_batches}")

    for ds_idx, ds_name in enumerate(datasets):
        cfg = src.DATASET_HYPERPARAMS[ds_name]
        cfg = replace(cfg, max_epochs=int(max_epochs))

        train_text, val_text, test_text, desc = src.load_text(ds_name, data_root)
        tokenizer = _build_tokenizer(ds_name, train_text)
        vocab_size = tokenizer.get_vocab_size()
        encode = lambda s: tokenizer.encode(s).ids

        val_data = torch.tensor(encode(val_text), dtype=torch.long)
        test_data = torch.tensor(encode(test_text), dtype=torch.long)
        get_val_batch = src.make_get_batch(val_data, cfg.block_size, cfg.batch_size)
        get_test_batch = src.make_get_batch(test_data, cfg.block_size, cfg.batch_size)

        print(f"={ds_name}=\n")
        print(f"  {desc}")
        print(f"  vocab {vocab_size}")

        for name, attn_cls, attn_kw in run_language.MODELS:
            path = checkpoint_dir / ds_name / f"{safe_name(name)}_best.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path}")

            checkpoint = torch.load(path, map_location=src.device)
            expected = {
                "vocab_size": vocab_size,
                "d_model": cfg.d_model,
                "n_head": cfg.n_head,
                "n_layer": cfg.n_layer,
                "block_size": cfg.block_size,
                "seed": seed + ds_idx * 1000,
            }
            _check_checkpoint_config(checkpoint, expected, path)

            model = src.LMTransformer(
                vocab_size, cfg.d_model, cfg.n_head, cfg.n_layer,
                cfg.block_size, attn_cls, **attn_kw,
            ).to(src.device)
            model.load_state_dict(checkpoint["model"])

            val_loss = _eval_split(model, get_val_batch, vocab_size, 2000, n_batches)
            test_loss = _eval_split(model, get_test_batch, vocab_size, 3000, n_batches)
            val_ppl = math.exp(val_loss)
            test_ppl = math.exp(test_loss)

            row = {
                "dataset": ds_name,
                "name": name,
                "checkpoint": str(path),
                "status": "ok",
                "best_epoch": checkpoint["epoch"],
                "checkpoint_val_loss": checkpoint["val_loss"],
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 2),
                "test_loss": round(test_loss, 4),
                "test_ppl": round(test_ppl, 2),
            }
            rows.append(row)
            print(
                f"  {name:<15} best_ep={row['best_epoch']!s:>4}  "
                f"val_ppl={val_ppl:>10.2f}  test_ppl={test_ppl:>10.2f}"
            )

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            fields = [
                "dataset", "name", "checkpoint", "status", "best_epoch",
                "checkpoint_val_loss", "val_loss", "val_ppl",
                "test_loss", "test_ppl",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {output_csv}")

    return rows


# Parse CLI arguments and run best-checkpoint evaluation.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--n-batches", type=int, required=True)
    args = parser.parse_args()

    if args.n_batches < 1:
        raise ValueError("--n-batches must be >= 1")

    evaluate_best_checkpoints(args.config, args.output_csv, args.n_batches)


if __name__ == "__main__":
    main()
