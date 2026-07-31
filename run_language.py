#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
import yaml
import torch
import src

MODELS = [
    ("Vanilla", src.VanillaAttention, {}),
    ("GroupQuery", src.GroupQueryAttention, {}),
    ("SplitEMA+aux", src.SplitEMAAttention, {}),
    ("SplitEMA", src.SplitEMAAttention, {"diversity_weight": 0.0}),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        c = yaml.safe_load(f)

    datasets  = c["datasets"]
    data_root = c["data_root"]
    output_csv = c["output_csv"]
    max_epochs = c["max_epochs"]
    seed = int(c["seed"])
    checkpoint_dir = Path(c["checkpoint_dir"])

    dataset_seed_offsets = {
    "wikitext2": 0,
    "ptb":  1000,
    "shakespeare": 2000,
}

    src.set_seed(seed)
    print(f"device: {src.device}")
    print(f"seed: {seed}")
    all_results = []

    for ds_idx, ds_name in enumerate(datasets):
        cfg = src.DATASET_HYPERPARAMS[ds_name]
        cfg = replace(cfg, max_epochs=int(max_epochs))
        print(f"={ds_name} block={cfg.block_size} d={cfg.d_model} heads={cfg.n_head} layers={cfg.n_layer}=\n")

        train_text, val_text, test_text, desc = src.load_text(ds_name, data_root)
        print(f"  {desc}")

        if ds_name == "shakespeare":
            tokenizer = src.CharTokenizer(train_text)
        else:
            tokenizer = src.build_bpe_tokenizer(train_text)
        vocab_size = tokenizer.get_vocab_size()
        encode = lambda s: tokenizer.encode(s).ids
        print(f"  vocab {vocab_size}  train {len(train_text):,} chars")

        train_data = torch.tensor(encode(train_text), dtype=torch.long)
        val_data   = torch.tensor(encode(val_text),   dtype=torch.long)
        test_data  = torch.tensor(encode(test_text),  dtype=torch.long)
        steps_per_epoch = max(
            1,
            math.ceil(max(1, len(train_data) - 1) / (cfg.batch_size * cfg.block_size)),
        )
        print(f"  tokens {len(train_data):,}  steps/epoch {steps_per_epoch:,}  epochs {cfg.max_epochs}")

        get_batch      = src.make_get_batch(train_data, cfg.block_size, cfg.batch_size)
        get_val_batch  = src.make_get_batch(val_data,   cfg.block_size, cfg.batch_size)
        get_test_batch = src.make_get_batch(test_data,  cfg.block_size, cfg.batch_size)

        for name, attn_cls, attn_kw in MODELS:
            r = src.train_and_eval_language(
                name, attn_cls, attn_kw,
                vocab_size, get_batch, get_val_batch, get_test_batch, tokenizer, cfg,
                steps_per_epoch=steps_per_epoch,
                seed=seed + dataset_seed_offsets[ds_name],
                save_dir=checkpoint_dir / ds_name,
                checkpoint_epochs=c["checkpoint_epochs"],
            )
            r["dataset"] = ds_name
            all_results.append(r)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fields = ["dataset", "name", "trainable", "total", "epochs",
                  "steps_per_epoch", "train_steps",
                  "time_s", "val_loss", "val_ppl", "test_loss", "test_ppl",
                  "best_epoch", "best_checkpoint_path", "checkpoint_paths"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    print(f"saved: {output_csv}")
    print(f"{'Dataset':<15} {'Model':<15} {'ValPPL':>10} {'TestPPL':>10}")
    for r in all_results:
        print(f"{r['dataset']:<15} {r['name']:<15} {r['val_ppl']:>10.2f} {r['test_ppl']:>10.2f}")


if __name__ == "__main__":
    main()
