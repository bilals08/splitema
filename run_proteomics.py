#!/usr/bin/env python3
import argparse
import csv
import gc
from pathlib import Path
import torch
import yaml
import src

MODELS = [
    ("Longformer", src.LongformerAttention, {"attention_window": 64}),
    ("Linformer", src.LinformerAttention, {"seq_len": src.MAX_PEAKS + 1, "k": 64}),
    ("Vanilla", src.VanillaAttention, {}),
    ("GroupQuery", src.GroupQueryAttention, {}),
    ("SplitEMA", src.SplitEMAAttention, {}),
    ("SplitEMA-NoAux", src.SplitEMAAttention, {"diversity_weight": 0.0}),
]


# Yield fixed-size batches of model specs for memory-limited training.
def _chunks(items, size):
    if size < 1:
        raise ValueError(f"max_concurrent_models must be >= 1, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


# Run proteomics contrastive experiments across configured attention variants.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        c = yaml.safe_load(f)
    seed = int(c["seed"])
    src.set_seed(seed)

    cfg = src.ProteomicsConfig(
        d_model=c["d_model"],
        n_head=c["n_head"],
        n_layer=c["n_layer"],
        batch_size=c["batch_size"],
        lr=c["lr"],
        weight_decay=c["weight_decay"],
        warmup_steps=c["warmup_steps"],
        train_steps=c["train_steps"],
        eval_batches=c["eval_batches"],
        temperature=c["temperature"],
    )

    train_ds, val_ds, test_ds = src.build_proteomics_datasets(Path(c["data_root"]))
    print(f"train {len(train_ds):,}  val {len(val_ds):,}  test {len(test_ds):,}")
    print(f"device: {src.device}  MAX_PEAKS: {src.MAX_PEAKS}")
    print(f"seed: {seed}")
    model_index = c["model_index"]
    model_specs = MODELS
    if model_index is not None:
        model_index = int(model_index)
        if not 0 <= model_index < len(MODELS):
            raise ValueError(
                f"model_index must be 0..{len(MODELS) - 1}, got {model_index}"
            )
        model_specs = [MODELS[model_index]]

    max_concurrent_models = int(c["max_concurrent_models"])
    model_batches = list(_chunks(model_specs, max_concurrent_models))

    print(f"models: {[name for name, _, _ in model_specs]}")
    print(f"training at most {max_concurrent_models} model(s) at a time")

    output_dir = Path(c["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for batch_idx, batch_specs in enumerate(model_batches, start=1):
        batch_names = [name for name, _, _ in batch_specs]
        print(f"=Model Batch {batch_idx}/{len(model_batches)}: {batch_names}=\n")

        results.extend(
            src.train_and_eval_contrastive_multi(
                batch_specs,
                train_ds,
                val_ds,
                test_ds,
                cfg,
                num_workers=c["num_workers"],
                persistent_workers=c["persistent_workers"],
                prefetch_factor=c["prefetch_factor"],
                save_dir=output_dir,
                epochs=c["epochs"],
                seed=seed,
            )
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_csv = output_dir / "results.csv"
    with open(output_csv, "w", newline="") as f:
        fields = [
            "name",
            "trainable",
            "total",
            "time_s",
            "val_loss",
            "val_retrieval_acc_pct",
            "test_loss",
            "test_retrieval_acc_pct",
            "weights_path",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"saved: {output_csv}")

    print(f"=Final Results=\n")
    print(f"{'Model':<18} {'ValAcc%':>8} {'TestAcc%':>9}")
    for r in results:
        print(
            f"{r['name']:<18} {r['val_retrieval_acc_pct']:>8.2f} {r['test_retrieval_acc_pct']:>9.2f}"
        )


if __name__ == "__main__":
    main()
