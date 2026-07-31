#!/usr/bin/env python3
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import src
from run_proteomics import MODELS
from src.train.utils import safe_name

_CACHE_ROOT = Path(tempfile.gettempdir()) / "transformer-eval-cache"
os.environ["MPLBACKEND"] = "Agg"
os.environ["XDG_CACHE_HOME"] = str(_CACHE_ROOT / "xdg")


# Convert YAML values into a proteomics training/evaluation config.
def _config_to_proteomics_config(c: dict) -> src.ProteomicsConfig:
    return src.ProteomicsConfig(
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


# Resolve and validate the Massive-KB test HDF5 path.
def _resolve_massivekb_test_file(test_h5: str) -> Path:
    path = Path(test_h5)
    if not path.exists():
        raise FileNotFoundError(f"Missing MassiveKB test file: {path}")
    return path


# Count raw sequence entries in an HDF5 dataset file.
def _h5_sequence_count(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return len(f["sequence"])


# Resolve a saved model checkpoint for the requested epoch.
def _resolve_checkpoint(
    checkpoint_dir: Path,
    model_name: str,
    checkpoint_epoch: int,
) -> Path:
    path = checkpoint_dir / f"{safe_name(model_name)}_epoch{checkpoint_epoch:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


# Rebuild a spectrum model and load checkpoint weights for evaluation.
def _load_model(name: str, attn_cls, attn_kwargs: dict, cfg: src.ProteomicsConfig, checkpoint: Path):
    model = src.SpectrumEmbeddingTransformer(
        attn_cls, cfg.d_model, cfg.n_head, cfg.n_layer, attn_kwargs
    ).to(src.device)
    raw = torch.load(checkpoint, map_location=src.device, weights_only=False)
    model.load_state_dict(raw["model"])
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, {"name": name, "trainable": trainable, "total": total}


# Compute bidirectional InfoNCE loss for experimental/theoretical embeddings.
def _infonce(ee: torch.Tensor, te: torch.Tensor, de: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(ee.size(0), device=ee.device)
    loss_fwd = F.cross_entropy(ee @ torch.cat([te, de], dim=0).T / temperature, labels)
    loss_bwd = F.cross_entropy(te @ torch.cat([ee, de], dim=0).T / temperature, labels)
    return 0.5 * (loss_fwd + loss_bwd)


# Evaluate contrastive loss plus top-1 and top-5 retrieval accuracy.
def _evaluate(model, loader, temperature: float, eval_batches: int, desc: str) -> dict:
    dev = next(model.parameters()).device
    losses = []
    top1 = 0
    top5 = 0
    total = 0

    n_batches = min(eval_batches, len(loader))
    with torch.no_grad():
        bar = tqdm(loader, total=n_batches, desc=desc, unit="batch")
        for i, (ep, _, el, tp, tl, dp, dl, _) in enumerate(bar):
            if eval_batches is not None and i >= eval_batches:
                break

            ep = ep.to(dev)
            el = el.to(dev)
            tp = tp.to(dev)
            tl = tl.to(dev)
            dp = dp.to(dev)
            dl = dl.to(dev)

            ee, _ = model(ep, el)
            te, _ = model(tp, tl)
            de, _ = model(dp, dl)

            labels = torch.arange(ee.size(0), device=dev)
            logits = ee @ torch.cat([te, de], dim=0).T
            losses.append(_infonce(ee, te, de, temperature).item())

            k = min(5, logits.size(1))
            topk = logits.topk(k, dim=-1).indices
            top1 += topk[:, 0].eq(labels).sum().item()
            top5 += topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.numel()

    return {
        "loss": sum(losses) / max(1, len(losses)),
        "retrieval_top1": top1 / max(1, total),
        "retrieval_top5": top5 / max(1, total),
        "evaluated_spectra": total,
        "evaluated_batches": len(losses),
    }


# Write evaluation rows to CSV and optional JSON files.
def _write_outputs(rows: list[dict], output_csv: Path, output_json: Path | None):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "checkpoint", "test_h5", "trainable", "total",
        "loss", "retrieval_top1", "retrieval_top1_pct",
        "retrieval_top5", "retrieval_top5_pct",
        "evaluated_spectra", "evaluated_batches",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved csv: {output_csv}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"saved json: {output_json}")


# Parse CLI arguments and evaluate selected proteomics checkpoints.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--test-h5", required=True)
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed = int(config["seed"])
    src.set_seed(seed)

    cfg = _config_to_proteomics_config(config)

    test_h5 = _resolve_massivekb_test_file(args.test_h5)
    raw_test_spectra = _h5_sequence_count(test_h5)
    test_ds = src.ProteomicsDataset(test_h5)
    test_loader = src.make_contrastive_loader(
        test_ds,
        cfg.batch_size,
        shuffle=False,
        num_workers=config["num_workers"],
        seed=seed,
    )

    model_specs = MODELS
    if args.model_index is not None:
        if not 0 <= args.model_index < len(MODELS):
            raise ValueError(f"model-index must be 0..{len(MODELS) - 1}, got {args.model_index}")
        model_specs = [MODELS[args.model_index]]
    if args.model_name is not None:
        model_specs = [spec for spec in model_specs if spec[0] == args.model_name]
        if not model_specs:
            raise ValueError(f"Unknown model name {args.model_name!r}")
    checkpoint_epoch = args.checkpoint_epoch
    checkpoint_dir = Path(config["output_dir"])
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json) if args.output_json else None

    print(f"test h5: {test_h5}")
    print(f"test spectra raw: {raw_test_spectra:,}")
    print(
        f"test spectra length-filtered "
        f"({src.MIN_PEPTIDE_LEN}-{src.MAX_PEPTIDE_LEN} aa): {len(test_ds):,}"
    )
    print(f"device: {src.device}")
    print(f"seed: {seed}")
    print(f"batch size: {cfg.batch_size}")
    print(f"checkpoint dir: {checkpoint_dir}")
    print(f"checkpoint epoch: {checkpoint_epoch}")
    print(f"eval batches: {cfg.eval_batches}\n")

    rows = []
    for name, attn_cls, attn_kwargs in model_specs:
        checkpoint = _resolve_checkpoint(checkpoint_dir, name, checkpoint_epoch)
        model, model_info = _load_model(name, attn_cls, attn_kwargs, cfg, checkpoint)
        print(f"[{name}] loaded {checkpoint}")
        metrics = _evaluate(model, test_loader, cfg.temperature, cfg.eval_batches, desc=f"{name} test")
        row = {
            **model_info,
            "checkpoint": str(checkpoint),
            "test_h5": str(test_h5),
            "loss": round(metrics["loss"], 6),
            "retrieval_top1": round(metrics["retrieval_top1"], 6),
            "retrieval_top1_pct": round(100 * metrics["retrieval_top1"], 2),
            "retrieval_top5": round(metrics["retrieval_top5"], 6),
            "retrieval_top5_pct": round(100 * metrics["retrieval_top5"], 2),
            "evaluated_spectra": metrics["evaluated_spectra"],
            "evaluated_batches": metrics["evaluated_batches"],
        }
        rows.append(row)
        print(
            f"  loss {row['loss']:.4f}  "
            f"top1 {row['retrieval_top1_pct']:.2f}%  "
            f"top5 {row['retrieval_top5_pct']:.2f}%  "
            f"n={row['evaluated_spectra']:,}\n"
        )

    _write_outputs(rows, output_csv, output_json)

    print(f"{'Model':<18} {'Top1%':>8} {'Top5%':>8} {'Loss':>10} {'Spectra':>10}")
    for row in rows:
        print(
            f"{row['name']:<18} {row['retrieval_top1_pct']:>8.2f} "
            f"{row['retrieval_top5_pct']:>8.2f} {row['loss']:>10.4f} "
            f"{row['evaluated_spectra']:>10,}"
        )


if __name__ == "__main__":
    main()
