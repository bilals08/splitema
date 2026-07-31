# SplitEMA Release Code

This repository contains the training and checkpoint evaluation code for the
SplitEMA language modeling and proteomics retrieval experiments.

## Setup

```bash
uv sync
```

## Language

Place the language datasets under the path configured by
`configs/language_config.yaml`:

```yaml
data_root: data/language_data
checkpoint_dir: weights/language
```

Download the required language datasets:

```bash
uv run python download_datasets.py --data-root data/language_data
```

Train:

```bash
uv run python run_language.py --config configs/language_config.yaml
```

Evaluate saved checkpoints:

```bash
uv run python eval_language_best.py --config configs/language_config.yaml
```

## Proteomics

Set `data_root` in `configs/proteomics_config.yaml` to the preprocessed
proteomics HDF5 directory.

Train:

```bash
uv run python run_proteomics.py --config configs/proteomics_config.yaml
```

Evaluate saved checkpoints:

```bash
uv run python eval_proteomics.py --config configs/proteomics_config.yaml
```

The default checkpoint directory is `weights/proteomics`.
