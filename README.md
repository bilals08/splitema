# SplitEMA Attention

Transformer self-attention is the standard architecture for sequence modeling because it models relationships between tokens to capture contextual relationships. However, the same key-value representations in attention must encode both recent information (short-term memory) and persistent long-term context (long-term memory). Since these representations do not explicitly separate the two timescales, the attention mechanism must balance transient and stable features with the same  memory representations. This places competing objectives on the same key-value representation. To address this limitation, we propose SplitEMA Attention, which divides each attention layer into two separate key-value streams. A trainable fast stream captures recent, rapidly changing patterns, whereas a slow exponential moving average (EMA) stream preserves stable long-term representations. The two streams share the same queries, compute attention independently, and their outputs are concatenated. This concatenation enables the model to use short-term memory and long-term memory to provide the final output (e.g., embeddings). We evaluate SplitEMA on language modelling benchmarks and peptide retrieval for proteomics to assess its applicability on diverse sequence modelling tasks.  On WikiText-2, SplitEMA lowered test perplexity from 117.89 to 114.02, and on Penn Treebank from 80.57 to 79.49, while using approximately 8% fewer trainable parameters. On the MassIVE-KB peptide retrieval benchmark, SplitEMA achieves competitive Top-1 accuracy of 95.74%. These results show that separating fast and slow memory within the attention mechanism improves parameter efficiency, and performance across language modeling and peptide retrieval tasks.

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then set up the project environment:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repository root
uv sync
```

If `uv` is already installed, only run `uv sync`. It will create the virtual environment and install the project dependencies from `pyproject.toml`.
Restart your shell if the `uv` command is not found immediately after installation.

## Language

1. Confirm the language data and output paths in `configs/language_config.yaml`:

```yaml
data_root: data/language_data
checkpoint_dir: weights/language
```

2. Download the required language datasets:

```bash
uv run python download_datasets.py --data-root data/language_data
```

3. Train the configured language experiments:

```bash
uv run python run_language.py --config configs/language_config.yaml
```

4. Evaluate saved best checkpoints:

```bash
uv run python eval_language_best.py \
  --config configs/language_config.yaml \
  --output-csv weights/language/eval_best.csv \
  --n-batches 20
```

## Proteomics

1. Download and preprocess the Mass Spectrometry proteomics dataset.

2. Set `data_root` in `configs/proteomics_config.yaml` to the preprocessed MassIVE-KB HDF5 directory:

```yaml
data_root: data/proteomicsdata/preprocessed_Massive_KB_v1
output_dir: weights/proteomics
```

3. Train the configured proteomics experiments:

```bash
uv run python run_proteomics.py --config configs/proteomics_config.yaml
```

4. Evaluate a saved checkpoint:

```bash
uv run python eval_proteomics.py \
  --config configs/proteomics_config.yaml \
  --checkpoint-epoch 20 \
  --test-h5 data/proteomicsdata/preprocessed_Massive_KB_v1/Massive-KB-v1_test.h5 \
  --output-csv weights/proteomics/eval_epoch020.csv
```

The default proteomics checkpoint directory is `weights/proteomics`. Use `--model-index` or `--model-name` with `eval_proteomics.py` to evaluate one attention variant at a time.

## License and Usage Terms
This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. If you are a commercial entity, please contact the corresponding author.
