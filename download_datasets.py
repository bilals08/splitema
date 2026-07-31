import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset

DATASETS = ("wikitext2", "ptb", "shakespeare")


def _save_wikitext2(root: Path) -> None:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    ds.save_to_disk(root / "wikitext2")


def _save_ptb(root: Path) -> None:
    ds = load_dataset("ptb_text_only", "penn_treebank")
    ds.save_to_disk(root / "ptb")


def _save_shakespeare(root: Path) -> None:
    raw = load_dataset("tiny_shakespeare")
    text = "\n".join(str(x) for x in raw["train"]["text"])
    ds = DatasetDict({"train": Dataset.from_dict({"text": [text]})})
    ds.save_to_disk(root / "shakespeare")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/language_data")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    downloaders = {
        "wikitext2": _save_wikitext2,
        "ptb": _save_ptb,
        "shakespeare": _save_shakespeare,
    }
    for name in args.datasets:
        target = root / name
        if target.exists():
            raise FileExistsError(f"Dataset already exists: {target}")
        print(f"{name}: downloading: {target}")
        downloaders[name](root)

    print(f"done: {root}")


if __name__ == "__main__":
    main()
