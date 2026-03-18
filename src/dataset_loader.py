from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset  # pip install datasets


@dataclass
class DatasetBatch:
    texts: list[str]
    references: list[str]


class CNNDailyMailDatasetLoader:
    """
    Data engineering for reproducible dataset experiments.

    Loads CNN/DailyMail using Hugging Face datasets.
    Output:
      - texts: article/document
      - references: human-written highlights summary
    """

    def __init__(self, version: str = "3.0.0"):
        self.version = version

    def load(
        self,
        split: str = "train",
        limit: int | None = 200,
        shuffle: bool = True,
        seed: int = 42,
    ) -> DatasetBatch:
        ds = load_dataset("cnn_dailymail", self.version, split=split)

        if shuffle:
            ds = ds.shuffle(seed=seed)

        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        texts = [row["article"] for row in ds]
        refs = [row["highlights"] for row in ds]

        return DatasetBatch(texts=texts, references=refs)
