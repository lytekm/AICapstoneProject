from __future__ import annotations

import json
from dataclasses import asdict
from itertools import product

from .evaluator import RougeEvaluator
from .summarizer_model import SummarizerConfig, TextRankMMRSummarizer


class SummarizerTrainer:
    """
    Training module: hyperparameter tuning (valid for non-neural extractive summarizers).
    """

    def __init__(self, metric: str = "rougeL_f"):
        self.metric = metric
        self.evaluator = RougeEvaluator()

    def tune(
        self,
        texts: list[str],
        references: list[str],
        k: int = 5,
        search_space: dict[str, list] | None = None,
    ) -> tuple[SummarizerConfig, dict[str, float]]:
        if search_space is None:
            search_space = {
                "mmr_lambda": [0.6, 0.75, 0.85],
                "blend_alpha": [0.5, 0.7, 0.9],
                "textrank_min_edge": [0.05, 0.1, 0.15],
            }

        keys = list(search_space.keys())
        best_cfg = None
        best_metrics = None
        best_score = -1e9

        for values in product(*[search_space[k] for k in keys]):
            params = dict(zip(keys, values, strict=False))
            cfg = SummarizerConfig(**params)
            model = TextRankMMRSummarizer(cfg)

            preds: list[str] = [str(model.summarize(t, k=k)["summary"]) for t in texts]
            metrics = self.evaluator.evaluate(preds, references)
            score = metrics.get(self.metric, 0.0)

            if score > best_score:
                best_score = score
                best_cfg = cfg
                best_metrics = metrics

        assert best_cfg is not None and best_metrics is not None
        return best_cfg, best_metrics

    def save_best(self, cfg: SummarizerConfig, metrics: dict[str, float], path: str) -> None:
        payload = {"best_config": asdict(cfg), "metrics": metrics}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
