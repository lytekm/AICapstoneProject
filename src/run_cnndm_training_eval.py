from __future__ import annotations

import json
import os

from src.dataset_loader import CNNDailyMailDatasetLoader
from src.diagram_generator import DiagramGenerator
from src.evaluator import RougeEvaluator
from src.summarizer_model import TextRankMMRSummarizer
from src.trainer import SummarizerTrainer


def ensure_outputs():
    os.makedirs("outputs", exist_ok=True)


def main():
    ensure_outputs()
    DiagramGenerator().generate_all()
    # ---- Load dataset (reproducible) ----
    loader = CNNDailyMailDatasetLoader(version="3.0.0")

    train_batch = loader.load(split="train", limit=200, shuffle=True, seed=42)
    test_batch = loader.load(split="test", limit=100, shuffle=True, seed=123)

    # ---- TRAINING (hyperparameter tuning) ----
    trainer = SummarizerTrainer(metric="rougeL_f")
    best_cfg, best_metrics = trainer.tune(
        texts=train_batch.texts,
        references=train_batch.references,
        k=3,
        search_space={
            "mmr_lambda": [0.6, 0.75, 0.85],
            "blend_alpha": [0.5, 0.7, 0.9],
            "textrank_min_edge": [0.05, 0.1, 0.15],
        },
    )

    best_path = os.path.join("outputs", "best_config_cnndm.json")
    trainer.save_best(best_cfg, best_metrics, best_path)
    print(f"Saved best config: {best_path}")
    print("Best (train) metrics:", best_metrics)

    # ---- EVALUATION (final on held-out test subset) ----
    model = TextRankMMRSummarizer(best_cfg)
    preds = [model.summarize(t, k=3)["summary"] for t in test_batch.texts]

    evaluator = RougeEvaluator()
    test_metrics = evaluator.evaluate(preds, test_batch.references)

    eval_path = os.path.join("outputs", "eval_report_cnndm.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Saved eval report: {eval_path}")
    print("Test metrics:", test_metrics)



if __name__ == "__main__":
    main()
