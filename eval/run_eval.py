"""Reproducible evaluation on CNN/DailyMail test split.

Runs extractive summarization, computes ROUGE scores, and saves
results as JSON for tracking over time.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.dataset_loader import CNNDailyMailDatasetLoader
from src.evaluator import RougeEvaluator
from src.summarizer_model import TextRankMMRSummarizer


def run_evaluation(
    samples: int = 50,
    seed: int = 42,
    k: int = 5,
) -> dict:
    """Run extractive eval on CNN/DailyMail and return results dict."""
    # load a reproducible slice of the test set
    loader = CNNDailyMailDatasetLoader()
    batch = loader.load(split="test", limit=samples, shuffle=True, seed=seed)

    model = TextRankMMRSummarizer()
    predictions = [
        str(model.summarize(text, k=k).get("summary", ""))
        for text in batch.texts
    ]

    evaluator = RougeEvaluator()
    scores = evaluator.evaluate(predictions, batch.references)

    return {
        "timestamp": datetime.now().isoformat(),
        "samples": samples,
        "seed": seed,
        "k": k,
        "model": "TextRankMMR (default config)",
        "scores": scores,
    }


def format_table(scores: dict[str, float]) -> str:
    """Pretty-print ROUGE scores as a simple table."""
    lines = [
        "Metric       Score",
        "---------    -----",
    ]
    # friendly names for the metrics
    labels = {
        "rouge1_f": "ROUGE-1 F1",
        "rouge2_f": "ROUGE-2 F1",
        "rougeL_f": "ROUGE-L F1",
    }
    for key, label in labels.items():
        val = scores.get(key, 0.0)
        lines.append(f"{label:<13}{val:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run extractive summarization evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5, help="Sentences per summary")
    parser.add_argument("--output", type=str, default="", help="Output JSON path (auto-dated if empty)")
    args = parser.parse_args(argv)

    # figure out where to save
    if args.output:
        out_path = Path(args.output)
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        out_path = Path("eval/results") / f"eval_{date_str}.json"

    print(f"Running evaluation: {args.samples} samples, seed={args.seed}, k={args.k}")
    print()

    results = run_evaluation(samples=args.samples, seed=args.seed, k=args.k)

    # save to disk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    print()

    # print the table
    print(format_table(results["scores"]))

    return results


if __name__ == "__main__":
    main()
