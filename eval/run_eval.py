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
from src.evaluator import BERTScoreEvaluator, RougeEvaluator
from src.summarizer_model import TextRankMMRSummarizer


def run_evaluation(
    samples: int = 50,
    seed: int = 42,
    k: int = 5,
    bertscore: bool = False,
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

    # ROUGE is always computed -- it's fast and lexical
    evaluator = RougeEvaluator()
    scores = evaluator.evaluate(predictions, batch.references)

    # BERTScore is opt-in because it pulls in torch + a big model
    if bertscore:
        try:
            bert_eval = BERTScoreEvaluator()
            bert_scores = bert_eval.evaluate(predictions, batch.references)
            scores.update(bert_scores)
        except RuntimeError as exc:
            # if bert-score isn't installed, warn but don't crash
            print(f"[warn] BERTScore skipped: {exc}")

    return {
        "timestamp": datetime.now().isoformat(),
        "samples": samples,
        "seed": seed,
        "k": k,
        "model": "TextRankMMR (default config)",
        "scores": scores,
    }


def format_table(scores: dict[str, float]) -> str:
    """Pretty-print evaluation scores as a simple table."""
    lines = [
        "Metric            Score",
        "--------------    -----",
    ]
    # friendly names for the metrics -- ROUGE first, then BERTScore if present
    labels = {
        "rouge1_f": "ROUGE-1 F1",
        "rouge2_f": "ROUGE-2 F1",
        "rougeL_f": "ROUGE-L F1",
        "bert_precision": "BERT Prec",
        "bert_recall": "BERT Recall",
        "bert_f1": "BERT F1",
    }
    for key, label in labels.items():
        if key in scores or key.startswith("rouge"):
            val = scores.get(key, 0.0)
            lines.append(f"{label:<18}{val:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run extractive summarization evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5, help="Sentences per summary")
    parser.add_argument("--output", type=str, default="", help="Output JSON path (auto-dated if empty)")
    parser.add_argument(
        "--bertscore", action="store_true", default=False,
        help="Compute BERTScore alongside ROUGE (slow, needs torch)",
    )
    args = parser.parse_args(argv)

    # figure out where to save
    if args.output:
        out_path = Path(args.output)
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        out_path = Path("eval/results") / f"eval_{date_str}.json"

    print(f"Running evaluation: {args.samples} samples, seed={args.seed}, k={args.k}")
    print()

    results = run_evaluation(
        samples=args.samples, seed=args.seed, k=args.k, bertscore=args.bertscore,
    )

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
