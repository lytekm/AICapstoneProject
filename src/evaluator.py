from __future__ import annotations

from dataclasses import dataclass

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None


@dataclass
class RougeScores:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float


class RougeEvaluator:
    """
    Evaluation module: computes ROUGE-1/2/L (F1).
    """

    def __init__(self):
        if rouge_scorer is None:
            raise RuntimeError("rouge-score not installed. Run: pip install rouge-score")
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def evaluate(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        r1 = r2 = rl = 0.0
        n = max(len(predictions), 1)

        for pred, ref in zip(predictions, references, strict=False):
            scores = self.scorer.score(ref, pred)
            r1 += scores["rouge1"].fmeasure
            r2 += scores["rouge2"].fmeasure
            rl += scores["rougeL"].fmeasure

        return {"rouge1_f": r1 / n, "rouge2_f": r2 / n, "rougeL_f": rl / n}
