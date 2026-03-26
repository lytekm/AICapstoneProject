from __future__ import annotations

from dataclasses import dataclass

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

# bert_score is heavy (torch + transformers), so we only import
# it when someone actually creates a BERTScoreEvaluator instance
try:
    import bert_score as _bert_score_lib
except Exception:
    _bert_score_lib = None


@dataclass
class RougeScores:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float


class RougeEvaluator:
    """
    Evaluation module: computes ROUGE-1/2/L (F1).

    ROUGE is a lexical overlap metric -- it counts n-gram matches
    between the prediction and reference. Fast and cheap but doesn't
    capture meaning. That's why we also have BERTScore below.
    """

    def __init__(self) -> None:
        if rouge_scorer is None:
            raise RuntimeError("rouge-score not installed. Run: pip install rouge-score")
        # stemming helps match "running" vs "ran" etc.
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def evaluate(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        r1 = r2 = rl = 0.0
        n = max(len(predictions), 1)

        for pred, ref in zip(predictions, references, strict=False):
            # rouge_score expects (reference, prediction) order
            scores = self.scorer.score(ref, pred)
            r1 += scores["rouge1"].fmeasure
            r2 += scores["rouge2"].fmeasure
            rl += scores["rougeL"].fmeasure

        return {"rouge1_f": r1 / n, "rouge2_f": r2 / n, "rougeL_f": rl / n}


class BERTScoreEvaluator:
    """Semantic evaluation using BERTScore.

    Unlike ROUGE, BERTScore uses contextual embeddings to compare
    meaning rather than exact word overlap. A summary that paraphrases
    well will score high here even if it uses different words.

    Uses DeBERTa-xmnli by default -- good balance of accuracy and speed.
    Set lang="en" so it picks the right tokenizer internally.
    """

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli") -> None:
        if _bert_score_lib is None:
            raise RuntimeError(
                "bert-score not installed. Run: pip install bert-score"
            )
        # store the model name; actual model loads lazily on first call
        self._model_type = model_type

    def evaluate(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute BERTScore precision, recall, and F1 across all pairs.

        Returns averaged scores as a flat dict, same pattern as RougeEvaluator
        so callers can merge both result dicts easily.
        """
        if not predictions or not references:
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

        # bert_score.score returns three tensors: (P, R, F1), one value per pair
        p_tensor, r_tensor, f1_tensor = _bert_score_lib.score(
            cands=predictions,
            refs=references,
            model_type=self._model_type,
            lang="en",
            verbose=False,
        )

        # average across all pairs
        return {
            "bert_precision": float(p_tensor.mean()),
            "bert_recall": float(r_tensor.mean()),
            "bert_f1": float(f1_tensor.mean()),
        }
