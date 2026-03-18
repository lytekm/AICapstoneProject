import pytest

from src.evaluator import RougeEvaluator


@pytest.fixture
def evaluator():
    return RougeEvaluator()


class TestRougeEvaluator:
    def test_perfect_score(self, evaluator):
        text = "The cat sat on the mat."
        scores = evaluator.evaluate([text], [text])
        assert scores["rouge1_f"] == pytest.approx(1.0, abs=0.01)
        assert scores["rougeL_f"] == pytest.approx(1.0, abs=0.01)

    def test_zero_overlap(self, evaluator):
        pred = "The quick brown fox jumps over the lazy dog."
        ref = "A completely unrelated sentence about quantum physics."
        scores = evaluator.evaluate([pred], [ref])
        assert scores["rouge1_f"] < 0.3

    def test_partial_overlap(self, evaluator):
        pred = "The bank held interest rates steady."
        ref = "The central bank held its key interest rate."
        scores = evaluator.evaluate([pred], [ref])
        assert 0.3 < scores["rouge1_f"] < 1.0

    def test_multiple_pairs(self, evaluator):
        preds = ["cats are great", "dogs are loyal"]
        refs = ["cats are wonderful", "dogs are faithful"]
        scores = evaluator.evaluate(preds, refs)
        assert "rouge1_f" in scores
        assert "rouge2_f" in scores
        assert "rougeL_f" in scores

    def test_returns_all_metrics(self, evaluator):
        scores = evaluator.evaluate(["hello world"], ["hello world"])
        assert set(scores.keys()) == {"rouge1_f", "rouge2_f", "rougeL_f"}
