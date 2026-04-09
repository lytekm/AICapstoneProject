from unittest.mock import MagicMock, patch

import pytest

from src.evaluator import BERTScoreEvaluator, RougeEvaluator


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


# ---------------------------------------------------------------------------
# BERTScore evaluator (mocked -- we don't want torch in CI)
# ---------------------------------------------------------------------------


class TestBERTScoreEvaluator:
    """Tests for BERTScore integration.

    We mock the bert_score library so tests run without torch/GPU.
    The real model loading is tested during vLLM validation (Phase 4.9).
    """

    def _make_mock_tensor(self, values: list[float]) -> MagicMock:
        """Create a mock tensor with a .mean() method."""
        tensor = MagicMock()
        tensor.mean.return_value = sum(values) / len(values)
        return tensor

    @patch("src.evaluator._bert_score_lib")
    def test_returns_expected_keys(self, mock_lib: MagicMock) -> None:
        mock_lib.score.return_value = (
            self._make_mock_tensor([0.9]),
            self._make_mock_tensor([0.85]),
            self._make_mock_tensor([0.87]),
        )

        evaluator = BERTScoreEvaluator()
        scores = evaluator.evaluate(["hello world"], ["hello world"])

        assert set(scores.keys()) == {"bert_precision", "bert_recall", "bert_f1"}

    @patch("src.evaluator._bert_score_lib")
    def test_scores_in_valid_range(self, mock_lib: MagicMock) -> None:
        mock_lib.score.return_value = (
            self._make_mock_tensor([0.92, 0.88]),
            self._make_mock_tensor([0.90, 0.85]),
            self._make_mock_tensor([0.91, 0.86]),
        )

        evaluator = BERTScoreEvaluator()
        scores = evaluator.evaluate(["pred1", "pred2"], ["ref1", "ref2"])

        for key in ("bert_precision", "bert_recall", "bert_f1"):
            assert 0.0 <= scores[key] <= 1.0

    @patch("src.evaluator._bert_score_lib")
    def test_empty_input_returns_zeros(self, mock_lib: MagicMock) -> None:
        evaluator = BERTScoreEvaluator()
        scores = evaluator.evaluate([], [])

        assert scores == {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
        # should not even call bert_score.score with empty lists
        mock_lib.score.assert_not_called()

    @patch("src.evaluator._bert_score_lib")
    def test_calls_library_with_correct_args(self, mock_lib: MagicMock) -> None:
        mock_lib.score.return_value = (
            self._make_mock_tensor([0.9]),
            self._make_mock_tensor([0.9]),
            self._make_mock_tensor([0.9]),
        )

        evaluator = BERTScoreEvaluator(model_type="distilbert-base-uncased")
        evaluator.evaluate(["the cat sat"], ["the cat sat"])

        mock_lib.score.assert_called_once_with(
            cands=["the cat sat"],
            refs=["the cat sat"],
            model_type="distilbert-base-uncased",
            lang="en",
            verbose=False,
        )

    def test_missing_library_raises(self) -> None:
        """If bert-score is not installed, constructor should raise."""
        with patch("src.evaluator._bert_score_lib", None), \
             pytest.raises(RuntimeError, match="bert-score not installed"):
            BERTScoreEvaluator()
