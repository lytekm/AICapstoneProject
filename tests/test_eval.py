"""Tests for the evaluation script."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from eval.run_eval import format_table, main, run_evaluation


class TestRunEvaluation:
    @patch("eval.run_eval.RougeEvaluator")
    @patch("eval.run_eval.TextRankMMRSummarizer")
    @patch("eval.run_eval.CNNDailyMailDatasetLoader")
    def test_returns_dict_with_expected_keys(self, mock_loader_cls, mock_model_cls, mock_eval_cls):
        # set up a tiny fake dataset batch
        batch = MagicMock()
        batch.texts = ["Article one content here. Second sentence here."]
        batch.references = ["Reference summary."]
        mock_loader_cls.return_value.load.return_value = batch

        mock_model_cls.return_value.summarize.return_value = {"summary": "Fake summary."}

        mock_eval_cls.return_value.evaluate.return_value = {
            "rouge1_f": 0.35,
            "rouge2_f": 0.12,
            "rougeL_f": 0.30,
        }

        result = run_evaluation(samples=1, seed=0, k=3)

        assert "timestamp" in result
        assert result["samples"] == 1
        assert result["seed"] == 0
        assert result["k"] == 3
        assert "scores" in result
        assert result["scores"]["rouge1_f"] == 0.35

    @patch("eval.run_eval.RougeEvaluator")
    @patch("eval.run_eval.TextRankMMRSummarizer")
    @patch("eval.run_eval.CNNDailyMailDatasetLoader")
    def test_calls_loader_with_correct_params(self, mock_loader_cls, mock_model_cls, mock_eval_cls):
        batch = MagicMock()
        batch.texts = ["Text."]
        batch.references = ["Ref."]
        mock_loader_cls.return_value.load.return_value = batch
        mock_model_cls.return_value.summarize.return_value = {"summary": "S."}
        mock_eval_cls.return_value.evaluate.return_value = {}

        run_evaluation(samples=10, seed=99, k=2)

        mock_loader_cls.return_value.load.assert_called_once_with(
            split="test", limit=10, shuffle=True, seed=99
        )

    @patch("eval.run_eval.RougeEvaluator")
    @patch("eval.run_eval.TextRankMMRSummarizer")
    @patch("eval.run_eval.CNNDailyMailDatasetLoader")
    def test_generates_predictions_for_each_text(self, mock_loader_cls, mock_model_cls, mock_eval_cls):
        batch = MagicMock()
        batch.texts = ["A.", "B.", "C."]
        batch.references = ["Ra.", "Rb.", "Rc."]
        mock_loader_cls.return_value.load.return_value = batch
        mock_model_cls.return_value.summarize.return_value = {"summary": "S."}
        mock_eval_cls.return_value.evaluate.return_value = {}

        run_evaluation(samples=3, seed=0, k=5)

        assert mock_model_cls.return_value.summarize.call_count == 3


class TestFormatTable:
    def test_includes_rouge_labels(self):
        scores = {"rouge1_f": 0.351, "rouge2_f": 0.123, "rougeL_f": 0.298}
        table = format_table(scores)
        assert "ROUGE-1 F1" in table
        assert "ROUGE-2 F1" in table
        assert "ROUGE-L F1" in table

    def test_formats_values_to_three_decimals(self):
        scores = {"rouge1_f": 0.351, "rouge2_f": 0.1, "rougeL_f": 0.0}
        table = format_table(scores)
        assert "0.351" in table
        assert "0.100" in table
        assert "0.000" in table

    def test_missing_keys_default_to_zero(self):
        scores = {}
        table = format_table(scores)
        assert "0.000" in table


class TestMainCLI:
    @patch("eval.run_eval.run_evaluation")
    def test_writes_output_json(self, mock_run, tmp_path):
        mock_run.return_value = {
            "timestamp": "2026-01-01T00:00:00",
            "samples": 5,
            "seed": 42,
            "k": 5,
            "model": "test",
            "scores": {"rouge1_f": 0.3, "rouge2_f": 0.1, "rougeL_f": 0.25},
        }

        out_file = tmp_path / "test_output.json"
        result = main(["--samples", "5", "--output", str(out_file)])

        assert out_file.exists()
        assert result["samples"] == 5

    @patch("eval.run_eval.run_evaluation")
    def test_default_output_path(self, mock_run, tmp_path, monkeypatch):
        mock_run.return_value = {
            "timestamp": "2026-01-01T00:00:00",
            "samples": 50,
            "seed": 42,
            "k": 5,
            "model": "test",
            "scores": {},
        }

        # run with default args but redirect cwd so it writes to tmp
        monkeypatch.chdir(tmp_path)
        main(["--samples", "1"])

        # should have created eval/results/ directory
        assert (tmp_path / "eval" / "results").exists()
