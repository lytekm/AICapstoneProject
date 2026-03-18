
from src.summarizer_model import SummarizerConfig
from src.trainer import SummarizerTrainer

TRAIN_TEXTS = [
    (
        "The Bank of Canada held its key interest rate steady at 4.5 percent on Wednesday. "
        "Governor Tiff Macklem said the economy is evolving broadly in line with projections. "
        "Inflation has come down significantly from its peak of 8.1 percent last summer. "
        "However, core inflation measures have not shown sustained decline. "
        "The central bank remains prepared to raise rates further if needed."
    ),
    (
        "Apple reported record quarterly revenue of $117 billion for the holiday quarter. "
        "iPhone sales accounted for more than half of the total revenue. "
        "Services revenue hit an all-time high of $23 billion. "
        "CEO Tim Cook highlighted strong demand across all product categories. "
        "The company returned over $25 billion to shareholders during the quarter."
    ),
]

TRAIN_REFS = [
    "The Bank of Canada held rates steady. Core inflation hasn't declined sustainably.",
    "Apple reported record $117B revenue. iPhone and services drove growth.",
]


class TestSummarizerTrainer:
    def test_tune_returns_config_and_metrics(self):
        trainer = SummarizerTrainer(metric="rougeL_f")
        search_space = {
            "mmr_lambda": [0.6, 0.75],
            "blend_alpha": [0.5, 0.7],
        }
        cfg, metrics = trainer.tune(TRAIN_TEXTS, TRAIN_REFS, k=2, search_space=search_space)

        assert isinstance(cfg, SummarizerConfig)
        assert "rougeL_f" in metrics
        assert "rouge1_f" in metrics
        assert "rouge2_f" in metrics

    def test_tune_picks_best_config(self):
        trainer = SummarizerTrainer(metric="rouge1_f")
        search_space = {
            "mmr_lambda": [0.5, 0.75, 0.9],
            "blend_alpha": [0.5, 0.9],
        }
        cfg, metrics = trainer.tune(TRAIN_TEXTS, TRAIN_REFS, k=2, search_space=search_space)

        assert cfg.mmr_lambda in [0.5, 0.75, 0.9]
        assert cfg.blend_alpha in [0.5, 0.9]
        assert metrics["rouge1_f"] > 0

    def test_tune_default_search_space(self):
        trainer = SummarizerTrainer()
        cfg, metrics = trainer.tune(TRAIN_TEXTS, TRAIN_REFS, k=2)

        assert isinstance(cfg, SummarizerConfig)
        assert metrics["rougeL_f"] >= 0

    def test_save_best(self, tmp_path):
        trainer = SummarizerTrainer()
        cfg = SummarizerConfig(mmr_lambda=0.75, blend_alpha=0.7)
        metrics = {"rouge1_f": 0.45, "rouge2_f": 0.20, "rougeL_f": 0.40}

        path = str(tmp_path / "best.json")
        trainer.save_best(cfg, metrics, path)

        import json
        with open(path) as f:
            data = json.load(f)

        assert data["best_config"]["mmr_lambda"] == 0.75
        assert data["metrics"]["rouge1_f"] == 0.45

    def test_custom_metric(self):
        trainer = SummarizerTrainer(metric="rouge2_f")
        search_space = {"mmr_lambda": [0.6, 0.8]}
        cfg, metrics = trainer.tune(TRAIN_TEXTS, TRAIN_REFS, k=2, search_space=search_space)

        assert "rouge2_f" in metrics
