import numpy as np

from src.summarizer_model import SummarizerConfig, TextRankMMRSummarizer
from tests.conftest import SAMPLE_ARTICLE


class TestSplitSentences:
    def test_splits_multiple_sentences(self, summarizer):
        sents = summarizer.split_sentences(SAMPLE_ARTICLE)
        assert len(sents) >= 5

    def test_empty_input(self, summarizer):
        assert summarizer.split_sentences("") == []

    def test_whitespace_only(self, summarizer):
        assert summarizer.split_sentences("   \n\t  ") == []

    def test_filters_short_fragments(self, summarizer):
        text = "Hi. Ok. This is a proper sentence that should survive the filter."
        sents = summarizer.split_sentences(text)
        assert len(sents) == 1
        assert "proper sentence" in sents[0]


class TestPageRank:
    def test_empty_matrix(self, summarizer):
        result = summarizer.pagerank(np.array([]).reshape(0, 0))
        assert len(result) == 0

    def test_single_node(self, summarizer):
        W = np.array([[1.0]])
        result = summarizer.pagerank(W)
        assert np.isclose(result[0], 1.0)

    def test_uniform_graph(self, summarizer):
        W = np.ones((3, 3)) - np.eye(3)
        result = summarizer.pagerank(W)
        assert len(result) == 3
        assert np.allclose(result, result[0])  # all equal for symmetric graph

    def test_scores_sum_to_one(self, summarizer):
        W = np.random.RandomState(42).rand(5, 5)
        np.fill_diagonal(W, 0)
        result = summarizer.pagerank(W)
        assert np.isclose(result.sum(), 1.0, atol=1e-6)


class TestTextRankScores:
    def test_returns_scores_matching_matrix_size(self, summarizer):
        sim = np.array([[1.0, 0.5, 0.2],
                        [0.5, 1.0, 0.3],
                        [0.2, 0.3, 1.0]])
        scores = summarizer.textrank_scores(sim)
        assert len(scores) == 3

    def test_empty_matrix(self, summarizer):
        scores = summarizer.textrank_scores(np.array([]).reshape(0, 0))
        assert len(scores) == 0

    def test_respects_min_edge(self, summarizer):
        sim = np.array([[1.0, 0.05], [0.05, 1.0]])
        scores = summarizer.textrank_scores(sim, min_edge=0.1)
        # edges below threshold are zeroed, so both nodes are dangling
        assert len(scores) == 2


class TestMMRSelect:
    def test_selects_k_items(self, summarizer):
        rel = np.array([0.9, 0.7, 0.5, 0.3])
        sim = np.eye(4)
        selected = summarizer.mmr_select(rel, sim, k=2)
        assert len(selected) == 2

    def test_first_selected_is_most_relevant(self, summarizer):
        rel = np.array([0.1, 0.9, 0.5])
        sim = np.eye(3)
        selected = summarizer.mmr_select(rel, sim, k=1)
        assert selected[0] == 1

    def test_k_larger_than_n(self, summarizer):
        rel = np.array([0.5, 0.3])
        sim = np.eye(2)
        selected = summarizer.mmr_select(rel, sim, k=10)
        assert len(selected) == 2

    def test_k_zero(self, summarizer):
        rel = np.array([0.5])
        sim = np.eye(1)
        assert summarizer.mmr_select(rel, sim, k=0) == []

    def test_diverse_selection(self, summarizer):
        rel = np.array([0.8, 0.79, 0.1])
        sim = np.array([[1.0, 0.95, 0.1],
                        [0.95, 1.0, 0.1],
                        [0.1, 0.1, 1.0]])
        selected = summarizer.mmr_select(rel, sim, k=2, lam=0.5)
        # with lam=0.5, MMR should prefer diversity over pure relevance
        # so it should pick 0 first (most relevant), then 2 (most diverse)
        assert selected[0] == 0
        assert 2 in selected


class TestSummarize:
    def test_returns_expected_keys(self, summarizer, sample_article):
        result = summarizer.summarize(sample_article, k=3)
        assert "summary" in result
        assert "sentences" in result
        assert "selected_indices" in result
        assert "scores" in result

    def test_summary_not_empty(self, summarizer, sample_article):
        result = summarizer.summarize(sample_article, k=3)
        assert len(result["summary"]) > 0

    def test_selected_count_matches_k(self, summarizer, sample_article):
        result = summarizer.summarize(sample_article, k=3)
        assert len(result["selected_indices"]) == 3

    def test_indices_are_sorted(self, summarizer, sample_article):
        result = summarizer.summarize(sample_article, k=4)
        indices = result["selected_indices"]
        assert indices == sorted(indices)

    def test_empty_article(self, summarizer):
        result = summarizer.summarize("")
        assert result["summary"] == ""
        assert result["sentences"] == []

    def test_single_sentence(self, summarizer):
        text = "This is a single sentence that is definitely long enough to not be filtered out by the minimum length check."
        result = summarizer.summarize(text, k=3)
        assert len(result["selected_indices"]) == 1

    def test_custom_config(self):
        cfg = SummarizerConfig(mmr_lambda=0.5, blend_alpha=0.9)
        s = TextRankMMRSummarizer(cfg)
        result = s.summarize(SAMPLE_ARTICLE, k=2)
        assert len(result["selected_indices"]) == 2

    def test_scores_length_matches_sentences(self, summarizer, sample_article):
        result = summarizer.summarize(sample_article, k=3)
        assert len(result["scores"]) == len(result["sentences"])
