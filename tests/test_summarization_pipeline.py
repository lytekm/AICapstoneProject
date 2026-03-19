import pytest

from src.abstractor import MockAbstractor
from src.pipeline import PipelineResult, SummarizationPipeline
from src.verifier import NERVerifier
from tests.conftest import SAMPLE_ARTICLE


class TestExtractiveMode:
    def test_returns_pipeline_result(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive")
        assert isinstance(result, PipelineResult)

    def test_summary_not_empty(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive")
        assert len(result.summary.strip()) > 0

    def test_confidence_is_one(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive")
        assert result.confidence == 1.0

    def test_mode_is_extractive(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive")
        assert result.mode == "extractive"

    def test_persona_stored(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive", persona="technical")
        assert result.persona == "technical"

    def test_extractive_sentences_populated(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive", k=3)
        assert len(result.extractive_sentences) > 0


class TestAbstractiveMode:
    def test_calls_abstractor(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock)
        result = pipe.run(SAMPLE_ARTICLE, mode="abstractive")
        assert "[Mock Summary]" in result.summary

    def test_persona_affects_prompt(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock)
        r1 = pipe.run(SAMPLE_ARTICLE, mode="abstractive", persona="technical")
        r2 = pipe.run(SAMPLE_ARTICLE, mode="abstractive", persona="executive")
        assert r1.persona == "technical"
        assert r2.persona == "executive"

    def test_mode_is_abstractive(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock)
        result = pipe.run(SAMPLE_ARTICLE, mode="abstractive")
        assert result.mode == "abstractive"

    def test_extractive_sentences_populated(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock)
        result = pipe.run(SAMPLE_ARTICLE, mode="abstractive", k=3)
        assert len(result.extractive_sentences) > 0


class TestHybridMode:
    def test_all_three_stages_run(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock)
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        assert "[Mock Summary]" in result.summary
        assert isinstance(result.confidence, float)

    def test_confidence_from_verifier(self):
        mock = MockAbstractor()
        verifier = NERVerifier()
        pipe = SummarizationPipeline(abstractor=mock, verifier=verifier)
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        assert 0.0 <= result.confidence <= 1.0

    def test_flagged_entities_is_list(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock)
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        assert isinstance(result.flagged_entities, list)

    def test_extractive_sentences_populated(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock)
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid", k=3)
        assert len(result.extractive_sentences) > 0


class TestErrorHandling:
    def test_abstractor_failure_falls_back(self):
        class FailingAbstractor(MockAbstractor):
            def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
                raise RuntimeError("LLM unavailable")

        pipe = SummarizationPipeline(abstractor=FailingAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        # Should fall back to extractive summary
        assert len(result.summary.strip()) > 0
        assert "abstractor_error" in result.metadata

    def test_abstractor_failure_in_abstractive_mode(self):
        class FailingAbstractor(MockAbstractor):
            def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
                raise RuntimeError("LLM unavailable")

        pipe = SummarizationPipeline(abstractor=FailingAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="abstractive")
        assert len(result.summary.strip()) > 0
        assert result.metadata.get("fallback") == "extractive"

    def test_invalid_mode_raises(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        with pytest.raises(ValueError, match="Unknown mode"):
            pipe.run(SAMPLE_ARTICLE, mode="invalid")

    def test_invalid_persona_raises(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        with pytest.raises(ValueError, match="Unknown persona"):
            pipe.run(SAMPLE_ARTICLE, mode="extractive", persona="nonexistent")

    def test_invalid_length_raises(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        with pytest.raises(ValueError, match="Unknown length"):
            pipe.run(SAMPLE_ARTICLE, mode="extractive", length="invalid")


class TestDefaults:
    def test_default_mode_is_extractive(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE)
        assert result.mode == "extractive"

    def test_default_persona_is_default(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE)
        assert result.persona == "default"
