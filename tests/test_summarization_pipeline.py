import pytest

from src.abstractor import MockAbstractor
from src.pipeline import PipelineResult, SummarizationPipeline
from src.verifier import VerificationResult
from tests.conftest import SAMPLE_ARTICLE


class AvailableVerifier:
    available = True

    def __init__(
        self,
        confidence: float = 0.73,
        flagged_entities: list[str] | None = None,
    ) -> None:
        self.confidence = confidence
        self.flagged_entities = flagged_entities or ["hallucinated entity"]

    def verify(self, source_text: str, summary_text: str) -> VerificationResult:
        return VerificationResult(
            confidence=self.confidence,
            flagged_entities=list(self.flagged_entities),
            source_entities=["source"],
            summary_entities=["summary"],
        )


class UnavailableVerifier:
    available = False

    def verify(self, source_text: str, summary_text: str) -> VerificationResult:
        raise AssertionError("verify() should not be called when the verifier is unavailable")


class TestExtractiveMode:
    def test_returns_pipeline_result(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive")
        assert isinstance(result, PipelineResult)

    def test_summary_not_empty(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive")
        assert len(result.summary.strip()) > 0

    def test_confidence_is_none(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        result = pipe.run(SAMPLE_ARTICLE, mode="extractive")
        assert result.confidence is None

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

    def test_length_scales_effective_k(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        brief = pipe.run(SAMPLE_ARTICLE, mode="extractive", length="brief", k=4)
        detailed = pipe.run(SAMPLE_ARTICLE, mode="extractive", length="detailed", k=4)

        assert brief.metadata["effective_k"] == 2
        assert detailed.metadata["effective_k"] == 8
        assert len(detailed.extractive_sentences) >= len(brief.extractive_sentences)


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

    def test_persona_changes_mock_output(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        technical = pipe.run(SAMPLE_ARTICLE, mode="abstractive", persona="technical")
        executive = pipe.run(SAMPLE_ARTICLE, mode="abstractive", persona="executive")

        assert "Technical Summary:" in technical.summary
        assert "Executive Brief:" in executive.summary
        assert technical.summary != executive.summary

    def test_length_changes_mock_output(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        brief = pipe.run(SAMPLE_ARTICLE, mode="abstractive", persona="casual", length="brief")
        detailed = pipe.run(SAMPLE_ARTICLE, mode="abstractive", persona="casual", length="detailed")

        assert brief.summary != detailed.summary

    def test_confidence_from_verifier_when_available(self):
        pipe = SummarizationPipeline(
            abstractor=MockAbstractor(),
            verifier=AvailableVerifier(confidence=0.61, flagged_entities=["year"]),
        )
        result = pipe.run(SAMPLE_ARTICLE, mode="abstractive")

        assert result.confidence == pytest.approx(0.61)
        assert result.flagged_entities == ["year"]

    def test_confidence_null_when_verifier_unavailable(self):
        pipe = SummarizationPipeline(
            abstractor=MockAbstractor(),
            verifier=UnavailableVerifier(),
        )
        result = pipe.run(SAMPLE_ARTICLE, mode="abstractive")

        assert result.confidence is None
        assert result.flagged_entities == []


class TestHybridMode:
    def test_all_three_stages_run(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock, verifier=AvailableVerifier())
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        assert "[Mock Summary]" in result.summary
        assert isinstance(result.confidence, float)

    def test_confidence_from_verifier(self):
        mock = MockAbstractor()
        verifier = AvailableVerifier(confidence=0.42, flagged_entities=["x"])
        pipe = SummarizationPipeline(abstractor=mock, verifier=verifier)
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        assert result.confidence == pytest.approx(0.42)
        assert result.flagged_entities == ["x"]

    def test_flagged_entities_is_list(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock, verifier=AvailableVerifier())
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        assert isinstance(result.flagged_entities, list)

    def test_extractive_sentences_populated(self):
        mock = MockAbstractor()
        pipe = SummarizationPipeline(abstractor=mock, verifier=AvailableVerifier())
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid", k=3)
        assert len(result.extractive_sentences) > 0


class TestErrorHandling:
    def test_abstractor_failure_falls_back(self):
        class FailingAbstractor(MockAbstractor):
            def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
                raise RuntimeError("LLM unavailable")

        pipe = SummarizationPipeline(abstractor=FailingAbstractor(), verifier=AvailableVerifier())
        result = pipe.run(SAMPLE_ARTICLE, mode="hybrid")
        # Should fall back to extractive summary
        assert len(result.summary.strip()) > 0
        assert "abstractor_error" in result.metadata
        assert result.confidence is None

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


class TestStreamExtractiveMode:
    def test_yields_single_done_event(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="extractive"))
        assert len(events) == 1
        assert "event: done" in events[0]

    def test_done_event_contains_summary(self):
        import json
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="extractive"))
        # parse the SSE data line
        data_line = events[0].split("data: ")[1].split("\n")[0]
        data = json.loads(data_line)
        assert len(data["summary"]) > 0
        assert data["mode"] == "extractive"

    def test_done_event_contains_effective_k(self):
        import json
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="extractive", length="detailed", k=4))
        data_line = events[0].split("data: ")[1].split("\n")[0]
        data = json.loads(data_line)
        assert data["effective_k"] == 8


class TestStreamHybridMode:
    def test_yields_meta_tokens_done(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor(), verifier=AvailableVerifier())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="hybrid", delay=0))
        event_types = [e.split("event: ")[1].split("\n")[0] for e in events]
        assert event_types[0] == "meta"
        assert "token" in event_types
        assert event_types[-1] == "done"

    def test_done_event_has_confidence(self):
        import json
        pipe = SummarizationPipeline(abstractor=MockAbstractor(), verifier=AvailableVerifier(confidence=0.58))
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="hybrid", delay=0))
        done_event = events[-1]
        data_line = done_event.split("data: ")[1].split("\n")[0]
        data = json.loads(data_line)
        assert "confidence" in data
        assert data["confidence"] == pytest.approx(0.58)

    def test_done_event_has_null_confidence_without_verifier(self):
        import json
        pipe = SummarizationPipeline(abstractor=MockAbstractor(), verifier=UnavailableVerifier())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="hybrid", delay=0))
        done_event = events[-1]
        data_line = done_event.split("data: ")[1].split("\n")[0]
        data = json.loads(data_line)
        assert data["confidence"] is None
        assert data["flagged_entities"] == []

    def test_token_events_build_summary(self):
        import json
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="hybrid", delay=0))
        tokens = []
        for e in events:
            if e.startswith("event: token"):
                data_line = e.split("data: ")[1].split("\n")[0]
                tokens.append(json.loads(data_line)["text"])
        full = "".join(tokens)
        assert len(full.strip()) > 0


class TestStreamAbstractiveMode:
    def test_yields_meta_tokens_done(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="abstractive", delay=0))
        event_types = [e.split("event: ")[1].split("\n")[0] for e in events]
        assert event_types[0] == "meta"
        assert event_types[-1] == "done"

    def test_done_event_has_confidence_when_verifier_available(self):
        import json
        pipe = SummarizationPipeline(
            abstractor=MockAbstractor(),
            verifier=AvailableVerifier(confidence=0.67, flagged_entities=["source"]),
        )
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="abstractive", delay=0))
        done_event = events[-1]
        data_line = done_event.split("data: ")[1].split("\n")[0]
        data = json.loads(data_line)
        assert data["confidence"] == pytest.approx(0.67)
        assert data["flagged_entities"] == ["source"]

    def test_done_event_has_null_confidence_without_verifier(self):
        import json
        pipe = SummarizationPipeline(
            abstractor=MockAbstractor(),
            verifier=UnavailableVerifier(),
        )
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="abstractive", delay=0))
        done_event = events[-1]
        data_line = done_event.split("data: ")[1].split("\n")[0]
        data = json.loads(data_line)
        assert data["confidence"] is None
        assert data["flagged_entities"] == []


class TestStreamErrorHandling:
    def test_invalid_mode_yields_error(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, mode="invalid"))
        assert len(events) == 1
        assert "event: error" in events[0]

    def test_invalid_persona_yields_error(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, persona="nonexistent"))
        assert len(events) == 1
        assert "event: error" in events[0]

    def test_invalid_length_yields_error(self):
        pipe = SummarizationPipeline(abstractor=MockAbstractor())
        events = list(pipe.run_stream(SAMPLE_ARTICLE, length="invalid"))
        assert len(events) == 1
        assert "event: error" in events[0]
