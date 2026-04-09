from unittest.mock import patch

from src.abstractor import (
    Abstractor,
    AbstractorConfig,
    MockAbstractor,
    create_abstractor,
)


class TestMockAbstractor:
    def test_returns_non_empty(self, mock_abstractor):
        result = mock_abstractor.generate("system", "- First sentence.\n- Second.")
        assert len(result) > 0

    def test_contains_mock_marker(self, mock_abstractor):
        result = mock_abstractor.generate("system", "- A sentence.")
        assert "[Mock Summary]" in result

    def test_respects_sentence_budget(self, mock_abstractor):
        prompt = "- One.\n- Two.\n- Three.\n- Four.\n- Five."
        result = mock_abstractor.generate("system", prompt, max_tokens=512)
        assert "One." in result
        assert "Two." in result
        assert "Three." in result
        assert "Four." not in result

    def test_detailed_budget_can_include_four_sentences(self, mock_abstractor):
        prompt = "- One.\n- Two.\n- Three.\n- Four.\n- Five."
        result = mock_abstractor.generate("system", prompt, max_tokens=1024)
        assert "Four." in result
        assert "Five." not in result

    def test_handles_empty_prompt(self, mock_abstractor):
        result = mock_abstractor.generate("system", "")
        assert "[Mock Summary]" in result
        assert "No content provided." in result

    def test_ignores_summarize_prefix_lines(self, mock_abstractor):
        prompt = "Summarize the following:\n- Actual content."
        result = mock_abstractor.generate("system", prompt)
        assert "Actual content." in result
        assert "Summarize" not in result.replace("[Mock Summary]", "")

    def test_deterministic(self, mock_abstractor):
        prompt = "- Sentence A.\n- Sentence B."
        r1 = mock_abstractor.generate("sys", prompt)
        r2 = mock_abstractor.generate("sys", prompt)
        assert r1 == r2

    def test_persona_style_affects_mock_output(self, mock_abstractor):
        prompt = "- One.\n- Two."
        technical = mock_abstractor.generate(
            "You are a technical writer.", prompt, max_tokens=256
        )
        executive = mock_abstractor.generate(
            "You are a business analyst.", prompt, max_tokens=256
        )
        casual = mock_abstractor.generate(
            "You are a friendly writer.", prompt, max_tokens=256
        )
        academic = mock_abstractor.generate(
            "You are an academic researcher.", prompt, max_tokens=256
        )

        assert "Technical Summary:" in technical
        assert "Executive Brief:" in executive
        assert "Plain-language Summary:" in casual
        assert "Academic Summary:" in academic

    def test_length_budget_affects_mock_output(self, mock_abstractor):
        prompt = "- One.\n- Two.\n- Three.\n- Four."
        brief = mock_abstractor.generate("You are a friendly writer.", prompt, max_tokens=128)
        detailed = mock_abstractor.generate("You are a friendly writer.", prompt, max_tokens=1024)

        assert "One." in brief
        assert "Two." not in brief
        assert "Four." in detailed


class TestAbstractorConfig:
    def test_default_base_url_from_env(self):
        with patch.dict("os.environ", {"VLLM_BASE_URL": "http://test:9000/v1"}):
            config = AbstractorConfig()
            assert config.base_url == "http://test:9000/v1"

    def test_default_model_from_env(self):
        with patch.dict("os.environ", {"VLLM_MODEL": "test-model"}):
            config = AbstractorConfig()
            assert config.model == "test-model"

    def test_default_base_url_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            config = AbstractorConfig()
            assert "localhost" in config.base_url

    def test_explicit_base_url(self):
        config = AbstractorConfig(base_url="http://custom:5000/v1")
        assert config.base_url == "http://custom:5000/v1"

    def test_default_temperature(self):
        config = AbstractorConfig()
        assert config.temperature == 0.3


class TestCreateAbstractorFactory:
    def test_mock_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            ab = create_abstractor()
            assert isinstance(ab, MockAbstractor)

    def test_explicit_use_mock_true(self):
        ab = create_abstractor(use_mock=True)
        assert isinstance(ab, MockAbstractor)

    def test_explicit_use_mock_false_requires_openai(self):
        # This may raise RuntimeError if openai not installed, which is fine
        try:
            ab = create_abstractor(use_mock=False)
            assert isinstance(ab, Abstractor)
        except RuntimeError:
            pass

    def test_env_var_false_creates_real(self):
        with patch.dict("os.environ", {"USE_MOCK_LLM": "0"}):
            try:
                ab = create_abstractor()
                assert isinstance(ab, Abstractor)
            except RuntimeError:
                pass

    def test_env_var_true_creates_mock(self):
        with patch.dict("os.environ", {"USE_MOCK_LLM": "1"}):
            ab = create_abstractor()
            assert isinstance(ab, MockAbstractor)


class TestMockAbstractorStream:
    def test_generate_stream_returns_iterator(self, mock_abstractor):
        tokens = list(mock_abstractor.generate_stream("sys", "- Hello world.", delay=0))
        assert len(tokens) > 0

    def test_joined_tokens_match_generate(self, mock_abstractor):
        prompt = "- First sentence.\n- Second sentence."
        full = mock_abstractor.generate("sys", prompt)
        tokens = list(mock_abstractor.generate_stream("sys", prompt, delay=0))
        joined = "".join(tokens).strip()
        assert joined == full

    def test_each_token_ends_with_space(self, mock_abstractor):
        tokens = list(mock_abstractor.generate_stream("sys", "- Content here.", delay=0))
        for token in tokens:
            assert token.endswith(" ")

    def test_empty_prompt_streams(self, mock_abstractor):
        tokens = list(mock_abstractor.generate_stream("sys", "", delay=0))
        joined = "".join(tokens).strip()
        assert "[Mock Summary]" in joined
