import pytest

from src.personas import (
    ACADEMIC,
    CASUAL,
    DEFAULT,
    EXECUTIVE,
    PERSONAS,
    TECHNICAL,
    Persona,
    format_prompt,
    get_persona,
)


class TestPersonaDefinitions:
    def test_all_four_personas_exist(self):
        assert "technical" in PERSONAS
        assert "casual" in PERSONAS
        assert "executive" in PERSONAS
        assert "academic" in PERSONAS

    def test_default_exists(self):
        assert "default" in PERSONAS
        assert DEFAULT is CASUAL

    def test_persona_is_frozen_dataclass(self):
        assert isinstance(TECHNICAL, Persona)
        with pytest.raises(AttributeError):
            TECHNICAL.name = "changed"  # type: ignore[misc]

    def test_all_personas_have_required_fields(self):
        for name, persona in PERSONAS.items():
            assert persona.name, f"{name} missing name"
            assert persona.system_prompt, f"{name} missing system_prompt"
            assert persona.style_instructions, f"{name} missing style_instructions"
            assert persona.max_tokens_hint > 0, f"{name} has invalid max_tokens_hint"

    def test_persona_names_match_keys(self):
        for key, persona in PERSONAS.items():
            if key == "default":
                continue
            assert persona.name == key


class TestGetPersona:
    def test_valid_name(self):
        assert get_persona("technical") is TECHNICAL
        assert get_persona("casual") is CASUAL
        assert get_persona("executive") is EXECUTIVE
        assert get_persona("academic") is ACADEMIC

    def test_case_insensitive(self):
        assert get_persona("TECHNICAL") is TECHNICAL
        assert get_persona("Casual") is CASUAL

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown persona"):
            get_persona("nonexistent")

    def test_default_returns_casual(self):
        assert get_persona("default") is CASUAL


class TestFormatPrompt:
    def test_contains_sentences(self):
        sentences = ["First sentence.", "Second sentence."]
        prompt = format_prompt(TECHNICAL, sentences)
        assert "First sentence." in prompt
        assert "Second sentence." in prompt

    def test_brief_length(self):
        prompt = format_prompt(CASUAL, ["A sentence."], length="brief")
        expected_tokens = int(CASUAL.max_tokens_hint * 0.5)
        assert str(expected_tokens) in prompt

    def test_standard_length(self):
        prompt = format_prompt(CASUAL, ["A sentence."], length="standard")
        assert str(CASUAL.max_tokens_hint) in prompt

    def test_detailed_length(self):
        prompt = format_prompt(CASUAL, ["A sentence."], length="detailed")
        expected_tokens = int(CASUAL.max_tokens_hint * 2.0)
        assert str(expected_tokens) in prompt

    def test_includes_style_instructions(self):
        prompt = format_prompt(EXECUTIVE, ["A sentence."])
        assert "bullet" in prompt.lower()

    def test_empty_sentences_handled(self):
        prompt = format_prompt(CASUAL, [])
        assert "Summarize" in prompt

    def test_whitespace_only_sentences_filtered(self):
        prompt = format_prompt(CASUAL, ["Real sentence.", "  ", ""])
        assert "Real sentence." in prompt
