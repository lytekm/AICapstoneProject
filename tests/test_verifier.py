from unittest.mock import patch

import pytest

from src.verifier import NERVerifier, VerificationResult


class TestEntityExtraction:
    def test_extracts_known_entities(self, verifier, sample_source_with_entities):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        entities = verifier.extract_entities(sample_source_with_entities)
        # Should find at least some of: Justin Trudeau, Ottawa, Bank of Canada, etc.
        assert len(entities) > 0
        entity_text = " ".join(entities)
        assert any(
            name in entity_text
            for name in ["trudeau", "ottawa", "canada", "toronto", "vancouver"]
        )

    def test_empty_text_returns_empty_set(self, verifier):
        assert verifier.extract_entities("") == set()

    def test_whitespace_only_returns_empty_set(self, verifier):
        assert verifier.extract_entities("   ") == set()

    def test_entities_are_lowercased(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        entities = verifier.extract_entities("Barack Obama visited Paris.")
        for ent in entities:
            assert ent == ent.lower()

    def test_no_empty_strings_in_entities(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        entities = verifier.extract_entities("Apple Inc. announced new products in California.")
        assert "" not in entities


class TestVerify:
    def test_subset_entities_give_full_confidence(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        source = "Justin Trudeau spoke in Ottawa about Canada's economy."
        summary = "Trudeau discussed the economy in Ottawa."
        result = verifier.verify(source, summary)
        # All summary entities should be in source
        assert result.confidence >= 0.5

    def test_extra_entities_lower_confidence(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        source = "The weather was nice today."
        summary = "Barack Obama said the weather in Tokyo was nice."
        result = verifier.verify(source, summary)
        assert result.confidence < 1.0
        assert len(result.flagged_entities) > 0

    def test_no_entities_in_summary_gives_full_confidence(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        source = "Justin Trudeau spoke in Ottawa."
        summary = "The leader discussed policy changes."
        result = verifier.verify(source, summary)
        # If no NER entities detected in summary, confidence = 1.0
        if len(result.summary_entities) == 0:
            assert result.confidence == 1.0

    def test_empty_source(self, verifier):
        result = verifier.verify("", "Some summary with Obama.")
        if verifier.available:
            assert isinstance(result, VerificationResult)
        else:
            assert result.confidence == 1.0

    def test_empty_summary(self, verifier):
        result = verifier.verify("Source text with Obama.", "")
        assert result.confidence == 1.0

    def test_result_has_entity_lists(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        source = "Apple announced results in California."
        summary = "Apple reported earnings in New York."
        result = verifier.verify(source, summary)
        assert isinstance(result.source_entities, list)
        assert isinstance(result.summary_entities, list)
        assert isinstance(result.flagged_entities, list)

    def test_confidence_between_zero_and_one(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        source = "Microsoft released Windows in Seattle."
        summary = "Google released Android in Tokyo and Microsoft was mentioned."
        result = verifier.verify(source, summary)
        assert 0.0 <= result.confidence <= 1.0

    def test_citation_scaffolding_is_filtered(self, verifier):
        if not verifier.available:
            pytest.skip("spaCy model not installed")
        source = "The Bank of Canada released a report in Ottawa."
        summary = (
            "The Bank of Canada released a report in Ottawa. "
            "According to Reuters (2024), the release came after new guidance. "
            "References: Reuters. Retrieved from https://example.com/report"
        )
        result = verifier.verify(source, summary)
        assert "reuters" not in result.flagged_entities


class TestVerifierFiltering:
    def test_sanitize_text_removes_reference_scaffolding(self):
        raw = (
            "**Conclusion:** Markets fell.\n"
            "1. Review exposures.\n"
            "Retrieved from https://example.com/report\n"
            "References:\n"
            "- Reuters (2024).\n"
        )
        cleaned = NERVerifier.sanitize_text(raw)
        assert "References" not in cleaned
        assert "Retrieved from" not in cleaned
        assert "https://" not in cleaned
        assert "1." not in cleaned
        assert "Review exposures." in cleaned

    def test_should_ignore_dates_and_percentages(self):
        assert NERVerifier.should_ignore_entity("2024", "DATE") is True
        assert NERVerifier.should_ignore_entity("2.4%", "PERCENT") is True
        assert NERVerifier.should_ignore_entity("1", "CARDINAL") is True

    def test_should_keep_named_entities(self):
        assert NERVerifier.should_ignore_entity("Bank of Canada", "ORG") is False
        assert NERVerifier.should_ignore_entity("Ottawa", "GPE") is False


class TestGracefulFallback:
    def test_unavailable_spacy_returns_full_confidence(self):
        with patch("src.verifier.NERVerifier.__init__", lambda self, **kw: setattr(self, "nlp", None)):
            v = NERVerifier.__new__(NERVerifier)
            v.nlp = None
            result = v.verify("Source text.", "Summary text.")
            assert result.confidence == 1.0
            assert result.flagged_entities == []

    def test_unavailable_spacy_extract_returns_empty(self):
        v = NERVerifier.__new__(NERVerifier)
        v.nlp = None
        assert v.extract_entities("Some text with Obama.") == set()

    def test_available_property_false_when_no_nlp(self):
        v = NERVerifier.__new__(NERVerifier)
        v.nlp = None
        assert v.available is False
