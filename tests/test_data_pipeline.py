from unittest.mock import MagicMock, patch

import pytest

from src.data_pipeline import NewsDataPipeline


@pytest.fixture
def pipeline():
    return NewsDataPipeline()


class TestNormalization:
    def test_removes_cbc_junk(self, pipeline):
        text = "Search Search Sign In Quick Links News Being Black in Canada More Actual Content"
        result = pipeline.normalize_text(text)
        assert result == "Actual Content"

    def test_removes_ads(self, pipeline):
        result = pipeline.normalize_text("Actual Content Advertisement")
        assert result == "Actual Content"

    def test_removes_copyright(self, pipeline):
        result = pipeline.normalize_text("Article text. © 2024 CBC All rights reserved")
        assert "CBC" not in result
        assert "Article text." in result

    def test_removes_social_follows(self, pipeline):
        result = pipeline.normalize_text("Content. Follow us on Twitter")
        assert "Twitter" not in result
        assert "Content." in result

    def test_collapses_whitespace(self, pipeline):
        result = pipeline.normalize_text("Hello    \n\n   World")
        assert result == "Hello World"

    def test_handles_nbsp(self, pipeline):
        result = pipeline.normalize_text("Hello\u00a0World")
        assert result == "Hello World"

    def test_empty_input(self, pipeline):
        assert pipeline.normalize_text("") == ""


class TestTokenizeStats:
    def test_basic_stats(self, pipeline):
        text = "Hello world. This is a test."
        stats = pipeline.tokenize_stats(text)
        assert stats["word_count"] == 6
        assert stats["sentence_count"] == 2

    def test_empty_string(self, pipeline):
        stats = pipeline.tokenize_stats("")
        assert stats["word_count"] == 0
        assert stats["sentence_count"] == 0

    def test_single_sentence(self, pipeline):
        stats = pipeline.tokenize_stats("The quick brown fox jumps over the lazy dog.")
        assert stats["word_count"] == 9
        assert stats["sentence_count"] == 1


class TestFetchRSS:
    @patch("src.data_pipeline.urllib.request.urlopen")
    def test_parses_rss_entries(self, mock_urlopen, pipeline):
        rss_xml = b"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Article One</title>
                    <link>https://example.com/1</link>
                    <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Article Two</title>
                    <link>https://example.com/2</link>
                </item>
            </channel>
        </rss>"""
        mock_resp = MagicMock()
        mock_resp.read.return_value = rss_xml
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        entries = pipeline.fetch_rss("https://example.com/feed")
        assert len(entries) == 2
        assert entries[0]["title"] == "Article One"
        assert entries[0]["url"] == "https://example.com/1"

    @patch("src.data_pipeline.urllib.request.urlopen")
    def test_handles_network_error(self, mock_urlopen, pipeline):
        mock_urlopen.side_effect = Exception("Connection refused")
        entries = pipeline.fetch_rss("https://bad-url.example.com")
        assert entries == []

    @patch("src.data_pipeline.urllib.request.urlopen")
    def test_respects_limit(self, mock_urlopen, pipeline):
        rss_xml = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel>
            <item><title>A1</title><link>https://e.com/1</link></item>
            <item><title>A2</title><link>https://e.com/2</link></item>
            <item><title>A3</title><link>https://e.com/3</link></item>
        </channel></rss>"""
        mock_resp = MagicMock()
        mock_resp.read.return_value = rss_xml
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        entries = pipeline.fetch_rss("https://example.com/feed", limit=2)
        assert len(entries) == 2
