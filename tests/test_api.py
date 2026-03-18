from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import _parse_k, app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestParseK:
    def test_default_value(self):
        assert _parse_k(None) == 5

    def test_valid_int(self):
        assert _parse_k(3) == 3

    def test_string_number(self):
        assert _parse_k("7") == 7

    def test_clamp_min(self):
        assert _parse_k(0) == 1

    def test_clamp_max(self):
        assert _parse_k(100) == 20

    def test_invalid_value(self):
        assert _parse_k("abc") == 5


class TestArticlesEndpoint:
    @patch("api.urllib.request.urlopen")
    def test_returns_articles(self, mock_urlopen, client):
        rss_xml = b"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                </item>
            </channel>
        </rss>"""
        mock_resp = MagicMock()
        mock_resp.read.return_value = rss_xml
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        resp = client.get("/api/articles")
        assert resp.status_code == 200
        articles = resp.json()
        assert len(articles) >= 1
        assert "title" in articles[0]
        assert "link" in articles[0]


class TestSummarizeEndpoint:
    def test_missing_url(self, client):
        resp = client.post("/api/summarize", json={})
        assert resp.status_code == 400

    def test_empty_url(self, client):
        resp = client.post("/api/summarize", json={"url": ""})
        assert resp.status_code == 400

    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_successful_summarize(self, mock_fetch, mock_extract, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = (
            "The Bank of Canada held its key interest rate steady at 4.5 percent. "
            "Governor Tiff Macklem said the economy is evolving broadly in line with projections. "
            "Inflation has come down significantly from its peak last summer. "
            "However, core inflation measures have not shown sustained decline. "
            "The central bank remains prepared to raise rates further if needed. "
            "Financial markets reacted positively to the announcement."
        )

        resp = client.post("/api/summarize", json={"url": "https://example.com/article", "k": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert len(data["summary"]) > 0

    @patch("api._fetch_url")
    def test_bad_url_returns_400(self, mock_fetch, client):
        from fastapi import HTTPException
        mock_fetch.side_effect = HTTPException(status_code=400, detail="Could not download")
        resp = client.post("/api/summarize", json={"url": "https://bad-url.example.com"})
        assert resp.status_code == 400
