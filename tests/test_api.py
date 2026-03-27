from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import _parse_k, app
from src.pipeline import PipelineResult


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestCatchAllRouting:
    def test_unknown_api_path_returns_404(self, client):
        resp = client.get("/api/does-not-exist")
        assert resp.status_code == 404


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


MOCK_ARTICLE_TEXT = (
    "The Bank of Canada held its key interest rate steady at 4.5 percent. "
    "Governor Tiff Macklem said the economy is evolving broadly in line with projections. "
    "Inflation has come down significantly from its peak last summer. "
    "However, core inflation measures have not shown sustained decline. "
    "The central bank remains prepared to raise rates further if needed. "
    "Financial markets reacted positively to the announcement."
)


class TestPersonasEndpoint:
    def test_returns_persona_list(self, client):
        resp = client.get("/api/personas")
        assert resp.status_code == 200
        data = resp.json()
        assert "personas" in data
        assert "technical" in data["personas"]
        assert "casual" in data["personas"]
        assert "executive" in data["personas"]
        assert "academic" in data["personas"]


class TestSummarizeWithMode:
    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_extractive_mode(self, mock_fetch, mock_extract, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "mode": "extractive",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "extractive"
        assert data["persona"] == "default"

    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_hybrid_mode(self, mock_fetch, mock_extract, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "mode": "hybrid",
            "persona": "executive",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "hybrid"
        assert data["persona"] == "executive"
        assert "confidence" in data

    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_abstractive_mode(self, mock_fetch, mock_extract, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "mode": "abstractive",
            "persona": "casual",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "abstractive"
        assert "confidence" in data

    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_backward_compat_no_mode(self, mock_fetch, mock_extract, client):
        """Old-style request without mode/persona still works."""
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert data["mode"] == "extractive"


class TestSummarizeWithPersona:
    def test_invalid_persona_returns_400(self, client):
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "persona": "nonexistent",
        })
        assert resp.status_code == 400
        assert "Unknown persona" in resp.json()["detail"]


class TestSummarizeResponseFormat:
    @patch("api.pipeline.run")
    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_extractive_response_includes_null_metadata(self, mock_fetch, mock_extract, mock_pipeline_run, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        mock_pipeline_run.return_value = PipelineResult(
            summary="Extractive summary.",
            mode="extractive",
            persona="default",
            confidence=0.91,
            flagged_entities=["ignored"],
        )
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "mode": "extractive",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "extractive"
        assert data["confidence"] is None
        assert data["flagged_entities"] == []

    @patch("api.pipeline.run")
    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_abstractive_response_includes_null_metadata(self, mock_fetch, mock_extract, mock_pipeline_run, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        mock_pipeline_run.return_value = PipelineResult(
            summary="Abstractive summary.",
            mode="abstractive",
            persona="casual",
            confidence=0.83,
            flagged_entities=["ignored"],
        )
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "mode": "abstractive",
            "persona": "casual",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "abstractive"
        assert data["confidence"] is None
        assert data["flagged_entities"] == []

    @patch("api.pipeline.run")
    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_hybrid_response_preserves_verified_metadata(self, mock_fetch, mock_extract, mock_pipeline_run, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        mock_pipeline_run.return_value = PipelineResult(
            summary="Hybrid summary.",
            mode="hybrid",
            persona="technical",
            confidence=0.64,
            flagged_entities=["entity-a", "entity-b"],
        )
        resp = client.post("/api/summarize", json={
            "url": "https://example.com/article",
            "mode": "hybrid",
            "persona": "technical",
            "length": "brief",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "hybrid"
        assert data["confidence"] == pytest.approx(0.64)
        assert data["flagged_entities"] == ["entity-a", "entity-b"]


class TestFeedbackEndpoint:
    @patch("api.feedback_store.record")
    @patch("api.profile_store.get")
    def test_feedback_accepts_real_boolean(self, mock_profile_get, mock_record, client):
        mock_profile_get.return_value = None
        resp = client.post("/api/user/feedback", json={
            "user_id": "u1",
            "article_title": "Article",
            "persona": "default",
            "mode": "extractive",
            "liked": True,
        })
        assert resp.status_code == 200
        assert resp.json() == {"status": "recorded"}
        assert mock_record.called

    def test_feedback_rejects_string_boolean(self, client):
        resp = client.post("/api/user/feedback", json={
            "user_id": "u1",
            "article_title": "Article",
            "persona": "default",
            "mode": "extractive",
            "liked": "false",
        })
        assert resp.status_code == 422


class TestStreamEndpoint:
    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_returns_sse_content_type(self, mock_fetch, mock_extract, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.get("/api/summarize/stream", params={
            "url": "https://example.com/article",
            "mode": "extractive",
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_extractive_stream_contains_done(self, mock_fetch, mock_extract, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.get("/api/summarize/stream", params={
            "url": "https://example.com/article",
            "mode": "extractive",
        })
        assert "event: done" in resp.text

    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_hybrid_stream_has_tokens(self, mock_fetch, mock_extract, client):
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.get("/api/summarize/stream", params={
            "url": "https://example.com/article",
            "mode": "hybrid",
            "persona": "executive",
        })
        assert "event: meta" in resp.text
        assert "event: token" in resp.text
        assert "event: done" in resp.text

    def test_missing_url_returns_422(self, client):
        resp = client.get("/api/summarize/stream")
        # FastAPI returns 422 for missing required query params
        assert resp.status_code == 422

    def test_invalid_persona_returns_400(self, client):
        resp = client.get("/api/summarize/stream", params={
            "url": "https://example.com/article",
            "persona": "nonexistent",
        })
        assert resp.status_code == 400
        assert "Unknown persona" in resp.json()["detail"]

    @patch("api._extract_main_text")
    @patch("api._fetch_url")
    def test_stream_done_contains_summary(self, mock_fetch, mock_extract, client):
        import json as json_mod
        mock_fetch.return_value = "<html>article</html>"
        mock_extract.return_value = MOCK_ARTICLE_TEXT
        resp = client.get("/api/summarize/stream", params={
            "url": "https://example.com/article",
            "mode": "extractive",
        })
        # parse the done event data
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                data = json_mod.loads(line[6:])
                assert "summary" in data
                assert len(data["summary"]) > 0
                break
