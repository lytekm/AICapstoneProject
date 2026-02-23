from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import feedparser
import urllib.request
import trafilatura

from main import summarize_textrank_mmr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RSS = "https://www.cbc.ca/cmlink/rss-business"

@app.get("/api/articles")
def get_articles():
    request = urllib.request.Request(RSS, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request) as response:
        feed = feedparser.parse(response.read())

    return [
        {"title": e.title, "link": e.link}
        for e in feed.entries[:10]
    ]


@app.post("/api/summarize")
def summarize(data: dict):
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body.")

    # 1) Download HTML with a browser-like User-Agent (CBC often needs this)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download article: {e}")

    # 2) Extract main text
    try:
        text = trafilatura.extract(html)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction failed: {e}")

    if not text or len(text.strip()) < 200:
        raise HTTPException(status_code=400, detail="Could not extract enough article text from this URL.")

    # 3) Summarize
    try:
        result = summarize_textrank_mmr(text, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarizer crashed: {e}")

    return {"summary": result["summary"]}