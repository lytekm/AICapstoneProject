from __future__ import annotations

import os
import csv
from typing import List

from .data_pipeline import NewsDataPipeline
from .summarizer_model import TextRankMMRSummarizer, SummarizerConfig
from src.diagram_generator import DiagramGenerator

def ensure_outputs():
    os.makedirs("outputs", exist_ok=True)


def main():
    ensure_outputs()

    feed_url = "https://www.cbc.ca/cmlink/rss-business"
    limit = 5
    k = 3
    DiagramGenerator().generate_all()
    pipeline = NewsDataPipeline()
    articles = pipeline.build_articles(feed_url=feed_url, limit=limit)

    model = TextRankMMRSummarizer(SummarizerConfig())

    out_path = os.path.join("outputs", "summaries.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "url",
                "published",
                "fetched_at",
                "word_count",
                "sentence_count",
                "k",
                "summary",
            ],
        )
        writer.writeheader()

        for a in articles:
            result = model.summarize(a.normalized_text, k=k)
            writer.writerow(
                {
                    "title": a.title,
                    "url": a.url,
                    "published": a.published,
                    "fetched_at": a.fetched_at,
                    "word_count": a.word_count,
                    "sentence_count": a.sentence_count,
                    "k": k,
                    "summary": result["summary"],
                }
            )

    print(f"Saved summaries to: {out_path}")
    print(f"Summarized {len(articles)} articles.")


if __name__ == "__main__":
    main()