"""
Main Entry Point: AI Capstone Project - Iteration 1
Orchestrates Multi-Source Fetching, Summarization, and Diagram Generation.
"""

import os
from src.data_pipeline import NewsDataPipeline
from src.summarizer_model import TextRankMMRSummarizer, SummarizerConfig
from src.diagram_generator import DiagramGenerator

def run_demonstration():
    # 1. Setup & Docs
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("docs/diagrams", exist_ok=True)
    
    print("Step 1: Refreshing Architecture Diagrams...")
    DiagramGenerator().generate_all()

    # 2. Universal Pipeline Config
    pipeline = NewsDataPipeline()
    sources = {
        "CBC Business": "https://www.cbc.ca/cmlink/rss-business",
        "CNN Top Stories": "http://rss.cnn.com/rss/cnn_topstories.rss"
    }

    # 3. Model Config (Optimized during CNN/DM Training)
    config = SummarizerConfig(mmr_lambda=0.75, blend_alpha=0.7)
    summarizer = TextRankMMRSummarizer(config)

    print("\nStep 2: Starting Live Multi-Source Processing...")

    for name, url in sources.items():
        print(f"\n{'='*10} FETCHING SOURCE: {name} {'='*10}")
        articles = pipeline.build_articles(feed_url=url, limit=2)
        
        if not articles:
            print(f"No articles retrieved from {name}. Checking connection...")
            continue

        for i, art in enumerate(articles, 1):
            print(f"\n[{i}] Processing: {art.title}")
            # Summary length k=3 for punchy extractive demo
            result = summarizer.summarize(art.normalized_text, k=3)
            
            print(f"SUMMARY:\n{result['summary']}")
            print(f"Efficiency: {art.word_count} words -> {len(result['summary'].split())} words.")
            print("-" * 40)

if __name__ == "__main__":
    try:
        run_demonstration()
        print("\nSuccess: Demo completed. Architecture diagrams available in docs/diagrams/")
    except Exception as e:
        print(f"\nFatal Error during demo: {e}")
        print("Tip: Run with: $env:PYTHONPATH = '.'; python main.py")