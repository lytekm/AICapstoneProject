"""
TextRank + MMR Extractive Summarizer (CPU-friendly)

Dependencies:
  pip install numpy scikit-learn rouge-score

Optional (better sentence splitting):
  pip install nltk
  python -c "import nltk; nltk.download('punkt')"

What you get:
- TextRank sentence importance (PageRank over sentence-similarity graph)
- MMR selection to minimize redundancy
- Multi-level summaries (k sentences)
- Simple ROUGE evaluation loop

Author: (you)
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


import re
from typing import List

import nltk
from nltk.tokenize import sent_tokenize


nltk.download("punkt_tab")


def split_sentences(text: str) -> List[str]:
    """NLTK sentence splitter + light cleanup."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sents = sent_tokenize(text)
    out = []
    for s in sents:
        s = s.strip()
        # filter junky/very short segments
        if len(s) < 20:
            continue
        out.append(s)
    return out




# TextRank (PageRank) on sentence graph


def pagerank(
    W: np.ndarray,
    d: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    PageRank over a weighted adjacency matrix W.
    W is NxN with non-negative weights. Diagonal can be 0.
    """
    n = W.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Row-normalize to get transition probabilities
    row_sums = W.sum(axis=1, keepdims=True)
    # Handle dangling rows (no outgoing edges) by making them uniform
    dangling = (row_sums.squeeze() == 0)
    P = np.zeros_like(W, dtype=np.float64)
    P[~dangling] = W[~dangling] / row_sums[~dangling]
    if np.any(dangling):
        P[dangling] = 1.0 / n

    pr = np.ones(n, dtype=np.float64) / n
    teleport = np.ones(n, dtype=np.float64) / n

    for _ in range(max_iter):
        prev = pr
        pr = d * (P.T @ pr) + (1 - d) * teleport
        if np.linalg.norm(pr - prev, ord=1) < tol:
            break

    # Normalize to sum to 1
    pr = pr / (pr.sum() + 1e-12)
    return pr


def textrank_scores(sim: np.ndarray, min_edge: float = 0.1) -> np.ndarray:
    """
    Build a graph from similarity matrix and run PageRank.
    - sim: cosine similarity NxN (0..1)
    - min_edge: threshold to reduce noise
    """
    n = sim.shape[0]
    if n == 0:
        return np.array([])
    W = sim.copy()
    np.fill_diagonal(W, 0.0)
    if min_edge is not None:
        W[W < min_edge] = 0.0
    return pagerank(W)



# MMR selection (diversity)


def mmr_select(
    rel: np.ndarray,
    sim: np.ndarray,
    k: int,
    lam: float = 0.75,
) -> List[int]:
    """
    Select k sentences using MMR:
      argmax_s lam*Rel(s) - (1-lam)*max_{s' in selected} Sim(s, s')
    - rel: normalized relevance scores (N,)
    - sim: similarity matrix (N,N)
    """
    n = rel.shape[0]
    k = max(0, min(k, n))
    if k == 0:
        return []

    selected: List[int] = []
    candidates = set(range(n))

    # Start with best relevance
    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_idx = None
        best_score = -1e18

        for i in candidates:
            redundancy = max(sim[i, j] for j in selected) if selected else 0.0
            score = lam * rel[i] - (1 - lam) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)  # type: ignore[arg-type]
        candidates.remove(best_idx)  # type: ignore[arg-type]

    return selected



# Summarizer configuration


@dataclass
class SummarizerConfig:
    max_features: int = 20000
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: str = "english"
    textrank_min_edge: float = 0.1
    mmr_lambda: float = 0.75
    # Blend TextRank with centroid similarity for stability
    blend_alpha: float = 0.7  # 0..1, higher = more TextRank



# Main summarizer


def summarize_textrank_mmr(
    article_text: str,
    k: int = 5,
    config: Optional[SummarizerConfig] = None,
) -> Dict[str, object]:
    """
    Returns:
      {
        "summary": str,
        "sentences": List[str],
        "selected_indices": List[int],
        "scores": List[float]  # final relevance scores per sentence
      }
    """
    config = config or SummarizerConfig()
    sents = split_sentences(article_text)
    n = len(sents)

    if n == 0:
        return {"summary": "", "sentences": [], "selected_indices": [], "scores": []}
    if n == 1:
        return {"summary": sents[0], "sentences": sents, "selected_indices": [0], "scores": [1.0]}

    vec = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        stop_words=config.stop_words,
    )
    X = vec.fit_transform(sents)  # (N, D)
    sim = cosine_similarity(X)    # (N, N)

    # TextRank importance
    tr = textrank_scores(sim, min_edge=config.textrank_min_edge)

    # Centroid similarity (often stabilizes TextRank on short/noisy docs)
    centroid = np.asarray(X.mean(axis=0))
    centroid_sim = cosine_similarity(X, centroid).reshape(-1)

    # Normalize both to 0..1
    def minmax(a: np.ndarray) -> np.ndarray:
        mn, mx = float(a.min()), float(a.max())
        if math.isclose(mx, mn):
            return np.ones_like(a)
        return (a - mn) / (mx - mn)

    tr_n = minmax(tr)
    cen_n = minmax(centroid_sim)

    rel = config.blend_alpha * tr_n + (1 - config.blend_alpha) * cen_n
    rel = rel / (rel.max() + 1e-12)

    # MMR selection for diversity
    selected = mmr_select(rel=rel, sim=sim, k=k, lam=config.mmr_lambda)

    # Output sentences in original order (readability)
    selected_sorted = sorted(selected)
    summary = " ".join(sents[i] for i in selected_sorted)

    return {
        "summary": summary,
        "sentences": sents,
        "selected_indices": selected_sorted,
        "scores": rel.tolist(),
    }



# ROUGE evaluation

def rouge_eval(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    if rouge_scorer is None:
        raise RuntimeError("rouge-score not installed. Run: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    n = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)  # note: (target, prediction)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure

    return {
        "rouge1_f": r1 / max(n, 1),
        "rouge2_f": r2 / max(n, 1),
        "rougeL_f": rl / max(n, 1),
    }


# Test

if __name__ == "__main__":
    article = """
    The Bank of Canada held its benchmark interest rate steady on Wednesday,
    citing easing inflation but warning that global uncertainty remains elevated.
    Economists say the decision reflects a balancing act between slowing growth
    and progress on price stability. The central bank noted that recent data show
    inflation moving closer to its target, while consumer spending has softened.
    Markets largely expected no change, though traders are watching for signals
    about the timing of future cuts. Analysts said any pivot would depend on
    continued disinflation and weaker labor market conditions.
    """

    out = summarize_textrank_mmr(article, k=1)
    print("Selected indices:", out["selected_indices"])
    print("Summary:\n", out["summary"])
