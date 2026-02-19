from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk = None
    sent_tokenize = None


@dataclass
class SummarizerConfig:
    max_features: int = 20000
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: str = "english"
    textrank_min_edge: float = 0.1
    mmr_lambda: float = 0.75
    blend_alpha: float = 0.7  # 0..1


class TextRankMMRSummarizer:
    """
    Build Model module (tokenize sentences + TextRank + MMR).
    """

    def __init__(self, config: Optional[SummarizerConfig] = None):
        self.config = config or SummarizerConfig()

    def split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        if sent_tokenize is not None:
            try:
                sents = sent_tokenize(text)
            except LookupError:
                # punkt not installed
                raise RuntimeError(
                    "NLTK punkt not found. Run: python -m nltk.downloader punkt"
                )
        else:
            # fallback
            sents = re.split(r"(?<=[.!?])\s+", text)

        out = []
        for s in sents:
            s = s.strip()
            if len(s) < 20:
                continue
            out.append(s)
        return out

    def pagerank(self, W: np.ndarray, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        n = W.shape[0]
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        row_sums = W.sum(axis=1, keepdims=True)
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

        pr = pr / (pr.sum() + 1e-12)
        return pr

    def textrank_scores(self, sim: np.ndarray, min_edge: float = 0.1) -> np.ndarray:
        n = sim.shape[0]
        if n == 0:
            return np.array([])
        W = sim.copy()
        np.fill_diagonal(W, 0.0)
        W[W < min_edge] = 0.0
        return self.pagerank(W)

    def mmr_select(self, rel: np.ndarray, sim: np.ndarray, k: int, lam: float = 0.75) -> List[int]:
        n = rel.shape[0]
        k = max(0, min(k, n))
        if k == 0:
            return []

        selected: List[int] = []
        candidates = set(range(n))

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
            selected.append(int(best_idx))  # type: ignore[arg-type]
            candidates.remove(int(best_idx))  # type: ignore[arg-type]

        return selected

    def summarize(self, article_text: str, k: int = 5) -> Dict[str, object]:
        cfg = self.config
        sents = self.split_sentences(article_text)
        n = len(sents)

        if n == 0:
            return {"summary": "", "sentences": [], "selected_indices": [], "scores": []}
        if n == 1:
            return {"summary": sents[0], "sentences": sents, "selected_indices": [0], "scores": [1.0]}

        vec = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
            stop_words=cfg.stop_words,
        )
        X = vec.fit_transform(sents)
        sim = cosine_similarity(X)

        tr = self.textrank_scores(sim, min_edge=cfg.textrank_min_edge)

        centroid = np.asarray(X.mean(axis=0))
        centroid_sim = cosine_similarity(X, centroid).reshape(-1)

        def minmax(a: np.ndarray) -> np.ndarray:
            mn, mx = float(a.min()), float(a.max())
            if math.isclose(mx, mn):
                return np.ones_like(a)
            return (a - mn) / (mx - mn)

        tr_n = minmax(tr)
        cen_n = minmax(centroid_sim)

        rel = cfg.blend_alpha * tr_n + (1 - cfg.blend_alpha) * cen_n
        rel = rel / (rel.max() + 1e-12)

        selected = self.mmr_select(rel=rel, sim=sim, k=k, lam=cfg.mmr_lambda)
        selected_sorted = sorted(selected)

        summary = " ".join(sents[i] for i in selected_sorted)

        return {
            "summary": summary,
            "sentences": sents,
            "selected_indices": selected_sorted,
            "scores": rel.tolist(),
        }