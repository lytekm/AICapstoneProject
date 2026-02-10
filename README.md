# AICapstoneProject
## TextRank + MMR Extractive Summarizer

### Overview
This is a module that implements an extractive text summarization pipeline that identifies the most important sentences in a news article while minimizing redundancy.

The approach combines TextRank, which identifies globally important sentences using graph-based ranking and Maximal Marginal Relevance (MMR) which selects a diverse subset of the sentances to avoid repetition.

Raw Article Text
      ↓
Sentence Segmentation (NLTK)
      ↓
TF-IDF Vectorization
      ↓
Sentence Similarity Matrix
      ↓
TextRank (Importance Scoring)
      ↓
MMR (Diversity-Aware Selection)
      ↓
Ordered Extractive Summary

### Text processing
Sentance segmantation is handled using NLTK's Punkt tokenizer which is good for english news-style text. Additional filtering also removes any very short or noisy sentences.
Each sentance is vectorized using TF-IDF which allows us to capture semantic similarity using cosine similarity.

We compute the cosine similarity matrix with:
Sim(Si, Sj) = Si x Sj/ ||Si||||Sj||
This matrix gets used by both TextRank and MMR.

TextRank scores the importance of the sentences by representing them as nodes in a graph, nodes being the sentences, edges being the sentence similarity and edge weights being the cosine similarity.
Edges with low similarity using a threshold to reduce noise.
PageRank Formula:
 \(PR(A)=\frac{1-d}{N}+d\sum _{i=1}^{n}\frac{PR(T_{i})}{C(T_{i})}\)

 We use cosine similarity to help stabalize TextRank on short or noisy articles by computing the aritcle centroid by averaging all the vectors we get from the cosine similarity:
 c=N1​i=1∑N​si
 CentroidSim(si​)=cos(si​,c)
 
 We then combine the TextRank score with the centroid similarity ​
Rel(si​)=α⋅TextRank(si​)+(1−α)⋅CentroidSim(si​)
The blended score gives us how important the sentence is to the whole article.

Instead of selecting the top-K sentences, we use MMR to ensure diversity.
MMR(s)=λ⋅Rel(s)−(1−λ)⋅s′∈Smax​Sim(s,s′)
where S is slected sentences, and lamda controls relevance vs diversity, high lamda is more relevance driven, and low lamda is more diversity driven.
We apply MMR iteratively until K sentences are selected.

### Usage
```
@dataclass
class SummarizerConfig:
    max_features: int = 20000
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: str = "english"
    textrank_min_edge: float = 0.1
    mmr_lambda: float = 0.75
    blend_alpha: float = 0.7
```
| Parameter           | Effect                           |
| ------------------- | -------------------------------- |
| `mmr_lambda`        | Higher = less redundancy penalty |
| `blend_alpha`       | Higher = more TextRank influence |
| `textrank_min_edge` | Higher = sparser graph           |

```
out = summarize_textrank_mmr(article_text, k=5)
print(out["summary"])
```

Output Structure
```
{
  "summary": "...",
  "sentences": [...],
  "selected_indices": [0, 3, 5],
  "scores": [...]
}
```

The module uses ROUGE evaluation for comparison agianst reference summaries, allowing for quantitative validation on datasets.
Going forward, we will improve the model to have a training process that we can use to tune the hyperparameters to maximise the rouge score. 
