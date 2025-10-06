# Topic Modeling

What we compute

- LDA topic modeling (via CountVectorizer + sklearn's LatentDirichletAllocation) to extract latent topics.
- For each topic, we compute the top keywords (by component weights) and compute correlation between topic prevalence and sentiment (VADER compound score) across documents.

Why it's useful

- Topics summarize recurring themes in the corpus (e.g., "billing issues", "network outages").
- Topic–sentiment correlation helps find which topics are associated with negative or positive sentiment (e.g., "outage" topics correlate with negative sentiment).

Where it's implemented

- `src/analysis.py` — `lda_topics`, `topic_sentiment_correlation`.

Output interpretation

- `topic_sentiment_correlation.csv` includes a `topic` id, `keywords` (human-readable label built from top words), and a correlation value with sentiment. Positive correlation means documents with that topic tend to have higher (more positive) VADER compound scores; negative correlation means more negative sentiment.

Notes and caveats

- LDA's topics are not guaranteed to be semantic labels; use the top keywords to create human-friendly labels.
- You can tune `n_topics` for granularity; fewer topics yield broader themes, more topics yield finer-grained themes.
- For short texts, topic models may be noisy. Consider aggregating messages (per user, per day) to stabilize topic prevalence.
