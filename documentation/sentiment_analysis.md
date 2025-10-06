# Sentiment Analysis

What we compute

- VADER sentiment scores per document (compound, pos/neu/neg scores).
- Aggregate sentiment over time (mean VADER compound by time bucket) and detect spikes (z-score based) to surface anomalous periods.

Why it's useful

- Quick measure of customer sentiment trends and early detection of negative spikes (e.g., outage events causing surges of negative posts).
- VADER is tuned for social-like text and short messages; it's lightweight and fast for demo purposes.

Where it's implemented

- `src/analysis.py` — `compute_vader_sentiment`, `aggregate_sentiment_over_time`, `detect_sentiment_spikes`.

Output interpretation

- `with_sentiment.csv` contains row-level VADER scores.
- `sentiment_over_time.png` or the interactive Plotly chart shows how mean sentiment evolves across time buckets.
- `sentiment_spikes.csv` lists detected anomalies (time buckets and z-scores).

Notes and caveats

- VADER is lexicon-based and may mis-handle domain-specific expressions. Use domain-tuned sentiment models for higher fidelity.
- For production, consider an ensemble of lexicon and supervised models, or a domain-labeled training set.
