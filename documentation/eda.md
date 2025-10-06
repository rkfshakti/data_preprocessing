# Exploratory Data Analysis (EDA)

What we compute

- Top unigrams/bigrams: frequency counts on the preprocessed `joined_tokens` column.
- Wordcloud: a visual summary of top words by frequency.
- Keyword trends: counts of selected important tokens over time (e.g., weekly), useful to spot rising/receding mentions.
- Correlation heatmap: pointwise correlation between top keywords (co-occurrence-based correlation), used for quick relationship detection.

Why it's useful

- Quickly surfaces dominant terms in the corpus and their temporal patterns.
- Helps detect topic shifts, sudden events (outages), or seasonal patterns.

Where it's implemented

- `src/analysis.py` — `top_ngrams`, `make_wordcloud`, `compute_keyword_trends`, `compute_correlation_heatmap`.
- Plots are saved as PNGs in `results/<domain>/` and embedded as images in the HTML report.

How to read outputs

- `top_unigrams.png` and `top_bigrams.png`: bar charts of the most frequent tokens/phrases.
- `wordcloud.png`: larger words indicate higher frequency; useful for quick visual cues.
- `keyword_trends.png`: lines by token showing how often tokens appeared in each time bucket.
- `heatmap.png`: darker cells mean stronger pairwise correlation/co-occurrence between token pairs.

Notes

- Because we remove stopwords at vectorizer time, the top tokens shown are more domain-specific.
- For multi-domain projects, compute trends per-domain to avoid mixing vocabulary across domains.
