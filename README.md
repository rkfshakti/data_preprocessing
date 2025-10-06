# NLP Data Preprocessing

This repository implements a comprehensive pipeline for domain-specific NLP analysis with enhanced capabilities. It:

- generates synthetic text per domain (banking, telecom, travel),
- preprocesses the text (comprehensive cleaning, tokenization, lemmatization),
- runs exploratory analyses (top terms, wordcloud, keyword trends, co-occurrence, sentiment analysis),
- runs multivariate analyses (predictive-term extraction, multiple classification algorithms, clustering, LDA topic modeling, topic–sentiment correlation),
- provides interactive Jupyter notebooks for different analysis techniques, and
- renders a single-page, tabbed HTML report (Plotly interactive charts + top-10 table snippets) per domain.

Read detailed conceptual documentation for each analysis in the `documentation/` folder.

## Quick start (macOS / zsh)

1. Create or select a Python environment. The project was developed with Python 3.13 but Python 3.8+ should work. Example using pyenv:

```bash
pyenv install 3.13.0   # optional
pyenv virtualenv 3.13.0 ml313
pyenv activate ml313
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the analysis for a single domain (telecom example). This will generate the domain CSV (if missing), run preprocessing and analyses, and write plots/CSV and the HTML report under `results/telecom/`.

```bash
PYTHONPATH=. python run_analysis.py --domain telecom --n 800 --embed-plotly
```

Flags:
- `--domain <banking|telecom|travel>`: analyze only the specified domain. If omitted, the combined dataset `data/synthetic_texts.csv` is used.
- `--n <int>`: number of synthetic rows to generate when generating per-domain data.
- `--embed-plotly`: embed Plotly JS into the output HTML so the report is standalone and viewable offline.

4. Open the generated report in your browser:

```bash
open results/telecom/analysis_report.html
```

## Project layout

- `generate_data.py` — synthetic data generator (domain vocabularies and templates). Produces CSVs under `data/`.
- `src/preprocess.py` — text cleaning, tokenization, and lemmatization utilities.
- `src/analysis.py` — EDA and multivariate analysis functions (vectorizers, LDA, clustering, logistic predictive terms, co-occurrence, VADER sentiment, plotting helpers).
- `src/report_template.html` — Jinja2 template used to render the single-page tabbed report.
- `run_analysis.py` — top-level orchestrator: generate → preprocess → analyze → render.
- `results/` — per-domain output folder with CSVs, PNGs, and `analysis_report.html`.
- `documentation/` — explanatory docs for each analysis component (added below).

## Outputs produced (per-domain)

Typical files produced under `results/<domain>/`:

- `preprocessed.csv` — cleaned and tokenized text plus metadata
- `with_sentiment.csv` — VADER sentiment scores per row
- `with_sentiment_and_clusters.csv` — sentiment plus cluster labels
- `predictive_terms.csv` — logistic model coefficients (full CSV saved; top-10 displayed in HTML)
- `topic_sentiment_correlation.csv` — top keywords per topic and correlation with sentiment
- `cooccurrence_matrix.csv` — saved co-occurrence counts
- `*.png` — static images (wordcloud, heatmap, top terms, etc.)
- `analysis_report.html` — the single-page interactive report

## Tests

Run unit tests with pytest:

```bash
python -m pytest -q
```

## Troubleshooting

- NLTK data: if you encounter missing NLTK corpora (punkt/wordnet/stopwords), install via:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

- If Plotly visualizations are not interactive in your environment, try `--embed-plotly` so the HTML contains the Plotly runtime.

## Documentation

See the `documentation/` folder for a conceptual write-up of each analysis component (what it computes, why it's useful, interpretation guidance, and pointers to the code that implements it). Start with `documentation/overview.md`.

## Next steps (suggestions)

- Enforce strict separation of keywords in the synthetic data generator (if you need absolute guarantees that positive examples do not contain specific negative tokens).
- Improve preprocessing to remove ticket/reference noise more aggressively.
- Add more domains or expand templates in `generate_data.py`.

If you'd like, I can add a `Makefile` or small wrapper scripts in a `scripts/` folder with one-liners for common operations.

## Notebooks

This repository includes a comprehensive set of notebooks for different NLP analysis techniques:

### Core Analysis Notebooks:
- `notebooks/preprocessing_demo.ipynb` — demonstrates comprehensive text cleaning, tokenization, lemmatization, and preprocessing pipeline with detailed explanations
- `notebooks/analysis_exploration.ipynb` — exploratory data analysis: label distributions, top n-grams, TF-IDF → LSA → t-SNE projection, keyword trends, and comprehensive visualizations
- `notebooks/modeling_demo.ipynb` — advanced text classification modeling with multiple algorithms, hyperparameter tuning, feature importance analysis, and error analysis

### Specialized Analysis Notebooks:
- `notebooks/sentiment_analysis.ipynb` — comprehensive sentiment analysis using VADER, domain-specific sentiment patterns, and sentiment trend analysis
- `notebooks/topic_modeling.ipynb` — LDA topic modeling with coherence analysis, topic visualization, and topic-sentiment correlation
- `notebooks/clustering_visualization.ipynb` — K-means clustering with silhouette analysis, cluster visualization, and cluster interpretation

### Notebook Features:
- **Interactive Visualizations**: Plotly charts for interactive exploration
- **Comprehensive Analysis**: Multiple algorithms and techniques compared
- **Hyperparameter Tuning**: Grid search for optimal model performance
- **Error Analysis**: Detailed examination of model mistakes
- **Domain-Specific Insights**: Analysis across different domains (banking, telecom, travel)

Open the notebooks in VS Code or Jupyter. Make sure the kernel's working directory is the project root so local imports (`src.*`) resolve. Example:

```bash
# From project root
jupyter lab notebooks/
```

Each notebook is self-contained and includes detailed explanations, making them ideal for learning and experimentation with different NLP techniques.

