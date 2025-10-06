"""Analysis utilities: richer EDA, topic extraction and lightweight modeling helpers.

This module builds on `src.preprocess` and provides convenience functions used
by the notebooks and the demo runner.
"""
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import re

from src.preprocess import clean_text, tokenize_and_lemmatize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df[text_col].fillna("").apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize_and_lemmatize)
    df["joined_tokens"] = df["tokens"].apply(lambda toks: " ".join(toks))
    return df


def top_ngrams(series: pd.Series, n=20, ngram_range=(1, 1)) -> List[Tuple[str, int]]:
    """Return top n (term, count) pairs using raw counts from CountVectorizer."""
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vec.fit_transform(series.fillna(""))
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    counts = list(zip(terms, sums.tolist()))
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts[:n]


def class_top_ngrams(df: pd.DataFrame, label_col: str = "label", text_col: str = "joined_tokens", n=15, ngram_range=(1, 1)) -> Dict[str, List[Tuple[str, int]]]:
    """Compute top ngrams per class label."""
    out: Dict[str, List[Tuple[str, int]]] = {}
    for lbl, grp in df.groupby(label_col):
        out[str(lbl)] = top_ngrams(grp[text_col], n=n, ngram_range=ngram_range)
    return out


def plot_top_terms(counts: List[Tuple[str, int]], out: Path, title: str = "Top Terms") -> None:
    terms, vals = zip(*counts) if counts else ([], [])
    plt.figure(figsize=(10, max(4, len(terms) * 0.4)))
    plt.barh(range(len(terms)), vals[::-1], color="tab:blue")
    plt.yticks(range(len(terms)), list(terms)[::-1])
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def make_wordcloud(text: str, out: Path, max_words: int = 200) -> None:
    wc = WordCloud(width=800, height=400, background_color="white", max_words=max_words).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def label_distribution(df: pd.DataFrame, label_col: str = "label") -> Counter:
    return Counter(df[label_col].fillna(""))


def compute_tfidf_matrix(series: pd.Series, max_features: int = 5000, ngram_range=(1, 2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vec.fit_transform(series.fillna(""))
    return vec, X


def compute_vader_sentiment(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Compute VADER sentiment scores and append columns to a copy of the dataframe.

    Returns a new dataframe with columns: vader_compound, vader_pos, vader_neu, vader_neg
    """
    df = df.copy()
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].fillna("").astype(str)
    scores = texts.apply(lambda t: analyzer.polarity_scores(t))
    df["vader_compound"] = scores.apply(lambda s: s.get("compound", 0.0))
    df["vader_pos"] = scores.apply(lambda s: s.get("pos", 0.0))
    df["vader_neu"] = scores.apply(lambda s: s.get("neu", 0.0))
    df["vader_neg"] = scores.apply(lambda s: s.get("neg", 0.0))
    return df


def aggregate_sentiment_over_time(df: pd.DataFrame, time_col: str = "timestamp", freq: str = "D", domain_col: str = "domain") -> pd.DataFrame:
    """Aggregate mean sentiment and counts per time period and domain.

    Returns a dataframe with columns [time, domain, vader_compound, count]
    """
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not in dataframe")
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    grp = d.groupby([pd.Grouper(key=time_col, freq=freq), domain_col]).agg(
        vader_compound=("vader_compound", "mean"), count=("vader_compound", "count")
    ).reset_index()
    grp = grp.rename(columns={time_col: "time"})
    return grp


def plot_sentiment_over_time(agg_df: pd.DataFrame, out: Path, title: str = "Sentiment over time") -> None:
    plt.figure(figsize=(12, 6))
    try:
        sns.lineplot(data=agg_df, x="time", y="vader_compound", hue="domain", marker="o")
    except Exception:
        # fallback: simple matplotlib plot grouping
        for dom, grp in agg_df.groupby("domain"):
            plt.plot(grp["time"], grp["vader_compound"], label=str(dom))
        plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mean VADER compound")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def compute_keyword_trends(df: pd.DataFrame, domain_col: str = "domain", text_col: str = "joined_tokens", top_k: int = 8, freq: str = "D") -> Dict[str, pd.DataFrame]:
    """Compute counts over time for top_k keywords per domain.

    Returns a dict mapping domain -> pivoted DataFrame with index=time and columns=term counts.
    """
    out: Dict[str, pd.DataFrame] = {}
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column required for trends")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    domains = df[domain_col].fillna("").unique()
    for dom in domains:
        sub = df[df[domain_col] == dom]
        # find top terms in this domain
        top = [t for t, _ in top_ngrams(sub[text_col], n=top_k, ngram_range=(1, 1))]
        if not top:
            out[str(dom)] = pd.DataFrame()
            continue
        records = []
        for t in top:
            # count occurrences per period
            mask = sub[text_col].fillna("").str.contains(rf"\b{re.escape(t)}\b", case=False, regex=True)
            tmp = sub[mask].groupby(pd.Grouper(key="timestamp", freq=freq)).size().rename(t)
            records.append(tmp)
        if records:
            dfp = pd.concat(records, axis=1).fillna(0).astype(int)
        else:
            dfp = pd.DataFrame()
        out[str(dom)] = dfp
    return out


def plot_keyword_trends(trends: Dict[str, pd.DataFrame], out: Path, max_plots: int = 4) -> None:
    """Plot keyword trends for each domain into a single PNG (stacked subplots)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(trends)
    if n == 0:
        return
    plt.figure(figsize=(12, max(3, n * 3)))
    for i, (dom, dfp) in enumerate(trends.items(), start=1):
        plt.subplot(n, 1, i)
        if dfp.empty:
            plt.text(0.5, 0.5, f"No data for {dom}", ha="center")
            continue
        dfp.plot(ax=plt.gca())
        plt.title(f"Keyword trends - {dom}")
        plt.xlabel("")
        plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def compute_correlation_heatmap(df: pd.DataFrame, keywords: List[str], out: Path) -> None:
    """Compute correlation between keyword presence and sentiment scores and save a heatmap."""
    if not keywords:
        return
    X = pd.DataFrame()
    for k in keywords:
        X[k] = df["joined_tokens"].fillna("").str.contains(rf"\b{re.escape(k)}\b", case=False, regex=True).astype(int)
    # include sentiment
    if "vader_compound" in df.columns:
        X["vader_compound"] = df["vader_compound"].fillna(0.0)
    corr = X.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Keyword-Sentiment Correlation")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def predictive_terms_logistic(series: pd.Series, labels: pd.Series, top_k: int = 20, max_features: int = 5000):
    """Train a logistic regression to predict negative label and return top_k predictive terms for negative class.

    Returns a list of (term, coef) sorted by coefficient descending (most predictive of negative).
    """
    y = (labels == "negative").astype(int)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vec.fit_transform(series.fillna(""))
    if X.shape[0] < 10:
        return []
    model = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    model.fit(X, y)
    coefs = model.coef_.ravel()
    terms = vec.get_feature_names_out()
    term_coefs = list(zip(terms, coefs))
    # filter out tokens that contain digits or are too short (likely IDs / noise)
    filtered = [(t, c) for t, c in term_coefs if (len(t) > 2 and not re.search(r"\d", t))]
    # terms with highest positive weight are predictive of negative (since y=1 means negative)
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_k]


def cluster_and_characterize(df: pd.DataFrame, text_col: str = "joined_tokens", n_clusters: int = 6, n_components: int = 100, max_features: int = 5000, samples_per_cluster: int = 3):
    """Cluster documents using LSA embeddings and return cluster assignments, top terms per cluster, and sample texts.

    Returns: (cluster_labels, cluster_top_terms: Dict[int, List[str]], cluster_samples: Dict[int, List[str]])
    """
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vec.fit_transform(df[text_col].fillna(""))
    emb, svd = lsa_embeddings(X, n_components=min(n_components, X.shape[1]-1 if X.shape[1]>1 else 1))
    k = min(n_clusters, max(1, emb.shape[0]))
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(emb)

    terms = vec.get_feature_names_out()
    cluster_top_terms = {}
    centroids = km.cluster_centers_
    # Map centroid in latent space back to term space approximately via svd components
    # compute pseudo-centroid in term-space: components_.T @ centroid
    components = svd.components_ if hasattr(svd, "components_") else None
    if components is not None:
        term_space_centroids = components.T @ centroids.T
        for i in range(k):
            arr = term_space_centroids[:, i]
            top_idx = arr.argsort()[::-1][:10]
            cluster_top_terms[i] = [terms[j] for j in top_idx]
    else:
        for i in range(k):
            cluster_top_terms[i] = []

    cluster_samples = {}
    for i in range(k):
        idxs = list(df.index[labels == i])
        samples = [df.loc[i_, text_col] for i_ in idxs[:samples_per_cluster]]
        cluster_samples[i] = samples

    return labels, cluster_top_terms, cluster_samples


def detect_sentiment_spikes(df: pd.DataFrame, time_col: str = "timestamp", sentiment_col: str = "vader_compound", freq: str = "7D", z_thresh: float = 2.0, out: Optional[Path] = None):
    """Detect spikes/dips in sentiment using rolling z-score on aggregated time series.

    If out is provided, save a plot of the time series and flagged spikes.
    Returns a DataFrame of aggregated time periods with zscore and flag.
    """
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    agg = d.groupby(pd.Grouper(key=time_col, freq=freq))[sentiment_col].agg(["mean", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "sentiment_mean"})
    agg["rolling_mean"] = agg["sentiment_mean"].rolling(3, min_periods=1).mean()
    agg["rolling_std"] = agg["sentiment_mean"].rolling(3, min_periods=1).std().fillna(0.0)
    # avoid division by zero
    agg["zscore"] = (agg["sentiment_mean"] - agg["rolling_mean"]) / (agg["rolling_std"] + 1e-9)
    agg["spike"] = agg["zscore"].abs() >= z_thresh

    if out is not None:
        plt.figure(figsize=(12, 4))
        plt.plot(agg[time_col], agg["sentiment_mean"], label="mean sentiment")
        spikes = agg[agg["spike"]]
        plt.scatter(spikes[time_col], spikes["sentiment_mean"], color="red", label="spike")
        plt.title("Sentiment over time with spikes")
        plt.legend()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    return agg


def cooccurrence_matrix(series: pd.Series, top_k: int = 50, out: Optional[Path] = None):
    """Compute co-occurrence matrix for top_k terms and optionally save a heatmap.

    Returns the DataFrame co-occurrence matrix (terms x terms).
    """
    vec = CountVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vec.fit_transform(series.fillna(""))
    terms = vec.get_feature_names_out()
    # pick top_k by document frequency
    freqs = np.asarray((X > 0).sum(axis=0)).ravel()
    idx = freqs.argsort()[::-1][:top_k]
    Xs = X[:, idx].toarray()
    co = (Xs.T @ Xs).astype(float)
    co_df = pd.DataFrame(co, index=[terms[i] for i in idx], columns=[terms[i] for i in idx])
    if out is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(co_df, cmap="mako")
        plt.title("Term co-occurrence (top terms)")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
    return co_df


def topic_sentiment_correlation(df: pd.DataFrame, text_col: str = "joined_tokens", sentiment_col: str = "vader_compound", n_topics: int = 6, max_features: int = 5000):
    """Fit LDA, compute document-topic matrix and correlate topic weights with sentiment.

    Returns a DataFrame with topics and correlation coefficients.
    """
    vec = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vec.fit_transform(df[text_col].fillna(""))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    W = lda.fit_transform(X)
    cols = [f"topic_{i}" for i in range(W.shape[1])]
    Wdf = pd.DataFrame(W, columns=cols)
    # compute top keywords per topic
    terms = vec.get_feature_names_out()
    topic_keywords = []
    for comp in lda.components_:
        top_idx = comp.argsort()[::-1][:6]
        topic_keywords.append(", ".join([terms[i] for i in top_idx]))
    corr = {}
    for i, c in enumerate(cols):
        corr_val = float(np.corrcoef(Wdf[c], df[sentiment_col].fillna(0.0))[0, 1])
        corr[c] = corr_val
    rows = []
    for i, c in enumerate(cols):
        rows.append({"topic": c, "keywords": topic_keywords[i], "corr_with_sentiment": corr[c]})
    return pd.DataFrame(rows)


def lda_topics(series: pd.Series, n_topics: int = 5, n_top_words: int = 10, max_features: int = 5000):
    vec = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b\w+\b", stop_words="english")
    X = vec.fit_transform(series.fillna(""))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    terms = vec.get_feature_names_out()
    topics: List[List[str]] = []
    for comp in lda.components_:
        top_idx = comp.argsort()[::-1][:n_top_words]
        topics.append([terms[i] for i in top_idx])
    return topics


def lsa_embeddings(X, n_components: int = 50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    emb = svd.fit_transform(X)
    return emb, svd


def tsne_project(embeddings: np.ndarray, n_components: int = 2, perplexity: int = 30):
    ts = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init="pca")
    proj = ts.fit_transform(embeddings)
    return proj


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels


def plot_scatter(coords: np.ndarray, labels: Optional[np.ndarray], out: Path, title: str = "Projection") -> None:
    plt.figure(figsize=(8, 6))
    if labels is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.7)
    else:
        uniq = np.unique(labels)
        palette = sns.color_palette("tab10", n_colors=max(3, len(uniq)))
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette=palette, s=30, alpha=0.8, legend=False)
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
