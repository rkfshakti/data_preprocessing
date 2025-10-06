"""Run the full demo: generate data, preprocess, analyze, and save results."""
from pathlib import Path
import pandas as pd

from generate_data import generate, generate_for_domain
from src.analysis import (
    load_data,
    preprocess_df,
    top_ngrams,
    plot_top_terms,
    make_wordcloud,
    compute_vader_sentiment,
    aggregate_sentiment_over_time,
    plot_sentiment_over_time,
    compute_keyword_trends,
    plot_keyword_trends,
    compute_correlation_heatmap,
    predictive_terms_logistic,
    cluster_and_characterize,
    detect_sentiment_spikes,
    cooccurrence_matrix,
    topic_sentiment_correlation,
)
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import argparse
import plotly.graph_objects as go
import plotly.io as pio
from generate_data import DOMAINS
import re


DATA = Path(__file__).parent / "data" / "synthetic_texts.csv"
RESULTS = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="optional domain to generate/analyze (banking|telecom|travel)")
    parser.add_argument("--n", type=int, default=500, help="number of rows to generate")
    parser.add_argument("--embed-plotly", action="store_true", help="embed Plotly JS into the HTML report for offline viewing")
    args = parser.parse_args()

    domain = args.domain
    n = args.n

    # generate dataset (per-domain if requested)
    if domain:
        out_file = Path(__file__).parent / "data" / f"synthetic_texts_{domain}.csv"
        if not out_file.exists():
            print(f"Generating synthetic data for domain={domain}...")
            generate_for_domain(domain, n=n, out_file=str(out_file))
        DATA_FILE = out_file
    else:
        DATA_FILE = DATA
        if not DATA_FILE.exists():
            print("Generating synthetic data (all domains)...")
            generate(n)

    print("Loading data...")
    df = load_data(DATA_FILE)
    print(f"Rows: {len(df)}")

    print("Preprocessing...")
    df = preprocess_df(df)

    # save preprocessed
    # results folder per-domain
    if domain:
        domain_results = RESULTS / domain
    else:
        domain_results = RESULTS / "all"
    domain_results.mkdir(parents=True, exist_ok=True)
    df.to_csv(domain_results / "preprocessed.csv", index=False)

    print("Computing top unigrams and bigrams...")
    uni = top_ngrams(df["joined_tokens"], n=20, ngram_range=(1, 1))
    bi = top_ngrams(df["joined_tokens"], n=20, ngram_range=(2, 2))

    plot_top_terms(uni, domain_results / "top_unigrams.png", title="Top Unigrams")
    plot_top_terms(bi, domain_results / "top_bigrams.png", title="Top Bigrams")

    print("Generating wordcloud...")
    all_text = " ".join(df["joined_tokens"].fillna(""))
    make_wordcloud(all_text, domain_results / "wordcloud.png")

    # Sentiment analysis
    print("Computing VADER sentiment...")
    df2 = compute_vader_sentiment(df, text_col="clean_text")
    df2.to_csv(domain_results / "with_sentiment.csv", index=False)

    print("Aggregating sentiment over time...")
    agg = aggregate_sentiment_over_time(df2, time_col="timestamp", freq="7D", domain_col="domain")
    plot_sentiment_over_time(agg, domain_results / "sentiment_over_time.png")

    print("Computing keyword trends...")
    trends = compute_keyword_trends(df2, domain_col="domain", text_col="joined_tokens", top_k=6, freq="7D")
    plot_keyword_trends(trends, domain_results / "keyword_trends.png")

    print("Computing correlation heatmap...")
    # choose a few top keywords across dataset
    top_terms = [t for t, _ in top_ngrams(df2["joined_tokens"], n=12, ngram_range=(1, 1))][:8]
    compute_correlation_heatmap(df2, top_terms, domain_results / "heatmap.png")

    # Build Plotly interactive charts
    # Top terms plotly (unigrams)
    terms, vals = zip(*uni) if uni else ([], [])
    top_terms_fig = go.Figure(go.Bar(x=list(vals), y=list(terms), orientation='h'))
    top_terms_fig.update_layout(height=400, margin=dict(l=120))
    include_js = True if args.embed_plotly else False
    top_terms_plotly = pio.to_html(top_terms_fig, include_plotlyjs=include_js, full_html=False)

    # Sentiment over time plotly
    try:
        agg_plot = agg.copy()
        # ensure time is datetime
        agg_plot['time'] = pd.to_datetime(agg_plot['time'])
        fig = go.Figure()
        for dom in agg_plot['domain'].unique():
            sub = agg_plot[agg_plot['domain'] == dom]
            fig.add_trace(go.Scatter(x=sub['time'], y=sub['vader_compound'], mode='lines+markers', name=str(dom)))
        fig.update_layout(height=420, xaxis_title='Time', yaxis_title='Mean VADER compound')
        sentiment_plotly = pio.to_html(fig, include_plotlyjs=include_js, full_html=False)
    except Exception:
        sentiment_plotly = ""

    # Keyword trends (combine domains or single domain)
    try:
        kt_fig = go.Figure()
        for dom, dfp in trends.items():
            if dfp.empty:
                continue
            for col in dfp.columns:
                kt_fig.add_trace(go.Scatter(x=dfp.index, y=dfp[col], mode='lines', name=f"{dom}:{col}"))
        kt_fig.update_layout(height=420, xaxis_title='Time', yaxis_title='Counts')
        keyword_trends_plotly = pio.to_html(kt_fig, include_plotlyjs=include_js, full_html=False)
    except Exception:
        keyword_trends_plotly = ""

    # (cooccurrence_plotly will be created after co matrix is computed)

    # deeper multivariate analyses
    print("Running multivariate analyses: predictive terms, clustering, spikes, cooccurrence, topic-sentiment")
    # predictive terms (what words predict negative labels)
    preds = predictive_terms_logistic(df2["joined_tokens"], df2["label"], top_k=30)
    if preds:
        pred_df = pd.DataFrame(preds, columns=["term", "coef"])
        # save full CSV
        pred_df.to_csv(domain_results / "predictive_terms.csv", index=False)
        # HTML snippet: top 10 only
        predictive_terms_html = pred_df.head(10).to_html(index=False)
    else:
        predictive_terms_html = ""

    # clustering and characterization
    labels, cluster_terms, cluster_samples = cluster_and_characterize(df2, text_col="joined_tokens", n_clusters=6)
    df2["cluster"] = labels
    df2.to_csv(domain_results / "with_sentiment_and_clusters.csv", index=False)
    # save cluster terms and samples
    with open(domain_results / "cluster_top_terms.txt", "w") as fh:
        for k, terms_list in cluster_terms.items():
            fh.write(f"Cluster {k}: {', '.join(terms_list)}\n")
    with open(domain_results / "cluster_samples.txt", "w") as fh:
        for k, samples in cluster_samples.items():
            fh.write(f"Cluster {k} samples:\n")
            for s in samples:
                fh.write(s.replace('\n', ' ')[:300] + "\n---\n")

    # prepare cluster HTML snippets
    cluster_terms_html = "<h4>Cluster top terms</h4>" + "".join([f"<p><strong>Cluster {k}</strong>: {', '.join(v)}</p>" for k, v in cluster_terms.items()])
    cluster_samples_html = "<h4>Cluster samples</h4>" + "".join([f"<p><strong>Cluster {k}</strong>:<br>" + "<br>".join([s.replace('\n',' ')[:200] for s in v]) + "</p>" for k, v in cluster_samples.items()])

    # sentiment spikes
    spikes_df = detect_sentiment_spikes(df2, time_col="timestamp", sentiment_col="vader_compound", freq="7D", z_thresh=2.0, out=domain_results / "spikes.png")
    spikes_df.to_csv(domain_results / "sentiment_spikes.csv", index=False)
    spikes_html = spikes_df.head(10).to_html(index=False)

    # co-occurrence heatmap
    co = cooccurrence_matrix(df2["joined_tokens"], top_k=40, out=domain_results / "cooccurrence.png")
    co.to_csv(domain_results / "cooccurrence_matrix.csv")

    # build cooccurrence plotly now that co exists
    try:
        co_small = co.copy()
        co_fig = go.Figure(data=go.Heatmap(z=co_small.values, x=co_small.columns.tolist(), y=co_small.index.tolist(), colorscale='Viridis'))
        co_fig.update_layout(height=600)
        cooccurrence_plotly = pio.to_html(co_fig, include_plotlyjs=include_js, full_html=False)
    except Exception:
        cooccurrence_plotly = ""

    # topic-sentiment correlation
    topic_corr = topic_sentiment_correlation(df2, text_col="joined_tokens", sentiment_col="vader_compound", n_topics=6)
    topic_corr.to_csv(domain_results / "topic_sentiment_correlation.csv", index=False)
    # show top 10 topics in HTML (there will usually be <= n_topics rows)
    topic_corr_html = topic_corr.head(10).to_html(index=False)

    # render HTML report
    print("Rendering HTML report...")
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "src"))
    tpl = env.get_template("report_template.html")
    label_counts = df2["label"].value_counts().to_dict()

    # summary metrics
    overall_mean = round(float(df2["vader_compound"].mean()), 3)
    overall_std = round(float(df2["vader_compound"].std()), 3)
    pct_pos = round(100 * (df2["label"] == "positive").mean(), 1)
    pct_neg = round(100 * (df2["label"] == "negative").mean(), 1)
    pct_neu = round(100 * (df2["label"] == "neutral").mean(), 1)

    pos_terms = [t for t, _ in top_ngrams(df2[df2["label"]=="positive"]["joined_tokens"], n=6)]
    neg_terms = [t for t, _ in top_ngrams(df2[df2["label"]=="negative"]["joined_tokens"], n=6)]

    # Remove domain negative keywords from displayed positive terms (robust matching)
    def contains_keyword(token: str, keyword: str) -> bool:
        tk = token.lower()
        for w in keyword.lower().split():
            if not re.search(r"\b" + re.escape(w) + r"\w*", tk):
                return False
        return True

    try:
        vocab = DOMAINS.get(domain, {}) if domain else {}
        negative_kw = vocab.get("negative_keywords", [])
        filtered_pos = []
        for t in pos_terms:
            if any(contains_keyword(t, kw) for kw in negative_kw):
                continue
            filtered_pos.append(t)
        # fallback to original if filtering removed everything
        if filtered_pos:
            pos_terms = filtered_pos
    except Exception:
        pass

    most_positive_time = "N/A"
    most_negative_time = "N/A"
    most_positive_val = "N/A"
    most_negative_val = "N/A"
    try:
        if not agg.empty:
            by_time = agg.groupby("time").vader_compound.mean().reset_index()
            best = by_time.loc[by_time["vader_compound"].idxmax()]
            worst = by_time.loc[by_time["vader_compound"].idxmin()]
            most_positive_time = str(best["time"])[:10]
            most_positive_val = round(float(best["vader_compound"]), 3)
            most_negative_time = str(worst["time"])[:10]
            most_negative_val = round(float(worst["vader_compound"]), 3)
    except Exception:
        pass

    # images are written into the same folder as the HTML report, so use filenames
    html = tpl.render(
        generated_on=datetime.now().isoformat(),
        domain_name=domain or "All domains",
        rows=len(df2),
        label_counts=label_counts,
        overall_sentiment_mean=overall_mean,
        overall_sentiment_std=overall_std,
        pct_positive=pct_pos,
        pct_negative=pct_neg,
        pct_neutral=pct_neu,
        top_pos_terms=pos_terms,
        top_neg_terms=neg_terms,
        most_positive_time=most_positive_time,
        most_negative_time=most_negative_time,
        most_positive_val=most_positive_val,
        most_negative_val=most_negative_val,
        top_unigrams="top_unigrams.png",
        top_bigrams="top_bigrams.png",
        wordcloud="wordcloud.png",
        sentiment_time="sentiment_over_time.png",
        keyword_trends="keyword_trends.png",
        heatmap="heatmap.png",
        # plotly html snippets
        top_terms_plotly=top_terms_plotly,
        sentiment_plotly=sentiment_plotly,
        keyword_trends_plotly=keyword_trends_plotly,
        cooccurrence_plotly=cooccurrence_plotly,
        # html tables/snippets
        predictive_terms_html=predictive_terms_html,
        cluster_terms_html=cluster_terms_html,
        cluster_samples_html=cluster_samples_html,
        spikes_html=spikes_html,
        topic_corr_html=topic_corr_html,
    )
    (domain_results / "analysis_report.html").write_text(html)
    print("Report written to:", domain_results / "analysis_report.html")

    print("Done. Results in:", RESULTS)


if __name__ == "__main__":
    main()
