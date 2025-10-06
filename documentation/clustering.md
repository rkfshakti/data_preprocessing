# Clustering

What we compute

- KMeans clustering over TF-IDF or LSA-reduced vectors to group documents into clusters.
- For each cluster we extract top terms (centroid-characterizing terms) and a few sample documents to help interpret each cluster.

Why it's useful

- Clustering groups similar messages together (billing-related, network-related, device activation complaints, etc.).
- Cluster-level summaries are easier to scan than individual messages when exploring large corpora.

Where it's implemented

- `src/analysis.py` — `cluster_and_characterize`.

Output and interpretation

- `cluster_top_terms.txt` contains the top descriptive terms for each cluster.
- `cluster_samples.txt` provides a few example messages from each cluster.

Notes

- The choice of `n_clusters` influences granularity. For demo, a moderate number (4–8) is a reasonable default.
- KMeans assumes spherical clusters in vector space—other algorithms (Agglomerative, HDBSCAN) may be preferable for non-spherical shapes.
