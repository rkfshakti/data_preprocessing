# Co-occurrence

What we compute

- A co-occurrence matrix of the top-k terms (how frequently two tokens appear together in the same document). We then visualize it as a heatmap.

Why it's useful

- Reveals terms that commonly co-occur ("dropped" and "call"), which helps identify compound issues (e.g., "dropped calls in area X").
- Useful for building multi-word detectors or for guiding phrase-based rules.

Where it's implemented

- `src/analysis.py` — `cooccurrence_matrix`.

Output and interpretation

- `cooccurrence_matrix.csv` — counts; row/column labels are token names. The heatmap (`cooccurrence.png`) shows high co-occurrence cells in darker colors.

Notes

- Use normalized co-occurrence (e.g., PMI or Jaccard) for some applications; raw counts are often adequate for inspection.
- Co-occurrence is sensitive to document length and preprocessing (if you break long text into smaller units, co-occurrence patterns change).
