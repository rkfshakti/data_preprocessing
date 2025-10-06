# Predictive Terms (Logistic)

What we compute

- A logistic regression classifier (with a linear penalty) trained on TF-IDF features to predict binary labels (e.g., negative vs not-negative). We then extract the model coefficients to identify terms most predictive of each class.

Why it's useful

- Coefficients provide interpretable signals ("dropped", "outage", "frustrating") that indicate which tokens are most associated with negative experiences.
- Useful for building alerts, keyword-based routing, or for product teams to focus on common pain points.

Where it's implemented

- `src/analysis.py` — `predictive_terms_logistic`.

Output interpretation

- `predictive_terms.csv`: each row includes `term` and `coef` (coefficient). Higher positive coef indicates association with the positive-coded class in the underlying training setup; negative coef indicates association with the negative-coded class (check how labels are encoded in the function call).
- The report displays the top-10 predictive terms (HTML snippet) while the full CSV is saved for downstream use.

Notes and caveats

- The logistic model and coefficient interpretation assume the vectorizer vocabulary aligns with the tokenization/normalization used in preprocessing.
- Rare tokens may have unstable coefficients. Consider frequency thresholds or L1 regularization for sparser solutions.
- We filter numeric tokens and stopwords to avoid noisy predictive features.
