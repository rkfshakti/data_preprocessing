# Preprocessing

What we do

- Basic cleaning: lowercasing, punctuation removal where appropriate, and stripping of extra whitespace.
- Tokenization and lemmatization: we use NLTK (if installed) to tokenize and lemmatize. There is a small fallback to a lightweight tokenizer/normalizer if NLTK data isn't available.
- Stopword handling: stopwords are removed in vectorizer-based analyses by passing `stop_words='english'` to scikit-learn vectorizers. The preprocessing step itself retains stopwords in the token lists (so they can be inspected), but analysis vectorizers will exclude them.
- Noise handling: the generator sometimes inserts reference strings like `REF-1234` or `ticket:...`. Preprocessing strips or normalizes these where possible.

Why it matters

- Normalized tokens reduce vocabulary size and create more meaningful frequency-based statistics (top terms, TF-IDF, LDA input).
- Lemmatization groups word forms ("activated" -> "activate"), improving topic and predictive-term quality.
- Removing stopwords from vectorizers avoids high-frequency but uninformative tokens ("the", "and").

Where it's implemented

- `src/preprocess.py` — functions `clean_text`, `tokenize_and_lemmatize`, `preprocess_df`.

Notes and extension ideas

- You may want more advanced normalization (e.g., phone number scrubbing, anonymizing email addresses, named-entity normalization). Add targeted regexes in `clean_text`.
- For production, consider using spaCy for faster, more accurate tokenization/lemmatization and optional NER.
