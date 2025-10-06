"""Text preprocessing utilities.

Functions:
- clean_text: remove HTML, urls, emails, emojis, punctuation normalization
- tokenize_and_lemmatize: tokenization, stopword removal, lemmatization
"""
import re
from typing import List


_nltk_ready = False


def ensure_nltk():
    """Lazily import nltk and download small models if needed.

    This avoids importing nltk at module import time so tests can import the
    module without all optional dependencies installed.
    """
    global _nltk_ready
    if _nltk_ready:
        return
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        _nltk_ready = True
    except Exception:
        # If nltk is not available, keep _nltk_ready False. callers should
        # handle missing nltk by raising a clear error later.
        _nltk_ready = False


def clean_text(text: str) -> str:
    """Basic cleaning for noisy text.

    - strip HTML (uses BeautifulSoup if available, otherwise a regex fallback)
    - remove URLs and emails
    - normalize whitespace
    - remove control chars
    - replace repeated punctuation
    - collapse repeated words
    """
    if not text:
        return ""
    # strip HTML using BeautifulSoup if available; otherwise strip tags simply
    try:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        # simple HTML tag stripper fallback
        text = re.sub(r"<[^>]+>", " ", text)

    # remove urls
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # remove emails
    text = re.sub(r"\S+@\S+", "", text)
    # remove code-like tokens (simple heuristic)
    text = re.sub(r"\bfor\s*\(.*?\)|console\.log\([^)]*\)", "", text, flags=re.I)
    # remove non-printable/control
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    # collapse repeated punctuation
    text = re.sub(r"([!?\.,]){2,}", r"\1", text)
    # remove long runs of the same word (e.g., "words words words") keep one
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", text, flags=re.IGNORECASE)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str, lang: str = "english") -> List[str]:
    """Tokenize and lemmatize text.

    This function attempts to use NLTK if available. If NLTK is not
    installed, a very small fallback tokenizer is used (split on whitespace,
    lowercase and remove short tokens) so tests can run without full NLP
    dependencies.
    """
    ensure_nltk()
    if _nltk_ready:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        stops = set(stopwords.words(lang))
        tokens = nltk.word_tokenize(text)
        tokens = [t.lower() for t in tokens if any(c.isalnum() for c in t)]
        tokens = [t for t in tokens if t not in stops]
        lemmas = [lemmatizer.lemmatize(t) for t in tokens]
        return lemmas
    else:
        # lightweight fallback
        toks = [t.lower() for t in re.findall(r"\w+", text) if len(t) > 1]
        # small builtin stopword list to mimic NLTK behavior for tests
        STOPWORDS = {
            'the','and','is','in','it','of','to','a','an','that','this','on','for','with','as','are','was','were','be','by','or','from','at','which','but','not'
        }
        toks = [t for t in toks if t not in STOPWORDS]
        return toks
