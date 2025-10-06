import pytest

from src.preprocess import clean_text, tokenize_and_lemmatize


def test_clean_text_basic():
    raw = "<p>Hello!!! Visit https://x.com now. \n Newlines\n\n and   spaces</p>"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "Hello" in cleaned
    assert "\n" not in cleaned


def test_tokenize_and_lemmatize():
    tokens = tokenize_and_lemmatize("The cats are running in the garden.")
    # cats -> cat, running -> running or run depending on lemmatizer; ensure lowercased and no stopwords
    assert all(t.islower() for t in tokens)
    assert "the" not in tokens
