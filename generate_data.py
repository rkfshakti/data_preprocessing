"""Generate synthetic domain-specific noisy NLP data.

Usage:
- generate(n, out_file) writes a CSV with columns: text,label,domain,timestamp
- generate_for_domain(domain, n, out_file) creates a CSV containing only that domain

This file intentionally keeps generation deterministic enough for demo purposes.
"""
import random
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import os
from tqdm import tqdm

# Domain vocabularies
BANKING_VOCAB = {
    "nouns": ["account", "transaction", "loan", "credit card", "fraud", "payment", "deposit", "balance", "statement", "interest rate", "mortgage", "overdraft"],
    "verbs": ["transferred", "deposited", "withdrew", "charged", "approved", "denied", "investigated", "resolved", "disputed", "flagged"],
    "adjectives": ["suspicious", "unauthorized", "pending", "declined", "successful", "high-value", "urgent", "erroneous"],
    "positive_keywords": ["approved", "resolved", "successful", "deposit", "helpful"],
    "negative_keywords": ["fraud", "denied", "disputed", "overdraft", "suspicious", "unauthorized", "erroneous"],
    "templates": [
        "A {adjective} transaction of ${amount} on my {noun} was {verb}.",
        "I am writing to report a {adjective} issue with my {noun}.",
        "My recent {noun} for ${amount} has not been processed correctly.",
        "There is a {adjective} hold on my {noun} that needs to be {verb}.",
        "I would like to inquire about the {noun} on my savings account.",
        "The {noun} was {verb} due to insufficient funds.",
    ],
}

TELECOM_VOCAB = {
    "nouns": ["network", "signal", "data plan", "billing", "roaming", "customer service", "dropped call", "outage", "SIM card", "coverage", "invoice"],
    "verbs": ["disconnected", "activated", "upgraded", "charged", "resolved", "escalated", "reported", "throttled", "restarted"],
    "adjectives": ["poor", "excellent", "unstable", "intermittent", "unlimited", "overpriced", "helpful", "frustrating"],
    "positive_keywords": ["excellent", "activated", "upgraded", "resolved", "helpful", "unlimited"],
    "negative_keywords": ["poor", "unstable", "overpriced", "frustrating", "outage", "throttled", "dropped call"],
    "templates": [
        "My {noun} has been {adjective} for the past few hours.",
        "I was {verb} an incorrect amount on my recent {noun}.",
        "The {adjective} {noun} in my area is causing frequent dropped calls.",
        "I need to speak with {noun} about a {adjective} charge.",
        "My mobile {noun} was suddenly {verb} without any warning.",
        "Can you check if there is a service {noun} in my location?",
    ],
}

TRAVEL_VOCAB = {
    "nouns": ["flight", "hotel", "booking", "reservation", "luggage", "check-in", "destination", "itinerary", "seat", "refund", "voucher"],
    "verbs": ["booked", "cancelled", "delayed", "rescheduled", "confirmed", "upgraded", "lost", "found", "enjoyed"],
    "adjectives": ["wonderful", "terrible", "non-refundable", "first-class", "scenic", "overbooked", "convenient", "disappointing"],
    "positive_keywords": ["wonderful", "first-class", "enjoyed", "upgraded", "convenient", "confirmed"],
    "negative_keywords": ["terrible", "cancelled", "delayed", "lost", "overbooked", "disappointing", "non-refundable"],
    "templates": [
        "Our {noun} to {destination} was {verb} and the experience was {adjective}.",
        "I have a {adjective} {noun} at the Grand Hyatt that I need to get a {noun} for.",
        "The airline {verb} my {noun} and offered a {adjective} travel {noun}.",
        "My {noun} was {verb} at the airport and I need assistance.",
        "The {noun} process was {adjective} and the staff were very helpful.",
        "I want to confirm my {noun} for the {adjective} {noun} to Paris.",
    ],
}

DOMAINS = {
    "banking": BANKING_VOCAB,
    "telecom": TELECOM_VOCAB,
    "travel": TRAVEL_VOCAB,
}

DESTINATIONS = ["Paris", "Tokyo", "New York", "London", "Sydney", "Rome", "Dubai"]


def make_sentence(domain: str, sentiment: str) -> str:
    vocab = DOMAINS[domain]
    # pick a template that is consistent with the requested sentiment when possible
    templates = vocab.get("templates", [])
    import re

    def template_contains_keyword(template: str, keyword: str) -> bool:
        # match each word of the keyword in the template allowing suffixes (e.g., 'call' matches 'calls')
        tpl = template.lower()
        for w in keyword.lower().split():
            # word boundary followed by the word and optional word characters
            if not re.search(r"\b" + re.escape(w) + r"\w*", tpl):
                return False
        return True

    if sentiment == "positive":
        # avoid templates that include known negative keywords (robust match)
        templates = [t for t in templates if not any(template_contains_keyword(t, kw) for kw in vocab.get("negative_keywords", []))]
    elif sentiment == "negative":
        # prefer templates that mention negative keywords if available
        neg_t = [t for t in templates if any(template_contains_keyword(t, kw) for kw in vocab.get("negative_keywords", []))]
        if neg_t:
            templates = neg_t

    template = random.choice(templates) if templates else random.choice(vocab.get("templates", []))

    # prefer sentiment keywords when possible
    # choose a specific token appropriate for the template slot
    if sentiment == "positive" and vocab.get("positive_keywords"):
        chosen = random.choice(vocab["positive_keywords"])
    elif sentiment == "negative" and vocab.get("negative_keywords"):
        chosen = random.choice(vocab["negative_keywords"])
    else:
        chosen = None

    # attempt to place the chosen word into a matching slot (noun/verb/adj)
    # avoid sampling tokens from the opposite sentiment pool when possible
    avoid_for_positive = set(vocab.get("negative_keywords", []))
    avoid_for_negative = set(vocab.get("positive_keywords", []))

    def sample(slot: str, avoid: set):
        candidates = [x for x in vocab.get(slot, []) if x not in avoid]
        if not candidates:
            candidates = vocab.get(slot, [])
        return random.choice(candidates)

    if chosen and chosen in vocab.get("nouns", []):
        noun = chosen
        verb = sample("verbs", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
        adj = sample("adjectives", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
    elif chosen and chosen in vocab.get("verbs", []):
        verb = chosen
        noun = sample("nouns", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
        adj = sample("adjectives", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
    elif chosen and chosen in vocab.get("adjectives", []):
        adj = chosen
        noun = sample("nouns", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
        verb = sample("verbs", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
    else:
        noun = sample("nouns", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
        verb = sample("verbs", avoid_for_positive if sentiment == "positive" else avoid_for_negative)
        adj = sample("adjectives", avoid_for_positive if sentiment == "positive" else avoid_for_negative)

    amount = round(random.uniform(10, 5000), 2)
    destination = random.choice(DESTINATIONS)

    sentence = template.format(noun=noun, verb=verb, adjective=adj, amount=amount, destination=destination)

    # add noise sometimes
    if random.random() > 0.85:
        sentence = sentence + random.choice(["...", "!!!", " ?!?"])
    if random.random() > 0.95:
        sentence = sentence.upper()

    if random.random() > 0.9:
        # insert a small filler
        fillers = [" umm ", " like ", " you know ", " ... "]
        pos = random.randint(0, max(0, len(sentence) - 1))
        sentence = sentence[:pos] + random.choice(fillers) + sentence[pos:]

    return sentence


def generate(n: int, out_file: str, start_days: int = 90, domain: str | None = None) -> None:
    """Generate n rows and write to out_file. If domain is provided, only generate that domain."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    data = []
    domains = [domain] if domain else list(DOMAINS.keys())
    start_date = datetime.now() - timedelta(days=start_days)

    for i in tqdm(range(n)):
        dom = random.choice(domains)
        label = random.choices(["positive", "negative", "neutral"], weights=[0.35, 0.4, 0.25])[0]
        text = make_sentence(dom, label)
        fraction = i / max(1, n - 1)
        timestamp = start_date + timedelta(days=fraction * start_days, minutes=random.randint(0, 1439))
        # add occasional reference or ticket noise, but avoid attaching to positive examples too frequently
        if random.random() > 0.92 and label != "positive":
            text = f"[REF-{random.randint(1000,9999)}] " + text
        if random.random() > 0.97 and label != "positive":
            text = text + f" (ticket:{random.randint(100000,999999)})"
        data.append({"text": text, "label": label, "domain": dom, "timestamp": timestamp})

    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)
    print(f"Wrote {len(df)} rows to {out_file}")


def generate_for_domain(domain: str, n: int, out_file: str) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    generate(n=n, out_file=out_file, domain=domain)


if __name__ == "__main__":
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    DATA_FILE = os.path.join(DATA_DIR, "synthetic_texts.csv")
    generate(n=1500, out_file=DATA_FILE)
