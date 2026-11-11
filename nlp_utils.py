"""
nlp_utils.py — Shared NLP utilities for PR 1: Airline Tweet Sentiment Analyzer.

Contains:
- Safe NLTK resource downloader (`ensure_nltk_resources`)
- Regular Expression text cleaning (`clean_text`)
- Tokenisation with negation-preserving stop-word removal (`tokenize_keep_negations`)
- Domain-specific airline sentiment lexicon (`POSITIVE_WORDS`, `NEGATIVE_WORDS`, `NEGATION_WORDS`)
- Rule-based heuristic sentiment classifier with 2-word negation window (`heuristic_sentiment`)
"""

import re
from typing import Dict, List, Set, Tuple

# Negation words to preserve during stop-word removal and use for polarity flipping
NEGATION_WORDS: Set[str] = {
    "no", "not", "nor", "never", "neither", "nobody", "none", "nothing",
    "nowhere", "hardly", "scarcely", "barely", "dont", "cant", "wont",
    "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    "doesnt", "didnt", "wouldnt", "shouldnt", "couldnt", "mustnt"
}

# 36 positive domain words tailored to airline tweets (Task 3.2: 30-40 words)
POSITIVE_WORDS: Set[str] = {
    "thanks", "thank", "great", "awesome", "amazing", "best", "love",
    "good", "nice", "helpful", "friendly", "excellent", "smooth", "early",
    "ontime", "comfortable", "upgrade", "upgraded", "appreciate", "appreciated",
    "wonderful", "fantastic", "pleasant", "kudos", "impressed", "fast",
    "quick", "easy", "safe", "safely", "happy", "glad", "polite", "caring",
    "outstanding", "superb"
}

# 38 negative domain words tailored to airline tweets (Task 3.2: 30-40 words)
NEGATIVE_WORDS: Set[str] = {
    "delay", "delayed", "delays", "cancelled", "canceled", "cancel", "lost",
    "rude", "worst", "bad", "terrible", "horrible", "awful", "late", "stuck",
    "waiting", "wait", "waited", "hours", "stranded", "missed", "missing",
    "broken", "damaged", "unhelpful", "unacceptable", "pathetic", "disappointed",
    "disappointing", "frustrated", "frustrating", "ridiculous", "poor", "slow",
    "overbooked", "luggage", "baggage", "hold"
}


def ensure_nltk_resources() -> None:
    """Download required NLTK resources safely at start-up (Task 8.4)."""
    try:
        import nltk
        resources = [
            "stopwords",
            "punkt",
            "punkt_tab",
            "vader_lexicon",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        ]
        for res in resources:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
    except ImportError:
        pass


def clean_text(s: str) -> str:
    """
    Clean a raw airline tweet using sequential Regular Expressions (Task 2.1 - 2.3):
    1. Convert to lowercase
    2. Remove URLs (http://, https://, www.)
    3. Remove @mentions (@username)
    4. Convert #hashtags to plain words (keep word, drop '#')
    5. Remove HTML character entities (&amp;, &lt;, &gt;)
    6. Expand common contractions before stripping punctuation so negations survive
    7. Remove numbers and non-alphabetic punctuation ([^a-z\\s])
    8. Collapse repeated whitespace and strip leading/trailing spaces
    """
    if not isinstance(s, str) or not s.strip():
        return ""

    # 1. Lowercase first
    text = s.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # 4. Handle hashtags: keep the captured word, drop '#'
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. Remove common HTML entities present in Twitter exports
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;", " ", text)

    # 6. Normalize common negation contractions (e.g., "didn't" -> "did not", "can't" -> "can not")
    text = re.sub(r"won't\b", "will not", text)
    text = re.sub(r"can't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)

    # 7. Remove numbers and punctuation (keep only lowercase letters and whitespace)
    text = re.sub(r"[^a-z\s]", " ", text)

    # 8. Collapse multiple spaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_stop_words_keep_negations() -> Set[str]:
    """
    Return English stop-words with negation words removed (Task 2.4).
    Falls back to a built-in English stop-word list if NLTK is unavailable.
    """
    try:
        from nltk.corpus import stopwords
        stop_set = set(stopwords.words("english"))
    except Exception:
        stop_set = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
            "or", "because", "as", "until", "while", "of", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "should", "now"
        }
    return stop_set - NEGATION_WORDS


def tokenize_keep_negations(cleaned_text: str) -> List[str]:
    """
    Tokenise cleaned text and remove English stop-words while preserving negations (Task 2.4).
    """
    if not cleaned_text:
        return []
    stop_set = get_stop_words_keep_negations()
    try:
        import nltk
        tokens = nltk.word_tokenize(cleaned_text)
    except Exception:
        tokens = cleaned_text.split()
    return [tok for tok in tokens if tok not in stop_set and len(tok) > 1 or tok in {"no"}]


def heuristic_sentiment_details(text: str) -> Tuple[str, int, int, List[str]]:
    """
    Compute heuristic sentiment score with a 2-word lookback negation window (Task 3.2).

    Returns:
        (label, pos_score, neg_score, matched_Breakdown)
    """
    cleaned = clean_text(text)
    tokens = cleaned.split()
    pos_score = 0
    neg_score = 0
    matches: List[str] = []

    for i, token in enumerate(tokens):
        # Check if either of the previous two words is a negation word
        window_start = max(0, i - 2)
        prev_two = tokens[window_start:i]
        is_negated = any(w in NEGATION_WORDS for w in prev_two)

        if token in POSITIVE_WORDS:
            if is_negated:
                neg_score += 1
                matches.append(f"NEGATED_POS({prev_two[-1]}..{token} -> -1)")
            else:
                pos_score += 1
                matches.append(f"POS({token} -> +1)")
        elif token in NEGATIVE_WORDS:
            if is_negated:
                pos_score += 1
                matches.append(f"NEGATED_NEG({prev_two[-1]}..{token} -> +1)")
            else:
                neg_score += 1
                matches.append(f"NEG({token} -> -1)")

    if pos_score > neg_score:
        label = "positive"
    elif neg_score > pos_score:
        label = "negative"
    else:
        label = "neutral"

    return label, pos_score, neg_score, matches


def heuristic_sentiment(text: str) -> str:
    """
    Return only the sentiment label ('positive', 'negative', or 'neutral') for compatibility
    with Task 3.2 specification.
    """
    label, _, _, _ = heuristic_sentiment_details(text)
    return label
