"""
app.py — Streamlit Web Application for PR 1: Airline Tweet Sentiment Analyzer
Red & White Skill Education — Natural Language Processing (PR 1)

Implements Task 8.3 & Task 8.4:
- st.set_page_config(page_title='Airline Tweet Sentiment Analyzer')
- Sidebar with project overview, dataset citation, and approach selector (Heuristic / VADER / ML Model)
- 3 Example tweet buttons that populate the input text area
- @st.cache_resource model loading (with auto-training fallback if sentiment_model.joblib is not yet saved)
- Robustness handling: empty input warning, >500 character truncation notice, emoji-only / URL-only empty-after-cleaning handling, safe NLTK download at start-up
- Coloured sentiment banner (st.success / st.warning / st.error), class probability bar chart (st.bar_chart), and collapsible POS tagger inspector (st.expander)
"""

import os
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st

from nlp_utils import (
    clean_text,
    ensure_nltk_resources,
    heuristic_sentiment_details,
)

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Airline Tweet Sentiment Analyzer",
    page_icon="✈️",
    layout="wide",
)

# 2. Safe NLTK Resource Initialization at Start-up (Task 8.4)
@st.cache_resource
def init_nltk() -> bool:
    ensure_nltk_resources()
    return True

init_nltk()


# 3. Cached Model & VADER Loader (Task 8.3)
@st.cache_resource
def load_ml_model():
    """
    Load 'sentiment_model.joblib' produced by NLP_PR1.ipynb.
    If not found locally but 'Tweets.csv' is present, train and save it automatically.
    Otherwise build a lightweight fallback pipeline trained on representative airline tweets
    so the app never crashes during testing.
    """
    model_path = "sentiment_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path), "Loaded pre-trained sentiment_model.joblib"

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    if os.path.exists("Tweets.csv"):
        df = pd.read_csv("Tweets.csv").drop_duplicates(subset=["text", "airline_sentiment"])
        df["clean_text"] = df["text"].astype(str).apply(clean_text)
        df.loc[df["clean_text"].str.strip() == "", "clean_text"] = "empty_tweet"
        X_tr, _, y_tr, _ = train_test_split(
            df["clean_text"],
            df["airline_sentiment"],
            test_size=0.2,
            random_state=42,
            stratify=df["airline_sentiment"],
        )
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        joblib.dump(pipe, model_path)
        return pipe, "Trained from Tweets.csv and saved to sentiment_model.joblib"

    # Fallback bootstrap corpus if neither sentiment_model.joblib nor Tweets.csv is in the directory yet
    bootstrap_data = [
        ("thank you so much great flight and friendly crew", "positive"),
        ("awesome service love flying with you best airline", "positive"),
        ("smooth landing early arrival polite staff thanks", "positive"),
        ("wonderful upgrade comfortable seats appreciate the help", "positive"),
        ("flight delayed hours rude customer service lost luggage", "negative"),
        ("cancelled flight stuck at gate waiting for hours terrible", "negative"),
        ("not happy with the delay worst airline experience never again", "negative"),
        ("on hold for two hours unhelpful agent missed connection", "negative"),
        ("what time does boarding start for flight tomorrow", "neutral"),
        ("can i change my seat assignment at the gate", "neutral"),
        ("does flight from chicago to miami have wifi", "neutral"),
        ("checking baggage allowance policy for domestic flights", "neutral"),
    ]
    X_boot = [x for x, _ in bootstrap_data]
    y_boot = [y for _, y in bootstrap_data]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_boot, y_boot)
    return pipe, "Fallback Demo Pipeline (Run NLP_PR1.ipynb to export full sentiment_model.joblib)"


@st.cache_resource
def load_vader():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# 4. Sidebar — Project Metadata & Approach Selector (Task 8.1 & 8.3)
with st.sidebar:
    st.title("✈️ About This Project")
    st.markdown(
        """
        **Practical Report 1 (PR 1)**  
        *Red & White Skill Education — NLP Fundamentals*

        **Dataset:**  
        [Twitter US Airline Sentiment (Kaggle)](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)  
        *(~14,640 real tweets across 6 US airlines: Negative, Neutral, Positive)*
        """
    )

    st.subheader("⚙️ Select NLP Approach")
    approach = st.radio(
        "Choose a sentiment engine to compare:",
        options=[
            "ML Pipeline (TF-IDF Bigrams + Logistic Regression)",
            "API-Based Ready-Made Engine (NLTK VADER)",
            "Heuristic Approach (Domain Lexicon + Negation Rules)",
        ],
        index=0,
    )

    st.subheader("🧪 Robustness Test Cases")
    st.caption(
        "Try testing edge cases:\n"
        "- Empty string\n"
        "- Only emojis (`✈️😡👍`)\n"
        "- URL-only (`https://t.co/xyz123`)\n"
        "- Negation (`not happy with the delay`)\n"
        "- All-caps angry complaint"
    )


# 5. Main Header & Example Tweet Buttons (Task 8.3)
st.title("✈️ US Airline Tweet Sentiment Analyzer")
st.markdown(
    "Analyse raw passenger tweets using **RegEx cleaning**, **Heuristic rules**, **NLTK VADER**, "
    "or our trained **TF-IDF (1,2) + Balanced Logistic Regression** machine learning pipeline."
)

if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = (
        "@VirginAmerica thank you for the amazing flight and super friendly cabin crew! #GreatService"
    )

st.markdown("##### Quick-Fill Example Tweets:")
col_ex1, col_ex2, col_ex3 = st.columns(3)

with col_ex1:
    if st.button("😊 Positive Example", use_container_width=True):
        st.session_state["tweet_input"] = (
            "@JetBlue thank you so much! Smooth flight, landed 20 minutes early, and the crew was awesome!"
        )
with col_ex2:
    if st.button("😡 Negative + Negation Example", use_container_width=True):
        st.session_state["tweet_input"] = (
            "@united I am NOT happy with the 4 hour delay!! Cancelled flight and lost my luggage https://t.co/bad123"
        )
with col_ex3:
    if st.button("😐 Neutral Inquiry Example", use_container_width=True):
        st.session_state["tweet_input"] = (
            "@AmericanAir what time does boarding begin for flight 415 from Chicago to Dallas tomorrow?"
        )

# 6. Tweet Input Text Area
raw_input = st.text_area(
    "Enter an Airline Tweet (up to 500 characters):",
    value=st.session_state["tweet_input"],
    height=120,
    placeholder="Type or paste a raw tweet here (e.g., @united flight delayed again, not happy!)...",
)

predict_clicked = st.button("🔍 Predict Sentiment", type="primary")

# 7. Prediction & Robustness Logic (Task 8.3 & Task 8.4)
if predict_clicked:
    # Edge Case 1: Empty or whitespace-only input
    if not raw_input or not raw_input.strip():
        st.warning("⚠️ Please enter a non-empty tweet before pressing Predict.")
        st.stop()

    # Edge Case 2: Very long input (> 500 characters) -> Truncate safely with notice
    working_text = raw_input
    if len(working_text) > 500:
        st.info(
            f"ℹ️ Your input was {len(working_text)} characters long and has been truncated to the first 500 characters."
        )
        working_text = working_text[:500]

    # Clean text with shared RegEx pipeline
    cleaned = clean_text(working_text)

    st.subheader("1️⃣ RegEx Cleaning Result")
    st.code(cleaned if cleaned else "[Empty after removing URLs, @mentions, emojis, and punctuation]", language="text")

    # Edge Case 3: Input becomes completely empty after RegEx cleaning (e.g., emoji-only or URL-only tweet)
    if not cleaned:
        st.warning(
            "⚠️ After removing `@mentions`, URLs, emojis, and non-alphabetic characters, no words remained! "
            "Returning **Neutral** by default (unless evaluating raw text with VADER)."
        )

    st.subheader("2️⃣ Sentiment Prediction")

    # Branch by chosen approach
    if approach.startswith("ML Pipeline"):
        model, status_msg = load_ml_model()
        st.caption(f"Model Status: {status_msg}")

        if not cleaned:
            pred_label = "neutral"
            proba_dict = {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
        else:
            pred_label = str(model.predict([cleaned])[0])
            probas = model.predict_proba([cleaned])[0]
            proba_dict = {str(cls): float(p) for cls, p in zip(model.classes_, probas)}

    elif approach.startswith("API-Based"):
        sia = load_vader()
        scores = sia.polarity_scores(working_text)
        compound = scores["compound"]
        if compound >= 0.05:
            pred_label = "positive"
        elif compound <= -0.05:
            pred_label = "negative"
        else:
            pred_label = "neutral"
        proba_dict = {
            "negative": float(scores["neg"]),
            "neutral": float(scores["neu"]),
            "positive": float(scores["pos"]),
        }
        st.caption(f"VADER Compound Score on raw text: `{compound:.4f}` (Thresholds: `>= 0.05` Positive, `<= -0.05` Negative)")

    else:
        pred_label, pos_cnt, neg_cnt, rule_matches = heuristic_sentiment_details(working_text)
        total_hits = max(pos_cnt + neg_cnt, 1)
        proba_dict = {
            "negative": round(neg_cnt / total_hits, 3) if (pos_cnt + neg_cnt) > 0 else 0.0,
            "neutral": 1.0 if pos_cnt == neg_cnt else 0.0,
            "positive": round(pos_cnt / total_hits, 3) if (pos_cnt + neg_cnt) > 0 else 0.0,
        }
        st.caption(
            f"Heuristic Scores → Positive: `{pos_cnt}`, Negative: `{neg_cnt}` | "
            f"Matched Rules: `{', '.join(rule_matches) if rule_matches else 'None (Neutral default)'}`"
        )

    # Display Coloured Result Banner (st.success / st.warning / st.error)
    if pred_label == "positive":
        st.success("### ✅ Predicted Sentiment: POSITIVE 😊")
    elif pred_label == "negative":
        st.error("### 🚨 Predicted Sentiment: NEGATIVE 😡")
    else:
        st.warning("### ⚖️ Predicted Sentiment: NEUTRAL 😐")

    # Display Class Probability / Score Bar Chart (st.bar_chart)
    st.subheader("3️⃣ Class Probability / Polarity Breakdown")
    chart_df = pd.DataFrame(
        {"Probability / Score": [proba_dict.get("negative", 0.0), proba_dict.get("neutral", 0.0), proba_dict.get("positive", 0.0)]},
        index=["Negative", "Neutral", "Positive"],
    )
    st.bar_chart(chart_df)

    # Collapsible POS Tagging Section (st.expander)
    with st.expander("🔤 View Part-of-Speech (POS) Tags of Cleaned Tweet (NLTK)"):
        if not cleaned:
            st.info("No alphabetic tokens available to tag after RegEx cleaning.")
        else:
            tokens = cleaned.split()
            try:
                pos_tags = nltk.pos_tag(tokens)
            except Exception:
                pos_tags = [(t, "N/A") for t in tokens]

            pos_Meanings = {
                "NN": "Noun, singular",
                "NNS": "Noun, plural",
                "NNP": "Proper noun",
                "VB": "Verb, base form",
                "VBD": "Verb, past tense",
                "VBG": "Verb, gerund/present participle",
                "VBN": "Verb, past participle",
                "VBP": "Verb, non-3rd person singular present",
                "VBZ": "Verb, 3rd person singular present",
                "JJ": "Adjective",
                "JJR": "Adjective, comparative",
                "JJS": "Adjective, superlative",
                "RB": "Adverb",
                "RBR": "Adverb, comparative",
                "PRP": "Personal pronoun",
                "PRP$": "Possessive pronoun",
                "IN": "Preposition / subordinating conjunction",
                "DT": "Determiner",
                "MD": "Modal auxiliary",
                "CC": "Coordinating conjunction",
            }
            pos_df = pd.DataFrame([
                {
                    "Token": word,
                    "Penn Treebank Tag": tag,
                    "Grammatical Role": pos_Meanings.get(tag, "Other POS Tag"),
                }
                for word, tag in pos_tags
            ])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
