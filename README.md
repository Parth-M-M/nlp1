# ✈️ US Airline Tweet Sentiment Analyzer — NLP Practical Report 1 (PR 1)

**Live Streamlit App URL:** `[Paste your deployed https://your-app-name.streamlit.app URL here after deploying in Step 4 below]`  
**Recorded Video Explanation (Face + Screen):** `[Paste your Google Drive (Anyone with link) or Unlisted YouTube link here]`  
**Institute:** Red & White Skill Education · Subject: Natural Language Processing (NLP Fundamentals)  
**Dataset Source:** [Twitter US Airline Sentiment (CrowdFlower / Kaggle)](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (`Tweets.csv`)

## 1. Project Overview & Pipeline Summary

This project builds an end-to-end **3-class Sentiment Analysis system** (`negative`, `neutral`, `positive`) on ~14,640 real passenger tweets covering six major US airlines from February 2015. Because raw tweets contain `@mentions`, URLs, hashtags, emojis, slang, and negations (`"not happy"`, `"never again"`), and suffer from strong class imbalance (~63% negative, ~21% neutral, ~16% positive), we evaluate the full NLP pipeline using **Macro-F1** as our primary metric:

1. **Data Collection & Deduplication (Task 1):** Load `Tweets.csv`, inspect missing values and exact duplicates, drop duplicate tweets before splitting to prevent data leakage, and analyse class imbalance and real-world NLP challenges.
2. **RegEx Text Cleaning & Negation-Preserving Tokenisation (Task 2):** Lowercase text, remove URLs (`https?://\S+`) and `@mentions` (`@\w+`), convert `#hashtags` into words, expand negation contractions (`n't` $\rightarrow$ `not`), strip punctuation/numbers (`[^a-z\s]`), and remove English stop-words while **explicitly preserving negation words** (`not`, `no`, `nor`, `never`).
3. **Heuristic & API-Based Sentiment Approaches (Task 3):** Compare a custom 74-word airline domain lexicon with a 2-word lookback negation flip (`heuristic_sentiment`) against **NLTK VADER** (`SentimentIntensityAnalyzer`) on the held-out 20% stratified test split.
4. **Feature Extraction — BoW, TF-IDF & N-Grams (Task 4):** Verify Bag-of-Words and TF-IDF ($\text{tf} \times (\ln(N/\text{df}) + 1)$) manually from scratch against `scikit-learn` (`np.allclose == True`), fit vectorisers strictly on training data, and evaluate unigram `(1,1)`, bigram `(1,2)`, and trigram `(1,3)` trade-offs.
5. **Word Embeddings — Word2Vec CBoW & Skip-gram (Task 5):** Train 100-dimensional CBoW (`sg=0`) and Skip-gram (`sg=1`) models with Gensim (`seed=42, workers=1`), compare nearest semantic neighbours and cosine similarities, project the top 60 words to 2-D with PCA, and build averaged document vectors.
6. **Supervised Text Classification & Model Selection (Task 6):** Compare 6 feature + classifier combinations alongside Heuristic and VADER, run 5-fold stratified cross-validation, analyse 8 test-set errors, and export the winning pipeline (`TfidfVectorizer(ngram_range=(1,2), min_df=2)` + `LogisticRegression(class_weight='balanced')`) to `sentiment_model.joblib`.
7. **Part-of-Speech (POS) Tagging (Task 7):** Tag tweets using NLTK's Averaged Perceptron tagger (`nltk.pos_tag`), demonstrate contextual disambiguation (`book`, `park`), plot POS tag distributions across sentiment classes, and compare POS-filtered features (`JJ*`, `RB*`, `VB*` + negations) against the full-text pipeline.
8. **Streamlit Web App Deployment (Task 8):** Deploy `app.py` sharing `nlp_utils.py` with interactive example buttons, approach selector, cached model loading (`@st.cache_resource`), probability bar chart, POS tag inspector, and edge-case robustness handling.

## 2. Benchmark Results Comparison Table (20% Stratified Test Split)

| Approach Family | Features | Classifier / Engine | Accuracy | Macro-F1 | Neg Precision | Pos Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Heuristic** | 74 Airline Domain Words + 2-Word Negation Flip | Hand-Crafted Rules (`heuristic_sentiment`) | ~0.61 | ~0.53 | ~0.81 | ~0.64 |
| **2. API-Based** | Social Media Lexicon + Punctuation/Caps Heuristics | NLTK VADER (`SentimentIntensityAnalyzer`) | ~0.54 | ~0.51 | ~0.89 | ~0.79 |
| **3. ML — BoW** | Bag-of-Words Unigrams (`min_df=2`) | Multinomial Naive Bayes | ~0.78 | ~0.69 | ~0.81 | ~0.67 |
| **3. ML — BoW** | Bag-of-Words Unigrams (`min_df=2`) | Logistic Regression (`class_weight='balanced'`) | ~0.78 | ~0.73 | ~0.88 | ~0.76 |
| **3. ML — TF-IDF** | TF-IDF Unigrams `(1,1)` (`min_df=2`) | Logistic Regression (`class_weight='balanced'`) | ~0.78 | ~0.73 | ~0.89 | ~0.78 |
| **3. ML — N-Grams (Deployed 🏆)** | **TF-IDF Unigrams + Bigrams `(1,2)` (`min_df=2`)** | **Logistic Regression (`class_weight='balanced'`)** | **~0.80** | **~0.75** | **~0.89** | **~0.79** |
| **3. ML — N-Grams** | TF-IDF Unigrams + Bigrams `(1,2)` (`min_df=2`) | Linear SVM (`class_weight='balanced'`) | ~0.80 | ~0.74 | ~0.87 | ~0.75 |
| **3. ML — Embeddings** | Word2Vec Skip-gram (100-d Averaged Vectors) | Logistic Regression (`class_weight='balanced'`) | ~0.72 | ~0.66 | ~0.86 | ~0.73 |
| **3. ML — POS-Filtered** | POS-Filtered (`JJ*`, `RB*`, `VB*` + Negations) TF-IDF `(1,2)` | Logistic Regression (`class_weight='balanced'`) | ~0.75 | ~0.69 | ~0.86 | ~0.74 |

*(Exact values are printed and plotted automatically when running `NLP_PR1.ipynb` on `Tweets.csv`.)*

## 3. Repository Structure

```text
├── NLP_PR1.ipynb            # Complete Jupyter Notebook (Tasks 1–7 + Task 8 Plan & Reflection)
├── NLP_PR1.html             # Exported HTML report of the executed notebook
├── app.py                   # Streamlit web application (Task 8)
├── nlp_utils.py             # Shared RegEx cleaning, stop-word & heuristic sentiment module
├── sentiment_model.joblib   # Saved scikit-learn Pipeline (TfidfVectorizer + LogisticRegression)
├── requirements.txt         # Pinned dependencies (scikit-learn==1.6.1)
├── README.md                # Project documentation, results, and deployment guide
└── screenshots/             # Saved EDA charts and local Streamlit app screenshots
```

## 4. How to Run Locally & Deploy on Streamlit Community Cloud

### Step 1: Clone & Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: `scikit-learn==1.6.1` is pinned in `requirements.txt` so the version used to train `sentiment_model.joblib` matches the deployment environment.)*

### Step 2: Download `Tweets.csv` & Run the Notebook
1. Download `Tweets.csv` from [Kaggle — Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) and place it in the project root directory.
2. Execute the notebook from top to bottom (**Kernel → Restart & Run All**) or via CLI:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace NLP_PR1.ipynb
   jupyter nbconvert --to html NLP_PR1.ipynb
   ```
   Running the notebook automatically generates `sentiment_model.joblib` and saves all analysis figures in `screenshots/`.

### Step 3: Run the Streamlit App Locally & Capture Screenshots (Task 8.5)
```bash
streamlit run app.py
```
Save 5 screenshots in `screenshots/`:
1. `screenshots/app_start_screen.png` — Main app interface on launch
2. `screenshots/app_positive_prediction.png` — Positive tweet prediction (`st.success`)
3. `screenshots/app_negative_prediction.png` — Negative + negation tweet prediction (`st.error`)
4. `screenshots/app_probability_chart.png` — Class probability bar chart (`st.bar_chart`)
5. `screenshots/app_pos_expander.png` — Expanded Part-of-Speech (POS) tags table (`st.expander`)

### Step 4: Recommended Git Commits (Criterion C3 — 3+ Meaningful Commits)
```bash
git init
git add nlp_utils.py NLP_PR1.ipynb
git commit -m "Add RegEx cleaning pipeline, shared nlp_utils, and EDA tasks"

git add NLP_PR1.ipynb NLP_PR1.html sentiment_model.joblib screenshots/
git commit -m "Add BoW, TF-IDF, N-gram, Word2Vec and POS experiments with saved model"

git add app.py requirements.txt README.md screenshots/
git commit -m "Add Streamlit app, pinned requirements, and deployment documentation"
```

### Step 5: Deploy on Streamlit Community Cloud (Task 8.7)
1. Push your repository to a **public GitHub repo**.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with GitHub, click **New app**, select your repository, branch (`main`), and main file path (`app.py`), and click **Deploy**.
3. Copy the live URL and paste it at the very top of this `README.md`.

## 5. Screenshots Gallery

| Screenshot | Description |
| :--- | :--- |
| `screenshots/app_start_screen.png` | Start screen showing sidebar approach selector and 3 example buttons |
| `screenshots/app_positive_prediction.png` | Green `st.success` banner predicting a positive praise tweet |
| `screenshots/app_negative_prediction.png` | Red `st.error` banner correctly handling negation (`"not happy with the delay"`) |
| `screenshots/app_probability_chart.png` | Calibrated class probability bar chart from the ML pipeline |
| `screenshots/app_pos_expander.png` | NLTK Penn Treebank POS tags inside the collapsible `st.expander` |

## 6. Video Explanation Guide (5–10 Minutes, Face PiP + Screen — Component B)

Save your recording as `NLP_PR1_YourName_GRID.mp4` and cover these 8 checkpoints verbally while showing your notebook outputs and live Streamlit app:
1. **NLP Pipeline Stages:** Walk through Data Collection $\rightarrow$ RegEx Cleaning $\rightarrow$ Tokenisation $\rightarrow$ Stop-Word Handling $\rightarrow$ Feature Extraction $\rightarrow$ Model Training $\rightarrow$ Evaluation $\rightarrow$ Deployment and why each stage is required.
2. **RegEx Cleaning & Keeping Negations:** Explain `https?://\S+`, `@\w+`, `#(\w+)`, and `[^a-z\s]`, and show why removing `"not"` from `"not happy with the delay"` flips a complaint into `"happy delay"`.
3. **Heuristic vs. API-Based (VADER) vs. ML Approach:** Show your test-set comparison table and explain why supervised ML outperforms static rules and general-purpose VADER on airline complaints.
4. **BoW vs. TF-IDF:** Show your manual calculation matching `TfidfVectorizer(smooth_idf=False, norm=None)` and explain why ubiquitous words like `"flight"` receive the lowest IDF weight ($\ln(N/\text{df}) + 1 = 1.0$).
5. **What N-Grams Add & What They Cost:** Show how bigrams capture `"not happy"` and `"customer service"` while trigrams `(1,3)` cause vocabulary explosion and high memory usage.
6. **Word2Vec (CBoW vs. Skip-gram):** Explain how dense 100-d vectors group words by context (*"a word is known by the company it keeps"*), contrast CBoW (context $\rightarrow$ centre) with Skip-gram (centre $\rightarrow$ context), and show the 2-D PCA plot.
7. **How POS Tagging Uses Context:** Show how NLTK's Averaged Perceptron tagger disambiguates `"book"` or `"park"` as a Verb (`VB`) vs. Noun (`NN`) using surrounding words.
8. **Why Vectorisers Are Fitted on Training Data Only + Live Streamlit Demo:** Explain how fitting on test data leaks vocabulary and IDF statistics, then test 3 live inputs (including negation/sarcasm) on your deployed Streamlit URL.
