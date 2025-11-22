import streamlit as st
import os
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# File paths
# -------------------------
MODEL_FILE = "news_model.pkl"
LABELS_FILE = "labels.pkl"

# -------------------------
# Function to train model
# -------------------------
def train_and_save_model():
    categories = [
        'rec.sport.baseball',
        'sci.space',
        'talk.politics.mideast',
        'comp.sys.mac.hardware'
    ]

    data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers","footers","quotes"))
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(data.target_names, LABELS_FILE)

    return model, data.target_names

# -------------------------
# Load or train model
# -------------------------
if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
    model = joblib.load(MODEL_FILE)
    labels = joblib.load(LABELS_FILE)
else:
    with st.spinner("Training model (this may take ~1 minute)..."):
        model, labels = train_and_save_model()
        st.success("Model trained and saved!")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="📰 AI News Categorizer", layout="wide")
st.title("📰 AI News Article Categorizer")
st.markdown("Paste any news article below and the AI will classify it automatically.")

text = st.text_area("Enter news article text here", height=200)

if st.button("Categorize"):
    if len(text.strip()) == 0:
        st.warning("Please enter some text to categorize.")
    else:
        probs = model.predict_proba([text])[0]
        idx = probs.argmax()
        confidence = probs[idx]

        st.success(f"### Category: **{labels[idx]}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        st.write("### All Categories Probabilities")
        for i, lbl in enumerate(labels):
            st.write(f"{lbl}: {probs[i]:.2f}")
