import streamlit as st
import joblib

# Load model & labels
model = joblib.load("news_model.pkl")
labels = joblib.load("labels.pkl")

st.title("📰 AI News Article Categorizer")
st.write("Paste any news article below and the AI will classify it.")

text = st.text_area("Enter news article text here")

if st.button("Categorize"):
    if len(text.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        probs = model.predict_proba([text])[0]
        idx = probs.argmax()
        confidence = probs[idx]

        st.success(f"### Category: **{labels[idx]}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        st.write("### Confidence Breakdown")
        for i, lbl in enumerate(labels):
            st.write(f"{lbl}: {probs[i]:.2f}")
