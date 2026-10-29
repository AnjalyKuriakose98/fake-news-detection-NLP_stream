
import re
import joblib
import streamlit as st
from pathlib import Path

# Page setup
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("Fake News Detection")
st.write("Enter a news article to get a machine-learning prediction.")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load model files from the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "best_model_week3.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_week3.pkl")

# User input
article = st.text_area(
    "Paste your news article here:",
    height=200
)

if st.button("Predict"):
    if not article.strip():
        st.warning("Please enter a news article.")
    else:
        cleaned_article = clean_text(article)
        article_tfidf = tfidf.transform([cleaned_article])
        prediction = model.predict(article_tfidf)[0]

        if str(prediction) == "0":
            st.error("Potentially FAKE News")
        elif str(prediction) == "1":
            st.success("Predicted REAL News")
        else:
            st.write("Model output:", prediction)

        st.caption(
            "This is a machine-learning prediction, not proof "
            "that the article is true or false."
        )
