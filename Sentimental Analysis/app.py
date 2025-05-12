import streamlit as st
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🛍️", layout="centered")

# Load the trained model
MODEL_PATH = "sentiment_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

# UI Elements
st.title("🛍️ Product Review Sentiment Analyzer")
st.write("Enter a product review below to predict its sentiment (positive, negative, or neutral).")

review = st.text_area("Product Review", height=150)

if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("⚠️ Please enter a review.")

    elif model is None:
        st.error("🚫 Model not found. Please train the model first by running review_classifier.py.")

    else:
        prediction = model.predict([review])

        if prediction == "positive" :
            st.success("👍 Sentiment: Positive")

        elif prediction == "Negative" :
            st.error("👎 Sentiment: Negative")

        else:
            st.info("😐 Sentiment: Neutral")
