import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Page config
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("🌀 Sentiment Analysis App (TensorFlow)")

# Load model
try:
    model = tf.keras.models.load_model("sentiment_tf_model.keras")
except Exception as e:
    st.error(f"TensorFlow model not found: {e}")
    st.stop()

# Load TF-IDF
try:
    with open("tfidf_vectorizer2.pkl", "rb") as f:
        tfidf = pickle.load(f)
except Exception as e:
    st.error(f"TF-IDF vectorizer not found: {e}")
    st.stop()

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\S*\d\S*', '', text)
    text = re.sub(r'(https|http)?:\/\/\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text

# UI
st.subheader("Enter a sentence:")
user_input = st.text_area(
    "Text",
    height=120,
    placeholder="Eg: Wow, another meeting that could have been an email."
)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess_text(user_input)
        vectorized = tfidf.transform([processed]).toarray()
        try:
            prediction = model.predict(vectorized)
            label = np.argmax(prediction)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.subheader("Prediction Result")
        if label == 2:
            st.success("😊 Positive")
        elif label == 1:
            st.info("😐 Neutral")
        else:
            st.error("😞 Negative")

st.markdown("---")
st.markdown("Built with ❤️ using **TensorFlow & Streamlit**")
