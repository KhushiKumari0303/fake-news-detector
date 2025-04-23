
import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load('logistic_fake_news.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Streamlit UI
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter the news content:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    st.markdown(f"### 🔍 Prediction: **{label}**")
